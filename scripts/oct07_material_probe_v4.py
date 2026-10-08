"""Fourteen bounded material measurements on the actual native UAV camera."""
import hashlib
import io
import json
import math
from pathlib import Path
import time

CANDIDATES=('vehicle.ford.mustang','vehicle.lincoln.mkz_2017')

def place_diagnostic_camera(ac,airsim,name,local_position,local_origin,q,offset):
    from render_replay import vehicle_for_camera
    import numpy as np
    # Pinned native runtime queues simSetVehiclePose while paused. The existing
    # replay diagnostic already verified immediate GLOBAL-NED actor placement.
    body,body_q=vehicle_for_camera(np.asarray(local_position)+local_origin,q,offset,q)
    if not ac.simSetObjectPose(name,airsim.Pose(airsim.Vector3r(*body),
        airsim.Quaternionr(body_q[1],body_q[2],body_q[3],body_q[0])),teleport=True):
        raise RuntimeError('Immediate diagnostic actor placement failed')
    return dict(api='simSetObjectPose',frame='global_ned',body_position=body.tolist(),body_quaternion_wxyz=body_q.tolist())

def run_uav_material(args,ac,cc,calibration,local_origins):
    import airsim
    import carla
    import cv2
    import numpy as np
    from PIL import Image
    from audit_color_language import COLOR_RGB,cue_mask
    from pixel_visibility import PixelVisibilityAudit
    from render_replay import vehicle_for_camera,pose_residual
    from runtime_frame_probe import xyz,save
    from renderer_weather import verify_static_visual_wind
    out=Path(args.output)/'material_rgb';out.mkdir(exist_ok=False)
    world=cc.get_world();library=world.get_blueprint_library();catalog={b.id for b in library}
    plans=[(model,color,z) for model in CANDIDATES for color in ('red','blue','white') for z in (26.,70.)]
    plans += [('vehicle.mercedes.sprinter','yellow',z) for z in (26.,70.)]
    rows=[];owned=[];original_weather=world.get_weather();original_settings=world.get_settings()
    inherited=None
    if getattr(args,'resume_receipt',None):
        from oct07_material_resume import inherited_prefix,link_prefix
        rows,files,inherited,binding=inherited_prefix(args.resume_receipt,plans)
        link_prefix(files,out)
    result=dict(run_class='renderer_uav_material_diagnostic',planned_frames=14,completed_frames=len(rows),
        attempts=rows,native_uav_camera=True,exact_actor_mask=True,benchmark_admitted=False,
        candidates=list(CANDIDATES),native_catalog=sorted(catalog),started_at=time.time(),
        asset_selection_uses_algorithm_scores=False,camera_settings_readback=['pose','fov'],
        gamma_and_exposure_readback=None)
    if hasattr(ac,'getSettingsString'):
        loaded=json.loads(ac.getSettingsString())
        if any(c.get('ImageType')==-1 for c in loaded.get('CameraDefaults',{}).get('CaptureSettings',[])):
            from oct07_native_exposure_profile import verify_loaded
            result['gamma_and_exposure_readback']=verify_loaded(loaded)
            result['loaded_settings_sha256']=hashlib.sha256(json.dumps(loaded,sort_keys=True).encode()).hexdigest()
    if inherited is not None:
        if result.get('loaded_settings_sha256')!=inherited.get('loaded_settings_sha256'):
            raise ValueError('Loaded camera/exposure differs from inherited prefix')
        result.update(inherited_measurements=7,new_measurement_limit=7,new_measurements=0,
            inherited_source_result_sha256=binding['source_result_sha256'],
            reused_files_sha256=binding['reused_files_sha256'])
    def write():result['observed_at']=time.time();save(out/'result.json',result)
    write();origin=np.asarray(calibration['world_origin_m']);axis=np.array([1.,1.,-1.]);name='UAV_0'
    mount=airsim.to_quaternion(-math.pi/2,0,0);q=np.array([mount.w_val,mount.x_val,mount.y_val,mount.z_val]);offset=np.array([.5,0,.1])
    ac.simSetCameraPose('0',airsim.Pose(airsim.Vector3r(*offset),mount),vehicle_name=name)
    point=world.get_map().get_spawn_points()[0]
    deadline=time.monotonic()+min(1500,getattr(args,'probe_timeout',1800)-120)
    try:
        for index,(model,color,height) in enumerate(plans):
            if inherited is not None and index<7:continue
            row=dict(index=index,blueprint=model,color=color,height=height,status='preparing');rows.append(row);write()
            if time.monotonic()>=deadline:raise TimeoutError('Original material diagnostic deadline exhausted')
            if model not in catalog:row.update(status='blueprint_absent');write();continue
            ac.simPause(False)
            pixel=PixelVisibilityAudit(ac);before=set(ac.simListSceneObjects());bp=library.find(model)
            rgb=COLOR_RGB[color];bp.set_attribute('color',','.join(map(str,rgb)))
            pose=carla.Transform(carla.Location(x=point.location.x,y=point.location.y,z=point.location.z+.3),point.rotation)
            actor=world.try_spawn_actor(bp,pose)
            if actor is None:raise RuntimeError('Fixed native spawn failed; no favorable-location search')
            owned.append(actor);actor.set_simulate_physics(False)
            for _ in range(3):world.wait_for_tick(20)
            pixel.register('target',set(ac.simListSceneObjects())-before)
            vertices=actor.bounding_box.get_world_vertices(actor.get_transform())
            box=np.array([[v.x,v.y,v.z] for v in vertices]);camera_world=np.r_[box[:,:2].mean(0),point.location.z+height]
            local=(camera_world-origin)*axis-np.asarray(local_origins[name])
            ac.simPause(True)
            row['placement']=place_diagnostic_camera(ac,airsim,name,local,np.asarray(local_origins[name]),q,offset);write()
            request=[airsim.ImageRequest('0',airsim.ImageType.Scene,False,True)]
            for _ in range(8):
                response=ac.simGetImages(request,vehicle_name=name)[0];time.sleep(.05)
            residual=pose_residual(response,local,q)
            row['pose_residual']=residual;write()
            if residual['position_m']>.05 or residual['quaternion_l2']>1e-4:raise RuntimeError('Actual native diagnostic camera differs from frozen pose')
            filename=f'material_{index:02d}.png';raw=bytes(response.image_data_uint8);(out/filename).write_bytes(raw)
            rot=response.camera_orientation;observation=dict(image=filename,agent=name,timestamp=int(response.time_stamp),
                camera_world_position=camera_world.tolist(),camera_local_position=xyz(response.camera_position),
                camera_orientation_ned_wxyz=[rot.w_val,rot.x_val,rot.y_val,rot.z_val])
            before_profile=verify_static_visual_wind(world)
            measure=pixel.capture(name,observation,{'target':dict(bbox_world_vertices=box.tolist())},out)['target']
            after_profile=verify_static_visual_wind(world)
            with Image.open(io.BytesIO(raw)) as image:rgb_image=np.array(image.convert('RGB'))
            hsv=cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)
            with Image.open(out/measure['visible_mask']) as image:mask=np.array(image)>0
            visible=int(mask.sum());cue=int((cue_mask(hsv,color)&mask).sum())
            readback=[int(x) for x in actor.attributes['color'].split(',')]
            passed=(visible>=16 and 2*cue>visible and readback==list(rgb) and
                    measure['visible_aba_exact'] and measure['isolated_aba_exact'] and measure['instance_mask_verified'])
            row.update(status='measured',file=filename,sha256=hashlib.sha256(raw).hexdigest(),measurement=measure,
                visible_actor_pixels=visible,cue_actor_pixels=cue,cue_fraction=cue/visible if visible else None,
                strict_majority_passed=bool(passed),readback_rgb=readback,camera=observation,pose_residual=residual,
                weather_before=before_profile,weather_after=after_profile)
            result['completed_frames']+=1;write()
            if inherited is not None:result['new_measurements']+=1;write()
            ac.simPause(False);actor.destroy();owned.remove(actor)
        passing=[model for model in CANDIDATES if len([r for r in rows if r['blueprint']==model and r.get('strict_majority_passed')])==6]
        reference=all(r.get('strict_majority_passed') for r in rows if r['color']=='yellow')
        result.update(stage='measured',passing_candidates=passing,reference_passed=reference,
            candidate_blueprint=passing[0] if passing and reference else None,native_rgb_review_pending=True)
        write()
    except BaseException as exc:result.update(stage='failed',error_type=type(exc).__name__,error=str(exc));write();raise
    finally:
        ac.simPause(False)
        for actor in reversed(owned):actor.destroy()
        world.apply_settings(original_settings);world.set_weather(original_weather)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=41451);p.add_argument('--resume-receipt',type=Path);a=p.parse_args()
    a.agents=2;a.carla_port=2000;a.expected_map='Town01';a.probe_timeout=1800;a.collect_scenes=0;a.stencil_control=True
    a.static_visual_wind=True;a.scene_repair=True;a.material_probe_v4=True
    from runtime_frame_probe import probe
    probe(a)
