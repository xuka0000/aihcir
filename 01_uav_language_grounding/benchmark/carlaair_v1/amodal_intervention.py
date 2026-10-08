"""Privileged A-B-A geometry-removal diagnostic, never a policy observation.

The visible/amodal distinction follows Mohan & Valada, CVPR 2022. This code
tests a local renderer implementation, not APSNet or a published algorithm.
Only registered CARLA environment geometry is disabled. Consequently agreement
is evidence about this intervention, NOT automatic amodal certification.
"""
import hashlib
import json
import math
import time
import numpy as np
from PIL import Image
from pixel_visibility import PixelVisibilityAudit
from render_replay import vehicle_for_camera, pose_residual
from runtime_frame_probe import save, xyz
from renderer_weather import verify_static_visual_wind


CASES = (
    ('multiscale_clear_2uav_v1', 'train_000000', 'round002_UAV_1.png', 'target'),
    ('multiscale_clear_2uav_v1', 'train_000006', 'round002_UAV_1.png', 'target'),
    ('multiscale_clear_4uav_v1', 'train_000001', 'round002_UAV_1.png', 'target'),
    ('static_visual_wind_2uav_v1', 'train_000004', 'round000_UAV_1.png', 'reference'),
)


def mask_checks(baseline, isolated_geometry, restored, old_visible, old_isolated):
    """Exact binary masks; no area-only or IoU-tolerance substitute."""
    v, a = baseline
    iv, ia = isolated_geometry
    rv, ra = restored
    arrays = [v, a, iv, ia, rv, ra, old_visible, old_isolated]
    if any(x.shape != (480, 640) or x.dtype != np.bool_ for x in arrays):
        raise ValueError('Expected binary native-resolution masks')
    return {
        'nonempty_reference': bool(a.any()),
        'visible_subset': bool(np.all(~v | a)),
        'archive_visible_exact': bool(np.array_equal(v, old_visible)),
        'archive_isolated_exact': bool(np.array_equal(a, old_isolated)),
        'geometry_removal_reveals_isolated_support_exact': bool(np.array_equal(iv, a)),
        'isolation_unchanged_under_geometry_removal': bool(np.array_equal(ia, a)),
        'restored_visible_exact': bool(np.array_equal(rv, v)),
        'restored_isolated_exact': bool(np.array_equal(ra, a)),
    }


def run_amodal_intervention(args, ac, cc, calibration, local_origins):
    import airsim
    import carla
    if args.agents != 2 or args.collect_scenes or args.render_replay or args.stencil_control:
        raise ValueError('Amodal diagnostic requires its own two-vehicle simulator')
    if not args.static_visual_wind:
        raise ValueError('Amodal diagnostic requires the verified static wind profile')
    root = args.output.parents[2]
    out = args.output / 'amodal_intervention'
    out.mkdir(exist_ok=False)
    status = {'stage': 'running', 'started_unix': time.time(), 'cases': [],
              'dataset_ready': False, 'runtime_validated_episodes': 0,
              'scope': 'Privileged registered-geometry-removal diagnostic; no policy inputs'}
    world = cc.get_world()
    environment = world.get_environment_objects(carla.CityObjectLabel.Any)
    environment_ids = {o.id for o in environment}
    if not environment_ids:
        raise RuntimeError('No registered environment geometry to test')
    status['environment_inventory'] = [{'id': int(o.id), 'name': o.name, 'label': str(o.type)}
                                       for o in environment]
    def update():
        save(out/'result.json', status)
    owned = []
    hidden = False
    paused = False
    try:
        for session, episode, filename, key in CASES:
            old_path = root/'sessions'/session/'probe/training_views'/episode/'observations.json'
            previous = json.loads(old_path.read_text())
            old_obs = next(o for o in previous['observations'] if o['image'] == filename)
            scene_path = root/'scene_assets_v3/train'/episode/'scene.json'
            if hashlib.sha256(scene_path.read_bytes()).hexdigest() != previous['source_sha256']:
                raise RuntimeError('Construction changed since archived observation')
            scene = json.loads(scene_path.read_text())
            folder = out/(episode+'_'+key)
            folder.mkdir()
            case = {'episode': episode, 'source_session': session, 'key': key,
                    'source_observation_file_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
                    'source_image': old_obs, 'conditions': [], 'stage': 'preparing',
                    'amodal_control_validated': False}
            status['cases'].append(case)
            update()
            pixel = PixelVisibilityAudit(ac)
            actors = {}
            for obj in scene['objects']:
                before = set(ac.simListSceneObjects())
                bp = world.get_blueprint_library().find(obj['blueprint'])
                if 'color' in obj and bp.has_attribute('color'):
                    bp.set_attribute('color', obj['color'])
                x,y,z,yaw = obj['transform']
                actor = world.spawn_actor(bp, carla.Transform(carla.Location(x=x,y=y,z=z),
                                                             carla.Rotation(yaw=yaw)))
                owned.append(actor)
                actors[obj['key']] = actor
                actor.set_simulate_physics(False)
                pixel.register(obj['key'], set(ac.simListSceneObjects())-before)
            for _ in range(3):
                world.wait_for_tick(20)
            originals = {k: a.get_transform() for k,a in actors.items()}
            target_actor = actors[key]
            instances = {key: {'bbox_world_vertices': [[v.x,v.y,v.z] for v in
                target_actor.bounding_box.get_world_vertices(originals[key])]}}
            agent = old_obs['agent']
            expected_q = np.asarray(old_obs['camera_orientation_ned_wxyz'])
            expected_position = ((np.asarray(old_obs['camera_world_position'])-calibration['world_origin_m'])
                                 *[1,1,-1]-local_origins[agent])
            mount_q = airsim.to_quaternion(-math.pi/2,0,0)
            mount_offset = np.array([.5,0,.1])
            body_p,body_q = vehicle_for_camera(expected_position+local_origins[agent],expected_q,
                mount_offset,[mount_q.w_val,mount_q.x_val,mount_q.y_val,mount_q.z_val])
            ac.simPause(True)
            paused = True
            ac.simSetCameraPose('0',airsim.Pose(airsim.Vector3r(*mount_offset),mount_q),vehicle_name=agent)
            if not ac.simSetObjectPose(agent,airsim.Pose(airsim.Vector3r(*body_p),
                    airsim.Quaternionr(body_q[1],body_q[2],body_q[3],body_q[0])),teleport=True):
                raise RuntimeError('Diagnostic camera placement failed')
            request = [airsim.ImageRequest('0',airsim.ImageType.Scene,False,True)]
            for _ in range(8):
                ac.simGetImages(request,vehicle_name=agent)
                time.sleep(.05)
            masks = []
            for condition in ('baseline','registered_geometry_removed','restored'):
                if condition == 'registered_geometry_removed':
                    world.enable_environment_objects(environment_ids,False)
                    hidden = True
                    for k,a in actors.items():
                        if k != key:
                            tf = originals[k]
                            a.set_transform(carla.Transform(carla.Location(x=tf.location.x,y=tf.location.y,z=-1000.),tf.rotation))
                elif condition == 'restored':
                    world.enable_environment_objects(environment_ids,True)
                    hidden = False
                    for k,a in actors.items():
                        if k != key:
                            a.set_transform(originals[k])
                weather = verify_static_visual_wind(world)
                for _ in range(8):
                    response = ac.simGetImages(request,vehicle_name=agent)[0]
                    time.sleep(.05)
                residual = pose_residual(response,expected_position,expected_q)
                if residual['position_m'] > .001 or residual['quaternion_l2'] > 1e-5:
                    raise RuntimeError(f'Original camera pose not reproduced: {residual}')
                path = folder/(condition+'.png')
                path.write_bytes(bytes(response.image_data_uint8))
                q = response.camera_orientation
                obs = {'agent':agent,'image':path.name,'timestamp':int(response.time_stamp),
                       'camera_local_position':xyz(response.camera_position),
                       'camera_world_position':((np.asarray(xyz(response.camera_position))+local_origins[agent])
                                                *[1,1,-1]+calibration['world_origin_m']).tolist(),
                       'camera_orientation_ned_wxyz':[q.w_val,q.x_val,q.y_val,q.z_val],
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                measurement = pixel.capture(agent,obs,instances,folder,debug=True)[key]
                case['conditions'].append({'name':condition,'observation':obs,'measurement':measurement,
                                           'weather':weather,'pose_residual':residual})
                def read_mask(path):
                    with Image.open(path) as image:
                        return np.asarray(image)>0
                masks.append((read_mask(folder/measurement['visible_mask']),
                              read_mask(folder/measurement['isolated_candidate_mask'])))
                case['stage'] = condition
                update()
            old_measure = previous['private_pixel_measurements'][filename][key]
            for path_key,hash_key in [('visible_mask','visible_mask_sha256'),
                                      ('isolated_candidate_mask','isolated_mask_sha256')]:
                if hashlib.sha256((old_path.parent/old_measure[path_key]).read_bytes()).hexdigest() != old_measure[hash_key]:
                    raise RuntimeError('Archived mask content drift')
            checks = mask_checks(*masks,read_mask(old_path.parent/old_measure['visible_mask']),
                                 read_mask(old_path.parent/old_measure['isolated_candidate_mask']))
            checks['same_paused_timestamp'] = len({c['observation']['timestamp'] for c in case['conditions']}) == 1
            case.update(checks=checks,intervention_agreement_passed=all(checks.values()),
                stage='intervention_complete_not_certified',
                remaining_limitation='Registered environment inventory may omit renderable occluders; no automatic amodal truth certification')
            update()
            ac.simPause(False)
            paused = False
            responses = cc.apply_batch_sync([carla.command.DestroyActor(a.id) for a in owned],False)
            if any(r.has_error() for r in responses):
                raise RuntimeError('Failed to clean owned diagnostic actors')
            owned=[]
        status.update(stage='intervention_completed_not_benchmark',ended_unix=time.time(),
                      agreement_passes=sum(c['intervention_agreement_passed'] for c in status['cases']))
        update()
    except Exception as error:
        status.update(stage='failed',error=f'{type(error).__name__}: {error}',ended_unix=time.time())
        update()
        raise
    finally:
        if hidden:
            world.enable_environment_objects(environment_ids,True)
        if paused:
            ac.simPause(False)
        if owned:
            cc.apply_batch_sync([carla.command.DestroyActor(a.id) for a in owned],False)
