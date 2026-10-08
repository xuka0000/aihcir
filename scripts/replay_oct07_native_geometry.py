"""Offline framing replay with measured boxes; new supports remain unverified."""
import collections
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_uav_language_grounding/benchmark/carlaair_v1'))
from prepare_oct07_scene_repair import OUT
from prepare_oct07_image_campaign import sha,write
from oct07_capture_protocol import recipe,SCENE_VERSION


def review_component(source):
    """Independently inspect actual native records; collision support is not truth."""
    from PIL import Image
    source=Path(source).resolve()
    raw=json.loads(source.read_text(encoding='utf-8-sig'))
    failures=[];checked=[];pairs=collections.defaultdict(list);planes=[]
    expected={('vehicle.carlamotors.firetruck',25),('vehicle.carlamotors.firetruck',35),
              ('vehicle.mercedes.sprinter',25),('vehicle.mercedes.sprinter',35)}
    rows=raw.get('rows',[])
    if len(rows)!=4 or raw.get('engineering_RGB_frames')!=4:
        failures.append('The exact four engineering frames are incomplete')
    seen=set()
    for index,row in enumerate(rows):
        try:
            key=(row['blueprint'],row['nominal_top_clearance_m'])
            if key not in expected or key in seen:raise ValueError('Unexpected or duplicate model/height')
            seen.add(key)
            image=(source.parent/row['image']).resolve()
            if source.parent not in image.parents or sha(image)!=row['image_sha256']:
                raise ValueError('Actual RGB path/hash differs')
            with Image.open(image) as rgb:
                if rgb.size!=(640,480) or rgb.mode not in ('RGB','RGBA'):
                    raise ValueError('Native RGB shape/mode differs')
                rgb.verify()
            root=row['actor_transform'];before=row.get('native_owner_before');after=row.get('native_owner_after')
            camera=row['actual_camera']['camera_world_position'];q=row['actual_camera']['camera_orientation_ned_wxyz']
            if len(root)!=4 or len(camera)!=3 or len(q)!=4:
                raise ValueError('Malformed native pose')
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                       for v in [*root,*camera,*q]):raise ValueError('Nonfinite native camera or actor root transform')
            if (row.get('native_owner_transform_readback_verified') is not True
                    or row.get('actual_image_pose_verified') is not True
                    or before is None or after is None or len(before)!=4 or len(after)!=4):
                raise ValueError('Same-time CARLA root/image readback evidence is absent')
            if not all(math.isfinite(v) for v in [*before,*after]):raise ValueError('Nonfinite before/after root')
            if (math.dist(before[:3],after[:3])>1e-4 or abs(before[3]-after[3])>.001
                    or math.dist(root[:3],after[:3])>1e-4 or abs(root[3]-after[3])>.001):
                raise ValueError('Actor moved during paused native readback')
            if (abs(math.sqrt(sum(v*v for v in q))-1)>1e-5
                    or abs(abs((q[0]-q[2])/math.sqrt(2))-1)>1e-5):
                raise ValueError('Native camera is not the original cardinal nadir camera')
            detections=row['actual_component_detections']
            if len(detections)!=1 or detections[0]['name'] not in row['mesh_owners']:
                raise ValueError('Native detection lacks a unique actor binding')
            box=detections[0]['box2D']
            x0,y0,x1,y1=[box[edge][axis+'_val'] for edge,axis in
                         [('min','x'),('min','y'),('max','x'),('max','y')]]
            if not all(math.isfinite(v) for v in (x0,y0,x1,y1)):
                raise ValueError('Nonfinite component pixel bounds')
            if not (0<x0<320<x1<640 and 0<y0<240<y1<480):
                raise ValueError('Clipped or off-center component bounds')
            record={'camera_relative':[a-b for a,b in zip(camera,root)],
                    'box':[x0,y0,x1,y1],'heading':root[3],'image':row['image'],
                    'image_sha256':row['image_sha256']}
            pairs[row['blueprint']].append(record)
            checked.append({'index':index,'image':row['image'],'image_sha256':row['image_sha256'],
                            'finite_stable_root_and_native_RGB_verified':True})
        except (ValueError,KeyError,TypeError,OSError) as error:
            failures.append('row '+str(index)+': '+str(error))
    if seen!=expected:failures.append('The predeclared two models/two heights are not covered')
    if not failures:
        for model,pair in sorted(pairs.items()):
            try:
                if len(pair)!=2:raise ValueError('Two heights per model are required')
                pair.sort(key=lambda p:p['camera_relative'][2]);a,b=pair
                if abs(a['heading']-b['heading'])>.001 or math.dist(a['camera_relative'][:2],b['camera_relative'][:2])>.001:
                    raise ValueError('Actor-relative camera XY or heading differs')
                h0,h1=a['camera_relative'][2],b['camera_relative'][2]
                if h1-h0<5:raise ValueError('Insufficient actual camera height separation')
                # Reciprocal pixel span is linear in camera height. Recover its
                # zero-height intercept without calling the acquisition estimator.
                top=[]
                for lower,upper in ((0,2),(1,3)):
                    s0=a['box'][upper]-a['box'][lower];s1=b['box'][upper]-b['box'][lower]
                    if not 0<s1<s0:raise ValueError('Support does not contract at the higher camera')
                    scale=(h1-h0)/(1/s1-1/s0)
                    top.append(h0-scale/s0)
                if abs(top[0]-top[1])>.02:raise ValueError('Axes identify inconsistent top planes')
                z=sum(top)/2;edges=[]
                for p in pair:
                    x,y,h=p['camera_relative'];x0,y0,x1,y1=p['box'];depth=h-z
                    if depth<=0:raise ValueError('Collision plane is behind camera')
                    edges.append([x-(y1-240)*depth/320,y+(x0-320)*depth/320,
                                  x-(y0-240)*depth/320,y+(x1-320)*depth/320])
                if max(abs(x-y) for x,y in zip(*edges))>.002:
                    raise ValueError('World edges differ between actual heights')
                planes.append({'blueprint':model,'top_z_offset_actor_root_m':z,
                               'top_plane_xyxy_offsets_in_world_axes_m':[(x+y)/2 for x,y in zip(*edges)],
                               'component_scope':'collision-enabled components only',
                               'visual_mesh_closure_verified':False})
            except (ValueError,ZeroDivisionError) as error:
                failures.append(model+': '+str(error))
    if raw.get('passed_operational') is not True or raw.get('error'):
        failures.append('Original component execution did not finish successfully')
    return {'observed_at':time.time(),'source':str(source),'source_sha256':sha(source),
            'review_entry_sha256':sha(Path(__file__)),
            'passed':not failures,'failures':failures,'engineering_RGB_verified':len(checked),
            'frames':checked,'independently_identified_collision_top_planes':planes,
            'raw_operational_flag_valid':raw.get('passed_operational') is True and not failures,
            'scientific_visual_closure':False,'benchmark_annotations_changed':False,
            'scientific_admission':False,'new_RGB':0,'new_algorithm_rows':0}


def main():
    source=OUT/'capture_scene_qualification1_geometrydiag1/effective/sessions/scene_geometry_Town01_2_scene_qualification1_geometrydiag1/geometry_diagnosis.json'
    box_source=OUT/'capture_scene_qualification1_templates1/effective/sessions/scene_geometry_Town01_2_scene_qualification1_templates1/geometry_diagnosis.json'
    d=json.loads(source.read_text());boxes=json.loads(box_source.read_text())['templates']
    specs=[json.loads(l) for l in (OUT/'specifications/train.specs.jsonl').read_text().splitlines()][:24]
    if any(s['map']!=d['actual_map'].rsplit('/',1)[-1] for s in specs):
        raise ValueError('Native map evidence does not match the requested scene map; no qualification replay claim')
    material=json.loads((OUT/'native_material_selection.json').read_text());rows=[]
    started=time.time()
    for spec in specs:
        try:
            scene=recipe(spec,d['spawn_points'],d['blueprint_ids'],SCENE_VERSION,material,boxes,
                lambda x,y,z:dict(x=x,y=y,z=0.,method='native_downward_ray'))
            rows.append(dict(episode_id=spec['episode_id'],geometry_success=True,scene=scene))
        except ValueError as exc:
            rows.append(dict(episode_id=spec['episode_id'],geometry_success=False,error=str(exc),rejections=getattr(exc,'geometry_rejections',None)))
        print(json.dumps({k:v for k,v in rows[-1].items() if k!='scene'}),flush=True)
    value=dict(started_at=started,observed_at=time.time(),planned=24,geometry_successes=sum(r['geometry_success'] for r in rows),
        rows=rows,actual_native_boxes_sha256=sha(box_source),actual_map_inputs_sha256=sha(source),
        ground_mode='flat-ground diagnostic approximation; shifted cargo supports require new native rays',
        runtime_admission=False,new_RGB_frames=0)
    write(OUT/'qualification_geometry_offline_replay.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='rows'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--component-source',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.component_source:
        result=review_component(args.component_source)
        if args.output:write(args.output,result)
        print(json.dumps(result,allow_nan=False))
        if not result['passed']:raise SystemExit(1)
    else:main()
