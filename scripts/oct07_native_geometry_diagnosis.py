"""Measure native construction inputs and rejection reasons, without RGB."""
import argparse
import copy
import json
import time
from pathlib import Path


def level_boxes(world):
    import carla
    return [dict(location=[b.location.x,b.location.y,b.location.z],
        extent=[b.extent.x,b.extent.y,b.extent.z],
        rotation=[b.rotation.pitch,b.rotation.yaw,b.rotation.roll],
        world_vertices=[[v.x,v.y,v.z] for v in b.get_world_vertices(carla.Transform())])
        for b in world.get_level_bbs(carla.CityObjectLabel.Any)]


def bounds_difference(before,after):
    from collections import Counter
    key=lambda row:json.dumps(row,sort_keys=True,separators=(',',':'))
    left=Counter(map(key,before));right=Counter(map(key,after))
    added=list((right-left).elements());removed=list((left-right).elements())
    unique=json.loads(added[0]) if len(added)==1 and not removed else None
    return dict(before_count=len(before),after_count=len(after),added_boxes=list(map(json.loads,added)),
        removed_boxes=list(map(json.loads,removed)),unique_added_box=unique,
        mesh_actor_identity_verified=unique is not None)


def verified_world(client,expected_map):
    if not expected_map:raise ValueError('An explicit native map is required before measuring geometry')
    world=client.get_world()
    if world.get_map().name.rsplit('/',1)[-1]!=expected_map:
        world=client.load_world(expected_map)
    if world.get_map().name.rsplit('/',1)[-1]!=expected_map:
        raise ValueError('Native geometry map readback differs from the requested map')
    return world


def main(root,output,templates_only=False,expected_map=None):
    import carla
    from oct07_capture_protocol import recipe,SCENE_VERSION,construction_modules
    from oct07_scene_geometry_v4 import native_templates,ground_hit,OCCLUDER
    root=Path(root);phase=root/'oct07_scene_repair'
    client=carla.Client('127.0.0.1',2000);client.set_timeout(15)
    world=verified_world(client,expected_map)
    points=[]
    for transform in world.get_map().get_spawn_points():
        p=transform.location;points.append([p.x,p.y,p.z,transform.rotation.yaw])
    material=json.loads((phase/'native_material_selection.json').read_text())
    car=construction_modules(SCENE_VERSION,material)
    templates=native_templates(world,[car,'vehicle.mercedes.sprinter','vehicle.carlamotors.firetruck',OCCLUDER])
    if templates_only:
        templates.update(native_templates(world,['vehicle.carlamotors.european_hgv','vehicle.mitsubishi.fusorosa']))
    blueprints=[b.id for b in world.get_blueprint_library()]
    ids=set(json.loads((phase/'contract.json').read_text())['scene_qualification']['qualification_scene_ids'])
    specs=[json.loads(l) for l in (phase/'specifications/train.specs.jsonl').read_text().splitlines() if json.loads(l)['episode_id'] in ids]
    if not templates_only and any(s['map']!=expected_map for s in specs):
        raise ValueError('Diagnostic specifications and actual native map differ')
    cache={};ground_records=[];rows=[]
    state=dict(run_class='native_geometry_diagnosis',new_RGB_frames=0,qualification_scenes_admitted=0,
        actual_map=world.get_map().name,expected_map=expected_map,map_readback_verified=True,spawn_points=points,blueprint_ids=blueprints,
        templates=templates,ground_records=ground_records,rows=rows,started_at=time.time())
    def save():Path(output).write_text(json.dumps(state,indent=2))
    def ground(x,y,z):
        key=(x,y,z)
        if key not in cache:
            hits=world.cast_ray(carla.Location(x=x,y=y,z=z+20),carla.Location(x=x,y=y,z=z-5))
            record=dict(x=x,y=y,near_z=z,raw_hits=[dict(label=str(h.label),location=[h.location.x,h.location.y,h.location.z]) for h in hits])
            try:cache[key]=ground_hit(world,x,y,z);record['accepted']=cache[key]
            except ValueError as exc:cache[key]=str(exc);record['error']=str(exc)
            ground_records.append(record)
        if isinstance(cache[key],str):raise ValueError(cache[key])
        return copy.deepcopy(cache[key])
    save()
    if templates_only:
        state.update(ended_at=time.time(),template_only=True,scientific_admission=False)
        save();print(json.dumps(dict(new_RGB_frames=0,measured_templates=list(templates))))
        return
    for spec in specs:
        try:
            scene=recipe(spec,points,blueprints,SCENE_VERSION,material,templates,ground)
            rows.append(dict(episode_id=spec['episode_id'],success=True,scene=scene))
        except ValueError as exc:
            rows.append(dict(episode_id=spec['episode_id'],success=False,error=str(exc),
                rejections=getattr(exc,'geometry_rejections',None)))
        save()
    state.update(ended_at=time.time(),geometry_successes=sum(r['success'] for r in rows),
        planned=len(specs),scientific_admission=False)
    save();print(json.dumps({k:v for k,v in state.items() if k not in ('spawn_points','blueprint_ids','templates','ground_records','rows')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-map',required=True);p.add_argument('--templates-only',action='store_true');a=p.parse_args();main(a.root,a.output,a.templates_only,a.expected_map)
