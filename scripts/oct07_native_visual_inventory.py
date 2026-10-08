"""Actor-bound vehicle mesh inventory, outside scientific RGB acquisition.

Use Vehicles only: the old Any inventory included moving UAV components.
Never infer mesh truth from collision extents or AirSim's component box3D.
The runtime result reports the native mesh API measurement scope; it does
not certify rendered closure, paint pixels, grounding, or qualification.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

MODELS = ('vehicle.carlamotors.firetruck', 'vehicle.mercedes.sprinter')
VEHICLE_LABELS = ('Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle')


def native_vehicle_labels(labels):
    """Bind the installed enum, retaining a vehicle-only census in both APIs."""
    if hasattr(labels, 'Vehicles'):
        return [('Vehicles', labels.Vehicles)]
    if not all(hasattr(labels, name) for name in ('Car', 'Truck')):
        raise ValueError('Installed native API lacks the required vehicle labels')
    return [(name, getattr(labels, name)) for name in VEHICLE_LABELS
            if hasattr(labels, name)]


def key(row):
    return json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)


def mesh_difference(before, after, actor_id):
    """Require a unique stable vehicle addition or an exact native actor ID."""
    left, right = Counter(map(key, before)), Counter(map(key, after))
    added = [json.loads(v) for v in (right-left).elements()]
    removed = [json.loads(v) for v in (left-right).elements()]
    identified = [v for v in added if v.get('actor_id') == actor_id]
    if len(identified) == 1:
        method = 'native_actor_id_and_added_vehicle_box'
        selected = identified[0]
    elif len(added) == 1 and not removed:
        selected = added[0]
        if selected.get('actor_id') not in (None, 0, actor_id):
            raise ValueError('Added mesh belongs to another native actor')
        method = 'unique_added_vehicle_box_stable_background'
    else:
        raise ValueError('Vehicle mesh addition is not uniquely actor-bound')
    if removed or len(added) != 1:
        raise ValueError('Unrelated vehicle inventory moved during measurement')
    return dict(box=selected, identity_method=method, added_count=1,
                removed_count=0, actor_id=actor_id)


def local_vertices(vertices, transform):
    x, y, z, yaw = transform
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    if len(vertices) != 8 or not all(math.isfinite(v) for p in vertices for v in p):
        raise ValueError('Native mesh must contain eight finite world vertices')
    return [[c*(p[0]-x)+s*(p[1]-y), -s*(p[0]-x)+c*(p[1]-y), p[2]-z]
            for p in vertices]


def verify_removal(before, after):
    if Counter(map(key, before)) != Counter(map(key, after)):
        raise ValueError('Destroyed actor did not restore the exact vehicle inventory')


def main(root, output, expected_map, carla_port, measurement_deadline):
    import carla
    from oct07_native_geometry_diagnosis import verified_world
    root, output = Path(root), Path(output)
    if root.resolve() != Path('/path/to/aihcir/benchmarks/carlaair_v1'):
        raise ValueError('Inventory is restricted to the original native root')
    started = time.time()
    deadline = min(started+90, measurement_deadline)
    if deadline-started < 30:
        raise TimeoutError('Insufficient bounded measurement time')
    client = carla.Client('127.0.0.1', carla_port)
    client.set_timeout(5)
    state = dict(run_class='original_native_vehicle_visual_inventory',
                 started_at=started, rows=[], color_attributes=[],
                 engineering_RGB_frames=0, scientific_RGB_frames=0,
                 algorithm_rows=0, scientific_admission=False,
                 complete_visual_closure_verified=False,
                 actual_material_pixel_verified=False,
                 client_version=client.get_client_version(),
                 server_version=client.get_server_version(),
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        temp=output.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
        temp.replace(output)

    def bound():
        remaining=deadline-time.time()
        if remaining<=0:
            raise TimeoutError('Original one-shot native inventory deadline')
        timeout=min(5,remaining)
        client.set_timeout(timeout)
        return timeout

    def boxes(world):
        result=[]
        for name,label in labels:
            bound()
            result.extend(dict(location=[b.location.x,b.location.y,b.location.z],
                     extent=[b.extent.x,b.extent.y,b.extent.z],
                     rotation=[b.rotation.pitch,b.rotation.yaw,b.rotation.roll],
                     semantic_label=name,
                     actor_id=getattr(b,'actor_id',None),
                     world_vertices=[[p.x,p.y,p.z] for p in b.get_world_vertices(carla.Transform())])
                          for b in world.get_level_bbs(label))
        return result

    world=verified_world(client,expected_map)
    labels=native_vehicle_labels(carla.CityObjectLabel)
    state.update(actual_map=world.get_map().name,
                 inventory_label='vehicle_only',
                 native_vehicle_labels=[name for name, _ in labels],
                 available_city_object_labels=[name for name in dir(carla.CityObjectLabel)
                                               if not name.startswith('_')],
                 full_visual_scope='Native vehicle mesh API, not all attached components',
                 original_sensor_exposure_or_thresholds_changed=False)
    save()
    actor=None
    try:
        library=world.get_blueprint_library()
        selection=json.loads((root/'oct07_scene_repair/native_material_selection.json').read_text())
        # Preserve the entire existing selection without manufacturing a runtime paint claim.
        state['original_material_selection']=selection
        paint=library.find('vehicle.ford.mustang')
        def attribute_value(a):
            if a.type==carla.ActorAttributeType.String:return a.as_str()
            if a.type==carla.ActorAttributeType.Int:return a.as_int()
            if a.type==carla.ActorAttributeType.Float:return a.as_float()
            if a.type==carla.ActorAttributeType.Bool:return a.as_bool()
            if a.type==carla.ActorAttributeType.RGBColor:
                color=a.as_color();return [color.r,color.g,color.b]
            raise ValueError('Unrecognized original native attribute type')
        state['color_attributes']=[dict(id=a.id,type=str(a.type),value=attribute_value(a),modifiable=a.is_modifiable,
                                        recommended_values=list(a.recommended_values)) for a in paint]
        weather=world.get_weather()
        state['original_weather']={k:getattr(weather,k) for k in
            ('sun_altitude_angle','sun_azimuth_angle','cloudiness','precipitation','fog_density')}
        save()
        point=world.get_map().get_spawn_points()[0]
        for model in MODELS:
            bound()
            before=boxes(world)
            verify_removal(before,boxes(world))
            blueprint=library.find(model)
            if blueprint.has_attribute('role_name'):
                blueprint.set_attribute('role_name','original_visual_inventory')
            pose=carla.Transform(carla.Location(x=point.location.x,y=point.location.y,z=point.location.z+3),
                                 carla.Rotation(yaw=point.rotation.yaw))
            actor=world.try_spawn_actor(blueprint,pose)
            if actor is None:
                raise ValueError('Original isolated metrology pose is occupied; no retry')
            actor.set_simulate_physics(False)
            for _ in range(2):world.wait_for_tick(bound())
            actor.set_transform(pose)
            for _ in range(2):world.wait_for_tick(bound())
            tf=actor.get_transform()
            transform=[tf.location.x,tf.location.y,tf.location.z,tf.rotation.yaw]
            if math.dist(transform[:3],[pose.location.x,pose.location.y,pose.location.z])>.001:
                raise ValueError('CARLA actor root failed exact placement readback')
            after=boxes(world)
            verify_removal(after,boxes(world))
            measurement=mesh_difference(before,after,actor.id)
            native=measurement['box']
            local=local_vertices(native['world_vertices'],transform)
            b=actor.bounding_box
            row=dict(blueprint=model,actor_id=actor.id,actor_transform=transform,
                     actual_mesh_box=measurement,
                     vehicle_inventory_before=before,vehicle_inventory_after=after,
                     measured_local_min=[min(p[i] for p in local) for i in range(3)],
                     measured_local_max=[max(p[i] for p in local) for i in range(3)],
                     original_preset=dict(location=[b.location.x,b.location.y,b.location.z],
                                          extent=[b.extent.x,b.extent.y,b.extent.z]),
                     ground_support_verified=False,complete_visual_closure_verified=False,
                     image_pose_or_ABA_verified=False)
            state['rows'].append(row)
            save()
            tf2=actor.get_transform()
            if math.dist(transform[:3],[tf2.location.x,tf2.location.y,tf2.location.z])>.0001:
                raise ValueError('Native actor moved during mesh inventory')
            if actor.destroy() is not True:
                raise ValueError('Exact diagnostic actor cleanup failed')
            actor=None
            restored=boxes(world)
            verify_removal(before,restored)
            row['vehicle_inventory_after_removal']=restored
            row['actor_removed_and_inventory_restored']=True
            save()
        state.update(ended_at=time.time(),passed_operational=True,
                     native_mesh_measurements=len(state['rows']))
        save()
        print(json.dumps(dict(native_mesh_measurements=len(state['rows']),new_RGB_frames=0,
                              scientific_admission=False,actual_material_pixel_verified=False)))
    except BaseException as exc:
        state.update(ended_at=time.time(),passed_operational=False,error=repr(exc))
        save()
        raise
    finally:
        if actor is not None:
            actor.destroy()


def review_inventory(path):
    """Recompute the preserved actor identity and local mesh bounds independently."""
    path=Path(path);state=json.loads(path.read_text(encoding='utf-8-sig'))
    failures=[]
    labels=state.get('native_vehicle_labels', [])
    label_scope=(labels==['Vehicles'] or
                 (all(name in labels for name in ('Car', 'Truck')) and
                  len(set(labels))==len(labels) and
                  all(name in VEHICLE_LABELS for name in labels)))
    if (state.get('passed_operational') is not True or state.get('inventory_label')!='vehicle_only'
        or not label_scope
        or any(state.get(k)!=0 for k in ('engineering_RGB_frames','scientific_RGB_frames','algorithm_rows'))
        or state.get('scientific_admission') is not False):
        failures.append('Execution or zero-RGB boundary differs')
    rows=state.get('rows',[])
    if [r.get('blueprint') for r in rows]!=list(MODELS):
        failures.append('Original two-model inventory incomplete')
    for row in rows:
        try:
            actual=mesh_difference(row['vehicle_inventory_before'],row['vehicle_inventory_after'],row['actor_id'])
            if actual!=row['actual_mesh_box']:
                raise ValueError('Saved unique actor binding differs')
            verify_removal(row['vehicle_inventory_before'],row['vehicle_inventory_after_removal'])
            local=local_vertices(actual['box']['world_vertices'],row['actor_transform'])
            for name,fn in (('measured_local_min',min),('measured_local_max',max)):
                recomputed=[fn(p[i] for p in local) for i in range(3)]
                if any(abs(a-b)>1e-9 for a,b in zip(recomputed,row[name])):
                    raise ValueError('Saved local vertex extremum differs')
            if row.get('actor_removed_and_inventory_restored') is not True:
                raise ValueError('Original actor cleanup unverified')
        except (ValueError,KeyError,TypeError) as exc:
            failures.append(row.get('blueprint','unknown')+': '+str(exc))
    return dict(observed_at=time.time(),passed=not failures,failures=failures,
                source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                actual_mesh_measurements=len(rows),new_RGB_frames=0,new_algorithm_rows=0,
                scientific_admission=False,complete_visual_closure_verified=False,
                actual_material_pixel_verified=False,
                interpretation='Operational native mesh inventory only; rendered/ground/material admission remains separate')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expected-map',required=True);p.add_argument('--carla-port',type=int,required=True)
    p.add_argument('--measurement-deadline',type=float,required=True)
    a=p.parse_args();main(a.root,a.output,a.expected_map,a.carla_port,a.measurement_deadline)
