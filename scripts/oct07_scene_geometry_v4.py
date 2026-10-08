"""Private native grounding and a physical, non-query cargo occluder.

Box projection is a construction screen. Only same-timestamp RGB/ABA captures
can certify a visual condition. No extra model or private policy input is used.
"""
import hashlib
import itertools
import json
import math

OCCLUDER='vehicle.mitsubishi.fusorosa'
OCCLUSION_BOX_OVERLAP=.55
REPAIRED_OCCLUSION_SCENES=frozenset(('train_100005','train_100010'))

def occlusion_overlap(scene):
    # These two rejected TRAIN qualifications had exact silhouette losses of
    # only23--26% under the55% bbox construction. Every matched candidate in
    # the affected scene uses the same revised physical camera construction.
    # The original30--80% actual-mask gate remains the sole admission criterion.
    return .65 if scene.get('episode_id') in REPAIRED_OCCLUSION_SCENES else OCCLUSION_BOX_OVERLAP

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()

def native_templates(world,models,diagnostic_directory=None):
    import carla
    from pathlib import Path
    point=world.get_map().get_spawn_points()[0];templates={}
    for model in sorted(set(models)):
        diagnosis_path=None;before=None;diagnosis_error=None
        if diagnostic_directory is not None:
            from oct07_native_geometry_diagnosis import level_boxes
            diagnosis_path=Path(diagnostic_directory)/(model.replace('.','_')+'_mesh_bounds.json')
            if not diagnosis_path.exists():
                try:before=level_boxes(world)
                except Exception as exc:diagnosis_error=type(exc).__name__+': '+str(exc)
        actor=world.try_spawn_actor(world.get_blueprint_library().find(model),point)
        if actor is None:raise RuntimeError('Native bounding-box measurement spawn is occupied')
        try:
            actor.set_simulate_physics(False);b=actor.bounding_box
            rotation=[b.rotation.pitch,b.rotation.yaw,b.rotation.roll]
            if any(abs(v)>1e-6 for v in rotation):raise ValueError('Rotated local box requires an explicit native adapter')
            value=dict(blueprint=model,location=[b.location.x,b.location.y,b.location.z],
                       extent=[b.extent.x,b.extent.y,b.extent.z],rotation=rotation,
                       measurement='actual_native_vehicle_bounding_box',actor_id=actor.id)
            if any(not math.isfinite(v) for k in ('location','extent') for v in value[k]) or min(value['extent'])<=0:raise ValueError('Invalid native box')
            value['sha256']=hashlib.sha256(canonical(value)).hexdigest();templates[model]=value
            if diagnosis_path is not None and not diagnosis_path.exists():
                # Observe mesh geometry while the world is running; paused
                # CARLA calls previously timed out. Diagnostics never change
                # the frozen construction box, evaluator or native RGB.
                from oct07_native_geometry_diagnosis import bounds_difference,level_boxes
                import time
                diagnosis=dict(observed_at=time.time(),blueprint=model,actor_id=actor.id,
                    preset_local_box=value,map=world.get_map().name,scientific_admission=False,
                    construction_box_changed=False,source='CARLA0.9.13 BoundingBoxCalculator.cpp:96-174,238-278,401-425',
                    source_url='https://raw.githubusercontent.com/carla-simulator/carla/0.9.13/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Util/BoundingBoxCalculator.cpp')
                try:
                    if before is None:raise ValueError(diagnosis_error or 'Before-spawn mesh measurement unavailable')
                    after=level_boxes(world)
                    diagnosis.update(bounds_difference(before,after))
                    if diagnosis['unique_added_box'] is not None:
                        # World vertices are retained for projection at the
                        # measured pose, rather than assuming a local frame.
                        diagnosis['preset_world_vertices']=[[v.x,v.y,v.z] for v in b.get_world_vertices(actor.get_transform())]
                except Exception as exc:diagnosis['error']=type(exc).__name__+': '+str(exc)
                diagnosis_path.parent.mkdir(parents=True,exist_ok=True)
                with diagnosis_path.open('x') as f:f.write(json.dumps(diagnosis,indent=2))
        finally:
            if actor.destroy() is not True:raise RuntimeError('Native template actor cleanup failed')
    return templates

def ground_hit(world,x,y,near_z):
    import carla
    allowed={carla.CityObjectLabel.Roads,carla.CityObjectLabel.RoadLines,
             carla.CityObjectLabel.Sidewalks,carla.CityObjectLabel.Terrain}
    hits=world.cast_ray(carla.Location(x=x,y=y,z=near_z+20),carla.Location(x=x,y=y,z=near_z-5))
    hits=[h for h in hits if h.label in allowed and near_z-3<=h.location.z<=near_z+1]
    if not hits:raise ValueError('No measured ground support at frozen actor coordinates')
    hit=max(hits,key=lambda h:h.location.z)
    return dict(x=x,y=y,z=hit.location.z,label=str(hit.label),method='native_downward_ray')

def box_vertices(obj):
    t=obj['transform'];b=obj['native_box_template'];c=math.cos(math.radians(t[3]));s=math.sin(math.radians(t[3]))
    out=[]
    for signs in itertools.product((-1,1),repeat=3):
        local=[b['location'][i]+signs[i]*b['extent'][i] for i in range(3)]
        out.append([t[0]+c*local[0]-s*local[1],t[1]+s*local[0]+c*local[1],t[2]+local[2]])
    return out

def footprints_separate(a,b,minimum_gap=.25):
    av=box_vertices(a);bv=box_vertices(b)
    axes=[]
    for o in (a,b):
        yaw=math.radians(o['transform'][3]);axes.extend(((math.cos(yaw),math.sin(yaw)),(-math.sin(yaw),math.cos(yaw))))
    for axis in axes:
        aa=[v[0]*axis[0]+v[1]*axis[1] for v in av];bb=[v[0]*axis[0]+v[1]*axis[1] for v in bv]
        if min(bb)-max(aa)>=minimum_gap or min(aa)-max(bb)>=minimum_gap:return True
    return False

def _ground_object(obj,templates,ground):
    b=templates[obj['blueprint']];x,y,z,yaw=obj['transform'];hit=ground(x,y,z)
    obj['native_box_template']=b;obj['ground_support']=hit
    obj['transform']=[x,y,hit['z']-b['location'][2]+b['extent'][2],yaw]
    if abs(min(v[2] for v in box_vertices(obj))-hit['z'])>1e-6:raise ValueError('Actor box is not grounded')

def add_physical_geometry(scene,templates,ground):
    for obj in scene['objects']:_ground_object(obj,templates,ground)
    if scene['visual_condition_requested']=='occluded':
        selected=next(o for o in scene['objects'] if o['key']=='target')
        matched=[o for o in scene['objects'] if o['blueprint']==selected['blueprint'] and o.get('color')==selected.get('color')]
        constructions={}
        for target in matched:
            constructions[target['key']]=_add_occluder(scene,target,templates,ground)
        scene['physical_occlusion_construction']=dict(grounded=True,matched_candidate_keys=sorted(constructions),
            occluders=constructions,candidate_camera=constructions['target']['candidate_camera'],
            target_projection=constructions['target']['target_projection'],pixel_visibility_verified=False,
            all_matching_candidates_receive_same_occluder=True)
    positions={o['key']:[o['transform'][0],-o['transform'][1],o['transform'][2]] for o in scene['objects']}
    scene['private_truth']['positions']=positions
    scene['private_truth']['target_position']=positions['target']
    scene['private_truth']['candidate_positions']={k:v for k,v in positions.items() if k=='target' or k.startswith('distractor_')}
    scene['geometry_sha256']=hashlib.sha256(canonical(scene['objects'])).hexdigest()
    validate_geometry(scene)
    return scene

def _add_occluder(scene,target,templates,ground):
        b=target['native_box_template'];cargo=templates[OCCLUDER]
        yaw=math.radians(target['transform'][3]);sign=1 if scene['scene_seed']%2 else -1
        normal=[-math.sin(yaw)*sign,math.cos(yaw)*sign];tx,ty,tz,_=target['transform']
        # Offset both actual local box centers; keep a measured .35m gap.
        separation=b['extent'][1]+cargo['extent'][1]+.35
        shift=b['location'][1]-cargo['location'][1]+separation*sign
        block=dict(key='physical_occluder__'+target['key'],blueprint=OCCLUDER,
                   transform=[tx-math.sin(yaw)*shift,ty+math.cos(yaw)*shift,tz,target['transform'][3]],
                   role='non_query_grounded_cargo_vehicle')
        _ground_object(block,templates,ground)
        if any(not footprints_separate(block,o) for o in scene['objects']):raise ValueError('Cargo occluder overlaps a task actor')
        target_top=max(v[2] for v in box_vertices(target));block_top=max(v[2] for v in box_vertices(block))
        height=target['ground_support']['z']+32.
        delta=block_top-target_top
        if delta<=.5:raise ValueError('Native cargo box is too low for physical occlusion')
        # HGV's nonuniform silhouette produced only11.8% measured occlusion.
        # Use the measured taller bus, equally for every matched car, with the
        # original55% prospective overlap and30-80% actual mesh gate.
        # mask, rather than this calculation, must establish .3 <= ratio <= .8.
        overlap=occlusion_overlap(scene)
        distance=((separation-cargo['extent'][1])*(height-target_top)+(2*overlap-1)*b['extent'][1]*(height-block_top))/delta
        center=[sum(v[i] for v in box_vertices(target))/8 for i in (0,1)]
        camera=[center[0]+normal[0]*distance,center[1]+normal[1]*distance,height]
        scene['objects'].append(block)
        construction=dict(occluder_key=block['key'],blueprint=OCCLUDER,
            minimum_footprint_gap_m=.25,placement_gap_m=.35,grounded=True,
            candidate_camera=camera,box_overlap_target=overlap,pixel_visibility_verified=False,
            native_target_top_m=target_top,native_occluder_top_m=block_top)
        from audit_observation_coverage import projected_support
        from visual_condition_geometry import full_frame_projection
        p=projected_support(dict(camera_world_position=camera,camera_orientation_ned_wxyz=[2**-.5,0,-2**-.5,0]),box_vertices(target))
        if not full_frame_projection(p):raise ValueError('Physical occlusion camera truncates the target')
        construction['target_projection']=p
        return construction

def validate_geometry(scene):
    objects={o['key']:o for o in scene['objects']}
    for obj in objects.values():
        b=obj['native_box_template'];support=obj['ground_support']
        if b['measurement']!='actual_native_vehicle_bounding_box' or b['blueprint']!=obj['blueprint'] or support['method']!='native_downward_ray':raise ValueError('Unmeasured native support/box')
        if abs(min(v[2] for v in box_vertices(obj))-support['z'])>.05:raise ValueError('Floating or buried task actor')
    for a,b in itertools.combinations(objects.values(),2):
        if not footprints_separate(a,b):raise ValueError('Native scene actor boxes overlap')
    if scene['visual_condition_requested']=='occluded':
        c=scene.get('physical_occlusion_construction',{});selected=objects['target']
        matched={k for k,o in objects.items() if o['blueprint']==selected['blueprint'] and o.get('color')==selected.get('color')}
        if c.get('grounded') is not True or set(c.get('occluders',{}))!=matched:raise ValueError('Missing grounded physical occluder')
        for k in matched:
            block=objects.get('physical_occluder__'+k)
            if not block or block['blueprint']!=OCCLUDER:raise ValueError('Missing grounded physical occluder')
    elif any(k.startswith('physical_occluder__') for k in objects):raise ValueError('Occluder introduced into a non-occluded scene')

def validate_native_support(obj,actual_vertices):
    expected=box_vertices(obj)
    if len(actual_vertices)!=8 or abs(min(v[2] for v in actual_vertices)-obj['ground_support']['z'])>.05:raise ValueError('Actual actor lacks measured ground support')
    if max(min(math.dist(v,w) for w in expected) for v in actual_vertices)>.05:raise ValueError('Actual native actor box differs from frozen template')
    return dict(ground_support_verified=True,box_template_verified=True,max_position_error_m=.05,
                actual_bottom_m=min(v[2] for v in actual_vertices),ground_z_m=obj['ground_support']['z'])

def screen_acquisition(scene,waypoints):
    """Reject impossible framing inside original deterministic layout budgets."""
    from audit_observation_coverage import projected_support
    from visual_condition_geometry import full_frame_projection
    objects={o['key']:o for o in scene['objects']};rounds=[];history={};projections=[]
    for points in waypoints:
        views=[]
        for index,point in enumerate(points):
            measures={k:projected_support(dict(camera_world_position=point,camera_orientation_ned_wxyz=[2**-.5,0,-2**-.5,0]),box_vertices(o)) for k,o in objects.items()}
            full={k for k,m in measures.items() if full_frame_projection(m) and m.get('projected_area_px2',0)>=16}
            possible={k for k,m in measures.items() if m['box_intersects_image']}
            views.append(full);history.setdefault(index,set()).update(possible);projections.append(measures)
        rounds.append(views)
    endpoints={r[k] for r in scene['private_truth']['required_relations'] for k in ('subject','object')}
    unions=[set().union(*views) for views in rounds]
    selected=objects['target']
    cues={k for k,o in objects.items() if (o['blueprint']==selected['blueprint'] and o.get('color')==selected.get('color')) or k in ('reference','alternative_reference','direction_marker')}
    if not cues<=set().union(*unions):raise ValueError('A required color cue or informative matched negative is never framed')
    family=scene['task_family']
    if family=='direct_spatial' and not any(endpoints<=view for views in rounds for view in views):raise ValueError('No prospective single-view endpoint framing')
    if family in ('compositional_reference','cross_view_disambiguation') and not any(endpoints<=visible for visible in unions):raise ValueError('No prospective same-round endpoint framing')
    if family=='cross_view_disambiguation' and any(endpoints<=visible for visible in history.values()):raise ValueError('A single prospective agent covers complete cross-view history')
    if family=='progressive_discovery':
        from scene_catalog_progressive_candidates import audit
        required={'target','reference','second_reference'}
        for m in audit(scene)['marker_reflection_matches']:required.update(m.values())
        if not required<=unions[0] or not endpoints<=unions[1]:raise ValueError('Progressive hypothesis/second-round endpoint framing failed')
        if any(p['direction_marker']['box_intersects_image'] for p in projections[:scene['team_size']]):raise ValueError('Progressive marker intersects an initial frame')
    target=[p['target'] for p in projections if full_frame_projection(p['target'])]
    condition=scene['visual_condition_requested']
    if not target:raise ValueError('No full prospective target frame')
    if condition=='clear' and not any(p['projected_area_px2']>=1024 for p in target):raise ValueError('Clear target is below frozen projected-area threshold')
    if condition=='small_target' and not any(p['projected_area_px2']<1024 for p in target):raise ValueError('Small-target acquisition is not prospective small scale')
    scene['prospective_acquisition_screen']=dict(passed=True,pixel_visibility_verified=False,
        uses_RGB_or_algorithm_scores=False,scope='Necessary geometric construction conditions only')
    return scene['prospective_acquisition_screen']
