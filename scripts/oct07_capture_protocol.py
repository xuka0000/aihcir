"""Private, prospective two-round construction and physical admission audit.

Nothing here may be imported by a policy. Geometry construction uses real map
spawn points; scene labels never certify image visibility or model success.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

VERSION='oct07_two_round_geometry_rgb_v2'
APPEARANCE_VERSION='oct07_two_round_geometry_rgb_v3_a2'
SCENE_VERSION='oct07_grounded_two_round_rgb_v4'
INSTRUCTION='Locate the white car near the yellow van whose east or west position relative to the fire engine matches that of the blue car.'

def construction_modules(version=VERSION,material=None):
    # Glass-roof Tesla captures failed the frozen observed-color screen.
    # Replace the shared appearance template, including every matched car and
    # blue marker, while preserving all geometry and relational constraints.
    import scene_catalog_v2 as base
    import scene_catalog_progressive_candidates as progressive
    import scene_catalog_crossview_v4 as cross
    if version not in (VERSION,APPEARANCE_VERSION,SCENE_VERSION):raise ValueError('Unknown frozen appearance construction')
    if version==SCENE_VERSION:
        from oct07_material_probe_v4 import CANDIDATES
        if (not material or material.get('candidate_blueprint') not in CANDIDATES or
            material.get('native_rgb_review_passed') is not True or not material.get('measurement_sha256')):
            raise ValueError('Actual UAV material selection and raw RGB review are required')
        car=material['candidate_blueprint']
    else:car='vehicle.audi.a2' if version==APPEARANCE_VERSION else 'vehicle.audi.tt'
    base.CAR=progressive.CAR=cross.CAR=car
    return car

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_scene(scene):
    version=scene.get('construction_version');construction_modules(version,scene.get('material_selection'))
    from scene_catalog_v2 import counterfactual_audit
    from scene_catalog_progressive_candidates import audit,direction
    split=scene['split'];eid=scene['episode_id'];number=int(eid.rsplit('_',1)[1])
    if scene.get('status')!='geometry_compiled':raise ValueError('Unexpected construction version')
    if not ((split=='train' and eid.startswith('train_') and 100000<=number<100120) or
            (split=='test' and eid.startswith('test_') and 100000<=number<100240) or
            (version==SCENE_VERSION and split=='validation' and eid.startswith('validation_') and 100000<=number<100024)):raise ValueError('Scene outside approved new split')
    if version==SCENE_VERSION:
        from oct07_scene_geometry_v4 import validate_geometry
        validate_geometry(scene)
    if scene['task_family']=='progressive_discovery':
        if scene['instruction']!=INSTRUCTION:raise ValueError('Latent direction must not be given in the instruction')
        checked=audit(scene)
        if not all(checked[k] is True for k in ('unique_target','every_task_constraint_informative','marker_side_changes_target')):raise ValueError('Latent semantics failed')
        objects={o['key']:o for o in scene['objects']};side=direction(objects['direction_marker']['transform'],objects['second_reference']['transform'])
        expected={('target','near','reference'),('reference',side,'second_reference'),('direction_marker',side,'second_reference')}
        actual={(r['subject'],r['relation'],r['object']) for r in scene['private_truth']['required_relations']}
        if side is None or len(scene['private_truth']['required_relations'])!=3 or actual!=expected:raise ValueError('Latent truth does not match geometry')
    else:
        checked=counterfactual_audit(scene)
        if not checked['unique_target'] or not checked['every_relation_target_informative']:raise ValueError('Noninformative scene')
    if scene['geometry_sha256']!=hashlib.sha256(canonical(scene['objects'])).hexdigest():raise ValueError('Geometry hash changed')
    return checked

def recipe(spec,points,blueprints,version=VERSION,material=None,templates=None,ground=None):
    car=construction_modules(version,material)
    if car not in blueprints:raise ValueError('Observed-color repair blueprint is absent in the actual map')
    from scene_catalog_v2 import recipe as base
    from scene_catalog_crossview_v7 import recipe as cross
    from scene_catalog_progressive_candidates import recipe as progressive
    prepared=[];rejected={}
    def geometric_accept(candidate):
        if version!=SCENE_VERSION:return True
        from oct07_scene_geometry_v4 import add_physical_geometry,screen_acquisition
        c=copy.deepcopy(candidate);c.update({k:v for k,v in spec.items()});c['parent_construction_version']=candidate['construction_version'];c['construction_version']=SCENE_VERSION;c['material_selection']=copy.deepcopy(material)
        try:
            add_physical_geometry(c,templates,ground)
            screen_acquisition(c,route(c))
        except ValueError as exc:
            reason=str(exc);rejected[reason]=rejected.get(reason,0)+1;return False
        prepared[:]=[c];return True
    def generate():
      if spec['task_family']=='progressive_discovery':
        construction_spec=copy.deepcopy(spec)
        # The inherited constructor restricts candidate generation to TRAIN.
        # This is a geometry interface only; restore the original split/ID
        # before storage and before the independent new version audit.
        construction_spec['split']='train';construction_spec['episode_id']='train_'+spec['episode_id'].rsplit('_',1)[1]
        return progressive(construction_spec,points,blueprints,max_attempts=256 if version==SCENE_VERSION else 64,
            candidate_accept=geometric_accept,training_candidate=True)
      elif spec['task_family']=='cross_view_disambiguation':return cross(spec,points,blueprints,max_layout_attempts=256,candidate_accept=geometric_accept)
      else:return base(spec,points,blueprints,candidate_accept=geometric_accept)
    try:scene=generate()
    except ValueError as exc:
        exc.geometry_rejections=copy.deepcopy(rejected)
        raise
    if version==SCENE_VERSION:scene=prepared[0]
    scene.update({k:v for k,v in spec.items()})
    scene.setdefault('parent_construction_version',scene['construction_version']);scene['construction_version']=version
    scene['runtime_admitted']=False;scene['status']='geometry_compiled'
    if version==SCENE_VERSION:
        scene['material_selection']=copy.deepcopy(material)
        scene['prospective_geometry_rejections']=rejected
    scene['construction_protocol']=dict(version=version,rounds=2,uses_private_geometry=True,policy_input=False,
        runtime_visibility=False,color_visibility=False,split_restored_before_audit=True)
    validate_scene(scene)
    return scene

def route(scene):
    """Fixed private acquisition geometry, distinct from an inference policy."""
    objs={o['key']:o['transform'] for o in scene['objects']};team=scene['team_size'];condition=scene['visual_condition_requested']
    # The smaller native A2 body needs a prospective closer clear acquisition;
    # the projected-area and visibility admission thresholds stay unchanged.
    clear_z=26. if scene.get('construction_version') in (APPEARANCE_VERSION,SCENE_VERSION) else 30.
    z={'clear':clear_z,'small_target':70.,'occluded':50.}[condition]
    def middle(keys):return [sum(objs[k][j] for k in keys)/len(keys) for j in (0,1)]
    if scene['task_family']=='cross_view_disambiguation':
        center=scene['search_region_center_xy'];first=[[center[0],center[1]+12,z],[center[0]-100,center[1]-10,z]]
        second=[[center[0],center[1]-12,z],[center[0]+100,center[1]-10,z]]
        if team==4:
            first=[[center[0],center[1]+12,z],[center[0]-100,center[1]+10,z],[center[0]+100,center[1]-10,z],[center[0],center[1]+65,z]]
            second=[[center[0],center[1]-12,z],[center[0]-100,center[1]-10,z],[center[0]+100,center[1]+10,z],[center[0],center[1]+65,z]]
        if scene.get('construction_version')==SCENE_VERSION:
            # Both road sectors are acquired in a spatial order independent of
            # which car satisfies the hidden relation. The central agent keeps
            # the fire-engine evidence throughout the complete history.
            cars=[o for o in scene['objects'] if o['key']=='target' or o['key'].startswith('distractor_')]
            sectors=[]
            for side in (-1,1):
                local=[o for o in scene['objects'] if (o['key'] in ('target','reference','alternative_reference') or o['key'].startswith('distractor_')) and (o['transform'][0]-center[0])*side>0]
                point=[sum(o['transform'][j] for o in local)/len(local) for j in (0,1)]+[z]
                if condition=='small_target':point[0]+=side*16
                if condition=='occluded':
                    car=next(o for o in cars if (o['transform'][0]-center[0])*side>0)
                    point=list(scene['physical_occlusion_construction']['occluders'][car['key']]['candidate_camera'])
                sectors.append(point)
            first[1]=sectors[0];second[1]=sectors[1]
            if team==4:
                first[1]=second[1]=sectors[0];first[2]=second[2]=sectors[1]
    elif scene['task_family']=='progressive_discovery':
        a=middle(['target','reference']);b=middle(['distractor_0','alternative_reference']);m=middle(['direction_marker','second_reference'])
        first=[[a[0],a[1],z],[b[0],b[1],z]]
        second=[[a[0]-2,a[1],z],[m[0],m[1],z]] if scene.get('construction_version')==SCENE_VERSION else [[m[0]-8,m[1],z],[m[0]+8,m[1],z]]
        if team==4:
            first += [[a[0],a[1]+16,z],[b[0],b[1]+16,z]]
            if scene.get('construction_version')==SCENE_VERSION:second=[[a[0],a[1],z],[b[0],b[1],z],[m[0]-8,m[1],z],[m[0]+8,m[1],z]]
            else:second += [[m[0],m[1]-16,z],[m[0],m[1]+16,z]]
    else:
        a=middle(['target','reference']);first=[[a[0]-8,a[1],z],[a[0]+8,a[1],z]]
        second=[[a[0]-8,a[1]+4,z],[a[0]+8,a[1]+4,z]]
        if scene.get('construction_version')==SCENE_VERSION and 'distractor_1' in objs:
            point=middle(['distractor_1','second_reference']) if scene['task_family']=='compositional_reference' else objs['distractor_1'][:2]
            second[1]=[point[0],point[1],z]
        if team==4:
            first += [[a[0],a[1]-16,z],[a[0],a[1]+16,z]]
            second += [[a[0]+4,a[1]-16,z],[a[0]+4,a[1]+16,z]]
    # CARLA spawn/road height is map-dependent; all normal flight waypoints use
    # measured road heights, rather than silently assuming world ground z=0.
    if scene.get('construction_version')==SCENE_VERSION:
        first=[list(p) for p in first];second=[list(p) for p in second]
        supports=[o['ground_support']['z'] for o in scene['objects'] if 'ground_support' in o and not o['key'].startswith('physical_occluder__')]
        ground_z=sum(supports)/len(supports) if supports else 0.
        for phase in (first,second):
            for p in phase:p[2]+=ground_z
        if scene['task_family']=='progressive_discovery':
            from oct07_prospective_camera_cover import cover
            first=cover(scene,z+ground_z,True)
            second=cover(scene,z+ground_z,False)
        if condition=='occluded':
            candidate=scene.get('physical_occlusion_construction',{}).get('candidate_camera')
            if candidate is not None and scene['task_family']!='cross_view_disambiguation':
                first[0]=list(candidate)
                if team==4:
                    if scene['task_family']=='progressive_discovery':first[2]=[a[0],a[1]-40,z+ground_z];first[3]=[b[0],b[1]+40,z+ground_z]
                    else:first[2]=[a[0],a[1]-40,z+ground_z];first[3]=[a[0],a[1]+40,z+ground_z]
            if condition=='occluded' and scene['task_family']=='cross_view_disambiguation':
                first[1]=list(sectors[0]);second[1]=list(sectors[1])
                if team==4:first[1]=list(sectors[0]);second[1]=list(sectors[0]);first[2]=list(sectors[1]);second[2]=list(sectors[1])
    for phase in (first,second):
        if any(math.dist(phase[i][:2],phase[j][:2])<12 for i in range(team) for j in range(i)):raise ValueError('Construction routes do not separate the UAVs')
    return [first,second]

def audit(record,scene,renderer):
    from audit_semantic_gate import audit_episode,validate_renderer_control
    validate_scene(scene)
    control=validate_renderer_control(renderer);legacy=audit_episode(record,scene,control['passed'])
    if len(legacy['rounds'])!=2:raise ValueError('New acquisition requires exactly two complete rounds')
    blockers=list(legacy['blockers']);latent=None
    if scene.get('construction_version')==SCENE_VERSION:
        from oct07_scene_geometry_v4 import validate_native_support
        try:
            for obj in scene['objects']:
                validate_native_support(obj,record['private_native_instances'][obj['key']]['bbox_world_vertices'])
        except (ValueError,KeyError) as exc:blockers.append('native_ground_or_box_failed:'+str(exc))
        if scene['visual_condition_requested']=='occluded':
            if not scene.get('physical_occlusion_construction',{}).get('grounded'):blockers.append('physical_occluder_not_grounded')
    if scene['task_family']=='progressive_discovery':
        blockers=[b for b in blockers if b!='implicit_relation_generation_not_implemented']
        from scene_catalog_progressive_candidates import audit as symbolic
        counter=symbolic(scene);required={'target','reference','second_reference'}
        for match in counter['marker_reflection_matches']:required.update(match.values())
        visible0=set(legacy['rounds'][0]['team_visible']);missing=sorted(required-visible0)
        from audit_semantic_gate import VIEW
        views0=[o for o in record['observations'] if int(VIEW.fullmatch(o['image'])[1])==0]
        hidden=bool(views0) and all(record['private_pixel_measurements'][o['image']]['direction_marker']['visible_pixels']==0 and
            all(record['private_pixel_measurements'][o['image']]['direction_marker'].get(k) is True for k in ('visible_aba_exact','isolated_aba_exact','visible_subset_of_isolated')) for o in views0)
        witnesses=[w for w in legacy['progressive_physical_witnesses'] if w['round']==1 and w['intermediary']=='direction_marker' and
            w['measured_displacement_m']>=.5 and 'second_reference' in legacy['rounds'][1]['team_visible']]
        if missing:blockers.append('initial_two_hypotheses_not_supported:'+','.join(missing))
        if not hidden:blockers.append('marker_not_hidden_from_entire_initial_team')
        if not witnesses:blockers.append('round1_moved_marker_and_anchor_not_observed')
        latent=dict(initial_required=sorted(required),initial_missing=missing,marker_hidden=hidden,round1_witnesses=witnesses,counterfactual=counter)
    return dict(schema='oct07_two_round_physical_admission_v1',episode_id=scene['episode_id'],legacy=legacy,
        latent=latent,renderer_control=control,blockers=blockers,passed=not blockers,
        model_success=False,user_acceptance=False,observability_only=True)

def compile_for_world(world,specs,destination,version=VERSION,material=None,diagnostic_directory=None):
    """Write only requested map/seed identities, never the historical 4800 set."""
    from benchmark import canonical as old_canonical
    destination=Path(destination);points=[]
    for transform in world.get_map().get_spawn_points():
        p=transform.location;points.append([p.x,p.y,p.z,transform.rotation.yaw])
    blueprints=[b.id for b in world.get_blueprint_library()];out=[];templates=None;ground=None
    if version==SCENE_VERSION:
        from oct07_scene_geometry_v4 import native_templates,ground_hit,OCCLUDER
        car=construction_modules(version,material)
        templates=native_templates(world,[car,'vehicle.mercedes.sprinter','vehicle.carlamotors.firetruck',OCCLUDER],diagnostic_directory)
        cache={}
        def ground(x,y,z):
            key=(x,y,z)
            if key not in cache:
                try:cache[key]=ground_hit(world,x,y,z)
                except ValueError as exc:cache[key]=str(exc)
            if isinstance(cache[key],str):raise ValueError(cache[key])
            return copy.deepcopy(cache[key])
    for spec in specs:
        if spec['map']!=world.get_map().name.rsplit('/',1)[-1]:raise ValueError('Wrong live map')
        path=destination/spec['split']/spec['episode_id']/'scene.json';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            scene=json.loads(path.read_text());validate_scene(scene)
            if scene['construction_version']!=version or (version==SCENE_VERSION and scene['material_selection']!=material):raise ValueError('Existing frozen construction differs')
            if scene['scene_seed']!=spec['scene_seed']:raise ValueError('Existing scene identity mismatch')
        else:
            try:scene=recipe(spec,points,blueprints,version,material,templates,ground)
            except ValueError as exc:
                exc.native_construction_failure=dict(spec=spec,templates=templates,spawn_points=points,
                    rejections=getattr(exc,'geometry_rejections',{}),
                    ground_cache=[dict(query=list(k),result=v) for k,v in cache.items()])
                raise
            with path.open('x') as f:f.write(old_canonical(scene)+'\n')
        out.append(dict(episode_id=spec['episode_id'],source_sha256=digest(path),waypoints=route(scene)))
    return out
