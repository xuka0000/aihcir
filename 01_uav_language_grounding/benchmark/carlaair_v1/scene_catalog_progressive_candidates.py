"""Experimental observation-dependent direction construction, NOT accepted data.

The instruction specifies a direction equality, not its east/west value. The
blue marker supplies that value. This differs from merely hiding an object
whose complete relation was already written in the instruction. Physical
disclosure and a versioned evaluator remain required before dataset admission.
"""
import copy
import hashlib
from benchmark import canonical
from scene_catalog_v2 import recipe as base_recipe, CAR, COLORS, enu, holds, distance

VERSION = 'latent_horizontal_direction_candidate_v1'
TRAINING_VERSION = 'latent_horizontal_direction_training_candidate_v2'


def direction(a, b):
    if a[0]-b[0] >= 5:
        return 'east'
    if a[0]-b[0] <= -5:
        return 'west'
    return None


def assignments(scene, use_near=True, use_side=True, marker_override=None):
    objects={o['key']:o for o in scene['objects']}
    target=objects['target']
    cars=[o for o in objects.values() if o['blueprint']==CAR and o.get('color')==target['color']]
    vans=[objects[k] for k in ('reference','alternative_reference')]
    fire=objects['second_reference']['transform']
    marker=marker_override if marker_override is not None else objects['direction_marker']['transform']
    side=direction(marker,fire)
    matches=[]
    for car in cars:
        for van in vans:
            if use_near and not holds(car['transform'],van['transform'],'near'):
                continue
            if use_side and (side is None or direction(van['transform'],fire)!=side):
                continue
            matches.append(dict(target=car['key'],reference=van['key']))
    return matches


def audit(scene):
    objects={o['key']:o for o in scene['objects']}
    fire=objects['second_reference']['transform']
    marker=objects['direction_marker']['transform']
    flipped=list(marker)
    flipped[0]=2*fire[0]-marker[0]
    full=assignments(scene)
    removed={name:assignments(scene,**kwargs) for name,kwargs in (
        ('near',dict(use_near=False)),('same_horizontal_side',dict(use_side=False)))}
    changed=assignments(scene,marker_override=flipped)
    return dict(unique_target={x['target'] for x in full}=={'target'},
                every_task_constraint_informative=all(len({x['target'] for x in rows})>1 for rows in removed.values()),
                marker_side_changes_target=bool(changed) and 'target' not in {x['target'] for x in changed},
                complete_matches=full,leave_one_constraint_out=removed,marker_reflection_matches=changed,
                reflection_is_symbolic_counterfactual_not_rendered=True)


def recipe(spec,points,blueprints,max_attempts=64,candidate_accept=None,*,training_candidate=False):
    required_split = 'train' if training_candidate else 'validation'
    if spec.get('task_family')!='progressive_discovery' or spec.get('split')!=required_split:
        raise ValueError('Experimental candidates are validation-only progressive scenes')
    if training_candidate and not str(spec.get('episode_id', '')).startswith('train_'):
        raise ValueError('Training candidate must preserve its original training identifier')
    if not 1<=max_attempts<=256:
        raise ValueError('Invalid deterministic search bound')
    for attempt in range(max_attempts):
        try:
            scene=base_recipe(copy.deepcopy(spec),points,blueprints,attempt=attempt)
        except ValueError:
            continue
        objects={o['key']:o for o in scene['objects']}
        fire=objects['second_reference']['transform']
        side=direction(objects['reference']['transform'],fire)
        opposite=direction(objects['alternative_reference']['transform'],fire)
        if side is None or opposite is None or side==opposite:
            continue
        occupied=[o['transform'] for o in scene['objects']]
        markers=[p for p in points if direction(p,fire)==side
                 and all(distance(p,q)>=12 for q in occupied)
                 and 25<=distance(p,fire)<=90]
        if not markers:
            continue
        # Keep the marker appearance distinct from all candidate targets.
        for obj in scene['objects']:
            if obj['blueprint']==CAR:
                obj['color']=COLORS['white']
        marker=list(markers[0])
        scene['objects'].append(dict(key='direction_marker',blueprint=CAR,color=COLORS['blue'],transform=marker))
        scene.update(construction_version=TRAINING_VERSION if training_candidate else VERSION,
                     instruction='Locate the white car near the yellow van whose east or west position relative to the fire engine matches that of the blue car.')
        positions={o['key']:enu(o['transform']) for o in scene['objects']}
        relations=[dict(relation_id='r0',subject='target',object='reference',relation='near'),
                   dict(relation_id='r1',subject='reference',object='second_reference',relation=side),
                   dict(relation_id='r2',subject='direction_marker',object='second_reference',relation=side)]
        truth=scene['private_truth']
        truth.update(positions=positions,required_relations=relations)
        truth.pop('counterfactual_audit',None)
        construction=audit(scene)
        if not all(construction[k] for k in ('unique_target','every_task_constraint_informative','marker_side_changes_target')):
            continue
        truth['latent_direction_construction']=dict(
            task_constraints=['near','same_horizontal_side'],latent_direction=side,
            inference_rule='direction(reference,fire_engine) = observed direction(blue_car,fire_engine)',
            latent_value_explicitly_given=False,counterfactual_audit=construction,
            visual_disclosure_verified=False,benchmark_admission=False,
            required_next_checks=['marker unavailable to entire team initially',
                'marker and fire-engine evidence acquired after measured motion',
                'all required endpoints observable', 'unchanged pixel and collision checks',
                'versioned relational scorer supporting this task semantics'])
        scene['geometry_sha256']=hashlib.sha256(canonical(scene['objects']).encode()).hexdigest()
        scene['progressive_boundary']='Unrendered experimental candidate; old auditor must continue rejecting it. No progressive success claim.'
        if candidate_accept is not None and not candidate_accept(scene):
            continue
        return scene
    raise ValueError('No valid latent-direction construction in bounded search')
