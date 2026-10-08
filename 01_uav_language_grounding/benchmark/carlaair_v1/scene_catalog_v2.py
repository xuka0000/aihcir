"""Native layouts with informative hard negatives, not validated episodes."""
import hashlib
import math
import random
from functools import lru_cache
from benchmark import canonical, relation_holds

VERSION = 'informative_relations_v2'
COLORS = {'white': '240,240,240', 'red': '180,20,20', 'blue': '20,40,180'}
CAR = 'vehicle.tesla.model3'
VAN = 'vehicle.mercedes.sprinter'
TRUCK = 'vehicle.carlamotors.firetruck'
DIRECTIONS = ('east', 'west', 'north', 'south')


def enu(point):
    return [point[0], -point[1], point[2]]


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def holds(a, b, relation):
    return relation_holds({'subject': 'a', 'object': 'b', 'relation': relation},
                          {'a': enu(a), 'b': enu(b)})


@lru_cache(maxsize=12)
def adjacency(points):
    return tuple(tuple(j for j, b in enumerate(points) if 9 <= distance(a, b) <= 19)
                 for a in points)


def counterfactual_audit(scene):
    """Enumerate matching car/van bindings; omit each relation in turn."""
    positions = scene['private_truth']['positions']
    relations = scene['private_truth']['required_relations']
    targets = [o for o in scene['objects'] if o['key'] == 'target']
    if len(targets) != 1:
        raise ValueError('Expected one constructed target actor')
    target_blueprint = targets[0]['blueprint']
    cars = [o['key'] for o in scene['objects'] if o['blueprint'] == target_blueprint]
    vans = [o['key'] for o in scene['objects'] if o['blueprint'] == VAN]
    def solutions(edges):
        matches = []
        for car in cars:
            for van in vans:
                binding = dict(positions, target=positions[car], reference=positions[van])
                if all(relation_holds(r, binding) for r in edges):
                    matches.append({'target': car, 'reference': van})
        return matches
    full = solutions(relations)
    dropped = {r['relation_id']: solutions(relations[:i] + relations[i+1:])
               for i, r in enumerate(relations)}
    return {'complete_matches': full, 'leave_one_relation_out_matches': dropped,
            'unique_target': {x['target'] for x in full} == {'target'},
            'every_relation_target_informative': all(
                len({x['target'] for x in matches}) > 1 for matches in dropped.values())}


def recipe(spec, points, blueprint_ids, attempt=0,candidate_accept=None):
    if not {CAR, VAN, TRUCK}.issubset(set(blueprint_ids)):
        raise ValueError('Required native assets absent')
    points = tuple(tuple(p) for p in points)
    if len(points) < 8:
        raise ValueError('Insufficient road spawn points')
    near = adjacency(points)
    eligible = [i for i, neighbors in enumerate(near) if len(neighbors) >= 2]
    if not eligible:
        raise ValueError('No native road cluster supports near hard negatives')
    rng = random.Random(int(spec['scene_seed']) ^ (attempt * 0x9E3779B97F4A7C15))
    family = spec['task_family']
    if family not in ('direct_spatial', 'compositional_reference', 'progressive_discovery',
                      'cross_view_disambiguation'):
        raise ValueError('Unknown family')
    for layout_attempt in range(160):
        anchor_index = rng.choice(eligible)
        anchor = points[anchor_index]
        ti = rng.choice(near[anchor_index])
        target = points[ti]
        near_candidates = [points[k] for k in near[anchor_index]
                           if k != ti and distance(points[k], target) >= 9]
        if not near_candidates:
            continue
        d1 = rng.choice(near_candidates)
        extra = []
        if family == 'progressive_discovery':
            alt_indices = [i for i, a in enumerate(points) if 28 <= distance(a, anchor) <= 75
                           and distance(a, target) >= 25 and near[i]]
            if not alt_indices:
                continue
            ai = rng.choice(alt_indices)
            alt = points[ai]
            options = [points[k] for k in near[ai] if distance(points[k], anchor) > 24
                       and distance(points[k], target) >= 12]
            if not options:
                continue
            d1 = rng.choice(options)
            extra = [{'key': 'alternative_reference', 'blueprint': VAN,
                      'color': '200,160,20', 'transform': alt}]
        occupied = [target, anchor, d1] + [o['transform'] for o in extra]
        second_candidates = [p for p in points if all(distance(p, a) >= 12 for a in occupied)
                             and distance(p, anchor) <= 90]
        rng.shuffle(second_candidates)
        selected = None
        for second in second_candidates:
            a, b, reference = (anchor, alt, second) if extra else (target, d1, second)
            if family in ('direct_spatial', 'cross_view_disambiguation'):
                reference = anchor
            directions = [r for r in DIRECTIONS if holds(a, reference, r) and not holds(b, reference, r)]
            if not directions:
                continue
            relation = rng.choice(directions)
            far_candidates = [p for p in points
                              if all(distance(p, q) >= 12 for q in occupied + [second])
                              and 24 < distance(p, anchor) <= 95
                              and (not extra or distance(p, alt) > 24)
                              and (bool(extra) or holds(p, reference, relation))]
            if far_candidates:
                selected = second, relation, rng.choice(far_candidates)
                break
        if selected is None:
            continue
        second, relation, d2 = selected
        color = rng.choice(sorted(COLORS))
        objects = [
            {'key': 'target', 'blueprint': CAR, 'color': COLORS[color], 'transform': target},
            {'key': 'reference', 'blueprint': VAN, 'color': '200,160,20', 'transform': anchor},
            {'key': 'second_reference', 'blueprint': TRUCK, 'transform': second},
            {'key': 'distractor_0', 'blueprint': CAR, 'color': COLORS[color], 'transform': d1},
            {'key': 'distractor_1', 'blueprint': CAR, 'color': COLORS[color], 'transform': d2},
        ] + extra
        relations = [{'relation_id': 'r0', 'subject': 'target', 'object': 'reference', 'relation': 'near'}]
        if family == 'compositional_reference':
            relations.append({'relation_id': 'r1', 'subject': 'target', 'object': 'second_reference', 'relation': relation})
            instruction = f'Find the {color} car near the yellow van and {relation} of the fire engine.'
        elif family == 'progressive_discovery':
            relations.append({'relation_id': 'r1', 'subject': 'reference', 'object': 'second_reference', 'relation': relation})
            instruction = f'Locate the {color} car near the yellow van that is {relation} of the fire engine.'
        else:
            relations.append({'relation_id': 'r1', 'subject': 'target', 'object': 'reference', 'relation': relation})
            instruction = f'Find the {color} car near and {relation} of the yellow van.'
            if family == 'cross_view_disambiguation':
                instruction += ' Use world directions, not image left or right.'
        positions = {o['key']: enu(o['transform']) for o in objects}
        scene = {**spec, 'construction_version': VERSION, 'status': 'geometry_compiled',
                 'instruction': instruction, 'objects': objects,
                 'geometry_sha256': hashlib.sha256(canonical(objects).encode()).hexdigest(),
                 'generation_attempt': attempt, 'layout_search_attempt': layout_attempt,
                 'private_truth': {'target_id': 'target', 'target_position': positions['target'],
                     'candidate_positions': {k: v for k, v in positions.items()
                                            if k == 'target' or k.startswith('distractor')},
                     'positions': positions, 'required_relations': relations},
                 'runtime_validation': {'spawn': False, 'visual_condition': False,
                                        'progressive_disclosure': False, 'flight': False},
                 'progressive_boundary': 'Nested reference ambiguity is only a candidate. Actual motion must reveal initially unseen task-relevant intermediary evidence. A template or preview is not progressive disclosure.'}
        audit = counterfactual_audit(scene)
        if audit['unique_target'] and audit['every_relation_target_informative']:
            scene['private_truth']['counterfactual_audit'] = audit
            if candidate_accept is not None and not candidate_accept(scene):continue
            return scene
    raise ValueError('No informative native layout within deterministic search bound')
