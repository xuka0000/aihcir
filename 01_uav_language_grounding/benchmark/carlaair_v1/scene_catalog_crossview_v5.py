"""Predeclared central-candidate revision of the two-sector development scenes.

The spare car is placed on a native road point nearest the fixed central
fourth-round viewpoint. The target and required relations stay unchanged.
Physical visibility and instruction recognition remain runtime questions.
"""

import hashlib
import math

from benchmark import canonical
from scene_catalog_crossview_v4 import recipe as previous_recipe
from scene_catalog_v2 import counterfactual_audit, enu


VERSION = 'crossview_two_sector_candidates_v5'
CLEAR_TWO_PROFILE = 'crossview_center_sectors_clear_v2_independent_clearance'


def recipe(spec, points, blueprint_ids, attempt=0, max_layout_attempts=256):
    scene = previous_recipe(spec, points, blueprint_ids, attempt, max_layout_attempts)
    center = scene['search_region_center_xy']
    occupied = [obj['transform'] for obj in scene['objects'] if obj['key'] != 'distractor_1']
    candidates = [tuple(float(v) for v in point) for point in points
                  if abs(point[0] - center[0]) <= 35
                  and 30 <= point[1] - center[1] <= 90
                  and all(math.dist(point[:2], other[:2]) >= 25 for other in occupied)]
    if not candidates:
        raise ValueError('No native road point in the fixed central candidate zone')
    spare = min(candidates, key=lambda point:
                (math.dist(point[:2], (center[0], center[1] + 55)), point))
    for obj in scene['objects']:
        if obj['key'] == 'distractor_1':
            obj['transform'] = spare
            break
    scene['construction_version'] = VERSION
    if scene['visual_condition_requested'] == 'clear' and scene['team_size'] == 2:
        scene['survey_profile_required'] = CLEAR_TWO_PROFILE
    scene['geometry_sha256'] = hashlib.sha256(canonical(scene['objects']).encode()).hexdigest()
    scene['private_truth']['positions']['distractor_1'] = enu(spare)
    scene['private_truth']['candidate_positions']['distractor_1'] = enu(spare)
    scene['private_truth']['crossview_construction']['spare_candidate_policy'] = (
        'nearest native road point to fixed central fourth-round view, '
        'subject to prespecified separation and region bounds')
    audit = counterfactual_audit(scene)
    if not audit['unique_target'] or not audit['every_relation_target_informative']:
        raise ValueError('Central candidate revision changed task semantics')
    scene['private_truth']['counterfactual_audit'] = audit
    return scene
