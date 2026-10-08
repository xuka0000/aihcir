"""Two informative car choices for the cross-view development task.

The v5 native-image review found its third, relation-redundant car unreadable
under a tree. Retain the target and the competing car near the other van;
both instruction relations remain counterfactually necessary.
"""

import hashlib
import copy

from benchmark import canonical
from scene_catalog_crossview_v4 import recipe as previous_recipe
from scene_catalog_v2 import counterfactual_audit


VERSION = 'crossview_two_sector_candidates_v6'
CLEAR_TWO_PROFILE = 'crossview_center_sectors_clear_v2_independent_clearance'


def recipe(spec, points, blueprint_ids, attempt=0, max_layout_attempts=256,candidate_accept=None):
    def accept(candidate):
        if candidate_accept is None:return True
        candidate=copy.deepcopy(candidate)
        candidate['objects']=[o for o in candidate['objects'] if o['key']!='distractor_1']
        for key in ('positions','candidate_positions'):candidate['private_truth'][key].pop('distractor_1',None)
        return candidate_accept(candidate)
    scene = previous_recipe(spec, points, blueprint_ids, attempt, max_layout_attempts,accept)
    scene['objects'] = [obj for obj in scene['objects'] if obj['key'] != 'distractor_1']
    scene['construction_version'] = VERSION
    if scene['visual_condition_requested'] == 'clear' and scene['team_size'] == 2:
        scene['survey_profile_required'] = CLEAR_TWO_PROFILE
    scene['geometry_sha256'] = hashlib.sha256(canonical(scene['objects']).encode()).hexdigest()
    del scene['private_truth']['positions']['distractor_1']
    del scene['private_truth']['candidate_positions']['distractor_1']
    audit = counterfactual_audit(scene)
    if not audit['unique_target'] or not audit['every_relation_target_informative']:
        raise ValueError('Two-candidate relation structure is not informative')
    scene['private_truth']['counterfactual_audit'] = audit
    return scene
