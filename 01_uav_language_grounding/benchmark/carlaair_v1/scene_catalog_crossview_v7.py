"""Shared 40 m clear-condition route for both fixed team sizes.

The v6 two-agent real capture met the clear-image area threshold at 40 m.
Four-agent clear candidates now use the same prespecified physical scale.
All other task geometry and relation semantics remain those of v6.
"""

from scene_catalog_crossview_v6 import recipe as previous_recipe


VERSION = 'crossview_two_sector_candidates_v7'
CLEAR_PROFILE = 'crossview_center_sectors_clear_v2_independent_clearance'


def recipe(spec, points, blueprint_ids, attempt=0, max_layout_attempts=256,candidate_accept=None):
    scene = previous_recipe(spec, points, blueprint_ids, attempt, max_layout_attempts,candidate_accept)
    scene['construction_version'] = VERSION
    if scene['visual_condition_requested'] == 'clear':
        scene['survey_profile_required'] = CLEAR_PROFILE
    return scene
