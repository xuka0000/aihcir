"""Strict split-aware dataset addressing; never relabel held-out scenes train."""
import re
from pathlib import Path


def episode_split(episode):
    match=re.fullmatch(r'(train|validation|test)_\d{6}',episode)
    if match is None:raise ValueError('Invalid frozen episode identifier')
    return match.group(1)


def scene_path(root,episode,assets_root=None):
    split=episode_split(episode)
    if assets_root is None:
        return Path(root)/'scene_assets_v3'/split/episode/'scene.json'
    base=Path(root).resolve()
    assets=Path(assets_root).resolve()
    if not assets.is_relative_to(base/'releases') or split=='test':
        raise ValueError('Candidate assets must be in an isolated release; test remains closed')
    return assets/split/episode/'scene.json'


def validate_collection_source(scene,candidate_release=False):
    if scene.get('status')=='scene_assets_instantiated':
        return
    if (candidate_release and scene.get('status') == 'geometry_compiled'
            and scene.get('construction_version') in ('crossview_two_sector_candidates_v4',
                                                      'crossview_two_sector_candidates_v5',
                                                      'crossview_two_sector_candidates_v6',
                                                      'crossview_two_sector_candidates_v7')
            and scene.get('split') in ('train', 'validation')
            and scene.get('task_family') == 'cross_view_disambiguation'):
        from scene_catalog_crossview_v4 import PROFILES_BY_CONDITION
        from scene_catalog_crossview_v5 import CLEAR_TWO_PROFILE
        from scene_catalog_v2 import counterfactual_audit
        truth = scene.get('private_truth', {})
        expected_profile = PROFILES_BY_CONDITION.get(scene.get('visual_condition_requested'))
        if (scene.get('construction_version') in ('crossview_two_sector_candidates_v5',
                                                 'crossview_two_sector_candidates_v6')
                and scene.get('visual_condition_requested') == 'clear'
                and scene.get('team_size') == 2):
            expected_profile = CLEAR_TWO_PROFILE
        if (scene.get('construction_version') == 'crossview_two_sector_candidates_v7'
                and scene.get('visual_condition_requested') == 'clear'):
            expected_profile = CLEAR_TWO_PROFILE
        if (scene.get('survey_profile_required') == expected_profile
                and truth.get('counterfactual_audit') == counterfactual_audit(scene)
                and truth['counterfactual_audit']['unique_target'] is True
                and truth['counterfactual_audit']['every_relation_target_informative'] is True):
            return  # Admission to physical measurement only; test remains closed.
    if (candidate_release and scene.get('status')=='geometry_compiled'
            and scene.get('construction_version') in ('color_asset_diagnostic_candidate_v1',
                                                       'color_asset_diagnostic_candidate_v2')
            and episode_split(scene.get('episode_id',''))=='train' and scene.get('split')=='train'
            and scene.get('diagnostic_evidence',{}).get('benchmark_admitted') is False):
        return  # collection_request checks exact source-bound reconstruction.
    if (candidate_release and scene.get('status')=='geometry_compiled'
            and scene.get('construction_version')=='condition_render_preview_candidate_v1'
            and scene.get('split')=='validation'):
        from scene_catalog_v2 import counterfactual_audit
        audit=counterfactual_audit(scene)
        if (audit.get('unique_target') is True
                and audit.get('every_relation_target_informative') is True
                and audit==scene.get('private_truth',{}).get('counterfactual_audit')
                and scene.get('preview_evidence',{}).get('result_sha256')):
            return  # Full source/hash verification remains mandatory in collection_request.
    if (candidate_release and scene.get('status')=='geometry_compiled'
            and scene.get('construction_version')=='crossview_three_endpoint_candidates_v3'
            and scene.get('split')=='train' and scene.get('task_family')=='cross_view_disambiguation'):
        from scene_catalog_v2 import counterfactual_audit
        checked=counterfactual_audit(scene)
        endpoints={r[k] for r in scene['private_truth']['required_relations'] for k in ('subject','object')}
        if (checked['unique_target'] and checked['every_relation_target_informative']
                and checked==scene['private_truth'].get('counterfactual_audit')
                and endpoints=={'target','reference','second_reference'}):
            return
    if (candidate_release and scene.get('status')=='geometry_compiled'
            and scene.get('construction_version')=='latent_horizontal_direction_training_candidate_v2'
            and scene.get('split')=='train' and scene.get('task_family')=='progressive_discovery'):
        from audit_latent_direction import validate_semantics
        validate_semantics(scene)
        return  # Geometry admission to measurement only; no runtime certification.
    if (candidate_release and scene.get('status')=='geometry_compiled'
            and scene.get('construction_version')=='latent_horizontal_direction_candidate_v1'
            and scene.get('split')=='validation' and scene.get('task_family')=='progressive_discovery'):
        from scene_catalog_progressive_candidates import audit
        result=audit(scene)
        if all(result[k] is True for k in ('unique_target','every_task_constraint_informative','marker_side_changes_target')):
            # Admission to physical measurement only. Existing semantic gate
            # continues to reject progressive scenes, including these candidates.
            return
    if (candidate_release and scene.get('status')=='geometry_compiled'
            and scene.get('construction_version')=='crossview_three_endpoint_candidates_v3'
            and scene.get('split')=='validation'
            and scene.get('task_family')=='cross_view_disambiguation'):
        audit=scene.get('private_truth',{}).get('counterfactual_audit',{})
        if audit.get('unique_target') is True and audit.get('every_relation_target_informative') is True:
            return
    raise ValueError('Source is neither instantiated nor an explicit audited construction candidate')


def selected_split(episodes):
    splits={episode_split(x) for x in episodes}
    if len(splits)!=1:raise ValueError('Collection must not mix dataset splits')
    return next(iter(splits))
