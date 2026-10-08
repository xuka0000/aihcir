"""Versioned, private audit of observation-dependent horizontal direction.

This leaves the historical semantic gate unchanged. A pass is a necessary
construction/observability screen, not dataset admission or model validation.
No model or public candidate builder may import this truth-assisted module.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_semantic_gate import audit_episode, validate_renderer_control, VIEW, MIN_PROGRESSIVE_MOTION_M
from scene_catalog_progressive_candidates import VERSION, TRAINING_VERSION, audit, direction
from latent_direction_control import INSTRUCTION

SCHEMA = 'latent_direction_observability_v1'


def initial_ambiguity_support(record, counterfactual, legacy):
    """Additional prospective construction check; never changes old profiles."""
    required={'target','reference','second_reference'}
    for assignment in counterfactual['marker_reflection_matches']:required.update(assignment.values())
    rounds={row['round']:set().union(*(set(v) for v in row['visible_by_agent'].values())) for row in legacy['rounds']}
    missing={str(t):sorted(required-rounds.get(t,set())) for t in (0,1)}
    early=[o for o in record['observations'] if int(VIEW.fullmatch(o['image'])[1])<2]
    marker_hidden=bool(early) and all(record['private_pixel_measurements'][o['image']]['direction_marker'].get('visible_pixels')==0
        and all(record['private_pixel_measurements'][o['image']]['direction_marker'].get(k) is True
                for k in ('visible_subset_of_isolated','visible_aba_exact','isolated_aba_exact')) for o in early)
    return dict(required_objects=sorted(required),missing_by_initial_round=missing,
        marker_hidden_for_first_two_rounds=marker_hidden,
        passed=all(not v for v in missing.values()) and marker_hidden,
        boundary='Verified pixel support for both symbolic target assignments before marker disclosure; not evidence of a model belief or an independent causal intervention.')


def validate_semantics(scene):
    allowed_split = {VERSION: 'validation', TRAINING_VERSION: 'train'}.get(scene.get('construction_version'))
    if (allowed_split is None
            or scene.get('task_family') != 'progressive_discovery'
            or scene.get('split') != allowed_split
            or scene.get('instruction') != INSTRUCTION):
        raise ValueError('Unsupported latent-direction task contract')
    objects = {o['key']: o for o in scene['objects']}
    if len(objects) != len(scene['objects']):
        raise ValueError('Duplicate scene object identity')
    side = direction(objects['direction_marker']['transform'], objects['second_reference']['transform'])
    expected = {('target', 'near', 'reference'), ('reference', side, 'second_reference'),
                ('direction_marker', side, 'second_reference')}
    relations = scene['private_truth']['required_relations']
    actual = {(r['subject'], r['relation'], r['object']) for r in relations}
    if side is None or len(relations) != 3 or actual != expected:
        raise ValueError('Stored relation truth differs from construction geometry')
    counterfactual = audit(scene)  # Recompute; do not trust cached success flags.
    if not all(counterfactual[k] is True for k in (
            'unique_target', 'every_task_constraint_informative', 'marker_side_changes_target')):
        raise ValueError('Latent task lacks uniqueness or relation necessity')
    return counterfactual


def audit_latent(record, scene, renderer_control_passed):
    counterfactual = validate_semantics(scene)
    legacy = audit_episode(record, scene, renderer_control_passed)
    # Replace only the explicit unimplemented-semantics blocker after checking
    # the exact new contract. Every other historical blocker remains effective.
    blockers = [b for b in legacy['blockers'] if b != 'implicit_relation_generation_not_implemented']
    if set(record['private_native_instances']) != {o['key'] for o in scene['objects']}:
        raise ValueError('Capture does not measure every constructed object')
    initial = [o for o in record['observations'] if int(VIEW.fullmatch(o['image'])[1]) == 0]
    hidden = all(record['private_pixel_measurements'][o['image']]['direction_marker'].get('visible_pixels') == 0
                 and all(record['private_pixel_measurements'][o['image']]['direction_marker'].get(k) is True
                         for k in ('visible_subset_of_isolated', 'visible_aba_exact', 'isolated_aba_exact'))
                 for o in initial)
    if not initial or not hidden:
        blockers.append('marker_not_demonstrably_absent_from_all_initial_views')
    visibility = {row['round']: row['visible_by_agent'] for row in legacy['rounds']}
    witnesses, same_view_witnesses = [], []
    for witness in legacy['progressive_physical_witnesses']:
        if (witness['intermediary'] == 'direction_marker'
                and witness['measured_displacement_m'] >= MIN_PROGRESSIVE_MOTION_M
                and 'second_reference' in set().union(*(
                    set(keys) for keys in visibility[witness['round']].values()))):
            witnesses.append(witness)
            if 'second_reference' in visibility[witness['round']][witness['agent']]:
                same_view_witnesses.append(witness)
    if not witnesses:
        blockers.append('no_moved_marker_view_with_same_round_anchor_evidence')
    stronger=record.get('private_construction_route',{}).get('initial_two_hypothesis_support_required') is True
    support=initial_ambiguity_support(record,counterfactual,legacy) if stronger else None
    if stronger and not support['passed']:blockers.append('initial_two_hypothesis_support_not_observed')
    if stronger and not any(w['round']>=2 for w in witnesses):blockers.append('late_marker_disclosure_not_observed')
    # Keep the historical necessary screen unchanged. Validation candidates
    # receive a separate, stricter diagnostic even when their old route did
    # not request initial two-hypothesis support.
    strict_support = (support if support is not None else
                      initial_ambiguity_support(record,counterfactual,legacy)) if scene['split']=='validation' else None
    strict_validation_candidate_screen = bool(
        scene['split']=='validation' and not blockers and strict_support['passed']
        and any(w['round']>=2 for w in witnesses))
    return dict(schema='latent_direction_initial_ambiguity_v2' if stronger else SCHEMA, episode_id=record['episode_id'],
                initial_two_hypothesis_support=support,
                strict_validation_initial_support=strict_support,
                strict_validation_candidate_screen_passed=strict_validation_candidate_screen,
                counterfactual=counterfactual, marker_initially_absent=hidden,
                disclosure_witnesses=witnesses, legacy_audit=legacy,
                same_view_direction_witnesses=same_view_witnesses,
                direction_evidence_scope='Marker and anchor may occupy different calibrated team views '
                                         'in the same round, as permitted by the multi-view task.',
                blockers=blockers, necessary_semantic_screen_passed=not blockers,
                benchmark_admission=False, independent_test=False,
                boundary='Private construction and physical evidence audit, not model success. '
                         'Language/visual shortcut review and dataset freeze remain separate.')


def audit_files(observation, scene, renderer):
    observation, scene, renderer = map(Path, (observation, scene, renderer))
    record = json.loads(observation.read_text(encoding='utf-8'))
    digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    if record['source_sha256'] != digest:
        raise ValueError('Scene changed after physical collection')
    control = validate_renderer_control(json.loads(renderer.read_text(encoding='utf-8')))
    result = audit_latent(record, json.loads(scene.read_text(encoding='utf-8')), control['passed'])
    result['source_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (observation, scene, renderer)}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('observations', 'scene', 'renderer', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    result = audit_files(a.observations, a.scene, a.renderer)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'passed': result['necessary_semantic_screen_passed'], 'blockers': result['blockers']}))
