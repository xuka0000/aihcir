"""Freeze the user-approved smaller cohort without changing parent task IDs."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'evidence/oct07_image_campaign_20261003_v3_scene_repair'
SCOPE = 'oct07_image_campaign_20261003_v3_scene_repair/tonight-core-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def selection(parent):
    with (parent / 'first_manuscript_scene_ids.csv').open(encoding='utf-8-sig', newline='') as stream:
        scenes = list(csv.DictReader(stream))
    selected = [r for r in scenes if r['split'] == 'train']
    cells = sorted({(r['task_family'], r['visual_condition']) for r in scenes if r['split'] == 'validation'})
    for index, cell in enumerate(cells):
        team = '2' if index % 2 == 0 else '4'
        selected.append(min((r for r in scenes if r['split'] == 'validation'
                             and (r['task_family'], r['visual_condition']) == cell
                             and r['team_size'] == team), key=lambda r: r['episode_id']))
    for stratum in range(24):
        selected.append(min((r for r in scenes if r['split'] == 'test' and int(r['stratum']) == stratum),
                            key=lambda r: (int(r['replicate']), r['episode_id'])))
    return selected


def contract(parent_contract, plan, specification_hashes):
    value = copy.deepcopy(parent_contract)
    value['campaign_id'] = SCOPE
    value['matrix_sha256'] = plan['matrix_sha256']
    value['models'] = {'qwen3': value['models']['qwen3']}
    value['data'].update(development_scenes=12, formal_specs=24, formal_per_stratum=1,
                         formal_runtime_admitted=0, capture_scenes=156)
    value['units'].update(image_episode_rows=5124, development_rows=1260, formal_rows=3864,
                          training_fits=12, native_bbox_frame_rows=432,
                          public_baseline_development=828, public_baseline_formal=1656)
    value['training']['fits'] = 12
    value['counts'] = plan['counts']
    value['baseline_first'].update(public_rows=828, baseline_stage_rows=900)
    value['specification_sha256'] = specification_hashes
    value['compute_stop'] = plan['phase_deadlines']['compute']
    value['delivery_stop'] = plan['phase_deadlines']['delivery']
    value['deadline'] = value['delivery_stop']
    value['sources']['native_frame_rows'] = dict(development=144, formal=288)
    value['concurrency'].update(model_slots_max=plan.get('model_slots_max',48),
                                 capture_slots_max=plan.get('capture_slots_max',32),
                                 effective_model_slots_measured=None, effective_capture_slots_measured=None)
    value['reporting_scope'] = dict(id=SCOPE, parent_directory='..',
        parent_contract_sha256=plan['parent']['contract_sha256'],
        parent_matrix_sha256=plan['parent']['matrix_sha256'],
        task_ledger='parent/queue.sqlite', parent_fit_ceiling=24,
        authorization_file='tonight_core_authorization_20261007.json',
        selection_basis=plan['selection_rule'], first_TEST_requires_complete_DEV=1260,
        first_TEST_requires_complete_formal=3864, parent_TEST_gate_unchanged=True,
        later_methods_must_match_first_DEV_freeze=True)
    return value


def verify(root):
    root = Path(root).resolve()
    parent = root.parent
    plan = read(parent / 'tonight_core_protocol_20261007.json')
    approval = read(parent / 'tonight_core_authorization_20261007.json')
    priority = read(ROOT / '.codex/task_priority.json')
    registered = priority.get('tonight_execution_authorization', {})
    if (root.name != 'tonight_runtime' or approval.get('approved') is not True
            or registered.get('receipt_sha256') != sha(parent / 'tonight_core_authorization_20261007.json')
            or registered.get('scope_id') != SCOPE):
        raise ValueError('Current user-authorized tonight scope is not registered')
    if (sha(parent / 'contract.json') != plan['parent']['contract_sha256']
            or sha(parent / 'matrix.csv') != plan['parent']['matrix_sha256']):
        raise ValueError('Parent scientific identity changed')
    spec_hashes = {split: sha(root / 'specifications' / (split + '.specs.jsonl'))
                   for split in ('train', 'validation', 'test')}
    if read(root / 'contract.json') != contract(read(parent / 'contract.json'), plan, spec_hashes):
        raise ValueError('Tonight contract differs from the frozen prospective cohort')
    if sha(root / 'matrix.csv') != plan['matrix_sha256']:
        raise ValueError('Tonight matrix changed')
    selected = selection(parent)
    for split, count in [('train', 120), ('validation', 12), ('test', 24)]:
        ids = {r['episode_id'] for r in selected if r['split'] == split}
        source = {s['episode_id']: s for s in map(json.loads,
            (parent / 'specifications' / (split + '.specs.jsonl')).read_text(encoding='utf-8').splitlines())}
        actual = list(map(json.loads, (root / 'specifications' / (split + '.specs.jsonl')).read_text(encoding='utf-8').splitlines()))
        if len(ids) != count or len(actual) != count or {s['episode_id'] for s in actual} != ids or any(s != source[s['episode_id']] for s in actual):
            raise ValueError('Prospective scene selection or frozen scene content changed')
    if (root / 'queue.sqlite').exists():
        raise ValueError('A second scientific task ledger is prohibited')
    return dict(scope_id=SCOPE, planned_algorithm_rows=5124, planned_scenes=156,
                fits=12, native_bbox=432, parent_paused=True, scientific_acceptance=False,
                shared_ledger=str(parent / 'queue.sqlite'))


def prepare(parent=PARENT):
    parent = Path(parent).resolve()
    root = parent / 'tonight_runtime'
    if root.exists():
        return verify(root)
    selected = selection(parent)
    ids = {r['episode_id'] for r in selected if r['split'] != 'train'}
    test = {r['episode_id'] for r in selected if r['split'] == 'test'}
    with (parent / 'matrix.csv').open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream))
    rows = [r for r in original if r['backbone'] == 'qwen3' and
            ((r['kind'] in ('main', 'ablation', 'all_view_reference') and r['case_id'] in ids)
             or (r['kind'] in ('sensitivity', 'robustness') and r['run_class'] == 'formal' and r['case_id'] in test))]
    from collections import Counter
    counts = dict(Counter(r['run_class'] + '/' + r['kind'] for r in rows))
    if len(rows) != 5124 or len({r['task_id'] for r in rows}) != 5124:
        raise ValueError('Exact approved matrix must contain 5124 unique original task IDs')
    root.mkdir()
    with (root / 'matrix.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / 'scene_ids.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    (root / 'specifications').mkdir()
    for split in ('train', 'validation', 'test'):
        chosen = {r['episode_id'] for r in selected if r['split'] == split}
        lines = [line for line in (parent / 'specifications' / (split + '.specs.jsonl')).read_text(encoding='utf-8').splitlines()
                 if json.loads(line)['episode_id'] in chosen]
        (root / 'specifications' / (split + '.specs.jsonl')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plan = dict(schema='oct07_tonight_core_v1', scope_id=SCOPE, observed_at=time.time(),
        authorization='User chose tonight priority smaller complete cohort, then Implement the plan',
        selection_rule='TRAIN120 unchanged; sorted12 task/visual DEV cells alternate2/4UAV; lowest TEST replicate/ID per original24 strata, no outcomes',
        parent=dict(contract_sha256=sha(parent / 'contract.json'), matrix_sha256=sha(parent / 'matrix.csv'),
                    rows=60816, paused=True, original_first_rows=7560),
        matrix_sha256=sha(root / 'matrix.csv'), counts=counts,
        units=dict(scenes=156, train=120, development=12, test=24, algorithm_rows=5124,
                   development_rows=1260, formal_rows=3864, fits=12, native_bbox=432),
        TEST_quality_opened=False, results_used_for_selection=False, scientific_acceptance=False,
        healthy_results_reused_only_after_source_data_model_evaluator_hash_verification=True,
        display_types=['cohort_fairness', 'visual_frontends', 'twelve_baseline_quality_cost',
            'paired_effect_uncertainty', 'conditioning_ablations', 'task_visual_team_conditions',
            'cap_horizon_sensitivity', 'drop_pose_order_robustness', 'quality_cost',
            'decision_packet_receiver_trace', 'RGB_episode_and_counterexample', 'training_reliability'],
        executor='scripts/execute_oct07_admitted_pipeline.py',
        phase_deadlines=dict(repair='2026-10-07T17:00:00+08:00', capture='2026-10-07T18:30:00+08:00',
            fits_DEV_freeze='2026-10-07T20:30:00+08:00', compute='2026-10-07T21:30:00+08:00',
            shutdown='2026-10-07T22:00:00+08:00', delivery='2026-10-07T23:59:59+08:00'))
    write(parent / 'tonight_core_protocol_20261007.json', plan)
    specs = {s: sha(root / 'specifications' / (s + '.specs.jsonl')) for s in ('train', 'validation', 'test')}
    write(root / 'contract.json', contract(read(parent / 'contract.json'), plan, specs))
    for name in ('processor_manifest.json', 'author_source_registry.json', 'manuscript_baseline_citations.json',
                 'runtime_dependency_contract.json', 'retrieval_model_source.json', 'feature_encoder.json', 'native_assets_manifest.json'):
        if (parent / name).is_file():
            os.link(parent / name, root / name)
    for name in ('author_sources', 'task_prompts', 'reference_pdfs'):
        for path in (parent / name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                target = root / path.relative_to(parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
    return verify(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(PARENT / 'tonight_runtime') if args.verify else prepare()))
