"""Freeze a reporting cohort and export existing development evidence, without jobs.

The parent contract/matrix, budgets, queue and TEST seal are never changed here.
This is a preparation command; missing execution gates remain explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'evidence/oct07_image_campaign_20261003_v3_scene_repair'
KINDS = {'main', 'ablation', 'all_view_reference'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows, fields):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def specifications(root, split):
    return [json.loads(line) for line in
            (root / 'specifications' / (split + '.specs.jsonl')).read_text(encoding='utf-8').splitlines() if line]


def select_test(specs):
    """Select from design metadata only; no result, timing or quality input."""
    groups = defaultdict(list)
    for spec in specs:
        groups[(spec['task_family'], spec['visual_condition_requested'], spec['team_size'])].append(spec)
    if len(groups) != 24 or any(len(group) != 10 for group in groups.values()):
        raise ValueError('Frozen parent TEST240/24-stratum design differs')
    chosen = [s for key in sorted(groups) for s in sorted(groups[key], key=lambda x: x['episode_id'])[:2]]
    if len({s['episode_id'] for s in chosen}) != 48:
        raise ValueError('Selected TEST IDs are not unique')
    return chosen


def scope_rows(rows, chosen):
    ids = {s['episode_id'] for s in chosen}
    return [r for r in rows if r['backbone'] == 'qwen3' and r['kind'] in KINDS
            and (r['run_class'] == 'development' or r['case_id'] in ids)]


def bootstrap(values, seed=20261004):
    """Descriptive scene bootstrap; degenerate empirical data have unknown CI."""
    if len(values) < 2 or len(set(values)) < 2:
        return None
    import numpy as np
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), (10000, len(data)))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def summarize(rows, group_fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_fields)].append(row)
    result = []
    for keys, group in sorted(groups.items(), key=lambda x: str(x[0])):
        scenes = defaultdict(list)
        for row in group:
            scenes[row['case_id']].append(row)
        item = dict(zip(group_fields, keys))
        item.update(rows=len(group), scene_contents=len(scenes), effective_independent_scenes=None)
        item['metrics'] = {}
        for metric in metrics:
            means = [statistics.mean(float(r[metric]) for r in records if r.get(metric) is not None)
                     for records in scenes.values() if any(r.get(metric) is not None for r in records)]
            item['metrics'][metric] = dict(scene_mean=statistics.mean(means) if means else None,
                descriptive_scene_bootstrap95=bootstrap(means), scene_contents_measured=len(means))
        result.append(item)
    return result


def historical_data(root):
    sources = {
        'component_original': ROOT / 'evidence/own_bundle_campaign_20261002_v1/remote_results/results.json',
        'component_extension': ROOT / 'evidence/component_extension/remote_results/results.json',
        'rgb': ROOT / 'evidence/own_rgb_baselines_20261002_v3/remote_results/results_rolemap_v2.json',
        'component_audit': ROOT / 'evidence/component_extension/audit.json',
        'rgb_audit': ROOT / 'evidence/own_rgb_baselines_20261002_v3/role_scoring_audit_v2.json',
    }
    component = read(sources['component_original'])['rows'] + read(sources['component_extension'])['rows']
    rgb_source = read(sources['rgb'])
    rgb = rgb_source['rows']
    if len(component) != 1105 or len(rgb) != 156:
        raise ValueError('Historical fixed denominators changed')
    for rows in (component, rgb):
        if len({r['id'] for r in rows}) != len(rows) or len({r['case_id'] for r in rows}) != 13:
            raise ValueError('Historical result ID or scene coverage differs')
        if any(not isinstance(r['target_correct'], bool) for r in rows):
            raise ValueError('Unknown target correctness cannot be imputed')
    cells = Counter((r['model'], r['mode'], r['case_id']) for r in rgb)
    if len(cells) != 78 or set(cells.values()) != {2}:
        raise ValueError('RGB complete two-presentation cell design differs')
    if rgb_source['metric_version'] != 'private-identity-to-public-role-v2' or rgb_source['formal_test_rows'] != 0:
        raise ValueError('Historical RGB metric or data-use identity differs')
    component_fields = ['id', 'case_id', 'method', 'seed', 'cap', 'rounds', 'noise', 'section',
                        'target_correct', 'exact_relation_success', 'communication_bytes', 'seconds',
                        'packet_count', 'control_bytes']
    rgb_fields = ['id', 'case_id', 'model', 'mode', 'presentation', 'target_correct', 'exact_role_success',
                  'grammar_valid', 'seconds', 'input_tokens', 'output_tokens', 'image_count',
                  'raw_pixel_count', 'peak_cuda_bytes']
    write_csv(root / 'first_existing_component_rows.csv', component, component_fields)
    write_csv(root / 'first_existing_rgb_rows.csv', rgb, rgb_fields)
    summary = dict(schema='retrospective_development_evidence_v1', observed_at=time.time(),
        component_rows=1105, rgb_rows=156, formal_test_rows=0, scene_contents=13,
        effective_independent_scenes=None, new_model_calls=0, excluded_canonical_rows=0,
        sources={k: dict(path=v.relative_to(ROOT).as_posix(), sha256=sha(v)) for k, v in sources.items()},
        component_cells=summarize(component, ['section', 'method', 'cap', 'rounds', 'noise'],
            ['target_correct', 'exact_relation_success', 'communication_bytes', 'seconds']),
        rgb_cells=summarize(rgb, ['model', 'mode'],
            ['target_correct', 'exact_role_success', 'seconds', 'image_count', 'input_tokens']),
        statistics=dict(unit='Scene content after seed/presentation averaging within a configuration',
            interval='10000 descriptive scene bootstrap resamples, seed20261004; degenerate empirical CI=null',
            interpretation='Retrospective development diagnosis; scene independence unverified; no confirmatory inference'),
        claims=dict(usable='Development evidence of role binding failures and component budget behavior',
            unsupported=['Twelve official baseline comparison', 'Full image-communication TRACE improvement',
                         'Independent TEST performance', 'General superiority or statistical significance']),
        datasets_are_separate=True, historical_metrics_not_mixed_with_new_formal_evaluator=True,
        scientific_acceptance=False, user_acceptance=False)
    write_new(root / 'first_existing_evidence_summary.json', summary)
    return summary


def freeze(root):
    root = Path(root)
    contract = read(root / 'contract.json')
    if sha(root / 'matrix.csv') != contract['matrix_sha256']:
        raise ValueError('Parent matrix binding changed')
    with (root / 'matrix.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if len(rows) != len({r['task_id'] for r in rows}) or len(rows) != 60816:
        raise ValueError('Parent unique task denominator differs')
    for split in ('train', 'validation', 'test'):
        name = split + '.specs.jsonl'
        expected = contract['specification_sha256'].get(split)
        if expected is not None and sha(root / 'specifications' / name) != expected:
            raise ValueError('Frozen scene specifications differ: ' + split)
    selected = select_test(specifications(root, 'test'))
    chosen = scope_rows(rows, selected)
    classes = Counter(r['run_class'] for r in chosen)
    public = set(contract['sources']['author_methods'])
    if classes != {'development': 2520, 'formal': 5040} or len(public) != 12:
        raise ValueError('First reporting scope or all12 baseline set differs')
    if {r['method'] for r in chosen if r['method'] in public} != public:
        raise ValueError('A public method was dropped')
    public_counts = Counter(r['run_class'] for r in chosen if r['method'] in public)
    if public_counts != {'development': 1656, 'formal': 3312}:
        raise ValueError('Public baseline fixed denominator differs')
    if any(r['cap_multiple'] != ('0' if r['kind'] == 'all_view_reference' else '4')
           or r['horizon'] != '2' or r['perturbation'] != 'none' for r in chosen):
        raise ValueError('First scope must preserve the common original budget')
    first_matrix = root / 'first_manuscript_matrix.csv'
    write_csv(first_matrix, chosen, fields)
    scene_rows = [dict(split=s['split'], episode_id=s['episode_id'], task_family=s['task_family'],
        visual_condition=s['visual_condition_requested'], team_size=s['team_size'], stratum=s['stratum'],
        scene_seed=s['scene_seed'], replicate=s['replicate']) for s in
        specifications(root, 'train') + specifications(root, 'validation') + selected]
    write_csv(root / 'first_manuscript_scene_ids.csv', scene_rows, list(scene_rows[0]))
    native = {split: 2 * 2 * sum(int(r['team_size']) for r in scene_rows if r['split'] == split)
              for split in ('validation', 'test')}
    historical = historical_data(root)
    seconds = read(root / 'dependency_protocol.json')['capacity']['old_interface_extrapolation_public_only_hours']
    # Parent full public36432 rows/8 slots estimates are historical only.
    extrapolation = [x * 4968 / 36432 * 8 / 4 for x in seconds]
    gate = read(root / 'material_independent_audit.json')
    observed = time.time()
    plan = dict(schema='oct07_first_manuscript_preparation_v1', scope_id=contract['campaign_id'] + '/first-qwen3-v1',
        observed_at=observed, status='frozen_reporting_scope_execution_not_admitted',
        authorization=dict(user_choice='Preserve12 baselines; complete one backbone first',
            new_material_window_authorized=False, fresh_budget_required_before_material_measurement=True),
        parent=dict(contract_sha256=sha(root / 'contract.json'), matrix_sha256=sha(root / 'matrix.csv'),
            image_rows=60816, training_fits=24, scientific_contract_modified=False,
            live_queue_detected=(root / 'queue.sqlite').exists()),
        backbone='qwen3', backbone_choice_basis='Existing native image path, locked8B checkpoint and runtime reuse; no TEST quality used',
        public_baselines=contract['sources']['author_methods'], coela_teams=[2],
        selection=dict(rule='First two frozen episode IDs in each of24 task/visual/team strata; no outcomes used',
            statistical_unit='Physical layout/geometry cluster, not method row, seed or presentation',
            selected_test_scene_ids=[s['episode_id'] for s in selected], stratum_count=24,
            tested_scene_count=48, effective_independent_test_scenes=None, later_test_scenes=192,
            scene_specification_sha256={split: sha(root / 'specifications' / (split + '.specs.jsonl'))
                                        for split in ('train', 'validation', 'test')}),
        units=dict(image_episode_rows=7560, development_rows=2520, formal_rows=5040,
            public_development_rows=1656, public_formal_rows=3312,
            training_fits_first=12, training_fits_later=12, training_fits_total_limit=24,
            admitted_scenes=0, planned_scenes=192, TRAIN=120, DEV=24, TEST=48,
            common_frontend_scene_bundles=192, native_bbox_frame_rows=sum(native.values()),
            native_bbox_development=native['validation'], native_bbox_formal=native['test']),
        matrix_sha256=sha(first_matrix), scene_ids_sha256=sha(root / 'first_manuscript_scene_ids.csv'),
        count_by_method=dict(Counter(r['method'] for r in chosen)),
        scientific_budget=dict(call_budget=contract['call_budget'], packet=contract['packet'],
            training=contract['training'], compute_stop=contract['compute_stop'], delivery_stop=contract['deadline'],
            mechanism_candidate_limit=contract['algorithm_development']['max_new_candidates'],
            qualification_attempts_used=0, qualification_limit_seconds=10800, qualification_scenes=24,
            budgets_reset=False, failed_attempts_preserved=True),
        stage_order=['Repair native material source', 'Admit native material using a separately authorized bounded protocol',
            'Qualify24 TRAIN scenes once and reuse them', 'Capture remaining96 TRAIN +24 DEV +48 TEST',
            'Common native proposal frontend192; two native bbox methods864',
            'Freeze TRAIN120 B0 and100 official planner demonstrations',
            'Audit all12 upstream code/RGB paths and execute complete public DEV1656 plus matched controls',
            'Derive mechanism targets using DEV; public TEST3312 predictions may run sealed',
            'TRAIN-only counterfactual labels and12 fits; complete allDEV2520',
            'Freeze code/models from DEV only, then remaining TEST1728 predictions',
            'Verify allTEST5040 terminal, frozen code and raw-output identity; independently score complete cohort',
            'Scene/geometry paired statistics, efficiency, mechanism traces and evidence-limited manuscript'],
        test_policy=dict(parent_seal_modified=False, test_quality_opened=False,
            current_gate='Existing parent release still requires all53088 formal rows; unchanged',
            scope_gate_implementation_ready=False,
            scope_gate_required='Predeclared first48-only gate; complete DEV2520 and frozen version, all5040 formal rows terminal, independent864 native outputs; release no other TEST IDs',
            later_tuning='No method changes selected using first48 TEST scores; later test192 must evaluate the same DEV-frozen version for aggregate claims'),
        reuse=dict(parent_task_ids_preserved=True, scientific_queue_count=1,
            parent_results_promoted_only_after='Identical source/model/data/config/evaluator hash and frozen scientific version verification',
            no_duplicate_live_experiment=True, training120_reusable=True,
            automatic_replay=False, fixtures_are_not_scientific_results=True),
        deferred=['Second backbone', 'Other192 TEST scenes', 'Cap/horizon sensitivity', 'Robustness expansion'],
        manuscript_claims=dict(required=['Joint role grounding under the fixed cap', 'Actual communication/calls/pixels/latency',
                'Matched single-agent repeat/no-communication controls', 'Task/receiver/visual feature ablations',
                'Decision-state-effect trace with observable contradictory predictions'],
            limits='One backbone and48 designed scene contents; independence and physical diversity must be audited. No240-scene, cross-backbone or general SOTA claim.',
            negative_results_included=True, measured_mechanism_success=None, statistical_power=None),
        capacity=dict(remaining_compute_hours=(datetime.fromisoformat(contract['compute_stop']).timestamp()-observed)/3600,
            authorized_model_slots_max=8, proposed_qwen_slots=4, authorized_capture_slots_max=4,
            effective_parallel_slots_measured=None, formal_ETA_hours=None,
            historical_public4968_rows_four_slot_no_overhead_hours=extrapolation,
            extrapolation_is_not_calibrated_ETA=True, capture_training_source_setup_not_included=True,
            training_counterfactual_call_count=None, training_call_census_requires_actual_proposal_bundles=True,
            deadline_guaranteed=False),
        readiness=dict(scientific_ready=False, actual_remote_job_ids=[], actual_algorithm_results=0,
            material_gate_passed=gate['material_numeric_gate_passed'], unused_material_windows=0,
            missing=['Native material color/binding diagnosis and approved correction measurement',
                     'Per-scene resume and192-scene deployment with reuse of qualification24',
                     'Current GPU/native environment architecture and864 bbox matrix',
                     'All12 upstream mechanism/RGB acceptance', 'Single authoritative shard ledger and real observer',
                     'One-backbone12-fit identity verification and first-scope selection/release implementation']),
        supervision=dict(no_subagents_or_Codex_workers=True, observed_live_scientific_jobs=0,
            observer_live=False, offline_AI_supervision=False, intended_interval_seconds=300),
        storage=dict(C_floor_GiB=10, D_floor_GiB=30, peak_writes_and_live_reservations_required=True,
            archived_whole_instance=False, duplicated_models=False),
        reference=dict(path='reference_pdfs/agentverse_iclr2024.pdf', sha256=sha(root / 'reference_pdfs/agentverse_iclr2024.pdf'),
            actual_experiment_pages_read=[4, 5, 6, 7, 18], tables=[1, 2],
            inherited='Single/group comparisons, fixed agent setups, task-specific metrics and mechanism case evidence; retain negative group results',
            necessary_changes='Aerial native-image grounding, paid message/call/pixel budgets, all12 source methods, prospective48-scene first reporting cohort',
            not_claimed='Reproduction of original benchmark values or original full experiment suite',
            all12_experiment_PDF_and_runtime_fidelity_complete=False),
        existing_data=dict(component_rows=historical['component_rows'], rgb_rows=historical['rgb_rows'],
            formal_test_rows=0, use='Retrospective development/negative evidence only',
            summary='first_existing_evidence_summary.json'), scientific_acceptance=False, user_acceptance=False)
    write_new(root / 'first_manuscript_protocol.json', plan)
    return plan


def verify(root):
    root = Path(root)
    p = read(root / 'first_manuscript_protocol.json')
    if sha(root / 'contract.json') != p['parent']['contract_sha256'] or sha(root / 'matrix.csv') != p['parent']['matrix_sha256']:
        raise ValueError('Original parent contract/matrix mutated')
    with (root / 'matrix.csv').open(encoding='utf-8', newline='') as f:
        original = {r['task_id']: r for r in csv.DictReader(f)}
    with (root / 'first_manuscript_matrix.csv').open(encoding='utf-8', newline='') as f:
        chosen = list(csv.DictReader(f))
    selected = select_test(specifications(root, 'test'))
    expected = scope_rows(list(original.values()), selected)
    if chosen != expected or len(chosen) != len({r['task_id'] for r in chosen}) or len(chosen) != 7560:
        raise ValueError('First matrix differs from predeclared parent subset')
    if sha(root / 'first_manuscript_matrix.csv') != p['matrix_sha256'] or sha(root / 'first_manuscript_scene_ids.csv') != p['scene_ids_sha256']:
        raise ValueError('First subset file hash differs')
    if {r['method'] for r in chosen if r['method'] in p['public_baselines']} != set(p['public_baselines']):
        raise ValueError('Baseline method coverage differs')
    evidence = read(root / 'first_existing_evidence_summary.json')
    for binding in evidence['sources'].values():
        if sha(ROOT / binding['path']) != binding['sha256']:
            raise ValueError('Historical source changed')
    if any((root / name).exists() for name in ('test_results_release.json', 'algorithm_selection.json')):
        raise ValueError('Preparation cannot silently inherit opened TEST or an unknown scientific selection')
    report = dict(observed_at=time.time(), preparation_integrity_passed=True, source_methods=12,
        unique_first_rows=7560, original_parent_rows=60816, scenes=192, training_fit_limit=24,
        retrospective_rows=dict(component=1105, rgb=156), new_model_calls=0,
        scientific_ready=False, public_algorithm_rows_completed=0, TEST_opened=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=CAMPAIGN)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify:
        freeze(args.root)
    print(json.dumps(verify(args.root), ensure_ascii=False))
