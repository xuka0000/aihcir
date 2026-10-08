"""Freeze the user-approved Oct7 image campaign; never launch a cloud job."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/oct07_image_campaign_20261003_v1'
ID = OUT.name
SEEDS = (17, 29, 43)
BACKBONES = ('qwen3', 'internvl35_hf')
AUTHORS = ('react', 'tot', 'self_refine', 'coela', 'llamar', 'debate')
FAMILIES = ('direct_spatial', 'compositional_reference', 'progressive_discovery', 'cross_view_disambiguation')
CONDITIONS = ('clear', 'small_target', 'occluded')
CELLS = tuple(itertools.product(FAMILIES, CONDITIONS, (2, 4)))

def sha(path):
    # Capture's author-compatible CARLA environment is Python 3.10.
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def write(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if exclusive:
        with path.open('xb') as f: f.write(data)
    else:
        # Shards and their heartbeat threads may write the same status path.
        # Each publication owns its temporary file; the final rename is atomic.
        import uuid
        tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp'); tmp.write_bytes(data)
        # Windows readers can briefly hold the destination without delete
        # sharing. A bounded local rename retry must not kill the heartbeat.
        for attempt in range(20):
            try:tmp.replace(path);break
            except PermissionError:
                if attempt==19:raise
                time.sleep(.025)

def specs(split, count):
    for i in range(count):
        family, condition, team = CELLS[i % 24]; replicate = i // 24
        key = f'{ID}/{split}/{i}'
        yield dict(episode_id=f'{split}_{100000+i:06d}', split=split,
            map=('Town01', 'Town02', 'Town04', 'Town05')[replicate % 4] if split == 'train' else 'Town03',
            task_family=family, visual_condition_requested=condition, team_size=team,
            scene_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'),
            recipe_key=hashlib.sha256(key.encode()).hexdigest(), replicate=replicate,
            stratum=i % 24, status='pending_scene_instantiation',
            agent_names=[f'UAV_{a}' for a in range(team)],
            budgets=dict(decision_rounds=2, communication_bytes=None, per_agent_path_m=1200, simulated_seconds=600))

def configurations(kind, team):
    if kind == 'main':
        for m in AUTHORS:
            if m == 'coela' and team != 2: continue
            for s in SEEDS: yield m, s, 4, 2, 'none'
        for m in ('trace_full', 'repeat_vote'):
            for s in SEEDS: yield m, s, 4, 2, 'none'
        for m in ('no_comm', 'fifo'): yield m, 17, 4, 2, 'none'
    elif kind == 'ablation':
        for m in ('trace_no_task', 'trace_no_receiver', 'trace_no_visual_features'):
            for s in SEEDS: yield m, s, 4, 2, 'none'
    elif kind == 'sensitivity':
        for cap, horizon in itertools.product((1, 4, 16), (1, 2)):
            if (cap, horizon) == (4, 2): continue
            for m, seeds in (('trace_full', SEEDS), ('fifo', (17,))):
                for s in seeds: yield m, s, cap, horizon, 'none'
    elif kind == 'robustness':
        for perturb in ('drop_0.1', 'pose_1m'):
            for m, seeds in (('trace_full', SEEDS), ('fifo', (17,))):
                for s in seeds: yield m, s, 4, 2, perturb
    elif kind == 'all_view_reference': yield 'all_view', 17, 0, 2, 'none'
    else: raise ValueError(kind)

def prepare():
    if (OUT / 'contract.json').exists(): raise FileExistsError('Existing phase: inspect/resume, never regenerate')
    old = json.loads((ROOT / 'evidence/own_rgb_baselines_20261002_v3/contract.json').read_text(encoding='utf-8'))
    dev = [dict(r, run_class='development', split='validation', stratum=None) for r in old['cases']]
    formal = [dict(r, run_class='formal', case_id=r['episode_id']) for r in specs('test', 240)]
    train = list(specs('train', 120))
    for split, rows in (('train', train), ('test', formal)):
        p = OUT / 'specifications' / f'{split}.specs.jsonl'; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x', encoding='utf-8', newline='\n') as f:
            for row in rows: f.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
    fields = ('task_id', 'run_class', 'kind', 'case_id', 'team_size', 'backbone', 'method', 'seed', 'presentation', 'cap_multiple', 'horizon', 'perturbation')
    counts = Counter(); ids = set()
    with (OUT / 'matrix.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for cohort in (dev, formal):
            for case in cohort:
                for kind in ('main', 'ablation', 'sensitivity', 'robustness', 'all_view_reference'):
                    if case['run_class'] == 'formal' and kind in ('sensitivity', 'robustness') and case['replicate'] != 0: continue
                    for config, backbone, order in itertools.product(configurations(kind, case['team_size']), BACKBONES, ('original', 'reverse')):
                        method, seed, cap, horizon, perturbation = config
                        row = dict(run_class=case['run_class'], kind=kind, case_id=case['case_id'], team_size=case['team_size'], backbone=backbone,
                            method=method, seed=seed, presentation=order, cap_multiple=cap, horizon=horizon, perturbation=perturbation)
                        identity = json.dumps(row, sort_keys=True, separators=(',', ':'))
                        row['task_id'] = hashlib.sha256((ID + identity).encode()).hexdigest()[:24]
                        assert row['task_id'] not in ids; ids.add(row['task_id']); w.writerow(row); counts[(case['run_class'], kind)] += 1
    expected = {('development','main'):1256, ('development','ablation'):468, ('development','sensitivity'):1040,
        ('development','robustness'):416, ('development','all_view_reference'):52, ('formal','main'):23520,
        ('formal','ablation'):8640, ('formal','sensitivity'):1920, ('formal','robustness'):768, ('formal','all_view_reference'):960}
    assert counts == Counter(expected) and len(ids) == 39040
    contract = dict(schema='oct07_image_campaign_v1', campaign_id=ID, created_at=time.time(),
        session=os.environ.get('CODEX_THREAD_ID'), scientific_claim='Task- and receiver-conditioned selection of actual RGB packets improves cooperative grounding quality or quality/cost tradeoff.',
        hypothesis=dict(primary_subset=['cross_view_disambiguation','progressive_discovery'],
            competition=['additional pixels/calls alone explain the gain', 'fixed geometry and task constraints explain the gain'],
            falsification='No paired gain or useful cost tradeoff over repeat-vote and FIFO, or no meaningful ablation decision/state effect.'),
        authorization=dict(source='User approved plan and instructed Implement the plan.', expense_ceiling_cny=None,
            cap_type='none', actual_fees_required=True, old_closed_budgets_reopened=False,
            allow_cloud_start_for_ready_jobs=True, allow_new_resources_for_ready_dependencies=True),
        deadline='2026-10-07T23:59:59+08:00', compute_stop='2026-10-07T12:00:00+08:00',
        call_budget=dict(generated_samples=32, max_new_tokens_per_sample=1024, failed_attempts_count=True,
            input_tokens_pixels_and_native_preprocessing='measured, no false equality', per_sample_timeout_seconds=180, episode_timeout_seconds=3600),
        packet=dict(crop_context=1.5, size=[224,224], jpeg_quality=85, framing_bytes=8,
            actual_jpeg_and_canonical_header_paid=True, controls=['advertisement','bid','grant','ack'],
            b0='TRAIN single complete packet P95 rounded up to 4096 bytes', b0_bytes=None, cap_multiples=[1,4,16]),
        training=dict(scenes=120, historical37_use='diagnostic_only', feature_encoder='frozen CLIP ViT-B/32 author checkpoint',
            head_width=96, epochs=20, optimizer='AdamW', learning_rate=.001, weight_decay=.01, gradient_clip=1,
            loss=['SmoothL1','all-pairs logistic ranking within scene/snapshot'],
            utility='joint_exact + fraction_correct_roles/(number_roles+1); measured VLM before/after on TRAIN only',
            checkpoint_selection='final_epoch', seeds=list(SEEDS), fits=24,
            ablations=['full','no_task','no_receiver','no_visual_features']),
        sources=dict(author_methods=list(AUTHORS), coela_teams=[2], author_fidelity_audit=None,
            native_localization=['AerialVG released Swin-T','GroundingDINO'], native_frame_rows=dict(development=168,formal=2880)),
        data=dict(train_specs=120, development_scenes=13, formal_specs=240, formal_runtime_admitted=0,
            strata=24, formal_per_stratum=10, raw_horizon=2, required_history_horizon=2,
            minimum_visible_pixels=16, private_renderer_same_time_ABA=True,
            statistics='Average method seeds and presentations within physical layout, paired clustered bootstrap over layouts and geographic tiles; 95% intervals.',
            legacy1200_test='later_target_not_current_denominator', failed_captures_retained=True,
            no_score_based_selection=True, formal_model_results_opened=False),
        concurrency=dict(model_slots_max=8, capture_slots_max=4, execution='single current Codex; scientific jobs only'),
        supervision=dict(check_interval_seconds=300, heartbeat_timeout_seconds=180,
            observer_verified=False, shutdown_guard_verified=False, offline_ai_or_phone_push=False),
        smoke=dict(max_units=14, data_split='train', timeout_seconds=3600,
            question='Author-path identity, real image input, paid access, sample budget and output preservation',
            after_pass='python -u scripts/run_oct07_image_campaign.py --root <deployed phase> --stage development_baselines'),
        repair=dict(max_evidence_based_rounds=2, automatic_scientific_retries=0, formal_scientific_code_frozen=True),
        units=dict(image_episode_rows=39040, development_rows=3232, formal_rows=35808, training_fits=24,
            native_bbox_frame_rows=3048, effective_independent_test_scenes=None),
        matrix_sha256=sha(OUT/'matrix.csv'), counts={f'{k[0]}/{k[1]}':v for k,v in counts.items()},
        inherited_evidence=dict(component_rows=1105, rgb_vlm_rows=156, not_new_algorithm_results=True),
        models=old['models'], runtime_qualification=old['runtime_qualification'],
        development_cases=dev, formal_cases=formal, train_cases=train,
        specification_sha256={s:sha(OUT/'specifications'/f'{s}.specs.jsonl') for s in ('train','test')})
    write(OUT/'contract.json', contract, True)
    write(OUT/'status.json', dict(phase='implementation', observed_at=time.time(), completed=0, total=39040,
        succeeded=0, failed=0, running=0, queued=0, not_submitted=39040, percent=0, eta_seconds=None))
    state_path = ROOT/'.codex/research_execution.json'; state = json.loads(state_path.read_text(encoding='utf-8'))
    state['previous_active_phase'] = {k:state.get(k) for k in ('active_campaign','live_progress','next_action')}
    state.update(session=os.environ.get('CODEX_THREAD_ID'), observed_at=time.time(), autonomous_subagents=False,
        active_campaign=str((OUT/'contract.json').relative_to(ROOT)).replace('\\','/'), active_work_packages=['Oct7 image campaign'],
        current_stage='S1', execution_mode='single_session', latest_authorization=contract['authorization'],
        experiment_deadline=contract['deadline'], live_progress=json.loads((OUT/'status.json').read_text()),
        next_action='Freeze author sources; implement and verify real-image adapters, TRAIN label generation and server dependencies; submit admitted full queue.')
    write(state_path,state)
    live = json.loads((ROOT/'.codex/autodl_current_list.json').read_text(encoding='utf-8')); rows=live['data']['data']['list']
    check_path=ROOT/'.codex/autodl_startup_check.json'; check=json.loads(check_path.read_text(encoding='utf-8'))
    check['previous_observation']={k:check.get(k) for k in ('observed_at','observation_window_utc','list_coverage','first_subsequent_failure')}
    check.update(recorded_at=time.time(), observed_at=live['observed_at'], observation_window_utc=None,
        first_subsequent_failure=None, time_precision='Timed current 30-second subprocess; sanitized full receipt retained.',
        list_coverage=dict(code='Success', returned=len(rows), result_total=live['data']['data'].get('result_total'),
            complete=len(rows)==16, source='.codex/autodl_current_list.json', total_command_timeout_seconds=30),
        next_action='Ready-code dependency first; start owned primary only under a verified independent shutdown guard.')
    check['counts']['other_or_unknown_report_only']=15
    write(check_path,check)
    print(json.dumps(dict(campaign=ID, matrix_rows=len(ids), counts=contract['counts'], gpu_calls=0)))

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args(); prepare()
