"""Read-only inventory for the original scientific executor; never admit a job."""
from collections import Counter
from datetime import datetime
import csv
import json
from pathlib import Path
import sqlite3
import time

from oct07_runtime_scope import parent_root, queue_path, paused, verify_runtime
from prepare_oct07_image_campaign import sha


def inspect(root):
    root = Path(root).resolve()
    contract = json.loads((root/'contract.json').read_text(encoding='utf-8'))
    parent = parent_root(root)
    errors = []
    def record(name):
        p = root/name
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError) as exc:
            errors.append(dict(path=str(p), error=str(exc)))
            return None
    identity = verify_runtime(root)
    with (root/'matrix.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    specs = {s: [json.loads(line) for line in
                (root/'specifications'/(s+'.specs.jsonl')).read_text(encoding='utf-8').splitlines()]
             for s in ('train', 'validation', 'test')}
    gates = {}
    bindings = {}
    def gate(key, passed, evidence, missing):
        gates[key] = dict(passed=passed is True, evidence=evidence,
                          missing=None if passed is True else missing)
    for split, name in [('train', 'train_admission.json'),
                        ('validation', 'development_admission.json'),
                        ('test', 'formal_admission.json')]:
        d = record(name)
        ids = {s['episode_id'] for s in specs[split]}
        admitted = [r.get('episode_id', r.get('case_id'))
                    for r in (d or {}).get('rows', []) if r.get('passed') is True]
        ok = bool(d and d.get('passed') is True and
                  d.get('runtime_admitted') == len(ids) and
                  len(admitted) == len(ids) and set(admitted) == ids)
        gate(split+'_admission', ok, name, f'Independent physical/RGB admission for all {len(ids)} exact {split} IDs')
    manifest = record('data_manifest.json')
    cases = (manifest or {}).get('cases', [])
    expected_ids = {s['episode_id'] for ss in specs.values() for s in ss}
    case_ids = [c.get('case_id', c.get('episode_id')) for c in cases]
    gate('public_cases', len(case_ids)==len(expected_ids) and set(case_ids)==expected_ids,
         'data_manifest.json', f'Native public proposals/RGB for all {len(expected_ids)} exact scene IDs')
    train_cases=[c.get('case_id',c.get('episode_id')) for c in cases if c.get('split')=='train']
    gate('public_TRAIN120', len(train_cases)==120 and set(train_cases)=={s['episode_id'] for s in specs['train']},
         'data_manifest.json', 'Native public TRAIN120 proposal/image assembly; DEV/TEST is not a preparation dependency')
    packet = record('packet_budget_freeze.json')
    gate('TRAIN_B0', bool(packet and packet.get('source_scenes')==120),
         'packet_budget_freeze.json', 'Actual TRAIN120 packet P95 and original4096-byte rounding')
    demos = record('llm_planner_train_examples.json')
    gate('planner100', bool(demos and len(demos.get('examples', []))==100),
         'llm_planner_train_examples.json', '100 official TRAIN demonstrations')
    smoke = record('smoke_results.json')
    gate('real_RGB_smoke', bool(smoke and smoke.get('passed') is True),
         'smoke_results.json', 'Existing bounded real TRAIN RGB author-path interface evidence')
    author = record('public_baseline_audit.json')
    methods = contract['sources']['author_methods']
    author_ids = [m.get('method') for m in (author or {}).get('methods', [])]
    gate('author_fidelity12', bool(author and author.get('all_runtime_gates_passed') is True
         and len(author_ids)==12 and set(author_ids)==set(methods)),
         'public_baseline_audit.json', 'All12 source/PDF/config/mechanism/realRGB fidelity reviews')
    citation = record('manuscript_baseline_citations.json')
    # That old report is useful provenance, but a stale source hash is not a pass.
    citation_current = False
    if citation:
        project = Path(__file__).resolve().parents[1]
        for name, digest in citation.get('sources', {}).items():
            path = project/Path(name.replace('\\', '/'))
            bindings[name] = dict(path=str(path), matches=path.is_file() and sha(path)==digest)
        pdf = project/'01_uav_language_grounding/manuscript/multi_uav/main.pdf'
        bindings['compiled_pdf'] = dict(path=str(pdf), matches=pdf.is_file()
            and sha(pdf)==citation.get('compiled_pdf_sha256'))
        citation_current = bool(bindings) and all(x['matches'] for x in bindings.values())
    gate('active_citations', bool(citation and citation.get('actual_active_source_verified') is True
         and citation_current), 'manuscript_baseline_citations.json',
         'Rebind audit to actual active manuscript source/PDF and verify all12 citations')
    dependency = record('runtime_dependency_contract.json')
    gate('remote_dependencies', bool(dependency and dependency.get('scientific_ready') is True
         and dependency.get('remote_environment_verified') is True
         and dependency.get('native_gpu_compile_verified') is True),
         'runtime_dependency_contract.json', 'Actual model/CLIP/BERT/native GPU frontend hashes and runtime')
    fits = record('fit_status.json')
    fit_ok = bool(fits and fits.get('completed')==12)
    if fit_ok:
        try:
            from oct07_fit_identity import verify_fits
            fit_ok = len(verify_fits(root))==12
        except (ValueError, OSError, KeyError) as exc:
            errors.append(dict(path='fit_status.json', error=str(exc))); fit_ok = False
    gate('fits12', fit_ok, 'fit_status.json', 'All12 version/seed/budget-bound real trained TRACE checkpoints')
    ledger = queue_path(root)
    states = {}
    if ledger.is_file():
        db = sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro', uri=True)
        try:
            states = dict(db.execute('SELECT id,state FROM jobs'))
        finally:
            db.close()
    progress = {split: dict(Counter(states.get(r['task_id'], 'not_submitted')
                for r in rows if r['run_class']==split))
                for split in ('development', 'formal')}
    for name, selected in [('DEV_baselines', [r for r in rows if r['run_class']=='development'
            and r['method'] in (*methods, 'fifo', 'no_comm', 'all_view')]),
            ('DEV_rows', [r for r in rows if r['run_class']=='development']),
            ('TEST_rows', [r for r in rows if r['run_class']=='formal'])]:
        gate(name, all(states.get(r['task_id']) in ('completed', 'failed') for r in selected),
             str(ledger), f'All{len(selected)} exact rows terminal, preserving failed/missing evidence')
    # Read-only inventory: it does not freeze baselines or release TEST.
    freeze_ok = False
    if (root/'baseline_freeze.json').is_file():
        try:
            from oct07_baseline_freeze import verify_baselines
            verify_baselines(root); freeze_ok = True
        except (ValueError, OSError, KeyError) as exc:
            errors.append(dict(path='baseline_freeze.json', error=str(exc)))
    gate('baseline_freeze', freeze_ok,
         'baseline_freeze.json', f'Original freeze requires all {len(expected_ids)} admissions and real interfaces')
    selected = record('algorithm_selection.json')
    gate('DEV_version_selection', bool(selected and selected.get('frozen') is True
         and selected.get('selection_split')=='development'
         and selected.get('development_rows_reviewed')==contract['units']['development_rows']),
         'algorithm_selection.json', 'Actual complete DEV mechanism review and pre-TEST scientific freeze')
    native = record('native_execution_complete.json')
    gate('native_outputs', bool(native and native.get('completed_proposals')==contract['data']['capture_scenes']
         and native.get('completed_bbox')==contract['units']['native_bbox_frame_rows']), 'native_execution_complete.json',
         f"{contract['data']['capture_scenes']} native proposals and {contract['units']['native_bbox_frame_rows']} native bbox rows")
    base = ['train_admission', 'public_TRAIN120']
    execution = ['remote_dependencies', 'real_RGB_smoke']
    dependencies = {
        'prepare': base+['remote_dependencies'],
        'interfaces': base+['remote_dependencies', 'TRAIN_B0'],
        'baseline': base+execution+['public_cases','validation_admission', 'test_admission',
            'TRAIN_B0', 'planner100', 'author_fidelity12', 'active_citations'],
        'targets': base+execution+['DEV_baselines', 'baseline_freeze'],
        'training': base+execution+['DEV_baselines', 'baseline_freeze'],
        'training_labels': base+execution+['DEV_baselines', 'baseline_freeze'],
        'training_fits': base+execution+['DEV_baselines', 'baseline_freeze'],
        'development': base+execution+['fits12', 'baseline_freeze'],
        'select': base+execution+['DEV_rows', 'fits12'],
        'formal_baselines': base+execution+['test_admission', 'baseline_freeze', 'DEV_baselines'],
        'formal': base+execution+['test_admission', 'baseline_freeze', 'fits12', 'DEV_rows', 'DEV_version_selection'],
        'finalize': ['native_outputs', 'DEV_rows', 'TEST_rows', 'DEV_version_selection']}
    if root.name!='tonight_runtime':
        for old,new in {'public_cases192':'public_cases','DEV_baseline1800':'DEV_baselines',
                        'DEV2520':'DEV_rows','TEST5040':'TEST_rows','native192_864':'native_outputs'}.items():
            gates[old]=gates[new]
    stopped = paused(root)
    deadline = datetime.fromisoformat(contract['compute_stop']).timestamp()
    stages = {stage: dict(missing=[n for n in names if not gates[n]['passed']],
        pause_present=stopped, compute_expired=stage!='finalize' and time.time()>=deadline,
        command=['python', '-B', 'scripts/execute_oct07_admitted_pipeline.py',
                 '--root', str(root), '--stage', stage, '--backbone', 'qwen3'],
        paid_admission=False, remote_process_lock_readback_required=True)
        for stage, names in dependencies.items()}
    return dict(schema='original_oct07_pipeline_preflight_v1', observed_at=time.time(),
        identity=identity, matrix_sha256=sha(root/'matrix.csv'), counts=contract['counts'],
        baseline_denominators=contract['baseline_first'], ledger_path=str(ledger),
        local_ledger_exists=ledger.is_file(), remote_ledger_status='not_observed_while_instances_off',
        local_scoped_rows=progress, gates=gates, stages=stages, errors=errors,
        active_citation_bindings=bindings, paid_admission=False, scientific_acceptance=False,
        boundary='Read-only inventory does not prove live job identity, budget, independent guard, scientific stage acceptance or manuscript claims')
