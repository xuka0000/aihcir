"""Public-baseline identity and TEST quality-access gates, independent of TRACE."""
import csv
import ast
import json
from pathlib import Path
import sqlite3
import time
from prepare_oct07_image_campaign import sha, write
from oct07_runtime_scope import queue_path,verify_runtime

def _json(root, name):return json.loads((Path(root)/name).read_text(encoding='utf-8'))

def baseline_files(code):
    names=['run_oct07_image_campaign.py','oct07_image_core.py','oct07_author_adapters.py',
        'oct07_additional_author_adapters.py','oct07_agentverse_adapter.py','oct07_llm_planner_adapter.py',
        'oct07_baseline_freeze.py']
    code=Path(code);files={};pending=names[:]
    while pending:
        n=pending.pop()
        if n in files:continue
        path=_source(code,n);tree=ast.parse(path.read_text(encoding='utf-8'));files[n]=sha(path)
        modules={node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module and not node.level}
        modules.update(a.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names)
        for module in sorted(modules):
            name=module+'.py'
            if name not in files and (_source(code,name,False) is not None):pending.append(name)
    return files

def _source(code,name,required=True):
    p=Path(code)/name
    if not p.exists() and name=='official_self_refine_task.py':
        p=Path(code).parent/'01_uav_language_grounding/benchmark/carlaair_v1'/name
    if p.is_file():return p
    if required:raise FileNotFoundError('Frozen public dependency is missing: '+name)
    return None

def _safe(root,name):
    p=Path(root)/name
    if Path(name).is_absolute() or '..' in Path(name).parts or Path(root).resolve() not in p.resolve().parents:raise ValueError('Unsafe frozen baseline path')
    return p

def freeze_baselines(root, code=None):
    root=Path(root);code=Path(code or Path(__file__).parent);c=_json(root,'contract.json')
    if 'reporting_scope' in c:verify_runtime(root)
    methods=set(c['sources']['author_methods']);audit=_json(root,'public_baseline_audit.json')
    if len(methods)!=12 or audit.get('all_runtime_gates_passed') is not True or {r['method'] for r in audit['methods']}!=methods:
        raise ValueError('All twelve source-fidelity and actual-RGB gates are required')
    citations=_json(root,'manuscript_baseline_citations.json')
    if citations.get('actual_active_source_verified') is not True or set(citations.get('methods',{}))!=methods:
        raise ValueError('Active manuscript citations are incomplete')
    if not _json(root,'smoke_results.json').get('passed'):raise ValueError('Real image interface is closed')
    budget=_json(root,'packet_budget_freeze.json')
    if budget.get('source_split')!='train' or budget.get('source_scenes')!=120:raise ValueError('TRAIN budget is not frozen')
    if len(_json(root,'llm_planner_train_examples.json').get('examples',[]))!=100:raise ValueError('Standard TRAIN demonstrations are incomplete')
    cases=_json(root,'data_manifest.json')['cases']
    if c.get('schema')=='oct07_image_campaign_v3_scene_repair':
        total=c['data']['capture_scenes'];test_count=c['data']['formal_specs'];dev_count=c['data']['development_scenes']
        if len(cases)!=total or len({r['case_id'] for r in cases})!=total or {k:sum(r['split']==k for r in cases) for k in ('train','validation','test')}!={'train':120,'validation':dev_count,'test':test_count}:raise ValueError('Complete frozen TRAIN/DEV/TEST public assembly is required')
        for split,name,count in [('train','train_admission.json',120),('validation','development_admission.json',dev_count),('test','formal_admission.json',test_count)]:
            g=_json(root,name)
            if g.get('passed') is not True or g.get('runtime_admitted')!=count or g.get('planned')!=count:raise ValueError('Native scene admission is incomplete: '+split)
    files=['contract.json','matrix.csv','data_manifest.json','processor_manifest.json','packet_budget_freeze.json',
        'llm_planner_train_examples.json','public_baseline_audit.json','author_source_registry.json','manuscript_baseline_citations.json']
    if c.get('schema')=='oct07_image_campaign_v3_scene_repair':files.extend(['train_admission.json','development_admission.json','formal_admission.json'])
    for directory in ('author_sources','task_prompts'):
        files.extend(p.relative_to(root).as_posix() for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for n in ('retrieval_model_source.json','runtime_dependency_contract.json','native_assets_manifest.json'):
        if (root/n).is_file():files.append(n)
    value=dict(schema='public_baseline_freeze_v1',campaign=c['campaign_id'],methods=sorted(methods),
        files_sha256={n:sha(root/n) for n in files},implementation_sha256=baseline_files(code),
        source_split='train',trace_fit_required=False,created_at=time.time())
    p=root/'baseline_freeze.json'
    if p.exists():verify_baselines(root,code);return _json(root,'baseline_freeze.json')
    write(p,value,True);return value

def verify_baselines(root, code=None):
    root=Path(root);code=Path(code or Path(__file__).parent);value=_json(root,'baseline_freeze.json')
    if value.get('trace_fit_required') is not False:raise ValueError('Public baseline freeze must not depend on TRACE')
    for name,digest in value['files_sha256'].items():
        if sha(_safe(root,name))!=digest:raise ValueError('Frozen baseline input changed: '+name)
    for name,digest in value['implementation_sha256'].items():
        if Path(name).name!=name or sha(_source(code,name))!=digest:raise ValueError('Frozen baseline implementation changed: '+name)
    return value

def require_test_release(root, run_class=None):
    root=Path(root);c=_json(root,'contract.json')
    if not c.get('test_seal',{}).get('required') or run_class=='development':return
    release=_json(root,'test_results_release.json') if (root/'test_results_release.json').exists() else {}
    selected=_json(root,'algorithm_selection.json') if (root/'algorithm_selection.json').exists() else {}
    if (release.get('released') is not True or selected.get('frozen') is not True or
        selected.get('selection_split')!='development' or
        selected.get('development_rows_reviewed')!=c['units']['development_rows'] or
        release.get('selection_sha256')!=sha(root/'algorithm_selection.json') or
        release.get('matrix_sha256')!=sha(root/'matrix.csv')):
        raise ValueError('TEST quality is sealed until DEV selection and all formal predictions are frozen')
    for name,digest in selected.get('implementation_sha256',{}).items():
        if Path(name).name!=name or sha(Path(__file__).parent/name)!=digest:raise ValueError('Selected implementation changed after TEST release')
    with (root/'matrix.csv').open(encoding='utf-8',newline='') as f:formal=[r['task_id'] for r in csv.DictReader(f) if r['run_class']=='formal']
    db=sqlite3.connect('file:'+str(queue_path(root).resolve())+'?mode=ro',uri=True)
    try:states=dict(db.execute('SELECT id,state FROM jobs'))
    finally:db.close()
    if any(states.get(i) not in ('completed','failed') for i in formal):raise ValueError('All formal predictions must be terminal before TEST quality access')
    _verify_selection(root,c)
    if 'reporting_scope' in c:require_first_native_complete(root,c)


def require_first_native_complete(root, contract):
    done=_json(root,'native_execution_complete.json')
    seals=_json(root,'native_output_seals.json')
    if (done.get('completed_bbox')!=contract['units']['native_bbox_frame_rows'] or
        done.get('completed_proposals')!=contract['data']['capture_scenes'] or
        done.get('output_seals_sha256')!=sha(root/'native_output_seals.json')):
        raise ValueError('Complete frozen first-scope native outputs are required before TEST access')
    bbox=[n for n in seals if n.startswith(('native_bbox/','.sealed_test/native_bbox/'))]
    bundles=[n for n in seals if n.startswith('proposal_outputs/') and n.endswith('/bundle.json')]
    if len(bbox)!=contract['units']['native_bbox_frame_rows'] or len(bundles)!=contract['data']['capture_scenes']:
        raise ValueError('Native seal count differs from first reporting scope')
    for name,digest in seals.items():
        source=_safe(root,name)
        if not source.exists() and name.startswith('.sealed_test/'):
            source=_safe(root,name.removeprefix('.sealed_test/'))
        if sha(source)!=digest:raise ValueError('First native output identity changed')
    from oct07_fit_identity import verify_fits
    selected=_json(root,'algorithm_selection.json')
    if verify_fits(root)!=selected.get('fit_identity'):
        raise ValueError('First DEV-frozen checkpoint identity changed')

def _verify_selection(root,contract):
    from run_oct07_image_campaign import require_selected_version
    db=sqlite3.connect('file:'+str(queue_path(root).resolve())+'?mode=ro',uri=True)
    try:require_selected_version(root,contract,db)
    finally:db.close()

def release_test(root):
    root=Path(root);selected=_json(root,'algorithm_selection.json')
    p=root/'test_results_release.json'
    value=dict(released=True,selection_sha256=sha(root/'algorithm_selection.json'),matrix_sha256=sha(root/'matrix.csv'),observed_at=time.time())
    # Complete DEV review and nonempty scientific code identity are prerequisites.
    if selected.get('frozen') is not True or selected.get('selection_split')!='development':raise ValueError('TRACE is not DEV-frozen')
    _verify_selection(root,_json(root,'contract.json'))
    c=_json(root,'contract.json')
    if 'reporting_scope' in c:require_first_native_complete(root,c)
    if p.exists():
        require_test_release(root)
        from oct07_output_store import publish_released
        publish_released(root)
        return _json(root,p.name)
    with (root/'matrix.csv').open(encoding='utf-8',newline='') as f:ids=[r['task_id'] for r in csv.DictReader(f) if r['run_class']=='formal']
    db=sqlite3.connect('file:'+str(queue_path(root).resolve())+'?mode=ro',uri=True)
    try:states=dict(db.execute('SELECT id,state FROM jobs'))
    finally:db.close()
    if any(states.get(i) not in ('completed','failed') for i in ids):raise ValueError('Formal predictions are incomplete')
    if selected.get('development_rows_reviewed')!=_json(root,'contract.json')['units']['development_rows']:raise ValueError('Incomplete DEV selection')
    for name,digest in selected.get('implementation_sha256',{}).items():
        if Path(name).name!=name or sha(Path(__file__).parent/name)!=digest:raise ValueError('Selected code changed')
    write(p,value,True)
    from oct07_output_store import publish_released
    publish_released(root)
    return value

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--release-test',action='store_true');a=p.parse_args()
    print(json.dumps(release_test(a.root) if a.release_test else freeze_baselines(a.root)))
