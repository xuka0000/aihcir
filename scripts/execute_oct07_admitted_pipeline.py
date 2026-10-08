"""Dependency-checked scientific stages; no cloud API, agent or waiting daemon."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from prepare_oct07_image_campaign import write,sha
from run_oct07_image_campaign import matrix,ADAPTERS
from oct07_runtime_scope import queue_path,paused,enforce_prior_first_freeze

def child_command(code,root,script,args):
    """Use each original child's CLI; an assets-only entry has no --root."""
    args=list(args)
    root_args=[] if script=='oct07_native_environment.py' or '--root' in args else ['--root',str(root)]
    return [sys.executable,'-B','-u',str(Path(code)/script),*root_args,*args]

def attempt_id(value):
    if not re.fullmatch(r'[a-zA-Z0-9_]{1,96}',value):
        raise argparse.ArgumentTypeError('Use one safe unique original execution attempt')
    return value

def require_json(root,name,test,description):
    p=root/name
    if not p.exists() or not test(json.loads(p.read_text())):raise ValueError(description)

def manuscript_analysis_handoff(root):
    """Hand verified numeric outputs to the original owner, never certify a claim."""
    root=Path(root)
    names=['contract.json','matrix.csv','development_independent_coverage.json',
        'development_independent_scores.json','development_paired_statistics.json',
        'formal_independent_coverage.json','formal_independent_scores.json',
        'formal_paired_statistics.json','native_bbox_scores.json']
    missing=[name for name in names if not (root/name).is_file()]
    if missing:raise ValueError('Manuscript handoff requires actual numeric outputs: '+', '.join(missing))
    contract=json.loads((root/'contract.json').read_text(encoding='utf-8-sig'))
    coverage={split:json.loads((root/(split+'_independent_coverage.json')).read_text(encoding='utf-8-sig'))
        for split in ('development','formal')}
    value=dict(schema='oct07_manuscript_analysis_handoff_v1',observed_at=time.time(),
        campaign=contract['campaign_id'],units=contract['units'],coverage=coverage,
        files_sha256={name:sha(root/name) for name in names},
        numeric_analysis_complete=all(c.get('complete') is True for c in coverage.values()),
        semantic_state='awaits_original_owner_evidence_interpretation_and_selection',
        manuscript_finalization_authorized=False,scientific_acceptance=False,
        original_executor='scripts/execute_oct07_admitted_pipeline.py --stage finalize',
        entry_sha256=sha(Path(__file__)),
        selection_record=('evidence/oct07_image_campaign_20261003_v3_scene_repair/tonight_evidence_selection_20261007.json'
            if contract.get('reporting_scope',{}).get('id','').endswith('/tonight-core-v1') else
            'evidence/oct07_image_campaign_20261003_v3_scene_repair/first_delivery_evidence_selection_20261006.json'),
        owner_handoff=dict(owner='existing_AIHCIR_Codex_only',
            required_review_output='manuscript_evidence_selection_review.json',
            required_fields=['artifact_id','research_question','claim_role','raw_statistic_hash_version_split',
                'compatible_conditions_and_fair_budget','effect_uncertainty_independent_unit_practical_meaning',
                'mechanism_or_absence','decisive_negative_competitor_tradeoff',
                'placement_reason_and_active_source_destination','claim_support_verdict'],
            review_binds='All numeric hashes above plus current manuscript sources and retained claims/displays',
            next_action='Original owner interprets actual results and records main/supplement/internal placement before R11 writing and separate numeric/semantic/PDF review',
            numeric_checks_do_not_pass_semantic_review=True),
        raw_accounting_unchanged=True,healthy_experiments_unaffected=True)
    write(root/'manuscript_analysis_handoff.json',value)
    return value

def execute(root,stage,backbone=None,attempt='initial',shard_index=0,shard_count=1,lease_deadline=None):
    root=Path(root);contract,rows=matrix(root);deadline=datetime.fromisoformat(contract['compute_stop']).timestamp()
    engineering=stage in ('engineering_prepare','engineering_frontend','engineering','engineering_score','engineering_batch')
    # The original coordinator stops this engineering process at lease-360
    # for extraction/shutdown. Share that boundary instead of subtracting
    # additional preservation windows in every nested subprocess.
    if lease_deadline is not None:deadline=min(deadline,lease_deadline-(360 if engineering else 240))
    if backbone is not None and backbone not in contract['models']:raise ValueError('Backbone is outside the frozen reporting scope')
    if paused(root):raise ValueError('Existing parent or first-scope pause/stop is preserved')
    enforce_prior_first_freeze(root)
    if time.time()>=deadline and stage!='finalize':raise TimeoutError('Fixed compute deadline reached')
    if not 1<=shard_count<=contract['concurrency']['model_slots_max'] or not 0<=shard_index<shard_count:raise ValueError('Invalid scientific shard')
    if not engineering and stage not in ('prepare','interfaces'):require_json(root,'smoke_results.json',lambda x:x.get('passed') is True,'Real TRAIN-only image interface gate is closed')
    if stage in ('prepare','interfaces','baseline','formal_baselines','training','training_labels','training_fits','development','formal','targets','select'):
        require_json(root,'train_admission.json',lambda x:x.get('passed') is True and x.get('runtime_admitted')==120,'TRAIN120 physical/native-RGB admission is incomplete')
        require_json(root,'data_manifest.json',lambda x:sum(c.get('split')=='train' for c in x['cases'])==120,'Public TRAIN120 proposal/image assembly is incomplete')
    code=Path(__file__).parent
    def run(script,args,name):
        if not (code/script).is_file():raise FileNotFoundError('Original child entry absent before subprocess: '+script)
        reserve=30 if engineering else 120
        if time.time()+reserve>=deadline:raise TimeoutError('Fixed protected child boundary')
        from oct07_output_store import directory
        log=(directory(root,'logs','formal') if stage in ('formal_baselines','formal') else root/'pipeline_logs')/(name+'_'+attempt+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('xb') as stream:
            cmd=child_command(code,root,script,args)
            job=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
            write(root/('pipeline_submission_'+stage+'_'+str(shard_index)+'.json'),dict(stage=name,pid=job.pid,started_at=time.time(),command=cmd,
                entry_sha256=sha(code/script),deadline=contract['compute_stop']))
            try:job.wait(timeout=max(1,deadline-time.time()-(0 if engineering else 120)))
            except BaseException:
                import signal
                os.killpg(job.pid,signal.SIGTERM)
                try:job.wait(20)
                except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait(10)
                raise
            if job.returncode:raise RuntimeError(name+' failed; inspect preserved log, do not replay blindly')
    bk=['--backbone',backbone] if backbone else []
    if stage=='engineering_batch':
        if lease_deadline is None:raise ValueError('Engineering batch requires an actual shared lease deadline')
        for part in ('engineering_frontend','engineering','engineering_score'):
            execute(root,part,backbone,attempt,shard_index,shard_count,lease_deadline)
    elif stage=='engineering_prepare':
        from oct07_engineering_pipeline import prepare
        prepare(root)
    elif stage=='engineering_frontend':
        run('oct07_native_environment.py',['--mode','prepare','--assets',str(root/'engineering_actual_train/native_assets')],
            'engineering_native_environment')
        run('download_oct07_native_weights.py',['--names','groundingdino,bert-base-uncased',
            '--transport-manifest',str(Path(__file__).with_name('oct07_native_weight_transport.json')),
            '--root',str(root/'engineering_actual_train')],'engineering_official_weights')
        # Original public detector, exact TRAIN-only inputs; never an oracle
        # candidate construction or full native execution-complete receipt.
        run('oct07_native_localization.py',['--root',str(root/'engineering_actual_train'),
            '--method','groundingdino','--mode','proposals',
            '--manifest','public_new_capture_manifest.json','--split','train'],'engineering_public_frontend')
    elif stage=='engineering':
        # Thirty seconds inside the existing compute boundary are reserved
        # for the separate engineering scorer, not another shutdown window.
        run('oct07_engineering_pipeline.py',['--mode','run','--deadline',str(deadline-30)],'engineering_actual_RGB')
    elif stage=='engineering_score':
        run('oct07_engineering_pipeline.py',['--mode','score'],'engineering_independent_score')
    elif stage=='interfaces':
        require_json(root,'packet_budget_freeze.json',lambda x:x.get('source_scenes')==120,'TRAIN budget is required before interfaces')
        if not (root/'smoke_additional1_results.json').exists():
            run('run_oct07_image_campaign.py',['--stage','smoke','--smoke-methods','llm_planner,adapt,critic,autogen,agentverse,dylan','--smoke-attempt','additional1'],'six_new_interfaces')
        if not (root/'smoke_results.json').exists():
            from merge_oct07_interface_evidence import merge
            merge(root)
        require_json(root,'public_baseline_audit.json',lambda x:x.get('all_runtime_gates_passed') is True,'Actual twelve-method source fidelity/RGB review remains required')
    elif stage=='targets':
        from derive_oct07_baseline_targets import derive
        derive(root)
    elif stage=='select':
        from oct07_stage_freeze import select
        select(root)
    elif stage=='finalize':
        from oct07_baseline_freeze import release_test
        from analyze_oct07_image_campaign import analyze
        require_json(root,'native_execution_complete.json',lambda x:x.get('completed_bbox')==contract['units']['native_bbox_frame_rows'] and x.get('completed_proposals')==contract['data']['capture_scenes'],'Actual complete native output matrix is required before finalization')
        release_test(root)
        analyze(root,root/'private_development_manifest.json','development')
        analyze(root,root/'private_evaluation_manifest.json','formal')
        from score_oct07_native_bbox import evaluate
        evaluate(root,root/'private_evaluation_manifest.json','public_new_capture_manifest.json','test')
        if contract.get('reporting_scope',{}).get('id','').endswith('/tonight-core-v1'):
            from export_oct07_tonight_evidence import export
            export(root)
        manuscript_analysis_handoff(root)
    elif stage=='prepare':
        run('train_oct07_image_selector.py',['--mode','budget'],'TRAIN_packet_budget')
        run('train_oct07_image_selector.py',['--mode','census'],'TRAIN_packet_census')
        if 'llm_planner' in contract['sources']['author_methods']:
            run('prepare_oct07_planner_examples.py',[],'TRAIN_planner_examples')
    elif stage=='baseline':
        require_json(root,'packet_budget_freeze.json',lambda x:x.get('source_scenes')==120,'Prepare TRAIN budget before the interface and baseline stages')
        if 'llm_planner' in contract['sources']['author_methods']:require_json(root,'llm_planner_train_examples.json',lambda x:len(x.get('examples',[]))==100,'Standard TRAIN demonstrations are incomplete')
        if contract.get('test_seal',{}).get('required'):
            from oct07_baseline_freeze import freeze_baselines
            freeze_baselines(root)
        run('run_oct07_image_campaign.py',['--stage','development_baselines',*bk,'--shard-count',str(shard_count),'--shard-index',str(shard_index)],'DEV_baselines'+('_'+backbone if backbone else '')+'_shard'+str(shard_index))
    elif stage=='formal_baselines':
        require_json(root,'formal_admission.json',lambda x:x.get('passed') is True and x.get('runtime_admitted')==contract['data']['formal_specs'],'Frozen TEST cohort admission is incomplete')
        from oct07_baseline_freeze import verify_baselines
        verify_baselines(root)
        import sqlite3
        db=sqlite3.connect('file:'+str(queue_path(root).resolve())+'?mode=ro',uri=True)
        try:states=dict(db.execute('SELECT id,state FROM jobs'))
        finally:db.close()
        if any(states.get(r['task_id']) not in ('completed','failed') for r in rows if r['run_class']=='development' and r['method'] in ADAPTERS):
            raise ValueError('Complete public DEV baseline rows must precede sealed public TEST')
        run('run_oct07_image_campaign.py',['--stage','formal_baselines',*bk,'--shard-count',str(shard_count),'--shard-index',str(shard_index)],'TEST_baselines_sealed'+('_'+backbone if backbone else '')+'_shard'+str(shard_index))
    elif stage in ('training','training_labels','training_fits'):
        import sqlite3
        db=sqlite3.connect('file:'+str(queue_path(root).resolve())+'?mode=ro',uri=True)
        required=[r for r in rows if r['run_class']=='development' and (r['method'] in ADAPTERS or r['method'] in ('fifo','no_comm','all_view'))]
        states=dict(db.execute('SELECT id,state FROM jobs'));db.close()
        if any(states.get(r['task_id']) not in ('completed','failed') for r in required):raise ValueError('Complete DEV baseline matrix must precede new selector training')
        if stage in ('training','training_labels'):
            label_shards=(['--shard-index',str(shard_index),'--shard-count',str(shard_count)] if stage=='training_labels' else [])
            for key in (list(contract['models']) if backbone is None else [backbone]):
                run('train_oct07_image_selector.py',['--mode','labels','--backbone',key,*label_shards],'TRAIN_labels_'+key+'_shard'+str(shard_index))
        if stage in ('training','training_fits'):
            if stage=='training_fits':
                for key in (list(contract['models']) if backbone is None else [backbone]):
                    run('train_oct07_image_selector.py',['--mode','merge_labels','--backbone',key,'--shard-count',str(shard_count)],'TRAIN_verified_label_merge_'+key)
            run('train_oct07_image_selector.py',['--mode','fits',*bk],'TRAIN_fits'+('_'+backbone if backbone else ''))
    elif stage in ('development','formal'):
        if stage=='formal':require_json(root,'formal_admission.json',lambda x:x.get('passed') is True and x.get('runtime_admitted')==contract['data']['formal_specs'],'Frozen TEST cohort admission is incomplete')
        if stage=='formal' and len(contract['sources']['author_methods'])==12:
            require_json(root,'algorithm_selection.json',lambda x:x.get('development_rows_reviewed')==contract['units']['development_rows'] and x.get('selection_split')=='development' and x.get('frozen') is True,'Complete twelve-baseline DEV diagnosis and scientific version freeze are required')
            selection=json.loads((root/'algorithm_selection.json').read_text())
            for file,digest in selection['implementation_sha256'].items():
                if sha(code/file)!=digest:raise ValueError('Selected scientific implementation changed before formal evaluation')
        require_json(root,'fit_status.json',lambda x:x.get('completed')==contract['units']['training_fits'],'Frozen fit matrix is incomplete')
        from oct07_fit_identity import verify_fits
        identities=verify_fits(root)
        write(root/'fit_identity_verified.json',dict(checkpoints=identities,observed_at=time.time()))
        if stage=='formal':
            import sqlite3
            db=sqlite3.connect('file:'+str(queue_path(root).resolve())+'?mode=ro',uri=True)
            states=dict(db.execute('SELECT id,state FROM jobs'));db.close()
            if any(states.get(r['task_id']) not in ('completed','failed') for r in rows if r['run_class']=='development'):raise ValueError('Complete frozen DEV matrix must precede formal evaluation')
        run('run_oct07_image_campaign.py',['--stage',stage,*bk,'--shard-count',str(shard_count),'--shard-index',str(shard_index)],stage+('_'+backbone if backbone else '')+'_shard'+str(shard_index))
    else:raise ValueError('Unknown scientific stage')
    if stage in ('baseline','formal_baselines','development','formal'):
        from run_oct07_image_campaign import shard_for
        import sqlite3
        db=sqlite3.connect('file:'+str(queue_path(root).resolve())+'?mode=ro',uri=True)
        try:states=dict(db.execute('SELECT id,state FROM jobs'))
        finally:db.close()
        expected=[r for r in rows if r['run_class']==('formal' if stage in ('formal_baselines','formal') else 'development') and (backbone is None or r['backbone']==backbone) and shard_for(r,shard_count)==shard_index]
        if stage=='baseline':expected=[r for r in expected if r['method'] in ADAPTERS or r['method'] in ('fifo','no_comm','all_view')]
        if stage=='formal_baselines':expected=[r for r in expected if r['method'] in ADAPTERS]
        pending=[r['task_id'] for r in expected if states.get(r['task_id']) not in ('completed','failed')]
        if pending:
            write(root/('pipeline_stage_pending_'+stage+'_'+str(shard_index)+'.json'),dict(stage=stage,backbone=backbone,planned=len(expected),pending_task_ids=pending,observed_at=time.time(),complete=False))
            raise RuntimeError('Stage still has unexecuted/dependency-waiting rows; not complete')
    write(root/('pipeline_stage_complete_'+stage+'_'+str(shard_index)+'.json'),dict(stage=stage,backbone=backbone,shard_index=shard_index,shard_count=shard_count,observed_at=time.time(),scientific_acceptance=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=('prepare','interfaces','baseline','targets','formal_baselines','training','training_labels','training_fits','development','select','formal','finalize','engineering_prepare','engineering_frontend','engineering','engineering_score','engineering_batch'));p.add_argument('--preflight',action='store_true');p.add_argument('--backbone',choices=('qwen3','internvl35_hf'));p.add_argument('--attempt',type=attempt_id,default='initial');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);p.add_argument('--lease-deadline',type=float);a=p.parse_args()
    if a.preflight:
        if a.stage:p.error('--preflight cannot execute a stage')
        from oct07_pipeline_preflight import inspect
        print(json.dumps(inspect(a.root),ensure_ascii=False,indent=2))
    else:
        if not a.stage:p.error('--stage is required for execution')
        execute(a.root,a.stage,a.backbone,a.attempt,a.shard_index,a.shard_count,a.lease_deadline)
