"""Resumable full image matrix with exclusive per-task claims and sealed output.

Private scorer runs as a separate command after predictions are sealed. A
dependency-waiting row is not an executed result. No uncertain attempt retries.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
import signal
from pathlib import Path
import sqlite3
import threading
import time
import traceback
from contextlib import contextmanager

from prepare_oct07_image_campaign import write,sha
from oct07_image_core import BudgetExhausted,EpisodeCalls,NativeVLM,Packet,PublicEpisode,ViewAccess,WireLedger
from oct07_author_adapters import ADAPTERS,repeat_vote
from oct07_trace_selector import FrozenCLIP,fifo,select_and_transfer
from oct07_output_store import directory,prediction_path
from oct07_runtime_scope import queue_path,paused,verify_runtime,enforce_prior_first_freeze

def matrix(root):
    root=Path(root)
    c=json.loads((root/'contract.json').read_text())
    if 'reporting_scope' in c:verify_runtime(root)
    if sha(root/'matrix.csv')!=c['matrix_sha256']:raise ValueError('Fixed matrix changed')
    with (root/'matrix.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    total=c['units']['image_episode_rows']
    if len(rows)!=total or len({r['task_id'] for r in rows})!=total:raise ValueError('Frozen matrix coverage or task identity differs')
    for r in rows:
        for k in ('team_size','seed','cap_multiple','horizon'):r[k]=int(r[k])
    return c,rows

def setup_db(root,rows):
    db=sqlite3.connect(queue_path(root),timeout=30);db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, config TEXT NOT NULL, state TEXT NOT NULL, pid INTEGER, start_ticks TEXT, started REAL, finished REAL, result_sha256 TEXT, error_type TEXT)')
    try:
        with db:
            for row in rows:
                config=json.dumps(row,sort_keys=True)
                existing=db.execute('SELECT config FROM jobs WHERE id=?',(row['task_id'],)).fetchone()
                if existing and existing[0]!=config:raise ValueError('Task ID reused with different configuration')
                db.execute('INSERT OR IGNORE INTO jobs(id,config,state) VALUES(?,?,?)',(row['task_id'],config,'pending'))
        db.execute('CREATE TEMP TABLE active_scope(id TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO active_scope VALUES(?)',[(r['task_id'],) for r in rows])
    except BaseException:
        db.close();raise
    return db

def start_ticks():
    p=Path(f'/proc/{os.getpid()}/stat')
    if p.exists():return p.read_text().split()[21]
    import psutil
    return str(psutil.Process().create_time())

def claim(db,row):
    with db:
        changed=db.execute("UPDATE jobs SET state='running',pid=?,start_ticks=?,started=? WHERE id=? AND state='pending'",(os.getpid(),start_ticks(),time.time(),row['task_id'])).rowcount
    return changed==1

def progress(db,root,**extra):
    counts=dict(db.execute('SELECT state,count(*) FROM jobs JOIN active_scope ON jobs.id=active_scope.id GROUP BY state'));completed=counts.get('completed',0)+counts.get('failed',0);total=sum(counts.values())
    write(root/'status.json',dict(phase='running',observed_at=time.time(),heartbeat=time.time(),
        total=total,completed=completed,percent=100*completed/total,succeeded=counts.get('completed',0),failed=counts.get('failed',0),
        running=counts.get('running',0),queued=0,not_submitted=counts.get('pending',0),eta_seconds=None,**extra))

@contextmanager
def heartbeat(root,stage):
    stop=threading.Event()
    def beat():
        while not stop.is_set():
            write(root/('heartbeat_'+str(os.getpid())+'.json'),dict(pid=os.getpid(),start_ticks=start_ticks(),stage=stage,at=time.time()))
            stop.wait(15)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join(2)

@contextmanager
def episode_timeout(seconds):
    """A hard wall-clock bound on the Linux scientific job, including adapters."""
    if not hasattr(signal,'setitimer'):raise RuntimeError('Formal jobs require Linux interval timers')
    def expired(*args):raise BudgetExhausted('Fixed episode wall-clock deadline')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,seconds))
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

def companion(row,rows):
    return next(r for r in rows if all(r[k]==row[k] for k in ('kind','case_id','backbone','seed','presentation','cap_multiple','horizon','perturbation')) and r['method']=='trace_full')

def ready(root,row,rows,data):
    if row['case_id'] not in data:return False
    if row['method'].startswith('trace_'):
        return (root/'fits'/row['backbone']/row['method'].removeprefix('trace_')/str(row['seed'])/'last.pt').is_file()
    if row['method']=='repeat_vote':
        return prediction_path(root,companion(row,rows)).is_file()
    return True


def require_selected_version(root,contract,db):
    """Enforce DEV selection even when the matrix runner is called directly."""
    if len(contract['sources']['author_methods'])!=12:return
    enforce_prior_first_freeze(root)
    selected=json.loads((root/'algorithm_selection.json').read_text())
    if not (selected.get('frozen') is True and selected.get('selection_split')=='development' and
            selected.get('development_rows_reviewed')==contract['units']['development_rows']):
        raise ValueError('Complete twelve-method DEV review and version freeze are required')
    _,rows=matrix(root);states=dict(db.execute('SELECT id,state FROM jobs'))
    if any(states.get(r['task_id']) not in ('completed','failed') for r in rows if r['run_class']=='development'):
        raise ValueError('DEV rows are not all terminal; TEST selection is prohibited')
    implementation=selected.get('implementation_sha256',{})
    if not implementation or Path(__file__).name not in implementation:
        raise ValueError('Selected version has no scientific entry-point hash')
    for name,digest in implementation.items():
        if Path(name).name!=name or sha(Path(__file__).parent/name)!=digest:
            raise ValueError('Selected scientific implementation changed')
    audit=json.loads((root/'public_baseline_audit.json').read_text(encoding='utf-8'))
    if audit.get('all_runtime_gates_passed') is not True or {r['method'] for r in audit['methods']}!=set(contract['sources']['author_methods']):
        raise ValueError('All twelve public controller fidelity/RGB gates must be accepted before formal comparison')
    citations=json.loads((root/'manuscript_baseline_citations.json').read_text(encoding='utf-8'))
    if citations.get('actual_active_source_verified') is not True or set(citations.get('methods',{}))!=set(contract['sources']['author_methods']):
        raise ValueError('R15 requires real citations in the active manuscript sources')

def require_public_development(rows,db,methods):
    states=dict(db.execute('SELECT id,state FROM jobs'))
    required=[r['task_id'] for r in rows if r['run_class']=='development' and r['method'] in methods]
    if not required or any(states.get(i) not in ('completed','failed') for i in required):
        raise ValueError('Complete public DEV baseline rows must precede sealed public TEST')

def run_row(root,row,vlm,data,b0,encoder=None,rgb_cache=None):
    if rgb_cache is not None and row['run_class']!='engineering':
        raise ValueError('Batch RGB reuse is restricted to the isolated engineering scope')
    import random
    import numpy as np
    # Author retrieval/example splits and graph traversal use Python RNG;
    # their seed must not depend on which rows preceded this one on a shard.
    random.seed(row['seed']);np.random.seed(row['seed'])
    case=data[row['case_id']];path=root/case['bundle']
    if sha(path)!=case['bundle_sha256']:raise ValueError('Bundle source changed')
    bundle=json.loads(path.read_text());ep=PublicEpisode(bundle,path.parent,row['horizon'],row['presentation'],row['perturbation'],row['seed'],rgb_cache=rgb_cache)
    cap=(2**63-1 if row['method']=='all_view' else row['cap_multiple']*b0)
    ledger=WireLedger(ep.agents,cap,row['seed'],.1 if row['perturbation']=='drop_0.1' else 0.)
    access=ViewAccess(ep,ledger,all_view_reference=row['method']=='all_view')
    cutoff=datetime.fromisoformat(json.loads((root/'contract.json').read_text())['compute_stop']).timestamp()
    calls=EpisodeCalls(vlm,access,directory(root,'calls',row['run_class'])/(row['task_id']+'.jsonl'),row['seed'],deadline=min(time.time()+3600,cutoff))
    begin=time.perf_counter();method=row['method']
    if method in ADAPTERS:
        try:prediction,trace=ADAPTERS[method](root,calls)
        except BudgetExhausted:
            prediction=dict(valid=False,assignment={},raw='',output_error='Fixed generated-sample budget exhausted')
            trace=dict(termination='generated_sample_budget',traceback=traceback.format_exc())
    elif method=='fifo':prediction,trace=fifo(calls)
    elif method=='no_comm':prediction,trace=fifo(calls,False)
    elif method=='all_view':
        prediction,trace=fifo(calls,False);trace['access_budget']='all available views; separate reference cohort'
    elif method=='repeat_vote':
        # Companion ID is found from the fixed matrix, not from result quality.
        _,rows=matrix(root)
        source=companion(row,rows)
        prior=json.loads(prediction_path(root,source).read_text())
        prediction,trace=repeat_vote(root,calls,prior['generated_samples'])
    elif method.startswith('trace_'):
        condition=method.removeprefix('trace_');fit=root/'fits'/row['backbone']/condition/str(row['seed'])/'last.pt'
        prediction,trace=select_and_transfer(calls,encoder,fit,condition)
    else:raise ValueError('Unregistered method')
    result=dict(row,status='completed',prediction=prediction,trajectory=trace,wire_events=ledger.events,
        communication_bytes=ledger.bytes,communication_cap=cap,generated_samples=calls.used,
        input_tokens=sum(r['input_tokens'] for r in calls.records),output_tokens=sum(r['output_tokens'] for r in calls.records),
        source_pixels=sum(r['source_pixels'] for r in calls.records),calls=calls.records,
        seconds=time.perf_counter()-begin,candidate_id_map=ep.original_ids,finished_at=time.time(),
        bundle_sha256=sha(path),contract_sha256=sha(root/'contract.json'))
    result['implementation_sha256']={p.name:sha(p) for p in Path(__file__).parent.glob('oct07_*.py')}
    result['implementation_sha256'][Path(__file__).name]=sha(Path(__file__))
    if not prediction.get('valid'):result['status']='invalid_output'
    return result

def smoke(root,methods=None,attempt='initial'):
    c=json.loads((root/'contract.json').read_text());m=json.loads((root/'smoke_manifest.json').read_text())
    interface_cutoff=min(time.time()+c['smoke']['timeout_seconds'],datetime.fromisoformat(c['compute_stop']).timestamp())
    if m['split']!='train' or m['purpose']!='interface_only':raise ValueError('Smoke source must be TRAIN and diagnostic only')
    processor=json.loads((root/'processor_manifest.json').read_text());results=[]
    methods=methods or c['sources']['author_methods']
    if not set(methods)<=set(c['sources']['author_methods']):raise ValueError('Smoke method outside frozen public list')
    if len(methods)==12:raise ValueError('Reuse the inherited six-method interface evidence; select only the six new methods')
    total=len(c['models'])*(len(methods)+(1 if c['schema']=='oct07_image_campaign_v1' else 0))
    if total>c['smoke']['max_units']:raise ValueError('Fixed interface-call unit allowance exceeded')
    target=root/('smoke_results.json' if attempt=='initial' else 'smoke_'+attempt+'_results.json')
    if target.exists():raise FileExistsError('No duplicate smoke')
    for key,spec in c['models'].items():
        vlm=NativeVLM(key,spec,processor)
        for method in (methods+(['fifo'] if c['schema']=='oct07_image_campaign_v1' else [])):
            row=dict(task_id=f'smoke_{attempt}_{key}_{method}',case_id=m['case']['case_id'],run_class='smoke',kind='interface',
                team_size=2,backbone=key,method=method,seed=17,presentation='original',cap_multiple=1,horizon=2,perturbation='none')
            try:
                if time.time()>=interface_cutoff:raise BudgetExhausted('Fixed total interface deadline')
                with episode_timeout(interface_cutoff-time.time()):
                    result=run_row(root,row,vlm,{m['case']['case_id']:m['case']},262144)
                write(root/'smoke_outputs'/(row['task_id']+'.json'),result,True)
                results.append(dict(method=method,backbone=key,executed=True,status=result['status'],calls=result['generated_samples'],
                    wire_bytes=result['communication_bytes'],seconds=result['seconds'],output_sha256=sha(root/'smoke_outputs'/(row['task_id']+'.json'))))
            except BaseException as exc:
                write(root/'smoke_outputs'/(row['task_id']+'_failure.json'),dict(type=type(exc).__name__,traceback=traceback.format_exc()),True)
                results.append(dict(method=method,backbone=key,executed=True,status='failure',error_type=type(exc).__name__))
            write(root/('smoke_'+attempt+'_status.json'),dict(completed=len(results),total=total,results=results,heartbeat=time.time()))
        del vlm
        import torch
        torch.cuda.empty_cache()
    passed=all(r['status'] in ('completed','invalid_output') for r in results)
    write(target,dict(results=results,total=total,passed=passed,methods=methods,attempt=attempt,
        boundary='Interface execution, not algorithm utility or formal evaluation',observed_at=time.time()),True)
    if not passed:raise RuntimeError('Diagnose preserved interface failures before admission; no automatic replay')

def shard_for(row,count):
    return int(hashlib.sha256(row['task_id'].encode()).hexdigest(),16)%count

def run(root,stage,backbone=None,shard_index=0,shard_count=1):
    c,rows=matrix(root)
    if backbone is not None and backbone not in c['models']:raise ValueError('Backbone is outside the frozen reporting scope')
    if paused(root):raise ValueError('Existing parent or first-scope pause/stop is preserved')
    enforce_prior_first_freeze(root)
    db=setup_db(root,rows)
    if not 1<=shard_count<=c['concurrency']['model_slots_max'] or not 0<=shard_index<shard_count:raise ValueError('Model shard exceeds authorized slots')
    if datetime.fromisoformat(c['compute_stop']).timestamp()<=time.time():raise TimeoutError('Fixed compute deadline')
    smoke_result=json.loads((root/'smoke_results.json').read_text())
    if not smoke_result['passed']:raise ValueError('Interface gate not passed')
    budget=json.loads((root/'packet_budget_freeze.json').read_text())
    if budget['source_split']!='train' or budget['source_scenes']!=120 or budget['b0_bytes']%4096:raise ValueError('TRAIN packet budget not frozen')
    b0=budget['b0_bytes'];manifest=json.loads((root/'data_manifest.json').read_text());data={r['case_id']:r for r in manifest['cases']}
    formal_stage=stage in ('formal','formal_baselines')
    eligible=[r for r in rows if r['run_class']==('formal' if formal_stage else 'development') and (backbone is None or r['backbone']==backbone) and shard_for(r,shard_count)==shard_index]
    if stage=='development_baselines':eligible=[r for r in eligible if r['method'] in ADAPTERS or r['method'] in ('fifo','no_comm','all_view')]
    if stage=='formal_baselines':eligible=[r for r in eligible if r['method'] in ADAPTERS]
    if stage in ('development_baselines','formal_baselines') and c.get('test_seal',{}).get('required'):
        from oct07_baseline_freeze import verify_baselines
        verify_baselines(root)
    if formal_stage:
        admission=json.loads((root/'formal_admission.json').read_text())
        if not admission['passed'] or admission['runtime_admitted']!=c['data']['formal_specs']:raise ValueError('Formal runtime gate closed')
        if stage=='formal':require_selected_version(root,c,db)
        else:require_public_development(rows,db,set(c['sources']['author_methods']))
    if stage in ('development','formal'):
        from oct07_fit_identity import verify_fits
        verify_fits(root)
    processor=json.loads((root/'processor_manifest.json').read_text());encoder=None
    if any(r['method'].startswith('trace_') for r in eligible):
        feature=json.loads((root/'feature_encoder.json').read_text());encoder=FrozenCLIP(feature['directory'],feature['revision'])
    for key,spec in c['models'].items():
        if backbone and key!=backbone:continue
        vlm=NativeVLM(key,spec,processor)
        # Companion TRACE rows precede matched-call repeat controls.
        ordered=sorted((r for r in eligible if r['backbone']==key),key=lambda r:(r['method']=='repeat_vote',r['kind']!='main',r['task_id']))
        for row in ordered:
            if time.time()>=datetime.fromisoformat(c['compute_stop']).timestamp() or paused(root):break
            if row['method']=='repeat_vote':
                source=companion(row,rows)
                dependency=db.execute('SELECT state FROM jobs WHERE id=?',(source['task_id'],)).fetchone()
                if dependency and dependency[0]=='failed' and not prediction_path(root,source).exists():
                    if claim(db,row):
                        path=directory(root,'failures',row['run_class'])/(row['task_id']+'.json')
                        write(path,dict(row,type='DependencyFailed',source_task_id=source['task_id'],
                            measured_call_count=None,at=time.time(),reason='No sealed TRACE call count; matched-call control cannot execute'),True)
                        with db:db.execute("UPDATE jobs SET state='failed',finished=?,error_type='DependencyFailed' WHERE id=?",(time.time(),row['task_id']))
                    continue
            if not ready(root,row,rows,data):continue
            if not claim(db,row):continue
            progress(db,root,current_task=row['task_id'],backbone=key,shard_index=shard_index,shard_count=shard_count)
            try:
                with episode_timeout(min(3600,datetime.fromisoformat(c['compute_stop']).timestamp()-time.time())):
                    result=run_row(root,row,vlm,data,b0,encoder)
                path=prediction_path(root,row);write(path,result,True)
                with db:db.execute('UPDATE jobs SET state=?,finished=?,result_sha256=? WHERE id=?',('completed' if result['status']=='completed' else 'failed',time.time(),sha(path),row['task_id']))
            except BaseException as exc:
                path=directory(root,'failures',row['run_class'])/(row['task_id']+'.json');write(path,dict(row,type=type(exc).__name__,traceback=traceback.format_exc(),at=time.time()),True)
                with db:db.execute("UPDATE jobs SET state='failed',finished=?,error_type=? WHERE id=?",(time.time(),type(exc).__name__,row['task_id']))
                if type(exc).__name__ in ('OutOfMemoryError','RuntimeError'):raise
            progress(db,root,current_task=None,backbone=key)
        del vlm
        import torch
        torch.cuda.empty_cache()
    progress(db,root,current_task=None);db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['smoke','development_baselines','development','formal_baselines','formal'],required=True);p.add_argument('--backbone',choices=['qwen3','internvl35_hf']);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);p.add_argument('--smoke-methods');p.add_argument('--smoke-attempt',default='initial');a=p.parse_args()
    if not a.smoke_attempt.replace('_','').isalnum():raise ValueError('Unsafe interface attempt')
    with heartbeat(a.root,a.stage):
        if a.stage=='smoke':smoke(a.root,a.smoke_methods.split(',') if a.smoke_methods else None,a.smoke_attempt)
        else:run(a.root,a.stage,a.backbone,a.shard_index,a.shard_count)
