"""Deploy a unique scientific job with a separate, bounded shutdown guard.

No Codex/team supervisor. Provider mutations are never retried automatically.
Only the registered owned UUID is controlled; passwords remain in memory.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from prepare_oct07_image_campaign import OUT,ROOT,sha,write

PYTHON='/path/to/aihcir/envs/qwen3/bin/python'
REMOTE='/path/to/aihcir/sessions/'+OUT.name
RELEASE='release'
CLOUD_FOLDER='cloud'
SCIENTIFIC_STAGES=('prepare','interfaces','baseline','targets','training_labels',
    'training_fits','development','select','formal_baselines','formal')

def admitted_stage_preflight(root,assessment,stage):
    """Fail before cloud/payment; engineering funds never admit scientific work."""
    if stage=='engineering_batch':return dict(stage=stage,engineering_only=True)
    if stage not in SCIENTIFIC_STAGES:raise ValueError('Unapproved original pipeline stage')
    root=Path(root).resolve()
    contract=json.loads((root/'contract.json').read_text(encoding='utf-8-sig'))
    if (assessment.get('run_class')!='scientific' or assessment.get('scientific_ready') is not True
        or assessment.get('engineering_window_authorization')
        or assessment.get('task_id')=='oct07-48h-model-argv-repair4-20261007'):
        raise ValueError('Separate ready scientific assessment/share required; engineering is not science')
    expected=dict(campaign_id=contract['campaign_id'],stage=stage,
        contract_sha256=sha(root/'contract.json'),matrix_sha256=sha(root/'matrix.csv'))
    if assessment.get('scientific_stage_binding')!=expected:
        raise ValueError('Scientific assessment differs from exact frozen stage/matrix')
    amount=assessment.get('preservation_maximum_bytes')
    if not isinstance(amount,int) or not 0<amount<=2*1024**3:
        raise ValueError('Scientific assessment must bound preservation writes before payment')
    from oct07_pipeline_preflight import inspect
    result=inspect(root);gate=result['stages'][stage]
    if gate['pause_present'] or gate['compute_expired'] or gate['missing'] or result['errors']:
        raise ValueError('Original scientific dependencies not ready: '+', '.join(gate['missing']))
    return dict(stage=stage,engineering_only=False,binding=expected,
        prerequisites_sha256={g['evidence']:sha(root/g['evidence'])
            for g in result['gates'].values() if g['passed'] and
            Path(g['evidence']).name==g['evidence'] and (root/g['evidence']).is_file()},
        actual_preflight=result,scientific_acceptance=False)

def require_available_allocation(pool,assessment):
    """Read the real ledger once before arming; closed ceilings are not refunds."""
    allocation=next(x for x in pool['allocations'] if x['workspace']=='AIHCIR')
    remaining=round(allocation['ceiling_cny']-allocation['used_upper_bound_cny'],2)
    if 'incremental_remaining_cny' in allocation:
        remaining=min(remaining,allocation['incremental_remaining_cny'])
    active=[x for x in pool.get('leases',[]) if x.get('workspace')=='AIHCIR'
        and x.get('state') not in ('closed','cancelled')
        and (x.get('instance')==assessment['owned_uuid'] or x.get('task')==assessment['task_id'])]
    if active:raise ValueError('Existing live/uncertain lease; inspect original identity instead of duplicating')
    if remaining+1e-8<assessment['all_in_ceiling_cny']:
        raise ValueError('Actual unspent share insufficient; no guard, reservation or paid start')
    return remaining

def deploy_cached_engineering_package(remote,package_file,target,folder,seconds):
    """Reuse only individually hash-verified original inputs; validate all after patch."""
    with zipfile.ZipFile(package_file) as archive:
        manifest=json.loads(archive.read('first_execution_release_manifest.json'))
    index=folder/'cache_probe_manifest.json';write(index,manifest,True)
    remote_index=target+'/'+folder.name+'_source_manifest.json'
    remote.upload(index,remote_index,min(30,seconds))
    probe="""import hashlib,json
from pathlib import Path
p=Path(TARGET);m=json.loads(Path(INDEX).read_text());missing=[];cached=0
for r in m['files']:
 n=Path(r['path']);assert not n.is_absolute() and '..' not in n.parts
 f=p/n
 assert not f.is_symlink(), 'Unexpected source symlink'
 if f.is_file() and f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']:cached+=1
 else:missing.append(r['path'])
print(json.dumps(dict(cached=cached,missing=missing)))
""".replace('TARGET',repr(target)).replace('INDEX',repr(remote_index))
    observed=json.loads(remote.command('python3 -c '+shlex.quote(probe),45))
    wanted={r['path']:r for r in manifest['files']}
    names=observed['missing']
    if len(set(names))!=len(names) or any(n not in wanted for n in names):
        raise ValueError('Cache probe did not bind to the exact current source manifest')
    raw_bytes=sum(wanted[n]['bytes'] for n in names)
    delta=raw_bytes<=8*1024**2
    if delta:
        transfer=folder/'current_source_delta.zip'
        with zipfile.ZipFile(package_file) as original,zipfile.ZipFile(transfer,'x',zipfile.ZIP_DEFLATED) as patch:
            for name in names:patch.writestr(name,original.read(name))
            patch.writestr('first_execution_release_manifest.json',json.dumps(manifest))
    else:transfer=package_file
    remote_zip=target+'/'+folder.name+'_source_update.zip'
    remote.upload(transfer,remote_zip,seconds)
    expected=sha(transfer)
    apply="""import hashlib,json,zipfile
from pathlib import Path
p=Path(TARGET);zpath=Path(ARCHIVE);assert hashlib.sha256(zpath.read_bytes()).hexdigest()==DIGEST
with zipfile.ZipFile(zpath) as z:
 assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and not (p/n).is_symlink() for n in z.namelist())
 z.extractall(p)
m=json.loads((p/'first_execution_release_manifest.json').read_text())
assert all((p/r['path']).stat().st_size==r['bytes'] and hashlib.sha256((p/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in m['files'])
print(json.dumps(dict(verified=True,files=len(m['files']))))
""".replace('TARGET',repr(target)).replace('ARCHIVE',repr(remote_zip)).replace('DIGEST',repr(expected))
    proof=json.loads(remote.command('python3 -c '+shlex.quote(apply),90))
    if proof.get('verified') is not True or proof.get('files')!=len(manifest['files']):
        raise ValueError('Actual entire current package source verification failed')
    write(folder/'remote_package_verification.json',dict(proof,
        individually_verified_reused=observed['cached'],changed_members=len(names),
        delta_transfer=delta,uploaded_archive_bytes=transfer.stat().st_size,
        current_full_package_sha256=sha(package_file),upload_sha256=expected,
        scientific_acceptance=False),True)
    return proof

def preserve_scientific_outputs(remote,target,root,folder,maximum_bytes,expected_queue_sha256):
    """Stream one effective copy, preserving the seal and avoiding whole-machine archives."""
    import importlib.util
    if not isinstance(maximum_bytes,int) or not 0<maximum_bytes<=2*1024**3:
        raise ValueError('Scientific preservation needs a bounded <=2GiB write assessment')
    script="""import hashlib,json,sqlite3
from pathlib import Path
p=Path(TARGET);r=p/'tonight_runtime';rows=[]
q=p/'queue.sqlite'
if q.exists():
 with sqlite3.connect(q) as db,sqlite3.connect(p/'queue_preservation.sqlite') as dest:db.backup(dest)
allowed={'predictions','failures','calls','pipeline_logs','fits','training_labels','native_bbox','analysis_exports','.sealed_test','proposal_outputs'}
paths=[f for f in r.rglob('*') if f.is_file() and (len(f.relative_to(r).parts)==1 and f.suffix=='.json' or f.relative_to(r).parts[0] in allowed) and f.suffix in ('.json','.jsonl','.log','.csv','.pt')]
paths += [p/'queue_preservation.sqlite'] if q.exists() else []
assert sum(f.stat().st_size for f in paths)<=MAXIMUM
for f in sorted(paths):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
 rows.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(files=rows,bytes=sum(x['bytes'] for x in rows),TEST_raw_remains_sealed=True)))
""".replace('TARGET',repr(target)).replace('MAXIMUM',str(maximum_bytes))
    receipt=json.loads(remote.command('python3 -c '+shlex.quote(script),90))
    guard_path=bridge_root()/'scripts/storage_guard.py'
    spec=importlib.util.spec_from_file_location('aihcirstorageguard',guard_path)
    storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
    reservation,capacity=storage.reserve(ROOT,root,max(4096,receipt['bytes']+4096))
    write(folder/'scientific_preservation_capacity.json',dict(reservation=reservation,**capacity),True)
    if reservation is None:raise ValueError('Live reservations and C10/D30 block preservation; retain remote copy')
    try:
        with remote.client.open_sftp() as sftp:
            for item in receipt['files']:
                relative=Path(item['path'])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe scientific preservation path')
                destination=(root.parent/'queue.sqlite' if relative.as_posix()=='queue_preservation.sqlite'
                    else root.parent/relative)
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists() and sha(destination)==item['sha256']:continue
                queue_copy=relative.as_posix()=='queue_preservation.sqlite'
                if queue_copy and (sha(destination) if destination.exists() else None)!=expected_queue_sha256:
                    raise ValueError('Local original ledger changed during remote work; do not overwrite')
                if destination.exists() and not queue_copy and any(
                    part in ('predictions','failures','calls','fits') for part in relative.parts):
                    raise ValueError('Existing effective scientific result identity conflict')
                temporary=destination.with_name(destination.name+'.preserving')
                if temporary.exists():raise ValueError('Inspect existing partial transfer before resuming')
                sftp.get(target+'/'+relative.as_posix(),str(temporary))
                if temporary.stat().st_size!=item['bytes'] or sha(temporary)!=item['sha256']:
                    raise ValueError('Independent scientific preservation hash differs')
                temporary.replace(destination)
        write(folder/'scientific_preservation_receipt.json',dict(receipt,observed_at=time.time(),
            independent_hash_verified=True,scientific_acceptance=False),True)
    finally:storage.release(reservation)

def bridge_root():
    """Resolve the verified shared entry in a fresh desktop command environment."""
    configured=os.environ.get('AIHCIR_BRIDGE_ROOT')
    path=(Path(configured) if configured else ROOT.parent/'具身智能大论文架构/cross_work_bridge').resolve()
    required=('scripts/cloud.py','scripts/storage_guard.py','tools/codex_control_center/cloud_services.py')
    if any(not (path/name).is_file() for name in required):
        raise ValueError('Shared cloud/budget/storage bridge path is missing; no paid action')
    return path


class Provider:
    def __init__(self,owned_uuid=None):
        bridge=bridge_root();sys.path.insert(0,str(bridge/'tools/codex_control_center'))
        from cloud_services import CloudClient
        self.client=CloudClient();self.uid=owned_uuid or json.loads((ROOT/'.codex/primary_cloud_node.json').read_text())['uuid']
        self.token_path=Path.home()/'.autodl/developer_token.txt';self.failed=False
        if self.uid!='OWNED_INSTANCE_UUID':
            registry=json.loads((ROOT/'.codex/oct07_cloud_ownership.json').read_text())
            owner=registry['instances'].get(self.uid,{})
            if owner.get('workspace')!='AIHCIR' or owner.get('created_receipt_verified') is not True:raise ValueError('Unapproved instance identity')
    def request(self,endpoint,method='GET',**body):
        if self.failed:raise RuntimeError('Cloud access stopped after first failure; no blind retry')
        try:
            payload=dict(body) if endpoint=='instance/pro/list' else dict(instance_uuid=self.uid,**body)
            r=self.client.request('autodl',method,'/api/v1/dev/'+endpoint,payload,timeout=(5,20))
            return r.get('data')
        except Exception:
            self.failed=True;raise
    def status(self):
        # The GET status endpoint timed out in two original guards while the
        # documented POST list endpoint returned complete fresh coverage.
        data=self.request('instance/pro/list','POST',page_index=1,page_size=100)
        rows=data.get('list',[]);total=data.get('result_total')
        if not isinstance(total,int) or len(rows)!=total or len({r.get('uuid') for r in rows})!=total:
            raise ValueError('Official status list coverage incomplete')
        owned=next((r for r in rows if r.get('uuid')==self.uid),None)
        if owned is None:raise ValueError('Owned instance absent from complete official list')
        return owned['status']
    def connection(self):return self.request('instance/pro/snapshot')
    def power_on(self):return self.request('instance/pro/power_on','POST',payload='gpu',start_command='sleep 1')
    def power_off(self):return self.request('instance/pro/power_off','POST')

def read_guard_json(path):
    """Bound only diagnosed local Windows sharing collisions, never cloud calls."""
    for attempt in range(20):
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def shutdown_once_and_verify(folder,context,state,save):
    """Never resend stop; resolve even a BadRequest by one fresh official readback."""
    from deploy_oct07_scene_repair import shared_cli
    if not state['power_off_attempts']:
        save(phase='stopping',power_off_attempts=1)
        try:
            shared_cli(folder,'protection_failure_stop','autodl','stop','--instance',context['owned_uuid'])
        except BaseException as stop_exc:
            save(shutdown_error_type=type(stop_exc).__name__,shutdown_error=str(stop_exc)[:500])
    try:
        time.sleep(10)
        result=shared_cli(folder,'protection_failure_stop_readback','autodl','list')
        data=result.get('data',{});rows=data.get('list',[]);total=data.get('result_total')
        complete=(result.get('code')=='Success' and isinstance(total,int) and
                  len(rows)==total and len({r.get('uuid') for r in rows})==total)
        owned=next((r for r in rows if r.get('uuid')==context['owned_uuid']),{})
        confirmed=complete and owned.get('status') in ('shutdown','shutdown_by_starting_error')
        save(phase='shutdown_confirmed' if confirmed else 'shutdown_unconfirmed',
             provider_status=owned.get('status'),provider_status_observed_at=time.time(),
             shutdown_confirmed=confirmed,terminal=True,coverage_complete=complete)
    except BaseException as read_exc:
        save(phase='shutdown_unconfirmed',terminal=True,shutdown_readback_error_type=type(read_exc).__name__,
             shutdown_readback_error=str(read_exc)[:500])


def retain_stop_protection(folder,context,state,save,exc,started):
    """Stop failed status polling, but retain the already armed stop obligation.

    No paid action or failed API call is repeated. The owner is told to stop;
    this original guardian issues at most one separate unpaid shutdown request.
    """
    save(phase='status_failed_protection_retained',cloud_checks_stopped=True,
         protection_degraded=True,error_type=type(exc).__name__,error=str(exc)[:500])
    while True:
        try:
            owner=read_guard_json(folder/'owner.json')
            stale=owner.get('nonce')!=context['nonce'] or time.time()-owner.get('heartbeat',0)>90
        except (OSError,ValueError):
            owner={};stale=True
        reserve=context.get('shutdown_reserve_seconds',180)
        due=(time.time()>=context['effective_deadline']-reserve or
             time.monotonic()-started>=context['max_powered_seconds']-reserve)
        if stale or owner.get('desired')=='stop' or due:break
        save();time.sleep(1)
    # A status timeout is not evidence that a prior mutation failed. Never
    # resend shutdown if an earlier shutdown request was already attempted.
    if state['power_off_attempts']:
        save(phase='shutdown_unconfirmed',terminal=True);return
    save(phase='stopping_after_status_failure',
         stop_reason='lost_status_protection_owner_stop_or_deadline')
    shutdown_once_and_verify(folder,context,state,save)


def attempt_powered_upper(folder):
    """Inherit failed guards and independently verified cleanup without rewriting them."""
    folder=Path(folder);proof=read_guard_json(folder/'guard.json')
    if proof.get('terminal') and proof.get('shutdown_confirmed'):
        return proof['elapsed_seconds']
    cleanup=read_guard_json(folder/'paid_attempt_stop_verification.json')
    context=read_guard_json(folder/'guard_context.json')
    lease=read_guard_json(folder/'budget_lease.json')
    stopped=read_guard_json(folder/'close_verified_stop.json')
    response=read_guard_json(folder/'close_cli_receipt.json')
    closed=json.loads(response['stdout'])
    actual=next((r for r in closed['leases'] if r['id']==lease['lease_id']),{})
    expected=context['effective_deadline']-context['max_powered_seconds']
    covered=stopped['provider_readback']
    owned=next((r for r in covered['instances'] if r['uuid']==context['owned_uuid']),{})
    if (cleanup.get('verified') is not True or cleanup.get('guard_sha256')!=sha(folder/'guard.json')
            or cleanup.get('stop_proof_sha256')!=sha(folder/'close_verified_stop.json')
            or cleanup.get('lease_id')!=lease['lease_id'] or actual.get('state')!='closed'
            or actual.get('instance')!=context['owned_uuid'] or not covered.get('coverage_complete')
            or owned.get('status') not in ('shutdown','shutdown_by_starting_error')
            or cleanup.get('accounting_start_upper')!=expected
            or abs(cleanup['elapsed_upper_seconds']-(covered['observed_at']-expected))>1e-6):
        raise ValueError('Failed protection has no exact independently verified stopped accounting')
    return cleanup['elapsed_upper_seconds']


def engineering_powered_commitment(folder,current_uuid):
    """Reserve a healthy other branch's full window; never duplicate its UUID."""
    folder=Path(folder);proof=read_guard_json(folder/'guard.json')
    if proof.get('terminal'):return attempt_powered_upper(folder)
    context=read_guard_json(folder/'guard_context.json')
    if context.get('owned_uuid')==current_uuid:
        raise ValueError('Existing live engineering attempt owns this UUID; do not duplicate')
    import psutil
    process=psutil.Process(proof['pid'])
    if (proof.get('nonce')!=context.get('nonce') or proof.get('protection_degraded')
            or abs(process.create_time()-proof.get('birth',0))>=.01
            or time.time()-proof.get('heartbeat',0)>30
            or proof.get('phase') not in ('armed','power_requested','running')
            or '--guard-folder' not in process.cmdline() or str(folder) not in process.cmdline()):
        raise ValueError('Other engineering branch protection is unknown; preserve it and block new payment')
    seconds=context['max_powered_seconds']
    if not isinstance(seconds,(int,float)) or isinstance(seconds,bool) or not 0<seconds<=7200:
        raise ValueError('Other engineering branch powered commitment differs')
    return seconds


def guard(folder):
    context=json.loads((folder/'guard_context.json').read_text());api=Provider(context.get('owned_uuid'));nonce=context['nonce'];started=time.monotonic()
    import psutil
    state=dict(pid=os.getpid(),birth=psutil.Process().create_time(),nonce=nonce,phase='initializing',power_on_attempts=0,power_off_attempts=0,
        shutdown_confirmed=False,terminal=False,heartbeat=time.time(),max_powered_seconds=context['max_powered_seconds'])
    def save(**kw):state.update(kw,heartbeat=time.time(),elapsed_seconds=time.monotonic()-started);write(folder/'guard.json',state)
    armed=False
    try:
        initial=api.status()
        save(provider_status=initial)
        attached=context.get('attach_existing') is True
        external=context.get('external_budget_start') is True
        if attached and external:raise ValueError('External start and live attachment are distinct leases')
        if external and not context.get('effective_deadline'):raise ValueError('External budget start requires an absolute protection deadline')
        if attached:
            if initial!='running':raise RuntimeError('Existing task attachment requires running owned instance')
            if not context.get('effective_deadline') or not context.get('existing_job_identity'):raise ValueError('Attachment requires exact job and inherited deadline')
            save(phase='running',provider_status=initial,inherited_powered_lease=True)
        elif context.get('created_already_powered'):
            if initial not in ('running','starting','powering_on','creating','restarting'):raise RuntimeError('Created resource status unexpected: '+str(initial))
            save(phase='running' if initial=='running' else 'power_requested',power_on_attempts=1,provider_status=initial)
        else:
            if initial!='shutdown':raise RuntimeError('Owned UUID is occupied; do not power or stop it')
            save(phase='armed')
        armed=True
        stopped_seen=None
        while not state['terminal']:
            owner=read_guard_json(folder/'owner.json')
            stale=owner.get('nonce')!=nonce or time.time()-owner.get('heartbeat',0)>90
            reserve=context.get('shutdown_reserve_seconds',180)
            if not 120<=reserve<=180:raise ValueError('Shutdown reserve outside bounded protection')
            stop=stale or owner.get('desired')=='stop' or time.monotonic()-started>=context['max_powered_seconds']-reserve or time.time()>=context.get('effective_deadline',float('inf'))-reserve
            if stop:
                if external:
                    if (folder/'budget_lease.json').exists() or (folder/'start_cli_receipt.json').exists():
                        shutdown_once_and_verify(folder,context,state,save)
                    else:
                        # This guardian's owner never acquired a lease or sent
                        # start. Verify the off node without a needless stop.
                        status=api.status()
                        save(phase='cancelled_before_paid_start',terminal=True,
                             provider_status=status,provider_status_observed_at=time.time(),
                             shutdown_confirmed=status in ('shutdown','shutdown_by_starting_error'))
                    break
                if not state['power_on_attempts'] and not attached and not external:save(phase='cancelled_before_start',terminal=True);break
                if not state['power_off_attempts']:
                    save(phase='stopping',power_off_attempts=1,stop_reason='owner_stale' if stale else 'owner_stop_or_deadline')
                    api.power_off() # Exactly one mutation attempt.
                status=api.status()
                save(provider_status=status,provider_status_observed_at=time.time())
                if status=='shutdown':
                    if stopped_seen is None:stopped_seen=time.monotonic()
                    elif time.monotonic()-stopped_seen>=10:save(phase='shutdown_confirmed',terminal=True,shutdown_confirmed=True)
                elif time.monotonic()-started>=context['max_powered_seconds']:save(phase='shutdown_unconfirmed',terminal=True)
            elif external:
                # Protection is an independent local deadline/owner heartbeat,
                # not repeated cloud status polling. The original coordinator
                # checks actual running state and SSH process/output identity.
                save(phase='external_start_pending' if (folder/'budget_lease.json').exists() else 'armed',
                     external_budget_start=True,continuous_cloud_status_polling=False)
            elif owner.get('desired')=='run' and not state['power_on_attempts'] and not attached:
                save(phase='starting',power_on_attempts=1);api.power_on();save(phase='power_requested')
            elif state['power_on_attempts'] and state['phase']!='running':
                status=api.status()
                if status=='running':save(phase='running',provider_status=status)
                else:save(provider_status=status)
            else:save()
            time.sleep(2)
    except BaseException as exc:
        if (armed and getattr(api,'failed',False) and context.get('external_budget_start')
                and context.get('effective_deadline')):
            retain_stop_protection(folder,context,state,save,exc,started);return
        save(phase='cloud_failure' if getattr(api,'failed',True) else 'protection_failure',terminal=True,error_type=type(exc).__name__,error=str(exc)[:500],cloud_checks_stopped=True,
            action='Current session must inspect the saved first failure; no automatic retry or additional mutation.')

def package():
    if (OUT/'release.zip').exists():raise FileExistsError('Inspect/resume existing immutable deployment')
    old=ROOT/'evidence/own_bundle_campaign_20261002_v1';contract=json.loads((old/'contract.json').read_text())
    case=next(r for r in contract['cases'] if r['split']=='train' and r['team_size']==2)
    b=json.loads((old/case['bundle']).read_text());b['views']=[v for v in b['views'] if v['round']<2]
    valid={v['id'] for v in b['views']};b['candidates']=[r for r in b['candidates'] if r['entity']['view_id'] in valid];b['cutoff_round']=1
    restored=json.loads((ROOT/'evidence/own_dataset_rgb_restored_20261002_v1/manifest.json').read_text())
    folder=OUT/'smoke_public'/case['case_id'];folder.mkdir(parents=True,exist_ok=True);write(folder/'bundle.json',b,True)
    smoke_case=dict(case,bundle='smoke_public/'+case['case_id']+'/bundle.json',bundle_sha256=sha(folder/'bundle.json'))
    write(OUT/'smoke_manifest.json',dict(case=smoke_case,split='train',purpose='interface_only',historical_training_diagnostic=True,
        formal_or_training_admission=False,interface_budget_bytes=262144,method_backbone_units=14),True)
    files={p.relative_to(OUT).as_posix():p for p in OUT.rglob('*') if p.is_file() and p.name not in ('boundary_tests.log','release.zip','release_manifest.json','queue.sqlite') and 'cloud' not in p.parts}
    for v in b['views']:
        src=ROOT/restored['sources'][v['sha256']]['path']
        if sha(src)!=v['sha256']:raise ValueError('Historical TRAIN RGB changed')
        files['smoke_public/'+case['case_id']+'/'+v['id']]=src
    names=['oct07_image_core.py','oct07_author_adapters.py','oct07_trace_selector.py','oct07_independent_score.py',
        'run_oct07_image_campaign.py','train_oct07_image_selector.py','prepare_oct07_image_campaign.py',
        'own_rgb_baselines_20261002.py','run_symmetric_full_campaign_20260930.py']
    for name in names:files['scripts/'+name]=ROOT/'scripts'/name
    files['scripts/official_self_refine_task.py']=ROOT/'01_uav_language_grounding/benchmark/carlaair_v1/official_self_refine_task.py'
    files['processor_manifest.json']=ROOT/'evidence/own_rgb_baselines_20261002_v3/processor_manifest.json'
    manifest=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())]
    write(OUT/'release_manifest.json',dict(files=manifest),True)
    with zipfile.ZipFile(OUT/'release.zip','x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name,p in files.items():z.write(p,name)
        z.write(OUT/'release_manifest.json','release_manifest.json')
    print(json.dumps(dict(package_bytes=(OUT/'release.zip').stat().st_size,files=len(files),smoke_case=case['case_id'])))

def model_engineering_deadline(parent,assessment,approval,now=None):
    """Apply an explicit model-only window update without resetting history."""
    from deploy_oct07_scene_repair import engineering_phase_deadline
    direct=assessment.get('direct_user_continuation_authorization')
    if direct:
        expected=Path(parent)/'continuation48h_model_child_root_user_authorization_20261008.json'
        if Path(direct.get('path','')).name!=expected.name or sha(expected)!=direct.get('sha256'):
            raise ValueError('Current user continuation authority changed')
        authority=json.loads(expected.read_text(encoding='utf-8-sig'))
        priority=json.loads((ROOT/'.codex/task_priority.json').read_text(encoding='utf-8-sig'))
        registered=priority.get('current_model_bounded_continuation_authorization',{})
        if (registered.get('sha256')!=direct['sha256']
                or (ROOT/registered.get('path','missing')).resolve()!=expected.resolve()
                or authority.get('approved') is not True
                or authority.get('user_quote')!='批准通过'
                or authority.get('workspace')!='AIHCIR'
                or authority.get('task_id')!='oct07-48h-model-argv-repair4-20261007'
                or assessment['task_id']!=authority['task_id']
                or authority.get('original_approved_cost_scope_not_expanded') is not True
                or authority.get('cumulative_attempts_time_and_failed_costs_preserved') is not True
                or authority.get('original_scientific_parent_deadlines_pauses_and_resource_quantities_preserved') is not True
                or authority.get('full_scientific_training_or_comparison_allowed_by_this_update') is not False
                or authority.get('previous_closed_grants_not_reused') is not True):
            raise ValueError('Exact current original model-only user authorization required')
        original=json.loads((ROOT/'.codex/bridge_admin_acceleration_20261007/engineering_window_authorization.json').read_text(encoding='utf-8-sig'))
        if (assessment['maximum_powered_seconds']>authority['maximum_single_paid_seconds']
                or authority['maximum_single_paid_seconds']>881
                or assessment['all_in_ceiling_cny']>authority['new_paid_scope_cny']+1e-8
                or authority['new_paid_scope_cny']>2.23+1e-8
                or authority['new_engineering_admission_deadline']>original['new_engineering_admission_deadline']
                or approval['engineering_model_units']!=24 or approval['algorithm_rows']!=5124):
            raise ValueError('Current bounded continuation expands original time/cost/science')
        current=time.time() if now is None else now
        deadline=authority['new_engineering_admission_deadline']
        if not authority['new_engineering_admission_not_before']<=current<deadline:
            raise TimeoutError('Current bounded user continuation window is not active')
        return deadline
    binding=assessment.get('engineering_window_authorization')
    if not binding:return engineering_phase_deadline(parent)
    expected='.codex/bridge_admin_acceleration_20261007/engineering_window_authorization.json'
    if binding.get('path')!=expected:raise ValueError('Unexpected engineering window authority')
    path=ROOT/binding['path']
    if sha(path)!=binding.get('sha256'):raise ValueError('Engineering window authority changed')
    window=json.loads(path.read_text(encoding='utf-8-sig'))
    decision_path=path.with_name('decision.json')
    if sha(decision_path)!=binding.get('decision_sha256'):raise ValueError('Engineering decision changed')
    decision=json.loads(decision_path.read_text(encoding='utf-8-sig'))
    task='oct07-48h-model-argv-repair4-20261007'
    original_decision=(decision.get('explicit_engineering_window_update')==window
        and decision.get('engineering_review_window_only_extended') is True
        and decision.get('grant',{}).get('registered') is True)
    grant=next((g for g in decision.get('grants',[]) if g.get('task_id')==task),{})
    narrow_decision=(decision.get('unchanged_parent_science') is True
        and decision.get('previous_closed_grants_not_reused') is True
        and decision.get('original_failed_costs_preserved') is True
        and decision.get('explicit_engineering_window_update',{}).get('new_engineering_admission_deadline')==decision.get('expires_at')
        and grant.get('receipt',{}).get('registered') is True
        and grant.get('receipt',{}).get('workspace')=='AIHCIR'
        and grant.get('receipt',{}).get('task_id')==task
        and grant.get('receipt',{}).get('grant_id')==decision.get('grant_id')
        and grant.get('receipt',{}).get('expires_at')==decision.get('expires_at')
        and grant.get('receipt',{}).get('amount_cny')==decision.get('all_in_ceiling_cny')
        and decision.get('maximum_paid_seconds',2401)<=window['maximum_single_paid_seconds']
        and decision.get('all_in_ceiling_cny',3.45)<=window['new_paid_scope_cny']
        and decision.get('expires_at',float('inf'))<=window['new_engineering_admission_deadline'])
    if (assessment['task_id']!=task or window['task_id']!=task or decision['task_id']!=task
            or window.get('workspace')!='AIHCIR' or decision.get('workspace')!='AIHCIR'
            or not (original_decision or narrow_decision)
            or window.get('old_engineering_review_superseded') is not True
            or window.get('cumulative_attempts_time_and_failed_costs_preserved') is not True
            or window.get('original_scientific_parent_deadlines_pauses_and_resource_quantities_preserved') is not True
            or window.get('full_scientific_training_or_comparison_allowed_by_this_update') is not False
            or window.get('fresh_guard_quote_wallet_and_source_still_required') is not True):
        raise ValueError('Exact original model-only continuation authority is required')
    original=json.loads((Path(parent)/'continuation48h_engineering_phase.json').read_text())
    if window['old_engineering_review_deadline']!=original['deadline']:
        raise ValueError('Window update refers to a different original phase')
    if (assessment['maximum_powered_seconds']>window['maximum_single_paid_seconds']
            or window['maximum_single_paid_seconds']>2400
            or assessment['all_in_ceiling_cny']>window['new_paid_scope_cny']+1e-8
            or window['new_paid_scope_cny']>3.44+1e-8
            or assessment['maximum_powered_seconds']>decision.get('maximum_paid_seconds',2400)
            or assessment['all_in_ceiling_cny']>decision['all_in_ceiling_cny']+1e-8):
        raise ValueError('Updated model engineering time or cost expanded')
    current=time.time() if now is None else now
    effective=min(window['new_engineering_admission_deadline'],decision['expires_at'])
    if not window['new_engineering_admission_not_before']<=current<effective:
        raise TimeoutError('Explicit model-only engineering admission window is not active')
    if approval['engineering_model_units']!=24 or approval['algorithm_rows']!=5124:
        raise ValueError('Parent engineering/scientific scope changed')
    return effective


def deploy_admitted(root,assessment_file,package_file,attempt,stage='engineering_batch'):
    """Original coordinator with fresh shared admission and exact process proof."""
    import re
    import psutil
    from oct07_runtime_scope import verify_runtime,paused
    from deploy_oct07_scene_repair import budgeted_start,close_verified_lease,engineering_phase_deadline,shared_cli,prepare_budgeted_start
    from oct07_ssh_once import connect_once
    from run_public_mcq_teacher_autodl import RemoteJob
    root=Path(root).resolve();parent=root.parent;package_file=Path(package_file).resolve()
    verify_runtime(root)
    if paused(root):raise ValueError('Existing scientific pause is preserved')
    if not re.fullmatch(r'[a-zA-Z0-9_]+',attempt):raise ValueError('Unsafe unique deployment attempt')
    a=json.loads(Path(assessment_file).read_text(encoding='utf-8'))
    if a.get('ready') is not True or a.get('owned_uuid')!='OWNED_INSTANCE_UUID':
        raise ValueError('Actual ready model-node assessment required')
    allowed=('engineering_batch',*SCIENTIFIC_STAGES)
    if stage not in allowed:raise ValueError('Unapproved original pipeline stage')
    engineering=stage=='engineering_batch'
    stage_proof=admitted_stage_preflight(root,a,stage)
    from oct07_runtime_scope import queue_path
    original_queue=queue_path(root)
    original_queue_sha256=sha(original_queue) if original_queue.is_file() else None
    binding=a['tonight_authorization'];authority=parent/binding['path']
    if sha(authority)!=binding['sha256']:raise ValueError('Model authority changed')
    approval=json.loads(authority.read_text(encoding='utf-8'))
    if approval.get('approved') is not True or approval.get('parent60816_paused') is not True:
        raise ValueError('Original scope/pause authority differs')
    if stage=='engineering_batch' and a['maximum_powered_seconds']>approval['engineering_model_powered_limit_seconds']:
        raise ValueError('Model engineering budget expanded')
    window_deadline=(model_engineering_deadline(parent,a,approval) if engineering
        else datetime.fromisoformat(approval['new_compute_stop']).timestamp())
    used_seconds=0.;ledger_closed_ceiling=0.;phase_compute_upper=0.;prior_attempt_count=0
    model_compute_upper=0.;prior_model_count=0
    for prior in list(parent.glob('model_continuation48h_*'))+list(parent.glob('capture_scene_qualification1_priority_recovery*')):
        ready=prior/'ready_budget_assessment.json';lease=prior/'budget_lease.json'
        if not ready.is_file() or not lease.is_file():continue
        assessment=json.loads(ready.read_text(encoding='utf-8'))
        if assessment.get('scientific_stage_binding'):continue
        if assessment.get('tonight_authorization',{}).get('sha256')!=binding['sha256']:continue
        ledger_closed_ceiling+=assessment['all_in_ceiling_cny'];prior_attempt_count+=1
        elapsed=engineering_powered_commitment(prior,a['owned_uuid'])
        phase_compute_upper+=elapsed*assessment['rate_cny_hour_upper']/3600
        if prior.name.startswith('model_'):
            used_seconds+=elapsed
            model_compute_upper+=elapsed*assessment['rate_cny_hour_upper']/3600
            prior_model_count+=1
    # Closed lease ceilings remain fully debited in the shared ledger. For
    # the physical20CNY phase cap, repeated leases of the same retained disks
    # count storage once. Historical ledger coverage needs coordinator funding;
    # it is never treated as refundable or bypassed by this local calculation.
    phase_upper=(phase_compute_upper+a['maximum_powered_seconds']*a['rate_cny_hour_upper']/3600
        +3.52+.12*(prior_attempt_count+1))
    if engineering and phase_upper>approval['engineering_combined_all_in_cap_cny']+1e-8:
        raise ValueError('Combined engineering20CNY ceiling exhausted; no cost reset')
    model_upper=(model_compute_upper+a['maximum_powered_seconds']*a['rate_cny_hour_upper']/3600
        +a['storage_ceiling_cny']+.12*(prior_model_count+1))
    if engineering and model_upper>10+1e-8:raise ValueError('Original model engineering branch10CNY ceiling exceeded')
    if engineering and a.get('phase_actual_cost_upper_cny') is not None and phase_upper>a['phase_actual_cost_upper_cny']+.02:
        raise ValueError('Actual cumulative phase quote/accounting differs from assessment')
    manifest=json.loads(package_file.with_name(package_file.stem+'_receipt.json').read_text(encoding='utf-8'))
    if sha(package_file)!=manifest['sha256']:raise ValueError('Package archive changed')
    with zipfile.ZipFile(package_file) as archive:
        m=json.loads(archive.read('first_execution_release_manifest.json'))
        for item in m['files']:
            raw=archive.read(item['path'])
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise ValueError('Package member changed')
            if item['path'].startswith('scripts/'):
                local=ROOT/'scripts'/Path(item['path']).name
                if local.is_file() and sha(local)!=item['sha256']:raise ValueError('Stale model source before payment')
        if not engineering:
            for name,digest in stage_proof['prerequisites_sha256'].items():
                if hashlib.sha256(archive.read('tonight_runtime/'+name)).hexdigest()!=digest:
                    raise ValueError('Scientific dependency evidence is absent or stale in package')
    folder=parent/('model_'+attempt);folder.mkdir(exist_ok=False)
    write(folder/'stage_admission_preflight.json',stage_proof,True)
    pool=shared_cli(folder,'before_guard_allocation','budget','status')
    try:remaining=require_available_allocation(pool,a)
    except ValueError as exc:
        write(folder/'admission_status.json',dict(observed_at=time.time(),paid_started=False,
            guard_started=False,error=str(exc),stage=stage,scientific_results=0),True)
        raise
    uid=a['owned_uuid'];api=Provider(uid)
    if api.status()!='shutdown':raise ValueError('Owned model instance occupied; inspect existing job')
    cutoff=min(a['absolute_batch_deadline'],time.time()+a['maximum_powered_seconds'],
        datetime.fromisoformat(approval['new_compute_stop']).timestamp())
    if engineering:cutoff=min(cutoff,window_deadline,
        time.time()+approval['engineering_model_powered_limit_seconds']-used_seconds)
    if cutoff-time.time()<600:raise TimeoutError('No complete protected model window')
    nonce=uuid.uuid4().hex
    write(folder/'guard_context.json',dict(nonce=nonce,owned_uuid=uid,max_powered_seconds=int(cutoff-time.time()),
        effective_deadline=cutoff,external_budget_start=True,shutdown_reserve_seconds=180),True)
    write(folder/'engineering_window_verification.json',dict(observed_at=time.time(),
        authorization=a.get('engineering_window_authorization'),effective_deadline=cutoff,
        original_model_used_powered_upper_seconds=used_seconds,cumulative_phase_cost_upper_cny=phase_upper,
        cumulative_model_branch_cost_upper_cny=model_upper,prior_closed_ledger_ceiling_cny=ledger_closed_ceiling,
        attempts_and_costs_not_reset=True,formal_rows_allowed=0 if engineering else approval['algorithm_rows'],
        engineering_cost_caps_apply=engineering,stage=stage,unspent_share_before_guard_cny=remaining),True)
    def owner(desired):write(folder/'owner.json',dict(nonce=nonce,pid=os.getpid(),desired=desired,heartbeat=time.time()))
    owner('run');log=(folder/'guard.log').open('xb')
    proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'guard','--guard-folder',str(folder)],
        stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(2):owner('run')
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    remote=RemoteJob(api,folder,parent.name,PYTHON,'',True);identity=None
    def beat():
        g=read_guard_json(folder/'guard.json')
        if (g['nonce']!=nonce or g['pid']!=proc.pid or g['terminal'] or
                g.get('protection_degraded') or time.time()-g['heartbeat']>30):
            raise RuntimeError('Exact independent model protection unavailable')
    remote.heartbeat=beat
    target='/path/to/aihcir/sessions/'+parent.name
    try:
        until=time.monotonic()+30
        while time.monotonic()<until:
            if (folder/'guard.json').exists():
                beat()
                if read_guard_json(folder/'guard.json')['phase']=='armed':break
            time.sleep(.25)
        else:raise TimeoutError('Model protection did not arm while off')
        wait_seconds=a.get('allocation_wait_maximum_seconds',0)
        if not isinstance(wait_seconds,(int,float)) or not 0<=wait_seconds<=600:
            raise ValueError('Admission wait exceeds this inline bounded batch')
        if wait_seconds:
            prepare_budgeted_start(folder,proc,uid,assessment_file,receipt_prefix='pending_')
            admission_stop=min(time.monotonic()+wait_seconds,time.monotonic()+cutoff-time.time()-600)
            check=0
            while True:
                beat()
                if read_guard_json(folder/'guard.json').get('provider_status')!='shutdown':
                    raise ValueError('Original off-node admission acquired another actual state; inspect ownership')
                pool=shared_cli(folder,'allocation_readback_'+str(check),'budget','status');check+=1
                allocation=next(x for x in pool['allocations'] if x['workspace']=='AIHCIR')
                available=round(allocation['ceiling_cny']-allocation['used_upper_bound_cny'],2)
                proof=dict(observed_at=time.time(),stage='awaiting_shared_allocation_while_off',
                    needed_cny=a['all_in_ceiling_cny'],available_cny=available,
                    admission_deadline=cutoff,paid_started=False,scientific_results=0)
                write(folder/'admission_status.json',proof);print(json.dumps(proof),flush=True)
                if available+1e-8>=a['all_in_ceiling_cny']:break
                if time.monotonic()>=admission_stop:
                    raise TimeoutError('Shared historical-ceiling coverage still pending; no paid action')
                time.sleep(min(30,max(0,admission_stop-time.monotonic())))
        budgeted_start(folder,proc,uid,assessment_file)
        until=time.monotonic()+180
        while time.monotonic()<until:
            beat()
            if api.status()=='running':break
            time.sleep(2)
        else:raise TimeoutError('Single shared model start did not reach running')
        connect_once(remote);remote.client.get_transport().set_keepalive(15)
        probe="import json,os,shutil;from pathlib import Path;rows=[]\nfor p in Path('/proc').iterdir():\n if not p.name.isdigit() or int(p.name)==os.getpid():continue\n try:a=(p/'cmdline').read_bytes().split(b'\\0')\n except OSError:continue\n if any(Path(x.decode(errors='replace')).name in ('run_oct07_image_campaign.py','execute_oct07_admitted_pipeline.py','train_oct07_image_selector.py','run_oct07_native_node.py') for x in a if x):rows.append(dict(pid=int(p.name),birth=(p/'stat').read_text().split()[21]))\nprint(json.dumps(dict(jobs=rows,free=shutil.disk_usage('/root/autodl-tmp').free,python=Path("+repr(PYTHON)+").is_file())))"
        inventory=json.loads(remote.command('python3 -c '+shlex.quote(probe),30));write(folder/'actual_inventory.json',inventory,True)
        if inventory['jobs'] or not inventory['python']:raise ValueError('Existing live scientific job or missing original interpreter')
        if inventory['free']<5*1024**3+2*package_file.stat().st_size:raise ValueError('Remote package peak/storage floor unavailable')
        remote.command('mkdir -p '+shlex.quote(target),15)
        transfer_seconds=min(600,max(1,int(cutoff-time.time()-480)))
        if engineering:
            deploy_cached_engineering_package(remote,package_file,target,folder,transfer_seconds)
        else:
            remote.upload(package_file,target+'/first_execution_release.zip',transfer_seconds)
            if remote.command('sha256sum '+shlex.quote(target+'/first_execution_release.zip'),15).split()[0]!=sha(package_file):
                raise ValueError('Uploaded original package hash differs')
            unpack="import zipfile,json,hashlib;from pathlib import Path;p=Path("+repr(target)+");z=zipfile.ZipFile(p/'first_execution_release.zip');assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist());z.extractall(p);m=json.loads((p/'first_execution_release_manifest.json').read_text());assert all(hashlib.sha256((p/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in m['files']);print(json.dumps(dict(verified=True,files=len(m['files']))))"
            write(folder/'remote_package_verification.json',json.loads(remote.command('python3 -c '+shlex.quote(unpack),180)),True)
        argv=[PYTHON,'-B','-u',target+'/scripts/execute_oct07_admitted_pipeline.py','--root',target+'/tonight_runtime',
               '--stage',stage,'--attempt',attempt,'--lease-deadline',str(cutoff)]
        if not engineering:
            # Read-only remote validation before a scientific subprocess. Package
            # reuse cannot establish the current native inputs/environment by itself.
            query="import json,sys;from pathlib import Path;sys.path.insert(0,"+repr(target+'/scripts')+");from oct07_pipeline_preflight import inspect;p=inspect(Path("+repr(target+'/tonight_runtime')+"));g=p['stages']["+repr(stage)+"];assert not(g['missing'] or g['pause_present'] or g['compute_expired'] or p['errors']),g;print(json.dumps(dict(passed=True,matrix_sha256=p['matrix_sha256'],stage="+repr(stage)+")))"
            proof=json.loads(remote.command('python3 -c '+shlex.quote(query),60))
            if proof['matrix_sha256']!=sha(root/'matrix.csv'):raise ValueError('Actual remote scientific matrix differs')
            write(folder/'remote_stage_admission.json',proof,True)
            argv+=['--backbone','qwen3']
        exit_path=target+'/'+attempt+'_exit.json';controller_log=target+'/'+attempt+'_controller.log'
        wrapper="import subprocess,json,time;from pathlib import Path;r=subprocess.run("+repr(argv)+");Path("+repr(exit_path)+").write_text(json.dumps(dict(exit_code=r.returncode,ended_at=time.time())));raise SystemExit(r.returncode)"
        launch="import subprocess,json;from pathlib import Path;f=Path("+repr(controller_log)+").open('xb');p=subprocess.Popen(["+repr(PYTHON)+",'-B','-u','-c',"+repr(wrapper)+"],stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True);print(json.dumps(dict(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21])))"
        identity=json.loads(remote.command('python3 -c '+shlex.quote(launch),15));write(folder/'job_identity.json',dict(identity,command=argv,source_sha256=sha(ROOT/'scripts/execute_oct07_admitted_pipeline.py')),True)
        while time.time()<cutoff-360:
            beat()
            progress_path=(target+'/tonight_runtime/engineering_actual_train/engineering_status.json' if engineering
                else target+'/tonight_runtime/fit_status.json')
            code="import json;from pathlib import Path;p=Path("+repr(exit_path)+");s=Path("+repr(progress_path)+");j=Path('/proc/"+str(identity['pid'])+"');print(json.dumps(dict(exit=json.loads(p.read_text()) if p.exists() else None,progress=json.loads(s.read_text()) if s.exists() else None,birth=(j/'stat').read_text().split()[21] if j.exists() else None)))"
            observed=json.loads(remote.command('python3 -c '+shlex.quote(code),30));write(folder/'actual_progress.json',dict(observed,observed_at=time.time()))
            print(json.dumps(dict(observed_at=time.time(),stage=stage,exit=observed['exit'],
                engineering_executed=(observed['progress'] or {}).get('executed'),formal_algorithm_rows=0 if stage=='engineering_batch' else None)),flush=True)
            if observed['exit'] is not None:
                write(folder/'actual_exit.json',observed['exit'],True);break
            if observed['birth']!=identity['start_ticks']:raise RuntimeError('Actual model job disappeared or PID was reused')
            time.sleep(30)
        else:raise TimeoutError('Actual model work reached preservation reserve')
        if observed['exit']['exit_code']!=0:
            raise RuntimeError('Actual original model entry failed; inspect preserved runtime evidence before any recovery')
    except BaseException as exc:
        write(folder/'failure.json',dict(observed_at=time.time(),error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()),True)
        raise
    finally:
        if remote.client is not None and not engineering and identity is not None:
            try:
                kill="import os,signal;from pathlib import Path;p=Path('/proc/"+str(identity['pid'])+"');alive=p.exists() and (p/'stat').read_text().split()[21]=="+repr(identity['start_ticks'])+"\nif alive:os.killpg("+str(identity['pid'])+",signal.SIGTERM)"
                remote.command('python3 -c '+shlex.quote(kill),15)
                preserve_scientific_outputs(remote,target,root,folder,
                    a['preservation_maximum_bytes'],original_queue_sha256)
            except BaseException as exc:write(folder/'preservation_failure.json',dict(error_type=type(exc).__name__,error=str(exc),observed_at=time.time()),True)
        if remote.client is not None and engineering and identity is not None:
            try:
                kill="import os,signal;from pathlib import Path;p=Path('/proc/"+str(identity['pid'])+"');alive=p.exists() and (p/'stat').read_text().split()[21]=="+repr(identity['start_ticks'])+"\nif alive:os.killpg("+str(identity['pid'])+",signal.SIGTERM)"
                remote.command('python3 -c '+shlex.quote(kill),15)
                preserve="import zipfile,hashlib,json;from pathlib import Path;p=Path("+repr(target)+");o=p/'engineering_effective_outputs.zip';paths=[]\nfor f in (p/'tonight_runtime/engineering_actual_train').rglob('*'):\n if f.is_file() and f.suffix in ('.json','.jsonl','.log') and not any(x in f.relative_to(p).parts for x in ('native_assets','author_sources','task_prompts','public_rgb','private_source')):paths.append(f)\npaths += [f for f in p.glob("+repr(attempt+'*')+") if f.is_file()];paths += list((p/'tonight_runtime/pipeline_logs').glob('engineering*'));paths=sorted(set(paths));assert sum(f.stat().st_size for f in paths)<64*1024**2\nrows=[dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in paths]\nwith zipfile.ZipFile(o,'w',zipfile.ZIP_DEFLATED) as z:\n for f in paths:z.write(f,f.relative_to(p).as_posix())\n z.writestr('preservation_manifest.json',json.dumps(dict(files=rows)))\nprint(json.dumps(dict(bytes=o.stat().st_size,sha256=hashlib.sha256(o.read_bytes()).hexdigest(),files=len(paths))))"
                receipt=json.loads(remote.command('python3 -c '+shlex.quote(preserve),60))
                with remote.client.open_sftp() as sftp:sftp.get(target+'/engineering_effective_outputs.zip',str(folder/'effective_outputs.zip'))
                if sha(folder/'effective_outputs.zip')!=receipt['sha256']:raise ValueError('Preserved engineering archive hash differs')
                with zipfile.ZipFile(folder/'effective_outputs.zip') as archive:
                    sealed=json.loads(archive.read('preservation_manifest.json'))['files']
                    if sum(r['bytes'] for r in sealed)>64*1024**2:raise ValueError('Model effective preservation exceeded bounded writes')
                    for item in sealed:
                        path=Path(item['path'])
                        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe model preservation path')
                        raw=archive.read(item['path'])
                        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                            raise ValueError('Independent model member audit differs')
                        dest=folder/'effective'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
                write(folder/'effective_independent_audit.json',dict(observed_at=time.time(),
                    files=len(sealed),passed=True,scientific_acceptance=False),True)
                write(folder/'preservation_receipt.json',dict(receipt,observed_at=time.time()),True)
            except BaseException as exc:write(folder/'preservation_failure.json',dict(error_type=type(exc).__name__,error=str(exc),observed_at=time.time()),True)
        stop.set();thread.join(3);owner('stop')
        until=time.monotonic()+130
        while proc.poll() is None and time.monotonic()<until:time.sleep(1)
        g=read_guard_json(folder/'guard.json') if (folder/'guard.json').exists() else {}
        remote.close();log.close()
        if g.get('shutdown_confirmed') and (folder/'budget_lease.json').is_file():
            lease=json.loads((folder/'budget_lease.json').read_text(encoding='utf-8'))
            close_verified_lease(folder,lease['lease_id'],uid)


def deploy():
    from run_public_mcq_teacher_autodl import RemoteJob,system_proxy_socket
    import paramiko
    folder=OUT/CLOUD_FOLDER;folder.mkdir(exist_ok=False);nonce=uuid.uuid4().hex
    c=json.loads((OUT/'contract.json').read_text());api=Provider()
    lease=ROOT/'evidence/cloud_leases'/('oct07_'+api.uid+'.json');lease.parent.mkdir(parents=True,exist_ok=True)
    write(lease,dict(nonce=nonce,owner_pid=os.getpid(),campaign=c['campaign_id'],created_at=time.time()),True)
    write(folder/'guard_context.json',dict(nonce=nonce,max_powered_seconds=7200,contract_sha256=sha(OUT/'contract.json')),True)
    def request(desired):write(folder/'owner.json',dict(nonce=nonce,pid=os.getpid(),desired=desired,heartbeat=time.time()))
    request('idle');log=(folder/'guard.log').open('xb')
    proc=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'guard','--release',RELEASE],stdout=log,stderr=subprocess.STDOUT,
        creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0,start_new_session=os.name=='posix')
    remote=RemoteJob(api,folder,c['campaign_id'],PYTHON,'',True);started=time.monotonic();pid=None
    state=dict(phase='guard_handshake',uuid=api.uid,guard_pid=proc.pid,submitted=False,release_attempted=False,
        check_interval_seconds=30,phone_push_verified=False,offline_ai_supervision=False,fee_ceiling_cny=None)
    def save(**kw):state.update(kw,observed_at=time.time());write(folder/'status.json',state);print(json.dumps(state),flush=True)
    def beat():
        request('run')
        g=json.loads((folder/'guard.json').read_text())
        if g['nonce']!=nonce or g['pid']!=proc.pid or time.time()-g['heartbeat']>30 or g['terminal']:raise RuntimeError('Independent shutdown guard unavailable')
        if time.monotonic()-started>6800:raise TimeoutError('Save/shutdown reserve reached')
    remote.heartbeat=beat
    try:
        until=time.monotonic()+30
        while time.monotonic()<until:
            if (folder/'guard.json').exists():
                g=json.loads((folder/'guard.json').read_text())
                if g['phase']=='armed':break
                if g['terminal']:raise RuntimeError('Guard first cloud check failed; no launch')
            time.sleep(.25)
        else:raise TimeoutError('Guard did not arm')
        request('run');save(phase='starting_owned_instance')
        until=time.monotonic()+180
        while time.monotonic()<until:
            beat();g=json.loads((folder/'guard.json').read_text())
            if g['phase']=='running':break
            time.sleep(2)
        else:raise TimeoutError('Provider startup unavailable')
        # Exactly one SSH handshake, strict known-host verification. A failure
        # is retained and ends cloud checking instead of a blind retry loop.
        snap=api.connection();client=paramiko.SSHClient();client.load_host_keys(str(api.token_path.parent/'known_hosts'));client.set_missing_host_key_policy(paramiko.RejectPolicy())
        stream=system_proxy_socket(snap['proxy_host'],int(snap['ssh_port']))
        client.connect(snap['proxy_host'],port=int(snap['ssh_port']),username='root',password=snap['root_password'],sock=stream,
            timeout=15,banner_timeout=15,auth_timeout=15,look_for_keys=False,allow_agent=False)
        price=snap.get('payg_price');del snap;remote.client=client;remote.proxy_socket=stream
        inventory_code="import json,os,shutil;from pathlib import Path;p=Path('/path/to/aihcir');jobs=[]\nfor x in Path('/proc').iterdir():\n if not x.name.isdigit() or int(x.name)==os.getpid():continue\n try: args=(x/'cmdline').read_bytes().split(b'\\0')\n except OSError:continue\n if any(Path(a.decode(errors='replace')).name in ('own_rgb_baselines_20261002.py','run_oct07_image_campaign.py','CarlaUE4-Linux-Shipping','train_oct07_image_selector.py') for a in args if a):jobs.append(dict(pid=int(x.name),cmd=b' '.join(args).decode(errors='replace')[:400]))\nprint(json.dumps(dict(python=Path("+repr(PYTHON)+").exists(),data_free=shutil.disk_usage(p).free,root_free=shutil.disk_usage('/').free,benchmark_runtime=(p/'benchmarks/carlaair_v1/runtime_preparation.json').exists(),processes=jobs)))"
        inventory=json.loads(remote.command('python3 -c '+shlex.quote(inventory_code),30));write(folder/'remote_inventory.json',dict(inventory,observed_at=time.time()),True)
        if inventory['processes']:raise RuntimeError('A scientific job exists; inspect identity before deployment')
        if inventory['data_free']<2*1024**3:raise RuntimeError('Remote peak-write reservation unavailable')
        remote.command('mkdir '+shlex.quote(REMOTE),15)
        remote.upload(OUT/(RELEASE+'.zip'),REMOTE+'/release.zip',90)
        if remote.command('sha256sum '+shlex.quote(REMOTE+'/release.zip'),15).split()[0]!=sha(OUT/(RELEASE+'.zip')):raise ValueError('Uploaded package hash differs')
        unpack="import zipfile,json,hashlib;from pathlib import Path;p=Path("+repr(REMOTE)+");z=zipfile.ZipFile(p/'release.zip');assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist());z.extractall(p);m=json.loads((p/'release_manifest.json').read_text());assert all(hashlib.sha256((p/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in m['files']);print(json.dumps(dict(files=len(m['files']),verified=True)))"
        verified=json.loads(remote.command('python3 -c '+shlex.quote(unpack),30));write(folder/'deployment_verification.json',verified,True)
        launch="import subprocess;from pathlib import Path;p=Path("+repr(REMOTE)+");f=(p/'controller.log').open('xb');c=subprocess.Popen(["+repr(PYTHON)+",'-B','-u',str(p/'scripts/run_oct07_image_campaign.py'),'--root',str(p),'--stage','smoke'],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(c.pid)"
        pid=int(remote.command('python3 -c '+shlex.quote(launch),15))
        identity="import json,os;from pathlib import Path;p=Path('/proc/"+str(pid)+"');assert b'run_oct07_image_campaign.py' in (p/'cmdline').read_bytes();print(json.dumps(dict(pid="+str(pid)+",pgid=os.getpgid("+str(pid)+"),start_ticks=(p/'stat').read_text().split()[21])))"
        ident=json.loads(remote.command('python3 -c '+shlex.quote(identity),15));write(folder/'job_identity.json',ident,True)
        write(folder/'submission.json',dict(at=time.time(),**ident,command=launch,remote_directory=REMOTE,log=REMOTE+'/controller.log',stage='smoke_then_admitted_full_queue',scientific_queue='matrix.csv:39040 rows, waiting for TRAIN120 cap/runtime gates'),True)
        save(phase='submitted',submitted=True,remote_pid=pid,price_milli_cny_per_hour=price,total=14,completed=0)
        while True:
            beat()
            query="import json;from pathlib import Path;p=Path("+repr(REMOTE)+");proc=Path('/proc/"+str(pid)+"');print(json.dumps(dict(live=proc.exists(),start_ticks=(proc/'stat').read_text().split()[21] if proc.exists() else None,status=json.loads((p/'smoke_status.json').read_text()) if (p/'smoke_status.json').exists() else None,done=json.loads((p/'smoke_results.json').read_text()) if (p/'smoke_results.json').exists() else None,log_tail=(p/'controller.log').read_text(errors='replace')[-1200:])))"
            observation=json.loads(remote.command('python3 -c '+shlex.quote(query),20));write(folder/'latest_observation.json',dict(observation,observed_at=time.time()))
            status=observation['status'] or {};save(phase='running_smoke',completed=status.get('completed',0),last_checked_at=time.time(),next_check_at=time.time()+30)
            if observation['done']:
                if not observation['done']['passed']:raise RuntimeError('Actual image interface failure; evidence preserved')
                save(phase='smoke_complete_full_dependencies_pending',completed=14);break
            if not observation['live'] or observation['start_ticks']!=ident['start_ticks']:raise RuntimeError('Scientific process exited without completion')
            until=time.monotonic()+30
            while time.monotonic()<until:beat();time.sleep(2)
    except BaseException as exc:
        save(phase='failed',error_type=type(exc).__name__,cloud_checks_stopped=isinstance(exc,ValueError) or api.failed)
        write(folder/'failure.json',dict(error_type=type(exc).__name__,at=time.time(),traceback=traceback.format_exc()),True)
    finally:
        # Preserve small effective outputs, never an entire machine archive.
        if remote.client is not None and pid is not None:
            try:
                if (folder/'job_identity.json').exists():
                    ident=json.loads((folder/'job_identity.json').read_text())
                    stop="import os,signal;from pathlib import Path;p=Path('/proc/"+str(pid)+"');\nif p.exists() and (p/'stat').read_text().split()[21]=="+repr(ident['start_ticks'])+": os.killpg("+str(ident['pgid'])+",signal.SIGTERM)\n"
                    remote.command('python3 -c '+shlex.quote(stop),15)
                archive="import zipfile;from pathlib import Path;p=Path("+repr(REMOTE)+");files=[f for d in ('smoke_outputs','calls') for f in (p/d).glob('*') if f.is_file()]+[p/n for n in ('controller.log','smoke_status.json','smoke_results.json') if (p/n).is_file()];z=zipfile.ZipFile(p/'effective_outputs.zip','w',zipfile.ZIP_DEFLATED);[z.write(f,f.relative_to(p).as_posix()) for f in files];z.close()"
                remote.command('python3 -c '+shlex.quote(archive),30)
                with remote.client.open_sftp() as sftp:
                    sftp.get_channel().settimeout(20)
                    if sftp.stat(REMOTE+'/effective_outputs.zip').st_size>256*1024**2:raise ValueError('Effective output transfer bound')
                    sftp.get(REMOTE+'/effective_outputs.zip',str(folder/'effective_outputs.zip'))
                save(effective_output_local_sha256=sha(folder/'effective_outputs.zip'))
            except BaseException as exc:save(output_preservation_error_type=type(exc).__name__)
        request('stop');until=time.monotonic()+130
        while time.monotonic()<until and proc.poll() is None:time.sleep(1)
        final=json.loads((folder/'guard.json').read_text()) if (folder/'guard.json').exists() else {}
        save(shutdown_confirmed=final.get('shutdown_confirmed',False),finished_at=time.time(),next_check_at=None,
            provisional_fee_estimate_cny=(state.get('price_milli_cny_per_hour') or 0)/1000*(time.monotonic()-started)/3600,actual_invoice_cny=None)
        remote.close();log.close()
        if final.get('shutdown_confirmed') or not final.get('power_on_attempts'):
            if json.loads(lease.read_text())['nonce']==nonce:lease.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['package','run','guard','admitted']);p.add_argument('--guard-folder',type=Path);p.add_argument('--release',choices=('release','repair1','repair2'),default='release');p.add_argument('--root',type=Path);p.add_argument('--assessment-file',type=Path);p.add_argument('--package-file',type=Path);p.add_argument('--attempt');p.add_argument('--stage',choices=('engineering_batch',*SCIENTIFIC_STAGES),default='engineering_batch');a=p.parse_args()
    RELEASE=a.release
    if RELEASE!='release':REMOTE+='_'+RELEASE;CLOUD_FOLDER='cloud_'+RELEASE
    if a.mode=='package':package()
    elif a.mode=='guard':guard(a.guard_folder or OUT/CLOUD_FOLDER)
    elif a.mode=='admitted':
        if not all((a.root,a.assessment_file,a.package_file,a.attempt)):p.error('Admitted execution requires root, exact assessment/package and unique attempt')
        deploy_admitted(a.root,a.assessment_file,a.package_file,a.attempt,a.stage)
    else:deploy()
