"""Inline native scene execution with the existing exact shutdown guard."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from prepare_oct07_scene_repair import OUT
from prepare_oct07_image_campaign import ROOT,sha,write
from deploy_oct07_image_campaign import Provider,bridge_root
from run_public_mcq_teacher_autodl import RemoteJob

REMOTE='/path/to/aihcir/benchmarks/carlaair_v1'
DEPLOY_CODE=('oct07_capture_protocol.py','oct07_capture_probe.py','oct07_scene_geometry_v4.py',
    'oct07_scene_preservation.py','oct07_material_probe_v4.py','oct07_material_resume.py',
    'oct07_material_budget.py','oct07_native_exposure_profile.py','oct07_native_detail_profile.py',
    'oct07_capture_scope.py','oct07_qualification_budget.py','oct07_native_geometry_diagnosis.py','prepare_oct07_tonight_core.py',
    'oct07_prospective_camera_cover.py','oct07_scene_observation.py','oct07_native_actor_restore.py','oct07_native_transit_profile.py','oct07_native_camera_settle.py','oct07_qualification_resume.py','oct07_native_group_lanes.py','run_oct07_scene_node.py','oct07_native_component_probe.py','oct07_native_visual_inventory.py')


def verify_package(source,mode,campaign=OUT,scripts=ROOT/'scripts'):
    """Reject stale or altered preparation before any provider/guard operation."""
    import hashlib
    source=Path(source);campaign=Path(campaign);scripts=Path(scripts)
    manifest=json.loads(source.with_name(source.stem+'_manifest.json').read_text(encoding='utf-8'))
    if sha(source)!=manifest['sha256']:raise ValueError('Deployment archive hash changed')
    with zipfile.ZipFile(source) as z:
        rows=manifest['files'];names={r['path'] for r in rows}
        if len(rows)!=len(names) or set(z.namelist())!=names|{'code_scene_repair/snapshot_manifest.json'}:
            raise ValueError('Deployment archive inventory changed')
        for row in rows:
            n=row['path']
            if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('Unsafe package path')
            raw=z.read(n)
            if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise ValueError('Deployment member hash changed')
        for n in DEPLOY_CODE:
            member='code_scene_repair/'+n
            if member not in names or hashlib.sha256(z.read(member)).hexdigest()!=sha(scripts/n):
                raise ValueError('Stale deployment code: '+n)
        required=['contract.json']
        if mode!='material':required+=['native_material_selection.json','native_detail_correction.json','native_source_correction.json']
        if mode=='qualification':
            required+=['qualification_execution_budget.json']
            if (campaign/'qualification_completion_authorization.json').exists():required+=['qualification_completion_authorization.json']
            if (campaign/'qualification_completion_resume_plan.json').exists():required+=['qualification_completion_resume_plan.json']
            if (campaign/'priority_recovery_authorization.json').exists():required+=['priority_recovery_authorization.json']
            if (campaign/'priority_capture_source_freeze.json').exists():required+=['priority_capture_source_freeze.json']
            frozen_path=campaign/'priority_capture_source_freeze.json'
            if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8')).get('scientific_capture_parameters_changed') is True:
                required+=['targeted_occlusion_pose_correction.json','qualification_recovery_cap_authorization_20261006.json']
        if mode=='capture':required+=['qualification_gate.json']
        if (campaign/'continuation48h_native_execution.json').is_file():
            required+=['continuation48h_native_execution.json','continuation48h_source_freeze.json',
                'tonight_core_authorization_20261007.json','tonight_core_protocol_20261007.json']
            policy=json.loads((campaign/'continuation48h_native_execution.json').read_text(encoding='utf-8'))
            for batch in policy.get('batches',{}).values():
                for name in (batch.get('source_freeze_path'),batch.get('operational_authorization',{}).get('path')):
                    if name:
                        if Path(name).name!=name:raise ValueError('Unsafe native continuation evidence path')
                        required.append(name)
            required=list(dict.fromkeys(required))
        for n in required:
            member='oct07_scene_repair/'+n
            if member not in names or hashlib.sha256(z.read(member)).hexdigest()!=sha(campaign/n):
                raise ValueError('Stale frozen deployment evidence: '+n)
        snapshot=json.loads(z.read('code_scene_repair/snapshot_manifest.json'))
        for n,h in snapshot['files'].items():
            if hashlib.sha256(z.read('code_scene_repair/'+n)).hexdigest()!=h:
                raise ValueError('Deployment source snapshot mismatch')
    return dict(package=source.name,sha256=manifest['sha256'],verified_members=len(rows),mode=mode)

def package(name):
    package_id=name
    base=ROOT/'01_uav_language_grounding/benchmark/carlaair_v1'
    files={'code_scene_repair/'+p.name:p for p in base.iterdir() if p.is_file() and p.suffix in ('.py','.json','.sh')}
    for n in DEPLOY_CODE:
        files['code_scene_repair/'+n]=ROOT/'scripts'/n
    for p in (OUT/'specifications').iterdir():files['oct07_scene_repair/specifications/'+p.name]=p
    files['oct07_scene_repair/contract.json']=OUT/'contract.json'
    if (OUT/'native_material_selection.json').exists():files['oct07_scene_repair/native_material_selection.json']=OUT/'native_material_selection.json'
    if (OUT/'material_budget.json').exists():files['oct07_scene_repair/material_budget.json']=OUT/'material_budget.json'
    if (OUT/'material_renewal_budget.json').exists():files['oct07_scene_repair/material_renewal_budget.json']=OUT/'material_renewal_budget.json'
    for asset_name in ('material_sourcefix_budget.json','native_source_correction.json','material_detail_budget.json','native_detail_correction.json','detail_interface_recovery.json','detail_launch_recovery.json','first_manuscript_protocol.json','first_manuscript_scene_ids.csv','tonight_core_authorization_20261007.json','tonight_native_execution_window.json'):
        if (OUT/asset_name).exists():files['oct07_scene_repair/'+asset_name]=OUT/asset_name
    for n in ('material_completion_budget.json','native_material_completion_protocol.json','qualification_execution_budget.json','qualification_completion_authorization.json','qualification_completion_resume_plan.json','priority_recovery_authorization.json','priority_capture_source_freeze.json','targeted_occlusion_pose_correction.json','qualification_recovery_cap_authorization_20261006.json'):
        if (OUT/n).exists():files['oct07_scene_repair/'+n]=OUT/n
    for name in ('tonight_core_protocol_20261007.json','continuation48h_native_execution.json','continuation48h_source_freeze.json'):
        if (OUT/name).is_file():files['oct07_scene_repair/'+name]=OUT/name
    if (OUT/'continuation48h_native_execution.json').is_file():
        policy=json.loads((OUT/'continuation48h_native_execution.json').read_text(encoding='utf-8'))
        for batch in policy.get('batches',{}).values():
            for name in (batch.get('source_freeze_path'),batch.get('operational_authorization',{}).get('path')):
                if name:
                    if Path(name).name!=name:raise ValueError('Unsafe native continuation evidence path')
                    files['oct07_scene_repair/'+name]=OUT/name
    if (OUT/'tonight_runtime').is_dir():
        for name in ('scene_ids.csv','contract.json','matrix.csv'):
            files['oct07_scene_repair/tonight_runtime/'+name]=OUT/'tonight_runtime'/name
        for p in (OUT/'tonight_runtime/specifications').glob('*.jsonl'):
            files['oct07_scene_repair/tonight_runtime/specifications/'+p.name]=p
    if (OUT/'qualification_gate.json').exists():files['oct07_scene_repair/qualification_gate.json']=OUT/'qualification_gate.json'
    manifest=[]
    for n,p in files.items():
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
        manifest.append(dict(path=n,bytes=p.stat().st_size,sha256=sha(p)))
    dest=OUT/(package_id+'.zip')
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('code_scene_repair/snapshot_manifest.json',json.dumps(dict(source=package_id,files={Path(n).name:sha(p) for n,p in files.items() if n.startswith('code_scene_repair/')})))
    write(OUT/(package_id+'_manifest.json'),dict(files=manifest,sha256=sha(dest),bytes=dest.stat().st_size),True)
    return dest

def tonight_native_authorization(root, assessment, uid):
    """Verify this session's explicit new window; never infer it from recharge."""
    binding = assessment.get('tonight_authorization')
    if not binding:
        return None
    root = Path(root).resolve()
    path = (root / binding['path']).resolve()
    if not path.is_relative_to(root) or sha(path) != binding.get('sha256'):
        raise ValueError('Tonight user authorization identity changed')
    approval = json.loads(path.read_text(encoding='utf-8-sig'))
    current = json.loads((ROOT / '.codex/task_priority.json').read_text(encoding='utf-8-sig'))
    registered = current.get('tonight_execution_authorization', {})
    from oct07_qualification_budget import continuation_batch
    batch=continuation_batch(root,assessment.get('attempt'))
    if (approval.get('approved') is not True or registered.get('approved') is not True
            or registered.get('receipt_sha256') != binding['sha256']
            or registered.get('scope_id') != approval.get('scope_id')
            or approval.get('owned_uuid') != uid
            or approval.get('parent60816_paused') is not True
            or approval.get('TEST_sealed') is not True
            or approval.get('prior_attempts_costs_and_elapsed_retained') is not True
            or not set(assessment.get('scene_ids', [])) <= set(batch['scene_ids'] if batch else approval['affected_scene_ids'])):
        raise ValueError('Current explicit tonight native window is not verified')
    from datetime import datetime
    cutoff = batch['absolute_deadline'] if batch else datetime.fromisoformat(approval['native_repair_stop']).timestamp()
    if (assessment.get('absolute_batch_deadline', float('inf')) > cutoff
            or assessment.get('maximum_powered_seconds', float('inf')) > (batch['maximum_powered_seconds'] if batch else approval['native_repair_maximum_new_powered_seconds'])):
        raise ValueError('Native request exceeds the approved tonight phase')
    return approval


def approved_capture_limits(root, assessment, uid):
    """Keep the original cap unless a hash-bound current user amendment exists."""
    tonight = tonight_native_authorization(root, assessment, uid)
    if tonight is not None:
        return tonight['new_native_cumulative_ceiling_seconds'], float('inf')
    binding = assessment.get('task_cap_amendment')
    if not binding:
        return 7200, 8
    if assessment.get('run_class') == 'native_runtime_diagnosis':
        raise ValueError('TRAIN24 cap amendment does not authorize a diagnostic grant')
    root = Path(root).resolve()
    path = (root / binding['path']).resolve()
    if not path.is_relative_to(root) or sha(path) != binding.get('sha256'):
        raise ValueError('User task cap amendment source is missing or changed')
    approval = json.loads(path.read_text(encoding='utf-8-sig'))
    current = json.loads((ROOT / '.codex/task_priority.json').read_text(encoding='utf-8-sig'))
    registered = current.get('task_budget_amendment', {})
    if (registered.get('approved') is not True
            or registered.get('receipt_sha256') != binding['sha256']
            or registered.get('authorization_id') != approval.get('authorization_id')
            or approval.get('approved') is not True
            or approval.get('owned_uuid') != uid
            or approval.get('logical_attempt') != 'scene_qualification1'
            or approval.get('original_budget_reset') is not False
            or approval.get('prior_cap_seconds') != 7200
            or approval.get('prior_actual_all_in_cny') != 8
            or approval.get('cumulative_cap_seconds') != 18000
            or approval.get('actual_all_in_cap_cny') != 18
            or approval.get('maximum_new_powered_seconds') != 10800
            or approval.get('planned_units') != 24
            or approval.get('parent60816_paused') is not True
            or approval.get('TEST_sealed') is not True):
        raise ValueError('Explicit current TRAIN24 task cap amendment is not verified')
    ids = assessment.get('scene_ids', [])
    if not ids or not set(ids) <= set(approval['affected_scene_ids']):
        raise ValueError('Task cap amendment does not cover the requested scene IDs')
    if (assessment.get('absolute_batch_deadline', float('inf')) > approval['absolute_fee_expiry']
            or assessment.get('maximum_powered_seconds', float('inf')) > approval['maximum_new_powered_seconds']
            or assessment.get('original_task_total_cost_upper_cny', float('inf')) > approval['actual_all_in_cap_cny']):
        raise ValueError('Amended task time/cost/deadline exceeds explicit user permission')
    # Both the revised cumulative ceiling and the NEW three-hour bound apply.
    limit = min(approval['cumulative_cap_seconds'],
                approval['prior_powered_upper_seconds'] + approval['maximum_new_powered_seconds'])
    return limit, approval['actual_all_in_cap_cny']


def remaining_powered_time(root,assessment,uid):
    """Deduct closed, actually started recovery guards from cumulative limits.

    Guard elapsed includes some preparation while off, so it is deliberately an
    upper bound. Read original receipts instead of trusting an assessment's old
    copied remaining-time field. An unfinished prior guard blocks a new start.
    """
    if not (assessment.get('cumulative_original_capture_powered_limit_seconds') or
            assessment.get('run_class')=='native_runtime_diagnosis'):
        return None
    total=0.0;diagnostic=0.0
    folders=sorted(set(root.glob('capture_scene_qualification1_priority*')) |
                   set(root.glob('recover_runtime_diag*')) |
                   set(root.glob('recover_native_entry_trace*')) |
                   set(root.glob('recover_preservation_priority_recovery*')))
    current_policy=root/'continuation48h_native_execution.json'
    batches=json.loads(current_policy.read_text(encoding='utf-8')).get('batches',{}) if current_policy.is_file() else {}
    folders=sorted(set(folders)|{root/('capture_'+name) for name in batches if (root/('capture_'+name)).is_dir()})
    new_engineering=0.;new_capture=0.
    for folder in folders:
        receipt=folder/'start_cli_receipt.json'
        if not receipt.exists():continue
        started=json.loads(receipt.read_text(encoding='utf-8-sig'))
        if started.get('exit_code')!=0:
            stopped=folder/'close_verified_stop.json'
            if not stopped.is_file():raise ValueError('Prior paid start unresolved; inspect actual state')
            settled=json.loads(stopped.read_text(encoding='utf-8'))
            readback=settled.get('provider_readback',{})
            owned=[r for r in readback.get('instances',[]) if r.get('uuid')==uid]
            if not readback.get('coverage_complete') or len(owned)!=1 or owned[0]['status']!='shutdown':
                raise ValueError('Failed prior start lacks independent stopped provider readback')
            # Retain its entire guard interval as a conservative powered upper;
            # a closed failed start never becomes zero-cost/zero-time history.
        context=json.loads((folder/'guard_context.json').read_text(encoding='utf-8-sig'))
        proof=json.loads((folder/'guard.json').read_text(encoding='utf-8-sig'))
        if context.get('owned_uuid')!=uid or context.get('nonce')!=proof.get('nonce'):
            raise ValueError('Prior paid guard ownership/identity mismatch')
        if not proof.get('terminal') or not proof.get('shutdown_confirmed'):
            raise ValueError('Prior paid recovery is still live or shutdown unknown')
        elapsed=proof.get('elapsed_seconds')
        if not isinstance(elapsed,(int,float)) or not 0<=elapsed<86400:
            raise ValueError('Prior paid elapsed time unknown')
        total+=elapsed
        batch=batches.get(folder.name.removeprefix('capture_'))
        if batch:
            if batch['mode']=='qualification':new_engineering+=elapsed
            else:new_capture+=elapsed
        if folder.name.startswith('recover_runtime_diag'):diagnostic+=elapsed
    permitted, _ = approved_capture_limits(root, assessment, uid)
    limit=min(permitted,assessment.get('cumulative_original_capture_powered_limit_seconds',7200))
    remaining=limit-total
    current=batches.get(assessment.get('attempt'))
    if current:
        approval=json.loads((root/'tonight_core_authorization_20261007.json').read_text(encoding='utf-8'))
        remaining=min(remaining,(approval['native_repair_maximum_new_powered_seconds']-new_engineering
            if current['mode']=='qualification' else approval['new_native_capture_powered_limit_seconds']-new_capture))
    if assessment.get('run_class')=='native_runtime_diagnosis':
        remaining=min(remaining,min(500,assessment.get('original_runtime_diagnostic_maximum_powered_seconds',500))-diagnostic)
    return int(remaining)


def engineering_phase_deadline(root,attempt=None):
    """One non-resetting wall window shared by the two existing executors."""
    from datetime import datetime
    root=Path(root);authority=root/'tonight_core_authorization_20261007.json'
    approval=json.loads(authority.read_text(encoding='utf-8'))
    if attempt is not None:
        from oct07_qualification_budget import continuation_batch,operational_window
        batch=continuation_batch(root,attempt)
        if batch is not None and batch['mode']=='qualification' and operational_window(root,batch) is not None:
            if batch['absolute_deadline']-time.time()<300:
                raise TimeoutError('Current bounded native operational window exhausted')
            return batch['absolute_deadline']
    path=root/'continuation48h_engineering_phase.json'
    value=dict(authorization_sha256=sha(authority),started_at=time.time())
    value['deadline']=min(value['started_at']+7200,
        datetime.fromisoformat(approval['native_repair_stop']).timestamp())
    try:write(path,value,True)
    except FileExistsError:value=json.loads(path.read_text(encoding='utf-8'))
    if value['authorization_sha256']!=sha(authority):raise ValueError('Original engineering phase authority changed; do not reset')
    if value['deadline']-time.time()<300:raise TimeoutError('Cumulative engineering wall window exhausted')
    return value['deadline']


def native_engineering_cost_admission(root,assessment):
    """Keep the shared physical engineering cap separate from closed leases."""
    from deploy_oct07_image_campaign import engineering_powered_commitment
    root=Path(root)
    current=json.loads((root/'continuation48h_native_execution.json').read_text(encoding='utf-8')).get('batches',{}).get(assessment.get('attempt'))
    if not current or current['mode']!='qualification':return None
    approval=json.loads((root/'tonight_core_authorization_20261007.json').read_text(encoding='utf-8'))
    binding=assessment['tonight_authorization']['sha256']
    compute=0.;native_compute=0.;count=0;native_count=0;closed=0.
    for prior in list(root.glob('model_continuation48h_*'))+list(root.glob('capture_scene_qualification1_priority_recovery*')):
        ready=prior/'ready_budget_assessment.json';lease=prior/'budget_lease.json'
        if not ready.is_file() or not lease.is_file():continue
        value=json.loads(ready.read_text(encoding='utf-8'))
        if value.get('scientific_stage_binding') or value.get('tonight_authorization',{}).get('sha256')!=binding:continue
        cost=engineering_powered_commitment(prior,assessment['owned_uuid'])*value['rate_cny_hour_upper']/3600
        compute+=cost;count+=1;closed+=value['all_in_ceiling_cny']
        if prior.name.startswith('capture_'):native_compute+=cost;native_count+=1
    next_compute=assessment['maximum_powered_seconds']*assessment['rate_cny_hour_upper']/3600
    phase_upper=compute+next_compute+3.52+.12*(count+1)
    native_upper=native_compute+next_compute+assessment['storage_ceiling_cny']+.12*(native_count+1)
    if phase_upper>approval['engineering_combined_all_in_cap_cny']+1e-8:
        raise ValueError('Original combined engineering20CNY cap exhausted; no reset')
    if native_upper>10+1e-8:raise ValueError('Original native engineering10CNY branch cap exhausted')
    return dict(phase_physical_upper_cny=phase_upper,native_branch_physical_upper_cny=native_upper,
                previous_closed_lease_ceiling_cny=closed,closed_ceilings_not_refunded=True)


def native_payload_directory(folder,assessment):
    """Keep raw outputs on the declared project-owned volume, metadata here."""
    target=assessment.get('local_preservation_root')
    if target is None:return Path(folder)
    expected=(Path.home()/'AIHCIR_effective'/OUT.name).resolve()
    if Path(target).resolve()!=expected:
        raise ValueError('Native preservation must use the exact project-owned effective-data root')
    return expected/Path(folder).name


def require_native_storage_reservation(folder,payload,export_limit):
    """Verify a real live ancestor reservation before any paid start."""
    import importlib.util,sqlite3,psutil
    bridge=ROOT.parent/'具身智能大论文架构/cross_work_bridge/scripts/storage_guard.py'
    spec=importlib.util.spec_from_file_location('aihcirstorage_guard',bridge)
    storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
    check=storage.check(payload)
    if not check['allowed']:raise ValueError('Current raw-data volume floor/live reservations are closed')
    metadata=storage.check(folder)
    if not metadata['allowed']:raise ValueError('Current metadata volume floor/live reservations are closed')
    if not storage.DATABASE.is_file():raise ValueError('Native write reservation is not registered')
    ancestors={p.pid for p in psutil.Process().parents()}
    conn=sqlite3.connect(storage.DATABASE.resolve().as_uri()+'?mode=ro',uri=True)
    try:rows=storage.active_rows(conn)
    finally:conn.close()
    # Archives are removed after independent decoded-file verification, so only
    # one bounded part, in addition to all retained source bytes, can be live.
    peak=export_limit+min(export_limit,256*1024**2)+8*1024**2
    owned=[r for r in rows if r[1]==check['id'] and Path(r[2]).resolve()==ROOT
           and r[3] in ancestors and r[5]>=peak]
    if not owned:raise ValueError('Actual ancestor storage_guard reservation does not cover native write peak')
    if metadata['id']!=check['id'] and not any(
            r[1]==metadata['id'] and Path(r[2]).resolve()==ROOT and r[3] in ancestors
            and r[5]>=8*1024**2 for r in rows):
        raise ValueError('Separate metadata volume requires its own live8MiB ancestor reservation')
    return dict(verified=True,observed_at=time.time(),reservation_id=owned[0][0],
                reservation_pid=owned[0][3],payload_directory=str(payload),declared_peak_bytes=peak)


def require_native_vulkan(inventory,minimal=None):
    query=inventory.get('commands',{}).get('owned_service_vulkan',{})
    lines=query.get('identity_lines',[])
    if query.get('exit_code')==0 and any(
            'NVIDIA' in s and ('deviceName' in s or 'GPU id' in s) for s in lines):
        return {'entry':'vulkaninfo','actual_NVIDIA_device':True}
    if minimal is not None:
        commands=minimal.get('commands',{})
        checked=commands.get('minimal_Vulkan10_instance',{})
        try:
            events=[json.loads(s) for s in checked.get('stdout','').splitlines()]
            stages={x['stage']:x for x in events}
            devices=stages['physical_devices']['devices']
            def gpu_uuid(source):
                row=source['commands']['gpu']['stdout'].strip().splitlines()
                if len(row)!=1:return None
                value=row[0].split(',')[1].strip()
                return value if value.startswith('GPU-') else None
            first=gpu_uuid(inventory);second=gpu_uuid(minimal)
            valid=(checked.get('exit_code')==0 and first is not None and first==second
                and 0<=time.time()-minimal['observed_at']<=120
                and 0<=time.time()-inventory['observed_at']<=120
                and stages['after_vkCreateInstance']['result']==0
                and stages['after_vkCreateInstance']['instance_created'] is True
                and stages['after_vkEnumeratePhysicalDevices_count']['result']==0
                and stages['after_vkEnumeratePhysicalDevices_count']['count']==len(devices)>0
                and stages['after_vkEnumeratePhysicalDevices_list']['result']==0
                and 'instance_destroyed' in stages
                and any(x['vendor_id']==0x10de and x['device_type'] in (1,2)
                        and 'NVIDIA' in x['device_name'] for x in devices))
        except (ValueError,KeyError,IndexError,TypeError):
            valid=False
        if valid:
            return {'entry':'same_boot_native_Vulkan10_device_inventory','actual_NVIDIA_device':True,
                    'gpu_uuid':first,'legacy_vulkaninfo_exit':query.get('exit_code'),
                    'rendering_or_scene_admission_proven':False}
    raise RuntimeError('Owned-service NVIDIA Vulkan enumeration failed; do not launch UE4')


def diagnose_observation_channel(remote,folder,identity,query,protection):
    """One diagnosed control-channel repair; never starts/replaces a native job."""
    from oct07_ssh_once import connect_once
    proof_path=folder/'observation_channel_repair.json'
    if proof_path.exists():raise RuntimeError('Observation channel repair already used; no blind reconnect loop')
    failure=json.loads(sorted(folder.glob('remote_command_failure_*.json'))[-1].read_text(encoding='utf-8'))
    if (failure.get('phase')!='channel_open' or failure.get('ssh_error_code')!='Timeout opening channel.'
            or failure.get('returncode') is not None or failure.get('output')
            or failure.get('ssh_transport_active') is not True):
        raise RuntimeError('Observation failure has no diagnosed pre-command channel boundary')
    protection()
    prior=json.loads((folder/'latest_observation.json').read_text(encoding='utf-8'))
    if (time.time()-prior['observed_at']>90 or prior.get('live') is not True or prior.get('identity_matches') is not True
            or prior.get('pid')!=identity['pid'] or prior.get('start_ticks')!=identity['start_ticks']):
        raise RuntimeError('Prior native observation is not recent and identity verified')
    context=json.loads((folder/'guard_context.json').read_text(encoding='utf-8'))
    if context['effective_deadline']-time.time()<360:raise TimeoutError('No protected time for channel diagnosis and preservation')
    probe=subprocess.run(['curl.exe' if os.name=='nt' else 'curl','--silent','--show-error',
        '--connect-timeout','5','--max-time','10','--output','NUL' if os.name=='nt' else '/dev/null',
        '--write-out','%{http_code}','https://www.gstatic.com/generate_204'],
        capture_output=True,text=True,timeout=12)
    record=dict(observed_at=time.time(),failure=failure,public_probe_exit=probe.returncode,
        public_probe_HTTP=probe.stdout.strip(),reconnect_attempts=0,native_starts=0,
        source_job=identity,root_cause_confirmed=False,
        diagnosis='Existing transport active but pre-command channel stalled; channel is isolated from original healthy native process. Source confirms each prior channel was closed. Public path is checked before one bounded new authenticated connection.')
    write(proof_path,record,True)
    print(json.dumps(dict(stage='observation_channel_failed_diagnosis',observed_at=time.time(),native_starts=0)),flush=True)
    if probe.returncode or probe.stdout.strip()!='204':
        raise RuntimeError('Actual public network probe failed; no reconnect or cloud retry')
    protection()
    record['reconnect_attempts']=1;write(proof_path,record)
    connect_once(remote)
    observed=json.loads(remote.command(shlex.join(query),15))
    if (observed.get('identity_matches') is not True or observed.get('pid')!=identity['pid']
            or observed.get('start_ticks') not in (identity['start_ticks'],None)):
        raise RuntimeError('New connection does not verify original controller identity')
    for group in observed.get('groups',[]):
        for label in ('child','simulator'):
            if group.get(label,{}).get('identity_matches') is False:
                raise RuntimeError('New connection native group identity differs')
    record.update(verified=True,observed_at=time.time(),native_observation=observed,
        same_native_job_retained=True,new_engine_or_model_starts=0)
    write(proof_path,record)
    return observed


def native_component_accounting(root):
    """Read terminal guard evidence; copied proposals do not reset elapsed time."""
    budget=json.loads((Path(root)/'qualification_execution_budget.json').read_text(encoding='utf-8-sig'))
    used=0.
    continuation=Path(root)/'continuation48h_native_execution.json'
    new_ids=set(json.loads(continuation.read_text(encoding='utf-8'))['batches']) if continuation.is_file() else set()
    for prior in budget.get('native_component_execution_ids',[]):
        if prior in new_ids:continue # New explicitly authorized phase; retain old1200s history.
        path=Path(root)/('capture_'+prior)/'guard.json'
        if not path.exists():continue
        state=json.loads(path.read_text(encoding='utf-8-sig'))
        if not state.get('terminal') or not state.get('shutdown_confirmed'):
            raise ValueError('Prior native engineering guard unresolved; inspect actual state')
        elapsed=state.get('elapsed_seconds')
        if not isinstance(elapsed,(float,int)) or isinstance(elapsed,bool) or not 0<=elapsed<86400:
            raise ValueError('Prior native engineering elapsed time unknown')
        used+=elapsed
    return dict(used_upper_seconds=used,remaining_seconds=1200-used,phase_cap_seconds=1200)


def native_inventory_admission(root, assessment):
    """Reject reserve/startup, elapsed accounting and expiry before a paid guard."""
    budget=json.loads((Path(root)/'qualification_execution_budget.json').read_text(encoding='utf-8-sig'))
    if assessment.get('attempt') not in budget.get('native_visual_inventory_execution_ids',[]):
        return None
    from run_oct07_scene_node import native_inventory_schedule
    schedule=native_inventory_schedule(assessment['maximum_powered_seconds'])
    minimum=schedule['minimum_window_seconds']
    from oct07_qualification_budget import continuation_batch
    current=continuation_batch(root,assessment['attempt'])
    if current:
        if current.get('native_visual_inventory_only') is not True:
            raise ValueError('Actual inventory mode does not match current allocation')
        if assessment['maximum_powered_seconds']<minimum or current['absolute_deadline']-time.time()<minimum:
            raise ValueError('Current native inventory cannot fit startup/preservation')
        return dict(**schedule,phase='explicit_new_engineering',original1200_accounting_retained=True)
    cap=assessment['component_phase_used_upper_seconds']+assessment['component_phase_remaining_seconds']
    if abs(cap-1200)>1e-6:
        raise ValueError('Original1200s engineering allowance differs; no reset')
    accounting=native_component_accounting(root)
    used,remaining=accounting['used_upper_seconds'],accounting['remaining_seconds']
    if assessment['maximum_powered_seconds']>remaining or remaining<minimum:
        raise ValueError(f'Native inventory reserve/startup conflict before payment: needs{minimum}s window; requested{assessment["maximum_powered_seconds"]}s, original engineering remaining{remaining:.3f}s')
    if abs(assessment['component_phase_used_upper_seconds']-used)>1e-6:
        raise ValueError('Assessment native engineering accounting is stale; no elapsed-time reset')
    now=time.time()
    cutoff=assessment.get('absolute_batch_deadline',assessment.get('deadline'))
    if not isinstance(cutoff,(int,float)) or isinstance(cutoff,bool) or cutoff-now<assessment['maximum_powered_seconds']:
        raise ValueError('Native inventory absolute authorization expired or cannot fit requested window before payment')
    auth_path=Path(root)/'priority_recovery_authorization.json'
    if auth_path.exists():
        batch=json.loads(auth_path.read_text(encoding='utf-8-sig'))['batches'][assessment['attempt']]
        if batch.get('paid_execution_authorized') is False and tonight_native_authorization(root, assessment, 'OWNED_INSTANCE_UUID') is None:
            raise ValueError('Native inventory is authorized for local preparation only')
    return dict(used_upper_seconds=used,remaining_seconds=remaining,**schedule)


def run(mode,attempt,uid='OWNED_INSTANCE_UUID',package_id=None,scope='parent',assessment_file=None,max_seconds=None):
    if assessment_file is None:raise ValueError('Paid native execution requires a shared budget assessment')
    requested=json.loads(Path(assessment_file).read_text(encoding='utf-8-sig'))
    if requested.get('ready') is not True:
        raise ValueError('Current batch is not ready; paid guard/start prohibited')
    native_inventory_admission(OUT,requested)
    tonight = tonight_native_authorization(OUT, requested, uid)
    if scope not in ('parent','first','tonight'):raise ValueError('Unknown frozen reporting scope')
    if scope in ('first','tonight'):
        from oct07_capture_scope import selected_ids
        selected_ids(OUT,scope)
    source=OUT/((package_id or attempt)+'.zip')
    if not source.is_file():raise FileNotFoundError('Concrete verified package is required before cloud mutation')
    package_audit=verify_package(source,mode)
    from oct07_qualification_budget import continuation_batch,verify_source_freeze
    new_batch=continuation_batch(OUT,attempt)
    if new_batch:
        with zipfile.ZipFile(source) as frozen_package:
            verify_source_freeze(OUT,attempt,ROOT/'scripts',lambda name:frozen_package.read('code_scene_repair/'+name))
        native_engineering_cost_admission(OUT,requested)
    if mode!='material':
        from admit_oct07_material import verify
        verify(OUT)
    if mode=='capture':
        q=json.loads((OUT/'qualification_gate.json').read_text())
        if q.get('passed') is not True or q.get('runtime_admitted')!=24:raise ValueError('Native TRAIN24 qualification must precede full capture')
    if mode=='qualification':
        from oct07_qualification_budget import deadline as qualification_deadline
        qualification_cutoff=qualification_deadline(OUT,attempt)
        if tonight is not None and new_batch is None:
            from datetime import datetime
            qualification_cutoff=datetime.fromisoformat(tonight['native_repair_stop']).timestamp()
        recovery=json.loads((OUT/'qualification_execution_budget.json').read_text())
        priority=recovery.get('priority_recovery_allocation',{})
        from oct07_qualification_budget import continuation_batch,verify_source_freeze
        new_batch=continuation_batch(OUT,attempt)
        if new_batch:
            with zipfile.ZipFile(source) as frozen_package:
                verify_source_freeze(OUT,attempt,ROOT/'scripts',lambda name:frozen_package.read('code_scene_repair/'+name))
        elif attempt in priority.get('execution_ids',[]):
            approved=json.loads((ROOT/'.codex/task_priority.json').read_text(encoding='utf-8'))['resume_authorization']
            frozen=json.loads((OUT/'priority_recovery_authorization.json').read_text(encoding='utf-8'))
            if any(approved.get(k)!=frozen.get(k) for k in ('approved','capture_stop_lifted','expired_capture_window_replaced','compute_deadline')):
                raise ValueError('Current user recovery authorization differs from package')
            from oct07_qualification_budget import verify_source_freeze
            with zipfile.ZipFile(source) as frozen_package:
                verify_source_freeze(OUT,attempt,ROOT/'scripts',
                    lambda name:frozen_package.read('code_scene_repair/'+name))
            original_recovery_lease(OUT,frozen['batches'][attempt]['source_attempt'],uid)
        elif attempt!='scene_qualification1':
            if recovery.get('zero_frame_deployment_failure_verified') is not True:
                raise ValueError('Qualification requires a diagnosed zero-frame infrastructure continuation')
            extension=recovery.get('explicit_completion_allocation',{})
            previous=(extension.get('repair_source_attempts',{}).get(attempt)
                if attempt in extension.get('diagnosed_repair_execution_ids',[]) else
                'scene_qualification1_actorfix1' if extension.get('execution_id')==attempt else 'scene_qualification1')
            if previous is None:raise ValueError('Diagnosed repair has no exact terminal source lease')
            original_recovery_lease(OUT,previous,uid)
    folder=OUT/('capture_'+attempt);folder.mkdir(exist_ok=False)
    contract=json.loads((OUT/'contract.json').read_text());nonce=uuid.uuid4().hex
    compute_cutoff=__import__('datetime').datetime.fromisoformat(tonight['new_compute_stop'] if tonight is not None else contract['compute_stop']).timestamp()
    maximum=min({'material':2400,'qualification':11400,'capture':41400}[mode],int(compute_cutoff-time.time()))
    if mode=='qualification':maximum=min(maximum,int(qualification_cutoff-time.time()))
    loader_plan=requested.get('native_loader_repair')
    if loader_plan:
        loader_manifest=json.loads(Path(loader_plan['manifest']).read_text(encoding='utf-8'))
        if sha(Path(loader_manifest['bundle']))!=loader_manifest['sha256']:
            raise ValueError('Pinned loader repair bundle differs before payment')
        if loader_plan.get('repair_helper_sha256')!=sha(ROOT/'scripts/oct07_isolated_vulkan_loader_repair.py'):
            raise ValueError('Private loader helper source differs before payment')
        if loader_plan.get('entry_adapter_sha256')!=sha(ROOT/'scripts/run_oct07_native_controller_entry.py'):
            raise ValueError('Exact native account-switch adapter differs before payment')
    export_limit=int(requested.get('maximum_effective_source_bytes',1024**3 if mode=='material' else 6*1024**3))
    if not 0<export_limit<=6*1024**3:raise ValueError('Invalid bounded native export/write allowance')
    payload=native_payload_directory(folder,requested)
    storage_proof=require_native_storage_reservation(folder,payload,export_limit)
    payload.mkdir(parents=True,exist_ok=True)
    write(folder/'verified_local_storage_reservation.json',storage_proof,True)
    maximum=min(maximum,int(requested['maximum_powered_seconds']))
    if requested.get('absolute_batch_deadline') is not None:
        maximum=min(maximum,int(requested['absolute_batch_deadline']-time.time()))
    if tonight and mode=='qualification' and tonight.get('authorization_id')=='USER-AIHCIR-48H-CONTINUATION-20261007':
        maximum=min(maximum,int(engineering_phase_deadline(OUT,attempt)-time.time()))
    carried=remaining_powered_time(OUT,requested,uid)
    if carried is not None:maximum=min(maximum,carried)
    if max_seconds is not None:maximum=min(maximum,max_seconds)
    if mode=='material' and (OUT/'material_budget.json').exists():
        from oct07_material_budget import deadline
        maximum=min(maximum,int(deadline(OUT,attempt)-time.time()))
    if maximum<300:raise TimeoutError('Fixed cutoff reached')
    write(folder/'package_preflight.json',package_audit,True)
    api=Provider(uid)
    if api.status()!='shutdown':raise RuntimeError('Owned instance is occupied; do not duplicate or stop its task')
    effective_deadline=time.time()+maximum
    if requested.get('absolute_batch_deadline') is not None:
        effective_deadline=min(effective_deadline,requested['absolute_batch_deadline'])
    maximum=int(effective_deadline-time.time())
    if maximum<300:raise TimeoutError('Protection arming reached the fixed cutoff')
    write(folder/'guard_context.json',dict(nonce=nonce,owned_uuid=uid,created_already_powered=False,max_powered_seconds=maximum,
        effective_deadline=effective_deadline,external_budget_start=True,
        shutdown_reserve_seconds=120 if attempt=='scene_material4_detailfix1_config1' else 180),True)
    def owner(desired):write(folder/'owner.json',dict(nonce=nonce,pid=os.getpid(),desired=desired,heartbeat=time.time()))
    owner('run');guardlog=(folder/'guard.log').open('xb')
    guard=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/deploy_oct07_image_campaign.py'),'guard','--guard-folder',str(folder)],stdout=guardlog,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    stop=threading.Event()
    def heart():
        while not stop.wait(2):owner('run')
    thread=threading.Thread(target=heart,daemon=True);thread.start()
    state=dict(uuid=uid,mode=mode,attempt=attempt,stage='starting_guard',started_at=time.time(),actual_invoice_cny=None,scientific_results=0)
    def save(**kw):state.update(kw,observed_at=time.time());write(folder/'status.json',state);print(json.dumps({k:state.get(k) for k in ('stage','observed_at','remote_stage','completed_scenes')}),flush=True)
    def beat():
        g=json.loads((folder/'guard.json').read_text())
        if g['nonce']!=nonce or g['pid']!=guard.pid or g['terminal'] or time.time()-g['heartbeat']>30:raise RuntimeError('Exact shutdown protection failed; inspect first failure')
    remote=RemoteJob(api,folder,OUT.name,'/root/miniconda3/bin/python','',True);remote.heartbeat=beat;identity=None;save()
    try:
        until=time.monotonic()+180
        while time.monotonic()<until:
            if (folder/'guard.json').exists():
                beat()
                if json.loads((folder/'guard.json').read_text())['phase']=='armed':break
            time.sleep(2)
        else:raise TimeoutError('Owned startup exceeded180s')
        budgeted_start(folder,guard,uid,assessment_file)
        until=time.monotonic()+180
        while time.monotonic()<until:
            beat()
            if api.status()=='running':break
            time.sleep(2)
        else:raise TimeoutError('Shared lease startup exceeded180s')
        from oct07_ssh_once import connect_once
        connect_once(remote)
        remote.client.get_transport().set_keepalive(15)
        inspect="import json,os,shutil;from pathlib import Path;jobs=[]\nfor p in Path('/proc').iterdir():\n if not p.name.isdigit() or int(p.name)==os.getpid():continue\n try:a=(p/'cmdline').read_bytes().split(b'\\0')\n except OSError:continue\n if any(Path(x.decode(errors='replace')).name in ('run_oct07_capture_node.py','run_oct07_scene_node.py','run_oct07_native_controller_entry.py','CarlaUE4-Linux-Shipping','run_oct07_image_campaign.py') for x in a if x):jobs.append(int(p.name))\nprint(json.dumps(dict(jobs=jobs,free_bytes=shutil.disk_usage('/root/autodl-tmp').free)))"
        inventory=json.loads(remote.command('python3 -c '+shlex.quote(inspect),15));write(folder/'native_preflight.json',inventory,True)
        peak=2*export_limit
        if inventory['jobs'] or inventory['free_bytes']<5*1024**3+peak:raise RuntimeError('Native jobs or peak storage prevent deployment')
        remote.upload(source,REMOTE+'/'+source.name,45)
        extraction="import hashlib,zipfile;from pathlib import Path;p=Path("+repr(REMOTE)+");f=p/"+repr(source.name)+";assert hashlib.sha256(f.read_bytes()).hexdigest()=="+repr(sha(source))+";z=zipfile.ZipFile(f);assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(p);print('verified')"
        remote.command('python3 -c '+shlex.quote(extraction),25)
        controller_path_prefix=None
        if loader_plan:
            for local,name in [(Path(loader_manifest['bundle']),'private_loader_build.zip'),
                    (Path(loader_plan['manifest']),'private_loader_build_manifest.json'),
                    (ROOT/'scripts/oct07_isolated_vulkan_loader_repair.py','private_loader_repair.py')]:
                remote.upload(local,REMOTE+'/'+name,90)
            args=['python3',REMOTE+'/private_loader_repair.py','--root',REMOTE,
                '--bundle',REMOTE+'/private_loader_build.zip','--manifest',REMOTE+'/private_loader_build_manifest.json']
            verify_helper="import hashlib,runpy,sys;from pathlib import Path;p=Path("+repr(args[1])+");assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(loader_plan['repair_helper_sha256'])+";sys.argv="+repr(args[1:])+";runpy.run_path(str(p),run_name='__main__')"
            actual=json.loads(remote.command('python3 -c '+shlex.quote(verify_helper),420))
            write(folder/'actual_private_loader_repair.json',actual,True)
            if actual.get('status')!='private_standard_loader_actual_device_admitted':
                raise RuntimeError('Private loader repair not admitted; retain exact actual dependency/exit evidence')
            controller_path_prefix=actual['controller_path_prefix']
        if attempt.startswith('scene_qualification1_priority'):
            # Provider aliases do not establish the physical GPU/driver assigned
            # on this boot. Preserve the same-boot inventory before UE4 starts.
            actual_runtime=diagnose_native_runtime(remote,folder,filename='native_runtime_before_launch.json')
            try:
                admission=require_native_vulkan(actual_runtime)
            except RuntimeError:
                minimum=diagnose_vulkan_instance(remote,folder,include_backtrace=False)
                admission=require_native_vulkan(actual_runtime,minimum)
            write(folder/'native_vulkan_admission.json',dict(admission,observed_at=time.time()),True)
        lease_deadline=json.loads((folder/'guard_context.json').read_text())['effective_deadline']
        launch="import subprocess,json,os;from pathlib import Path;p=Path("+repr(REMOTE)+");f=(p/'oct07_scene_repair'/"+repr(attempt+'_controller.log')+").open('xb');c=subprocess.Popen(['/root/miniconda3/bin/python','-B','-u',str(p/'code_scene_repair/run_oct07_scene_node.py'),'--root',str(p),'--mode',"+repr(mode)+",'--attempt',"+repr(attempt)+",'--scope',"+repr(scope)+",'--lease-deadline',"+repr(str(lease_deadline))+"],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True);print(json.dumps(dict(pid=c.pid,start_ticks=Path('/proc/'+str(c.pid)+'/stat').read_text().split()[21])))"
        if controller_path_prefix:
            wrapper=ROOT/'scripts/run_oct07_native_controller_entry.py'
            wrapper_path=REMOTE+'/run_oct07_native_controller_entry.py'
            remote.upload(wrapper,wrapper_path,30)
            command=['/root/miniconda3/bin/python','-B','-u',wrapper_path,
                     '--private-loader',actual['library'],'--loader-sha256',actual['library_sha256'],
                     '--node-sha256',sha(ROOT/'scripts/run_oct07_scene_node.py'),
                     '--root',REMOTE,'--mode',mode,'--attempt',attempt,'--scope',scope,
                     '--lease-deadline',str(lease_deadline)]
            launch="import subprocess,json,os,hashlib;from pathlib import Path;p=Path("+repr(REMOTE)+");assert hashlib.sha256(Path("+repr(wrapper_path)+").read_bytes()).hexdigest()=="+repr(sha(wrapper))+";f=(p/'oct07_scene_repair'/"+repr(attempt+'_controller.log')+").open('xb');c=subprocess.Popen("+repr(command)+",stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True);print(json.dumps(dict(pid=c.pid,start_ticks=Path('/proc/'+str(c.pid)+'/stat').read_text().split()[21])))"
        identity=json.loads(remote.command('python3 -c '+shlex.quote(launch),15));write(folder/'job_identity.json',identity,True);save(stage='native_job_submitted',**identity)
        inventory_only=attempt in json.loads((OUT/'qualification_execution_budget.json').read_text(encoding='utf-8-sig')).get('native_visual_inventory_execution_ids',[])
        observation_interval=2 if inventory_only else 30
        while True:
            beat()
            query=['python3',REMOTE+'/code_scene_repair/oct07_scene_observation.py','--root',REMOTE,'--attempt',attempt,'--pid',str(identity['pid']),'--ticks',str(identity['start_ticks'])]
            try:
                observed=json.loads(remote.command(shlex.join(query),15))
            except Exception as error:
                import paramiko
                if not isinstance(error,paramiko.SSHException) or str(error)!='Timeout opening channel.':raise
                observed=diagnose_observation_channel(remote,folder,identity,query,beat)
            write(folder/'latest_observation.json',dict(observed,observed_at=time.time()))
            status=observed.get('status') or {};save(stage='observing_native_job',remote_stage=status.get('stage'),completed_scenes=status.get('completed_scenes'),next_check_at=time.time()+observation_interval)
            if observed['live'] and observed['start_ticks']!=identity['start_ticks']:raise RuntimeError('Native process identity changed')
            for group in observed.get('groups',[]):
                for label in ('child','simulator'):
                    if group.get(label,{}).get('identity_matches') is False:raise RuntimeError('Native group process identity changed')
            if not observed['live'] or status.get('ended_at'):break
            time.sleep(observation_interval)
        # Preserve only this attempt's effective output, not old runtime trees.
        receipt=json.loads(remote.command('python3 '+shlex.quote(REMOTE+'/code_scene_repair/oct07_scene_preservation.py')+' --root '+shlex.quote(REMOTE)+' --attempt '+shlex.quote(attempt)+' --max-bytes '+str(export_limit),120))
        total=0
        verified_files=0;decoded_bytes=0
        from oct07_scene_preservation import extract
        for index,part in enumerate(receipt['parts']):
            target=payload/('effective_outputs_'+str(index).zfill(3)+'.zip')
            if Path(part['path']).is_absolute() or '..' in Path(part['path']).parts:raise ValueError('Unsafe remote sensor export path')
            with remote.client.open_sftp() as sftp:
                sftp.get_channel().settimeout(30);sftp.get(REMOTE+'/'+part['path'],str(target))
            if sha(target)!=part['sha256']:raise ValueError('Native output transfer hash changed')
            total+=part['bytes']
            if total>export_limit:raise ValueError('Effective sensor transfer exceeds local live reservation')
            verified_files+=extract(target,payload/'effective')['verified_files']
            # A second read checks decoded outputs independently of transport.
            with zipfile.ZipFile(target) as z:
                for row in json.loads(z.read('preservation_manifest.json'))['files']:
                    decoded=payload/'effective'/row['path']
                    if sha(decoded)!=row['sha256'] or decoded.stat().st_size!=row['bytes']:
                        raise ValueError('Independent decoded native asset check failed')
                    decoded_bytes+=row['bytes']
            target.unlink()
            beat()
        write(folder/'effective_independent_audit.json',dict(observed_at=time.time(),passed=True,failures=[],
            independent_decoded_files_checked=verified_files,bytes_checked=decoded_bytes,
            source_attempt=attempt,effective_directory=str(payload/'effective'),scientific_admission=False),True)
        receipt['local_effective_directory']=str(payload/'effective')
        write(folder/'preservation_receipt.json',receipt,True);save(stage='native_outputs_preserved',native_terminal=status.get('stage'))
        component_budget_path=OUT/'qualification_execution_budget.json'
        component_budget=(json.loads(component_budget_path.read_text(encoding='utf-8'))
                          if mode=='qualification' and component_budget_path.is_file() else {})
        if attempt in component_budget.get('native_component_execution_ids',[]):
            save(component_review_passed=False)
            records=list((payload/'effective'/'sessions').glob('*/component_metrology.json'))
            if len(records)!=1:raise RuntimeError('Preserved component metrology record is missing or duplicated')
            if attempt in component_budget.get('native_visual_inventory_execution_ids',[]):
                from oct07_native_visual_inventory import review_inventory
                review=review_inventory(records[0])
            else:
                from replay_oct07_native_geometry import review_component
                review=review_component(records[0])
            write(folder/'component_independent_review.json',review,True)
            save(component_review_passed=review['passed'])
            if not review['passed']:
                raise RuntimeError('Independent native component review rejected the preserved measurement')
        if status.get('stage')=='failed':
            if attempt.startswith('scene_qualification1_priority'):
                diagnose_native_runtime(remote,folder,filename='native_runtime_after_failure.json')
            raise RuntimeError('Native batch failed; exact failure outputs preserved before shutdown')
    except BaseException as exc:
        import traceback
        failures=sorted(folder.glob('remote_command_failure_*.json'))
        detail=json.loads(failures[-1].read_text()) if failures else {}
        access_failure=bool(detail.get('ssh_error') or
            ('ssh_transport_active' in detail and not detail['ssh_transport_active']))
        runtime_rejection=bool(not access_failure and (state.get('native_terminal')=='failed' or state.get('component_review_passed') is False or
            (detail.get('ssh_transport_active') and detail.get('returncode') is not None)))
        save(stage='failed',error_type=type(exc).__name__,traceback=traceback.format_exc(),cloud_checks_stopped=not runtime_rejection,
            failure_class='native_command_rejection' if runtime_rejection else 'access_or_unclassified_failure',
            next_action='Inspect and preserve this failure before any diagnosed bounded continuation; never reconnect or replay blindly')
        raise
    finally:
        stop.set();thread.join(3);owner('stop');until=time.monotonic()+130
        while guard.poll() is None and time.monotonic()<until:time.sleep(1)
        g=json.loads((folder/'guard.json').read_text()) if (folder/'guard.json').exists() else {}
        save(shutdown_confirmed=g.get('shutdown_confirmed',False),finished_at=time.time(),next_check_at=None)
        remote.close();guardlog.close()
        if g.get('shutdown_confirmed') and (folder/'budget_lease.json').exists():
            reserved=json.loads((folder/'budget_lease.json').read_text())
            close_verified_lease(folder, reserved['lease_id'], uid)

def original_recovery_lease(root,source_attempt,uid):
    """Resolve a completed capture/inline repair without changing either record."""
    root=Path(root)
    if not source_attempt.replace('_','').isalnum():raise ValueError('Unsafe source attempt')
    candidates=[root/(prefix+source_attempt) for prefix in ('capture_','repair_')
        if (root/(prefix+source_attempt)/'job_identity.json').is_file()]
    if len(candidates)!=1:raise ValueError('One unambiguous original scientific lease is required')
    original=candidates[0]
    identity=json.loads((original/'job_identity.json').read_text(encoding='utf-8'))
    from oct07_active_protection import protection_folder
    protection=protection_folder(original)
    prior=json.loads((protection/'guard.json').read_text(encoding='utf-8'))
    if not prior.get('terminal') or not prior.get('shutdown_confirmed'):
        raise ValueError('Original protection is not terminal')
    context=json.loads((protection/'guard_context.json').read_text(encoding='utf-8'))
    if (not prior.get('terminal') or not prior.get('shutdown_confirmed')
        or prior.get('nonce')!=context.get('nonce') or context.get('owned_uuid')!=uid):
        raise ValueError('Original protection/ownership is not verified terminal')
    if not identity.get('pid') or not identity.get('start_ticks'):
        raise ValueError('Original scientific process identity is incomplete')
    return original,identity,prior


def shared_cli(folder,name,*arguments):
    """Fresh shared CLI process; retain each response and never resend a mutation."""
    command=bridge_root()/'scripts/cloud.py'
    result=subprocess.run([sys.executable,'-B',str(command),*map(str,arguments)],
        capture_output=True,text=True,encoding='utf-8',timeout=30)
    write(folder/(name+'_cli_receipt.json'),dict(observed_at=time.time(),exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr),True)
    if result.returncode:raise RuntimeError('Shared CLI rejected '+name+'; see retained response')
    return json.loads(result.stdout)


def close_verified_lease(folder, lease_id, uid, receipt_name='close'):
    """Use the current shared admission's complete official stop readback."""
    observed=shared_cli(folder,receipt_name+'_official_stop','autodl','list')
    data=observed.get('data',{})
    rows=data.get('list',[])
    total=data.get('result_total', data.get('total'))
    if (observed.get('code')!='Success' or not isinstance(total,int)
            or len(rows)!=total or len({r.get('uuid') for r in rows})!=total):
        raise ValueError('Stop list coverage incomplete; retain active reservation')
    match=next((r for r in rows if r.get('uuid')==uid),None)
    if not match or match.get('status') not in ('shutdown','shutdown_by_starting_error'):
        raise ValueError('Exact owned instance not officially stopped; retain reservation')
    assessment=json.loads((folder/'ready_budget_assessment.json').read_text(encoding='utf-8-sig'))
    proof=dict(task_id=assessment['task_id'],instance_uuid=uid,lease_id=lease_id,
               provider_readback=dict(observed_at=time.time(),source='/api/v1/dev/instance/pro/list',
                 coverage_complete=True,provider_total=total,
                 instances=[dict(uuid=r['uuid'],status=r['status']) for r in rows]),
               actual_guard_evidence=str(folder/'guard.json'),
               actual_guard_sha256=sha(folder/'guard.json'))
    path=folder/(receipt_name+'_verified_stop.json')
    write(path,proof,True)
    return shared_cli(folder,receipt_name,'budget','close','--lease',lease_id,
                      '--instance',uid,'--evidence',path)


def prepare_budgeted_start(folder,guard,uid,assessment_file,receipt_prefix=''):
    """Quote and submit this protected off-node batch without reserving or starting."""
    import hashlib,math,psutil
    from deploy_oct07_image_campaign import read_guard_json
    assessment=json.loads(Path(assessment_file).read_text(encoding='utf-8-sig'))
    preservation=assessment.get('run_class','maintenance')=='maintenance'
    if not preservation and assessment.get('ready') is not True:
        raise ValueError('Only a dependency-ready scientific batch can reserve paid execution')
    if assessment['workspace']!='AIHCIR' or Path(assessment['workspace_root']).resolve()!=ROOT:
        raise ValueError('Budget assessment ownership differs')
    context=read_guard_json(folder/'guard_context.json')
    proof=read_guard_json(folder/'guard.json')
    process=psutil.Process(guard.pid)
    if (proof['pid']!=guard.pid or proof['nonce']!=context['nonce'] or proof['terminal']
        or proof['phase']!='armed' or time.time()-proof['heartbeat']>10
        or '--guard-folder' not in process.cmdline() or str(folder) not in process.cmdline()):
        raise ValueError('Actual independent protection not armed for this exact lease')
    write(folder/(receipt_prefix+'verified_independent_guard.json'),dict(verified=True,guard_deadline=context['effective_deadline'],
        observed_at=time.time(),pid=guard.pid,birth=process.create_time(),nonce=context['nonce'],
        evidence=[str(folder/'guard.json'),str(folder/'guard_context.json')]),True)
    quote_path=folder/(receipt_prefix+'storage_quote.json')
    storage=shared_cli(folder,receipt_prefix+'storage_quote','autodl','storage-quote','--instance',uid,'--quote-output',quote_path)
    snapshot=shared_cli(folder,receipt_prefix+'compute_quote','autodl','snapshot','--instance',uid)['data']
    rate=snapshot['payg_price']/1000
    hours=(context['effective_deadline']-time.time())/3600
    if hours<=0:raise TimeoutError('Protection deadline already reached')
    storage_ceiling=storage['storage_cny_day_upper']*(math.ceil(hours/24)+1)
    ceiling=math.ceil((rate*context['max_powered_seconds']/3600+storage_ceiling+.12)*100)/100
    if ceiling>assessment.get('requested_conditional_allocation_cny',assessment['all_in_ceiling_cny']):
        raise ValueError('Fresh complete cost exceeds the assessed bounded batch ceiling')
    body=dict(instance_uuid=uid,payload='gpu',start_command='sleep 1')
    write(folder/(receipt_prefix+'start_request.json'),body,True)
    assessment.update(ready=True,observed_at=time.time(),deadline=context['effective_deadline'],
        stop_protection_ready=True,execution_ready=False,
        quote_observed_at=time.time(),quote_evidence=str(folder/(receipt_prefix+'compute_quote_cli_receipt.json')),
        stop_evidence=str(folder/(receipt_prefix+'verified_independent_guard.json')),
        rate_cny_hour_upper=rate,storage_ceiling_cny=storage_ceiling,all_in_ceiling_cny=ceiling,
        storage_accounting='account_retention',storage_instance_uuid=uid,storage_quote_evidence=str(quote_path),
        operation_path='/api/v1/dev/instance/pro/power_on',
        request_body_sha256=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        current_batch_dependencies=[dict(name='original_terminal_owned_source',ready=True),
            dict(name='fresh_storage_quote_and_real_shutdown_guard',ready=True)],
        downstream_scientific_dependencies=assessment.pop('scientific_dependencies',[]),
        current_batch=assessment.get('current_batch') or 'Original asset preservation only; no simulator/model start')
    if not preservation:
        native_engineering_cost_admission(OUT,assessment)
    if preservation:
        assessment['resource_analysis']=assessment['resources']
        assessment['resources']=dict(cpu_cores=1,ram_gib=2,gpu_required=False,gpu_count=1,storage_gib=1,
            allocated_GPU_gib=48,task_GPU_compute_required=False,
            platform_GPU_reason='Only supported Pro API mode for retrieving this existing unique source; no simulator/model started',
            peak_archive_and_extraction_gib=1)
    assessment['resume_authorization_evidence']=str(ROOT/'.codex/task_priority.json')
    if preservation:
        assessment['notes']='Current priority recovery authorization supersedes the old capture stop; this lease remains preservation-only. Subsequent capture requires its own ready complete-cost assessment. Original failed attempts and scientific gates are retained.'
    ready=folder/(receipt_prefix+'ready_budget_assessment.json');write(ready,assessment,True)
    shared_cli(folder,receipt_prefix+'assessment','budget','request','--assessment-file',ready)
    shared_cli(folder,receipt_prefix+'wallet','budget','refresh')
    return ready


def budgeted_start(folder,guard,uid,assessment_file):
    """Arm real protection first, then quote/reserve and issue one exact CLI start."""
    ready=prepare_budgeted_start(folder,guard,uid,assessment_file)
    reserved=shared_cli(folder,'reserve','budget','reserve','--assessment-file',ready)
    write(folder/'budget_lease.json',reserved,True)
    shared_cli(folder,'start','autodl','start','--instance',uid,'--body-file',folder/'start_request.json',
        '--budget-lease',reserved['lease_id'])
    return reserved


def diagnose_native_runtime(remote,folder,filename='native_runtime_diagnosis.json'):
    """Read the actual GPU/runtime and recent saved crash evidence; no engine launch."""
    probe=r'''
import json,os,subprocess,time,shutil,hashlib,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path('/path/to/aihcir/benchmarks/carlaair_v1')
prepared=json.loads((root/'runtime_preparation.json').read_text())
result={'observed_at':time.time(),'run_class':'native_runtime_diagnosis','new_RGB_frames':0,'engine_starts':0,'model_calls':0,'prepared':prepared,'commands':{},'crash_files':[],'diagnostic_core_dump_limit_bytes':0}
def command(name,args,environment=None):
 try:
  p=subprocess.run(args,capture_output=True,text=True,errors='replace',timeout=25,env=environment)
  result['commands'][name]={'args':args,'exit_code':p.returncode,'stdout':p.stdout[-24000:],'stderr':p.stderr[-12000:]}
  if name.startswith('owned_service_vulkan') or name=='root_vulkan_default':
   # The actual installed tool predates --summary. Keep enumeration identity
   # lines even when extension/feature lists exceed the bounded log tail.
   keys=('Vulkan Instance Version','GPU id','GPU0','deviceName','deviceID','vendorID','driverVersion','driverName','apiVersion','VkResult','VK_ERROR','ERROR')
   result['commands'][name]['identity_lines']=[s for s in p.stdout.splitlines() if any(k in s for k in keys)][:100]
   result['commands'][name]['stdout_bytes']=len(p.stdout.encode('utf-8'))
 except Exception as exc:result['commands'][name]={'error':type(exc).__name__+': '+str(exc)}
command('gpu',['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'])
command('tools',['sh','-c','command -v vulkaninfo; command -v gdb; command -v ldd'])
result['selected_environment']={k:os.environ.get(k) for k in ('DISPLAY','WAYLAND_DISPLAY','SDL_VIDEODRIVER','XDG_RUNTIME_DIR','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES','VK_ICD_FILENAMES')}
command('graphics_component_versions',['dpkg-query','-W','libvulkan1','vulkan-tools'])
result['graphics_component_files']=[]
library_directory=Path('/usr/lib/x86_64-linux-gnu')
for pattern in ('libvulkan.so.*','libnvidia-vulkan*','libnvidia-glvkspirv*','libnvidia-eglcore*','libnvidia-gpucomp*'):
 for p in sorted(library_directory.glob(pattern))[:8]:
  try:
   st=p.stat();row={'path':str(p),'resolved':str(p.resolve()),'bytes':st.st_size,'mode':oct(st.st_mode&0o777)}
  except OSError as exc:row={'path':str(p),'error':type(exc).__name__+': '+str(exc)}
  result['graphics_component_files'].append(row)
result['device_nodes']=[]
for directory,pattern in (('/dev','nvidia*'),('/dev/dri','*')):
 for p in sorted(Path(directory).glob(pattern))[:16]:
  try:
   st=p.stat();row={'path':str(p),'mode':oct(st.st_mode&0o777),'uid':st.st_uid,'gid':st.st_gid}
  except OSError as exc:row={'path':str(p),'error':type(exc).__name__+': '+str(exc)}
  result['device_nodes'].append(row)
command('service_groups',['id',prepared['service_user']])
for name in ('nvidia_egl_icd.json','settings_2uav.json','settings_4uav.json'):
 p=root/'code_scene_repair'/name
 if p.is_file():result[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text(errors='replace')[:16000]}
binary=Path(prepared['binary']);result['binary']={'path':str(binary),'bytes':binary.stat().st_size,'header':binary.read_bytes()[:512].decode(errors='replace') if binary.stat().st_size<65536 else None}
runtime=Path(prepared['runtime'])
elfs=list(runtime.glob('**/CarlaUE4-Linux-Shipping'))[:1]
for elf in elfs:command('engine_ldd',['ldd',str(elf)])
vulkan=shutil.which('vulkaninfo')
if vulkan:
 env=dict(os.environ,VK_ICD_FILENAMES=str(root/'code_scene_repair/nvidia_egl_icd.json'),XDG_RUNTIME_DIR=str(root/'service_home/xdg'));env.pop('DISPLAY',None);env.pop('WAYLAND_DISPLAY',None)
 # Apply the limit after PAM switches accounts; keep failed tool probes small.
 wrapper="import os,resource,sys;resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.execv(sys.argv[1],[sys.argv[1]])"
 command('owned_service_vulkan',['runuser','-u',prepared['service_user'],'--','env',*[k+'='+v for k,v in env.items() if k in ('VK_ICD_FILENAMES','XDG_RUNTIME_DIR')],'python3','-c',wrapper,vulkan],env)
 # A second read-only enumeration separates the frozen ICD override from the
 # provider's installed loader configuration on this exact boot. No file edit.
 default=dict(env);default.pop('VK_ICD_FILENAMES',None)
 command('owned_service_vulkan_default',['runuser','-u',prepared['service_user'],'--','env','-u','VK_ICD_FILENAMES','XDG_RUNTIME_DIR='+default['XDG_RUNTIME_DIR'],'python3','-c',wrapper,vulkan],default)
 command('root_vulkan_default',[vulkan],default)
result['installed_icd_files']=[]
for directory in ('/usr/share/vulkan/icd.d','/etc/vulkan/icd.d'):
 for p in sorted(Path(directory).glob('*.json'))[:8]:
  result['installed_icd_files'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text(errors='replace')[:4000]})
result['nvidia_graphics_libraries']=[]
for p in (Path('/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0'),Path('/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0')):
 if p.exists():
  result['nvidia_graphics_libraries'].append({'path':str(p),'resolved':str(p.resolve()),'bytes':p.stat().st_size})
  command(p.name+'_ldd',['ldd',str(p)])
saved=list(runtime.glob('**/Saved/Logs/*.log'))+list(runtime.glob('**/Saved/Crashes/*/*.runtime-xml'))
for p in sorted(saved,key=lambda x:x.stat().st_mtime,reverse=True)[:6]:
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-64000));tail=f.read(64000)
 result['crash_files'].append({'path':str(p),'bytes':p.stat().st_size,'modified_at':p.stat().st_mtime,'tail':tail.decode(errors='replace')})
print(json.dumps(result))
'''
    observed=json.loads(remote.command('python3 -c '+shlex.quote(probe),150))
    write(folder/filename,observed,True)
    return observed


def diagnose_graphics_loader(remote,folder):
    """Discriminate missing dynamic components from ICD interface negotiation."""
    probe=r'''
import json,os,subprocess,time,resource,shutil
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path('/path/to/aihcir/benchmarks/carlaair_v1')
prepared=json.loads((root/'runtime_preparation.json').read_text())
result={'observed_at':time.time(),'run_class':'native_runtime_diagnosis','diagnostic_stage':'loader_interface','new_RGB_frames':0,'engine_starts':0,'model_calls':0,'commands':{},'component_files':[]}
def query(name,arguments,environment=None):
 try:
  p=subprocess.run(arguments,capture_output=True,text=True,errors='replace',timeout=15,env=environment)
  lines=p.stderr.splitlines()
  result['commands'][name]={'args':arguments,'exit_code':p.returncode,'stdout':p.stdout[-24000:],'stderr':p.stderr[-48000:],
    'dynamic_failure_lines':[s for s in lines if any(k in s.lower() for k in ('error','not found','file too short','cannot open','undefined symbol'))][-100:],
    'component_load_lines':[s for s in lines if any(k in s for k in ('glvkspirv','eglcore','allocator','glcore','gpucomp','glsi','vulkan'))][-100:]}
 except Exception as exc:result['commands'][name]={'error':type(exc).__name__+': '+str(exc)}
query('gpu',['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'])
query('linker_cache',['ldconfig','-p'])
for pattern in ('libnvidia-allocator*','libnvidia-cfg*','libnvidia-glvkspirv*','libnvidia-eglcore*','libnvidia-glcore*','libnvidia-gpucomp*','libnvidia-glsi*'):
 for p in sorted(Path('/usr/lib/x86_64-linux-gnu').glob(pattern))[:8]:
  try:result['component_files'].append({'path':str(p),'resolved':str(p.resolve()),'bytes':p.stat().st_size})
  except OSError as exc:result['component_files'].append({'path':str(p),'error':str(exc)})
layer=Path('/etc/vulkan/implicit_layer.d/nvidia_layers.json')
if layer.is_file():result['implicit_layer_manifest']=layer.read_text(errors='replace')[:12000]
# Child-side limit runs after the account switch so PAM cannot raise it later.
child="import ctypes,json,resource,sys;resource.setrlimit(resource.RLIMIT_CORE,(0,0));d=ctypes.CDLL(sys.argv[1],mode=2);n=getattr(d,'vk_icdNegotiateLoaderICDInterfaceVersion',None);v=ctypes.c_uint32(int(sys.argv[2]));rc=None\nif n is not None:n.argtypes=[ctypes.POINTER(ctypes.c_uint32)];n.restype=ctypes.c_int;rc=n(ctypes.byref(v))\ng=d.vk_icdGetInstanceProcAddr;g.argtypes=[ctypes.c_void_p,ctypes.c_char_p];g.restype=ctypes.c_void_p;p=g(None,b'vkCreateInstance');print(json.dumps(dict(requested_interface=int(sys.argv[2]),negotiated_interface=v.value,negotiate_result=rc,vkCreateInstance_address=p,library=sys.argv[1])))"
environment=dict(os.environ,LD_DEBUG='libs,files');environment.pop('DISPLAY',None);environment.pop('WAYLAND_DISPLAY',None)
for library in ('libGLX_nvidia.so.0','libEGL_nvidia.so.0'):
 for version in (2,7):
  query(library+'_interface_'+str(version),['runuser','-u',prepared['service_user'],'--','python3','-c',child,'/usr/lib/x86_64-linux-gnu/'+library,str(version)],environment)
vulkan=shutil.which('vulkaninfo')
if vulkan:
 env=dict(environment,VK_ICD_FILENAMES=str(root/'code_scene_repair/nvidia_egl_icd.json'),XDG_RUNTIME_DIR=str(root/'service_home/xdg'),VK_LOADER_DEBUG='all')
 wrapper="import os,resource,sys;resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.execv(sys.argv[1],[sys.argv[1]])"
 query('frozen_loader_dynamic_trace',['runuser','-u',prepared['service_user'],'--','python3','-c',wrapper,vulkan],env)
print(json.dumps(result))
'''
    observed=json.loads(remote.command('python3 -c '+shlex.quote(probe),150))
    write(folder/'graphics_loader_diagnosis.json',observed,True)
    return observed


def diagnose_vulkan_instance(remote,folder,include_backtrace=True):
    """Locate failure at instance creation/device enumeration, with optional native backtrace."""
    child=(ROOT/'scripts/oct07_vulkan_instance_inventory.py').read_text(encoding='utf-8')
    probe=r'''
import json,os,subprocess,time,shutil
from pathlib import Path
root=Path('/path/to/aihcir/benchmarks/carlaair_v1')
prepared=json.loads((root/'runtime_preparation.json').read_text())
env=dict(os.environ,VK_ICD_FILENAMES=str(root/'code_scene_repair/nvidia_egl_icd.json'),XDG_RUNTIME_DIR=str(root/'service_home/xdg'))
env.pop('DISPLAY',None);env.pop('WAYLAND_DISPLAY',None)
out={'observed_at':time.time(),'run_class':'native_runtime_diagnosis','diagnostic_stage':'minimal_instance','engine_starts':0,'new_RGB_frames':0,'model_calls':0,'commands':{}}
def query(name,args):
 try:
  p=subprocess.run(args,capture_output=True,text=True,errors='replace',timeout=25,env=env)
  out['commands'][name]={'exit_code':p.returncode,'stdout':p.stdout[-24000:],'stderr':p.stderr[-16000:]}
 except Exception as exc:out['commands'][name]={'error':type(exc).__name__+': '+str(exc)}
query('gpu',['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'])
query('minimal_Vulkan10_instance',['runuser','-u',prepared['service_user'],'--','python3','-c',CHILD])
gdb=shutil.which('gdb');vulkan=shutil.which('vulkaninfo')
out['gdb_available']=bool(gdb)
if BACKTRACE and gdb and vulkan:
 query('vulkaninfo_crash_backtrace',['runuser','-u',prepared['service_user'],'--',gdb,'--batch','--nx','-ex','set pagination off','-ex','set confirm off','-ex','run','-ex','bt 20','--args',vulkan])
print(json.dumps(out))
'''.replace('CHILD',repr(child)).replace('BACKTRACE',repr(include_backtrace))
    observed=json.loads(remote.command('python3 -c '+shlex.quote(probe),90))
    write(folder/'vulkan_instance_diagnosis.json',observed,True)
    return observed


def recover(source_attempt,lease_id,max_seconds=600,uid='OWNED_INSTANCE_UUID',assessment_file=None,diagnose_runtime=False):
    """One protected maintenance start; never executes a scientific entry."""
    from datetime import datetime
    from oct07_scene_preservation import extract
    if not all(n.replace('_','').isalnum() for n in (source_attempt,lease_id)):
        raise ValueError('Unsafe recovery identity')
    original,identity,prior=original_recovery_lease(OUT,source_attempt,uid)
    priority=json.loads((ROOT/'.codex/task_priority.json').read_text(encoding='utf-8'))
    authorization=priority.get('resume_authorization',{})
    if authorization.get('approved') is not True or assessment_file is None:
        raise ValueError('Current explicit recovery authorization and shared budget assessment required')
    contract=json.loads((OUT/'contract.json').read_text())
    maximum=min(max_seconds,int(datetime.fromisoformat(contract['compute_stop']).timestamp()-time.time()))
    assessment=json.loads(Path(assessment_file).read_text(encoding='utf-8-sig'))
    engine_trace=assessment.get('diagnostic_stage')=='native_entry_backtrace'
    if engine_trace:
        approved=json.loads((OUT/'priority_capture_window_authorization_20261006.json').read_text(encoding='utf-8'))
        if (approved.get('approved') is not True or assessment.get('run_class')!='qualification_entry_repair'
            or assessment.get('absolute_batch_deadline')!=json.loads((OUT/'priority_capture_continuation_assessment_20261006.json').read_text(encoding='utf-8'))['absolute_batch_deadline']):
            raise ValueError('Entry trace requires the exact newly authorized capture window, not expired diagnosis funding')
        manifest=json.loads(Path(assessment['debugger_manifest']).read_text(encoding='utf-8'))
        if sha(Path(manifest['bundle']))!=manifest['sha256']:
            raise ValueError('Official isolated debugger bundle changed before paid start')
    carried=remaining_powered_time(OUT,assessment,uid)
    if carried is not None:maximum=min(maximum,carried)
    if assessment.get('absolute_batch_deadline') is not None:
        maximum=min(maximum,int(assessment['absolute_batch_deadline']-time.time()))
    if not 300<=maximum<=1200:raise ValueError('Recovery requires the approved bounded20-minute maintenance allowance')
    folder=OUT/('recover_'+lease_id)
    if folder.exists():raise FileExistsError('Recovery already attempted; diagnose its retained first failure')
    api=Provider(uid)
    if api.status()!='shutdown':raise RuntimeError('Owned instance occupied; do not duplicate or stop its task')
    folder.mkdir();nonce=uuid.uuid4().hex
    effective_deadline=time.time()+maximum
    if assessment.get('absolute_batch_deadline') is not None:
        effective_deadline=min(effective_deadline,assessment['absolute_batch_deadline'])
    maximum=int(effective_deadline-time.time())
    if maximum<300:raise TimeoutError('Protection arming reached the fixed cutoff')
    write(folder/'guard_context.json',dict(nonce=nonce,owned_uuid=uid,created_already_powered=False,max_powered_seconds=maximum,
        effective_deadline=effective_deadline,external_budget_start=True,
        run_class='qualification_entry_repair' if engine_trace else 'native_runtime_diagnosis' if diagnose_runtime else 'effective_output_preservation',source_lease=original.name,new_native_frame_limit=0,
        authorization='.codex/task_priority.json.resume_authorization',
        original_measurement_budget_reset=False),True)
    def owner(desired):write(folder/'owner.json',dict(nonce=nonce,pid=os.getpid(),desired=desired,heartbeat=time.time()))
    owner('run');log=(folder/'guard.log').open('xb')
    guard=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/deploy_oct07_image_campaign.py'),'guard','--guard-folder',str(folder)],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    stop=threading.Event()
    def heart():
        while not stop.wait(2):owner('run')
    thread=threading.Thread(target=heart,daemon=True);thread.start()
    def beat():
        g=json.loads((folder/'guard.json').read_text())
        if g['nonce']!=nonce or g['pid']!=guard.pid or g['terminal'] or time.time()-g['heartbeat']>30:
            raise RuntimeError('Recovery shutdown protection is not healthy')
    remote=RemoteJob(api,folder,OUT.name,'/root/miniconda3/bin/python','',True);remote.heartbeat=beat
    state=dict(source_attempt=source_attempt,lease_id=lease_id,uuid=uid,started_at=time.time(),stage='starting_guard',experiment_started=False,actual_invoice_cny=None)
    error=None
    def save(**kw):
        state.update(kw,observed_at=time.time());write(folder/'status.json',state)
        print(json.dumps({k:state.get(k) for k in ('stage','observed_at','verified_files','shutdown_confirmed')}),flush=True)
    try:
        until=time.monotonic()+180
        while time.monotonic()<until:
            if (folder/'guard.json').exists():
                beat()
                if json.loads((folder/'guard.json').read_text())['phase']=='armed':break
            time.sleep(2)
        else:raise TimeoutError('Recovery protection arming exceeded180seconds')
        reserved=budgeted_start(folder,guard,uid,assessment_file)
        save(stage='shared_lease_started',budget_lease_id=reserved['lease_id'])
        until=time.monotonic()+180
        while time.monotonic()<until:
            beat()
            if json.loads((folder/'guard.json').read_text())['phase']=='running':break
            time.sleep(2)
        else:raise TimeoutError('One budgeted startup exceeded180seconds')
        from oct07_ssh_once import connect_once
        connect_once(remote)
        remote.client.get_transport().set_keepalive(5)
        query="import json,os,shutil;from pathlib import Path;root=Path("+repr(REMOTE)+");jobs=[]\nfor p in Path('/proc').iterdir():\n if not p.name.isdigit() or int(p.name)==os.getpid():continue\n try:a=(p/'cmdline').read_bytes().split(b'\\0')\n except OSError:continue\n if any(Path(x.decode(errors='replace')).name in ('run_oct07_scene_node.py','run_oct07_capture_node.py','run_oct07_native_controller_entry.py','CarlaUE4-Linux-Shipping','run_oct07_image_campaign.py','train_oct07_image_selector.py') for x in a if x):jobs.append(int(p.name))\ns=root/'oct07_scene_repair'/"+repr(source_attempt+'_status.json')+";print(json.dumps(dict(jobs=jobs,free_bytes=shutil.disk_usage(root).free,status=json.loads(s.read_text()) if s.exists() else None)))"
        inv=json.loads(remote.command('python3 -c '+shlex.quote(query),15));write(folder/'native_preflight.json',dict(inv,original_identity=identity),True)
        if inv['jobs']:raise RuntimeError('Unexpected live scientific processes; recovery must not replace them')
        if inv['free_bytes']<5*1024**3+512*1024**2:raise ValueError('Recovery export would breach remote floor')
        if diagnose_runtime:
            stage=assessment.get('diagnostic_stage')
            if engine_trace:
                for local,name in [(Path(manifest['bundle']),'entry_debugger.zip'),
                    (Path(assessment['debugger_manifest']),'entry_debugger_manifest.json'),
                    (ROOT/'scripts/oct07_engine_crash_inventory.py','entry_crash_inventory.py')]:
                    remote.upload(local,REMOTE+'/'+name,45)
                arguments=['python3',REMOTE+'/entry_crash_inventory.py','--root',REMOTE,
                    '--bundle',REMOTE+'/entry_debugger.zip','--manifest',REMOTE+'/entry_debugger_manifest.json']
                if assessment.get('private_loader'):
                    arguments+=['--private-loader',assessment['private_loader']]
                diagnosis=json.loads(remote.command(' '.join(map(shlex.quote,arguments)),170))
                write(folder/'native_entry_crash_inventory.json',diagnosis,True)
            else:
                diagnosis=(diagnose_graphics_loader(remote,folder) if stage=='loader_interface' else
                       diagnose_vulkan_instance(remote,folder) if stage=='minimal_instance' else
                       diagnose_native_runtime(remote,folder))
            save(stage=diagnosis['status'] if engine_trace else 'actual_native_runtime_diagnosed',new_RGB_frames=0,
                engine_starts=diagnosis.get('engine_starts',0),diagnostic_commands=len(diagnosis['commands']))
        helper=ROOT/'scripts/oct07_scene_preservation.py'
        remote_helper=REMOTE+'/oct07_scene_repair/'+lease_id+'_preservation_helper.py'
        remote.upload(helper,remote_helper,30)
        verify_helper="import hashlib,runpy,sys;from pathlib import Path;p=Path("+repr(remote_helper)+");assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(sha(helper))+";sys.argv=[str(p),'--root',"+repr(REMOTE)+",'--attempt',"+repr(source_attempt)+",'--export-id',"+repr(lease_id)+",'--max-bytes','268435456'];runpy.run_path(str(p),run_name='__main__')"
        cmd='python3 -c '+shlex.quote(verify_helper)
        receipt=json.loads(remote.command(cmd,60));write(folder/'preservation_receipt.json',receipt,True)
        total=0;verified=0
        for index,part in enumerate(receipt['parts']):
            if Path(part['path']).is_absolute() or '..' in Path(part['path']).parts:raise ValueError('Unsafe recovery export')
            total+=part['bytes']
            if total>256*1024**2:raise ValueError('Recovery transfer exceeds live reservation')
            target=folder/('effective_outputs_'+str(index).zfill(3)+'.zip')
            with remote.client.open_sftp() as sftp:
                sftp.get_channel().settimeout(15);sftp.get(REMOTE+'/'+part['path'],str(target))
            if sha(target)!=part['sha256']:raise ValueError('Recovery transfer hash differs')
            verified+=extract(target,folder/'effective')['verified_files'];beat()
        save(stage='original_outputs_recovered',verified_files=verified,native_status=inv.get('status'),scientific_results=0)
    except BaseException as exc:
        import traceback
        error=exc;save(stage='failed',error_type=type(exc).__name__,traceback=traceback.format_exc(),cloud_checks_stopped=True,automatic_retries=0)
    finally:
        stop.set();thread.join(3);owner('stop');until=time.monotonic()+130
        while guard.poll() is None and time.monotonic()<until:time.sleep(1)
        g=json.loads((folder/'guard.json').read_text()) if (folder/'guard.json').exists() else {}
        save(shutdown_confirmed=g.get('shutdown_confirmed',False),finished_at=time.time());remote.close();log.close()
        if g.get('shutdown_confirmed') and (folder/'budget_lease.json').exists():
            reserved=json.loads((folder/'budget_lease.json').read_text())
            close_verified_lease(folder, reserved['lease_id'], uid)
    if error:raise RuntimeError('Original-output recovery failed; inspect retained first failure, no automatic retry') from error
    if not state.get('shutdown_confirmed'):raise RuntimeError('Recovery shutdown readback is unresolved')
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('package','run','recover','diagnose'));p.add_argument('--mode',choices=('material','qualification','capture'),default='material');p.add_argument('--attempt');p.add_argument('--package-id');p.add_argument('--source-attempt');p.add_argument('--lease-id');p.add_argument('--assessment-file',type=Path);p.add_argument('--max-seconds',type=int,default=600);p.add_argument('--scope',choices=('parent','first','tonight'),default='parent');a=p.parse_args()
    for n in (a.attempt,a.package_id,a.source_attempt,a.lease_id):
        if n is not None and not n.replace('_','').isalnum():raise ValueError('Unsafe unique identity')
    if a.action in ('recover','diagnose'):
        if not a.source_attempt or not a.lease_id:p.error('Recovery requires source-attempt and lease-id')
        recover(a.source_attempt,a.lease_id,a.max_seconds,assessment_file=a.assessment_file,diagnose_runtime=a.action=='diagnose')
    elif not a.attempt:p.error('Package/run requires attempt')
    elif a.action=='package':
        if a.mode=='material' and a.attempt=='scene_material2_authorized1':
            from oct07_material_budget import begin_renewal
            begin_renewal(OUT,a.attempt)
        print(json.dumps(dict(package=str(package(a.package_id or a.attempt)))))
    else:run(a.mode,a.attempt,package_id=a.package_id,scope=a.scope,assessment_file=a.assessment_file)
