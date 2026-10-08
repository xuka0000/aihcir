"""Bounded, terminal-supervised repair of the existing primary teacher node.

The local request queue contains deterministic commands from the current Codex
session, not an AI worker. A 180-second idle deadline prevents forgotten GPU
sessions. Every request is archived. The original powered budget is debited
from both previous attempts; their one startup recovery is not reset.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import time
import socket

from public_mcq_cloud_guard import OwnedControl, read, write
from run_public_mcq_teacher_autodl import LocalGuard, RemoteJob, system_proxy_socket


def connect_ready(api, remote, folder, heartbeat, *, sleep=time.sleep,proxy_factory=None):
    """At most three pre-command SSH handshakes; never replay authentication errors."""
    import paramiko
    connection_id=str(time.time_ns())
    if proxy_factory is None:proxy_factory=system_proxy_socket
    for attempt in range(1,4):
        heartbeat();snapshot=api.connection()
        candidate=paramiko.SSHClient();stream=None
        candidate.load_host_keys(str(api.token_path.parent/'known_hosts'))
        candidate.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            stream=proxy_factory(snapshot['proxy_host'],int(snapshot['ssh_port']))
            candidate.connect(snapshot['proxy_host'],port=int(snapshot['ssh_port']),username='root',
                password=snapshot['root_password'],timeout=10,auth_timeout=10,banner_timeout=10,
                look_for_keys=False,allow_agent=False,sock=stream)
            candidate.get_transport().set_keepalive(15)
            remote.client=candidate;remote.proxy_socket=stream
            write(folder/'ssh_connection_status.json',dict(stage='connected',connection_id=connection_id,attempt=attempt,observed_unix=time.time()))
            return
        except Exception as error:
            transient=(isinstance(error,(EOFError,socket.timeout,ConnectionResetError))
                or type(error) is paramiko.SSHException and str(error).startswith('Error reading SSH protocol banner'))
            write(folder/f'ssh_{connection_id}_attempt_{attempt}.json',dict(stage='failed',attempt=attempt,
                error_type=type(error).__name__,pre_command=True,banner_or_transport_failure=transient,observed_unix=time.time()))
            try:candidate.close()
            finally:
                if stream is not None:stream.close()
            if not transient or attempt==3:raise
            for _ in range(5):heartbeat();sleep(1)
        finally:del snapshot


def remaining_budget(previous, node_hash):
    charged=0;receipts=[];expected=2400
    for folder in previous:
        raw=(folder/'cloud_status.json').read_bytes();state=json.loads(raw)
        context=read(folder/'guard_context.json');guard=read(folder/'guard.json')
        elapsed=state.get('elapsed_seconds')
        if (state.get('max_powered_seconds')!=expected or not state.get('shutdown_confirmed')
                or state.get('teacher_launches')!=0 or state.get('remote_launch_intent') is not False
                or context.get('node_hash')!=node_hash or context.get('nonce')!=guard.get('nonce')
                or not guard.get('terminal') or not guard.get('shutdown_confirmed')
                or type(elapsed) not in (float,int) or not math.isfinite(elapsed) or elapsed<=0):
            raise ValueError('Prior powered evidence is incomplete or budget is discontinuous')
        used=math.ceil(elapsed);charged+=used;expected-=used
        receipts.append(dict(path=str(folder.resolve()),sha256=hashlib.sha256(raw).hexdigest(),charged_seconds=used))
    if not 120<=expected<=2400:raise ValueError('Powered budget exhausted')
    return expected,dict(original_seconds=2400,charged_seconds=charged,remaining_seconds=expected,attempts=receipts)


def remaining_after_provision(folder, node_hash):
    """Explicit terminal-directed repair; debit the complete previous cycle."""
    raw=(folder/'provision_status.json').read_bytes();state=json.loads(raw)
    context=read(folder/'guard_context.json');guard=read(folder/'guard.json')
    budget=state.get('budget',{});elapsed=state.get('elapsed_seconds');cap=state.get('max_powered_seconds')
    if (state.get('stage')!='failed' or not state.get('shutdown_confirmed')
            or state.get('teacher_launches')!=0 or state.get('actual_model_calls')!=0
            or state.get('powered_deadline_breached') is not False
            or context.get('node_hash')!=node_hash or context.get('nonce')!=guard.get('nonce')
            or not guard.get('terminal') or not guard.get('shutdown_confirmed')
            or type(cap) is not int or budget.get('original_seconds')!=2400
            or budget.get('remaining_seconds')!=cap or budget.get('charged_seconds',-2400)+cap!=2400
            or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<elapsed<=cap):
        raise ValueError('Prior provision evidence does not authorize a continuous prelaunch budget')
    remaining=cap-math.ceil(elapsed)
    if remaining<120:raise ValueError('Powered budget exhausted')
    return remaining,dict(original_seconds=2400,charged_seconds=2400-remaining,remaining_seconds=remaining,
        previous_path=str(folder.resolve()),previous_sha256=hashlib.sha256(raw).hexdigest(),
        reason='Current terminal diagnosed infrastructure failure; no scientific retry')


def remaining_after_deployment(folder,node_hash):
    raw=(folder/'provision_status.json').read_bytes();state=json.loads(raw)
    context=read(folder/'guard_context.json');guard=read(folder/'guard.json');budget=state.get('budget',{})
    cap=state.get('max_powered_seconds');elapsed=state.get('elapsed_seconds')
    charged=budget.get('scope_charged_seconds',0)
    if (budget.get('scope')!='primary_model_deployment' or state.get('stage')!='failed'
            or not state.get('shutdown_confirmed') or state.get('teacher_launches')!=0
            or state.get('actual_model_calls')!=0 or state.get('powered_deadline_breached') is not False
            or context.get('node_hash')!=node_hash or context.get('nonce')!=guard.get('nonce')
            or not guard.get('terminal') or not guard.get('shutdown_confirmed')
            or type(cap) is not int or type(charged) is not int or cap+charged!=2100
            or budget.get('prior_charged_seconds')!=1576
            or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<elapsed<=cap):
        raise ValueError('Deployment continuation evidence invalid')
    remaining=cap-math.ceil(elapsed)
    if remaining<120:raise ValueError('Deployment scope exhausted')
    return remaining,dict(scope='primary_model_deployment',scope_ceiling_seconds=2100,
        scope_charged_seconds=2100-remaining,max_powered_seconds=remaining,prior_charged_seconds=1576,
        previous_path=str(folder.resolve()),previous_sha256=hashlib.sha256(raw).hexdigest(),scientific_budget_changed=False)


INVENTORY="""import json,os,shutil,subprocess
from pathlib import Path
roots=[Path('/path/to/aihcir'),Path('/root/gpufree-data/AIHCIR')]
result={'roots':[],'python':[]}
for root in roots:
 if not root.exists():continue
 row={'root':str(root),'disk':dict(zip(('total','used','free'),shutil.disk_usage(root))),'models':[],'envs':[]}
 for folder in sorted((root/'models').glob('*')):
  if folder.is_dir():
   marker=folder/'download_status.json'
   files=[{'name':p.name,'size':p.stat().st_size} for p in folder.iterdir() if p.is_file()]
   row['models'].append({'path':str(folder),'files':files,'download_status':json.loads(marker.read_text()) if marker.exists() else None})
 for folder in sorted((root/'envs').glob('*')):
  row['envs'].append({'path':str(folder),'python_exists':(folder/'bin/python').exists()})
 result['roots'].append(row)
for candidate in ['/root/miniconda3/envs/experiment_env_gpu/bin/python','/path/to/aihcir/envs/qwen251/bin/python','/path/to/aihcir/envs/qwen3/bin/python']:
 if not Path(candidate).exists():continue
 code='import json,importlib.metadata as m; ds={d.metadata["Name"].lower():d.version for d in m.distributions()}; print(json.dumps({n:ds.get(n.lower()) for n in ["torch","torchvision","transformers","accelerate","safetensors","Pillow"]}))'
 p=subprocess.run([candidate,'-c',code],capture_output=True,text=True,timeout=15)
 result['python'].append({'path':candidate,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
print(json.dumps(result))
"""


def run(api,guard,remote,folder,cap,receipt,bootstrap_new_host=False):
    started=time.monotonic();last_request=started;guard_started=False;powered=False
    state=dict(stage='starting',max_powered_seconds=cap,budget=receipt,teacher_launches=0,
               actual_model_calls=0,shutdown_confirmed=False,automatic_retries=0,
               unattended_supervision_available=False,notification_path='Current terminal plus provision_status.json',
               started_unix=time.time(),requests_completed=0)
    def save(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-started,heartbeat_unix=time.time())
        write(folder/'provision_status.json',state)
        print(json.dumps({k:state.get(k) for k in ['stage','elapsed_seconds','requests_completed','shutdown_confirmed']}),flush=True)
    def heartbeat():
        guard.request('run');value=guard.sample()
        if value and (value.get('terminal') or value.get('state') in ('stopping','guard_failed_before_power')):
            raise RuntimeError('Guard unavailable')
        if time.monotonic()-started>=cap-120:raise TimeoutError('Shutdown reserve')
        return value
    try:
        save()
        if api.status()!='shutdown':raise RuntimeError('Node not available')
        guard.start();guard_started=True
        while True:
            value=guard.sample()
            if value and value['state']=='armed':break
            if time.monotonic()-started>30:raise TimeoutError('Guard readiness')
            time.sleep(.2)
        powered=True;guard.request('run');save(stage='starting_node')
        boot=time.monotonic()
        while True:
            value=heartbeat()
            if value and value.get('node_status')=='running':break
            if time.monotonic()-boot>180:raise TimeoutError('Node startup')
            time.sleep(2)
        import paramiko
        remote.heartbeat=heartbeat
        snapshot=api.connection();remote.client=paramiko.SSHClient()
        remote.client.load_host_keys(str(api.token_path.parent/'known_hosts'))
        remote.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        if bootstrap_new_host:
            host=snapshot['proxy_host'];port=int(snapshot['ssh_port']);key_name=host if port==22 else '['+host+']:'+str(port)
            if key_name not in remote.client.get_host_keys():
                # First-use key is bound to the freshly created instance and
                # two authenticated provider endpoint snapshots. Existing keys
                # are never replaced. Authentication then uses RejectPolicy.
                stream=system_proxy_socket(host,port);transport=paramiko.Transport(stream)
                try:
                    transport.start_client(timeout=15);key=transport.get_remote_server_key()
                    check=api.connection()
                    if check['proxy_host']!=host or int(check['ssh_port'])!=port:
                        raise RuntimeError('Provider endpoint changed during initial key binding')
                    remote.client.get_host_keys().add(key_name,key.get_name(),key)
                    remote.client.save_host_keys(str(api.token_path.parent/'known_hosts'))
                    write(folder/'new_node_host_key.json',dict(node_hash=api.node_hash,key_type=key.get_name(),
                        public_key_sha256=hashlib.sha256(key.asbytes()).hexdigest(),observed_unix=time.time(),
                        trust_basis='First-use key of newly created node; provider endpoint checked twice',existing_key_overwritten=False))
                finally:transport.close();stream.close()
        del snapshot
        connect_ready(api,remote,folder,heartbeat)
        save(stage='inventory')
        inventory=json.loads(remote.command('python3 -c '+shlex.quote(INVENTORY),60))
        write(folder/'inventory.json',inventory)
        # Current terminal may inspect the inventory before issuing repair.
        last_request=time.monotonic();save(stage='awaiting_terminal_request')
        index=1
        while True:
            heartbeat()
            request=folder/f'request_{index:03d}.json'
            if not request.exists():
                if time.monotonic()-last_request>=180:raise TimeoutError('Terminal request idle deadline')
                time.sleep(1);continue
            data=read(request)
            if data.get('action')=='finish':save(stage='provisioning_finished');break
            seconds=data.get('seconds')
            if type(seconds) is not int or not 1<=seconds<=1850:raise ValueError('Invalid request bound')
            if time.monotonic()-started+seconds+130>cap:raise TimeoutError('Request does not fit remaining powered budget')
            save(stage='executing_terminal_request',request=index)
            if data['action']=='teacher':
                from acquire_primary_on_guard import acquire
                remote.close()
                remote=RemoteJob(api,folder,'train10_primary_20260928_03',
                    '/path/to/aihcir/envs/qwen3/bin/python',
                    '/path/to/aihcir/models/Qwen3-VL-8B-Instruct_0c351dd_modelscope',True)
                save(teacher_launches=None,actual_model_calls=None)
                teacher=acquire(remote,folder,heartbeat,seconds)
                state.update(teacher_launches=teacher['teacher_launches'],actual_model_calls=teacher['actual_teacher_calls'])
                write(folder/f'response_{index:03d}.json',teacher)
                save(stage='provisioning_finished' if teacher['stage']=='teacher_completed' else 'failed')
                break
            elif data['action']=='detached_command':
                from observe_detached_runtime import observe
                output=observe(remote,folder,heartbeat,lambda:connect_ready(api,remote,folder,heartbeat),data)
            elif data['action']=='command':
                output=remote.command(data['command'],seconds)
            elif data['action']=='upload':
                source=Path(data['source']).resolve();destination=data['destination']
                if not destination.startswith('/path/to/aihcir/') or '..' in Path(destination).parts:
                    raise ValueError('Upload outside primary project')
                if hashlib.sha256(source.read_bytes()).hexdigest()!=data['sha256']:raise ValueError('Upload source changed')
                before=time.monotonic()
                def progress(*_):
                    if time.monotonic()-before>seconds:raise TimeoutError('Upload deadline')
                    heartbeat()
                with remote.client.open_sftp() as sftp:
                    sftp.get_channel().settimeout(min(15,seconds))
                    sftp.put(str(source),destination,callback=progress,confirm=True)
                output=remote.command('sha256sum '+shlex.quote(destination),10)
                if output.split()[0]!=data['sha256']:raise ValueError('Remote upload digest changed')
            else:raise ValueError('Unknown provisioning action')
            write(folder/f'response_{index:03d}.json',dict(request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),output=output,observed_unix=time.time()))
            state['requests_completed']=index;index+=1;last_request=time.monotonic()
            save(stage='awaiting_terminal_request')
    except BaseException as exc:
        save(stage='failed',error_type=type(exc).__name__)
    finally:
        remote.close()
        if guard_started:
            try:
                guard.request('stop');stop=time.monotonic()
                while time.monotonic()-stop<120:
                    value=guard.sample()
                    if value and value.get('terminal'):
                        state['guard_terminal']=value;state['shutdown_confirmed']=value.get('shutdown_confirmed',False);break
                    time.sleep(2)
            except Exception as exc:state['shutdown_error_type']=type(exc).__name__
            if powered and not state['shutdown_confirmed']:
                try:
                    recovered=guard.recover();state['emergency_shutdown']=recovered
                    state['shutdown_confirmed']=recovered['shutdown_confirmed']
                except Exception as exc:state['emergency_shutdown_error_type']=type(exc).__name__
            guard.close()
        state['powered_deadline_breached']=time.monotonic()-started>cap
        if (folder/'cloud_status.json').exists():
            teacher=read(folder/'cloud_status.json');teacher['shutdown_confirmed']=state['shutdown_confirmed']
            teacher['powered_cycle_seconds']=time.monotonic()-started;teacher['powered_budget']=receipt
            if (teacher['stage']=='teacher_completed' and teacher.get('artifact_sha256')
                    and state['shutdown_confirmed'] and not state['powered_deadline_breached']):teacher['stage']='completed'
            write(folder/'cloud_status.json',teacher)
        save(ended_unix=time.time())
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-script',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--previous',type=Path,nargs=2)
    group.add_argument('--replacement-creation',type=Path)
    group.add_argument('--previous-provision',type=Path)
    group.add_argument('--deployment-scope',type=Path)
    group.add_argument('--previous-deployment',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();api=OwnedControl(a.control_script)
    if a.previous:
        cap,receipt=remaining_budget(a.previous,api.node_hash);claim=a.previous[-1]/'provisioning_claim.json'
    elif a.previous_provision:
        cap,receipt=remaining_after_provision(a.previous_provision,api.node_hash)
        claim=a.previous_provision/'continuation_claim.json'
    elif a.previous_deployment:
        cap,receipt=remaining_after_deployment(a.previous_deployment,api.node_hash)
        claim=a.previous_deployment/'continuation_claim.json'
    elif a.deployment_scope:
        raw=a.deployment_scope.read_bytes();scope=json.loads(raw)
        if (scope.get('kind')!='primary_model_deployment' or scope.get('max_powered_seconds')!=2100
                or scope.get('node_hash')!=api.node_hash or scope.get('teacher_calls')!=0
                or scope.get('authorization_basis')!='User-directed autonomous server repair'
                or not scope.get('previous_shutdown_verified') or not scope.get('prior_envelope_closed')):
            raise ValueError('Deployment scope incomplete')
        cap=2100;receipt=dict(scope='primary_model_deployment',max_powered_seconds=cap,
            prior_charged_seconds=scope['prior_charged_seconds'],prior_envelope_seconds=2400,
            scope_path=str(a.deployment_scope.resolve()),scope_sha256=hashlib.sha256(raw).hexdigest(),
            scientific_budget_changed=False)
        claim=a.deployment_scope.with_suffix('.claim.json')
    else:
        raw=a.replacement_creation.read_bytes();creation=json.loads(raw)
        if (creation.get('stage')!='created_and_stopped' or not creation.get('shutdown_confirmed')
                or creation.get('node_hash')!=api.node_hash or not 600<=creation.get('max_powered_seconds',0)<=2061
                or creation.get('original_powered_seconds')!=2400
                or creation.get('prior_seconds_charged')+creation['max_powered_seconds']!=2400):
            raise ValueError('Replacement creation evidence invalid')
        cap=creation['max_powered_seconds']-math.ceil(creation['elapsed_seconds'])
        if not 120<=cap<=2061:raise ValueError('Replacement exhausted remaining budget')
        receipt=dict(original_seconds=2400,charged_seconds=2400-cap,remaining_seconds=cap,
                     creation_sha256=hashlib.sha256(raw).hexdigest(),creation_path=str(a.replacement_creation.resolve()))
        claim=a.replacement_creation.parent/'provisioning_claim.json'
    a.output.mkdir(parents=True,exist_ok=False)
    # An exclusive scope claim prevents accidental repetition with old receipts.
    with claim.open('x',encoding='utf-8') as f:
        json.dump(dict(output=str(a.output.resolve()),budget=receipt,reason='User explicitly directs autonomous server repair'),f)
    guard=LocalGuard(a.output,a.control_script,api,cap)
    remote=RemoteJob(api,a.output,'primary_provision_20260928_v1','','',True)
    result=run(api,guard,remote,a.output,cap,receipt,bootstrap_new_host=bool(a.replacement_creation))
    raise SystemExit(0 if result['stage']=='provisioning_finished' and result['shutdown_confirmed'] else 1)


if __name__=='__main__':main()
