"""Acquire assets and run the fixed ten-question primary teacher on the owned node.

No new instance is created here. A separate deterministic guard owns every
power mutation. The current terminal observes failures and emits notifications;
workstation/network loss still prevents guaranteed off-machine supervision.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import threading
import uuid
from urllib.parse import urlsplit
from urllib.request import getproxies

from public_mcq_cloud_guard import OwnedControl,emergency_shutdown,read,write
from prepare_public_mcq_teacher import fetch_images
from build_public_mcq_teacher_release import assemble
from supervise_public_mcq_teacher import load_frozen_batch

ROOT=Path(__file__).resolve().parents[1]
FROZEN='7579bd635cc5c39eb849dbc0e495852a68675cdb3fdcf2128aef2ddee71d077d'
REMOTE_ROOT='/path/to/aihcir/sessions/public_mcq_teacher'


def system_proxy_socket(host,port):
    """Use only the user's configured local HTTP proxy; SSH still verifies keys."""
    raw=getproxies().get('https')
    if not raw:raise ValueError('No configured HTTPS proxy')
    proxy=urlsplit(raw if '://' in raw else 'http://'+raw)
    if (proxy.scheme!='http' or proxy.hostname not in ('127.0.0.1','localhost','::1')
            or not proxy.port or proxy.username or proxy.password or proxy.path not in ('','/')):
        raise ValueError('Require the existing local unauthenticated HTTP proxy')
    if not re.fullmatch(r'[A-Za-z0-9.-]+',host) or type(port) is not int or not 1<=port<=65535:
        raise ValueError('Invalid provider SSH endpoint')
    deadline=time.monotonic()+15
    stream=socket.create_connection((proxy.hostname,proxy.port),timeout=15)
    try:
        target=host+':'+str(port)
        stream.sendall(('CONNECT '+target+' HTTP/1.1\r\nHost: '+target+'\r\n\r\n').encode('ascii'))
        header=bytearray()
        # Buffered file readers can consume and discard the first SSH banner.
        # Read exactly the CONNECT header, leaving all tunnel bytes untouched.
        while not header.endswith(b'\r\n\r\n'):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Proxy CONNECT deadline')
            if len(header)>=4096:raise ValueError('Proxy CONNECT header exceeds bound')
            stream.settimeout(remaining);chunk=stream.recv(1)
            if not chunk:raise ConnectionError('Proxy closed before CONNECT response')
            header.extend(chunk)
        status=header.split(b'\r\n',1)[0].split()
        if len(status)<2 or status[1]!=b'200':raise ConnectionError('Proxy CONNECT rejected')
        stream.settimeout(15)
        return stream
    except BaseException:
        stream.close();raise


def continuation_budget(previous,node_hash):
    """One explicit recovery before any teacher launch; never reset paid time."""
    raw=(previous/'cloud_status.json').read_bytes();state=json.loads(raw)
    context=read(previous/'guard_context.json');guard=read(previous/'guard.json')
    elapsed=state.get('elapsed_seconds')
    if (state.get('stage')!='failed' or state.get('failure_phase')!='runtime_preflight'
            or state.get('teacher_launches')!=0 or state.get('remote_launch_intent') is not False
            or not state.get('shutdown_confirmed') or state.get('continuation')
            or state.get('max_powered_seconds')!=2400 or guard.get('power_on_attempts')!=1
            or not guard.get('shutdown_confirmed') or not guard.get('terminal')
            or context.get('node_hash')!=node_hash or context.get('nonce')!=guard.get('nonce')
            or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<elapsed<=360):
        raise ValueError('Prior attempt is not eligible for one bounded prelaunch continuation')
    used=math.ceil(elapsed)
    return 2400-used,dict(previous=str(previous.resolve()),previous_status_sha256=hashlib.sha256(raw).hexdigest(),
        prior_elapsed_seconds_charged=used,total_powered_seconds_ceiling=2400,prelaunch_recovery_number=1)


def prepare(assets,protocol,output,*,fetch=True):
    if hashlib.sha256((assets/'batch.json').read_bytes()).hexdigest()!=FROZEN:
        raise ValueError('Only the existing frozen ten-question batch is authorized by this entry')
    batch=load_frozen_batch(assets);episodes=read(assets/'public_episodes.json')
    payload,report=assemble(assets,protocol)
    if payload is None and fetch:
        fetch_images(episodes,assets,time.monotonic()+900)
        payload,report=assemble(assets,protocol)
    write(output/'release_manifest.json',report)
    if payload is None:raise FileNotFoundError('Frozen source images are incomplete')
    with (output/'release.zip').open('xb') as stream:stream.write(payload)
    return batch


class LocalGuard:
    def __init__(self,folder,control_script,api,max_powered_seconds=2400):
        self.folder,self.script,self.api=folder,control_script,api
        self.max_powered_seconds=max_powered_seconds
        self.nonce=uuid.uuid4().hex;self.process=None;self.log=None
        self.lease=ROOT/'evidence/cloud_leases'/('public_teacher_'+api.node_hash+'.json')

    def start(self):
        self.lease.parent.mkdir(parents=True,exist_ok=True)
        with self.lease.open('x',encoding='utf-8') as handle:
            json.dump(dict(nonce=self.nonce,folder=str(self.folder.resolve()),created_unix=time.time()),handle)
        try:
            write(self.folder/'guard_context.json',dict(nonce=self.nonce,node_hash=self.api.node_hash,
                lease=str(self.lease.resolve()),max_powered_seconds=self.max_powered_seconds))
            self.request('idle')
            self.log=(self.folder/'guard.log').open('xb')
            self.process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('public_mcq_cloud_guard.py')),
                '--folder',str(self.folder.resolve()),'--control-script',str(self.script.resolve())],
                stdout=self.log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix',
                creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0)
        except Exception:
            if self.process is None:
                if self.log is not None:self.log.close()
                if read(self.lease)['nonce']==self.nonce:self.lease.unlink()
            raise

    def request(self,desired):
        write(self.folder/'owner.json',dict(nonce=self.nonce,pid=os.getpid(),desired=desired,heartbeat_unix=time.time()))

    def sample(self):
        path=self.folder/'guard.json'
        value=read(path) if path.is_file() else None
        if value is not None and (value.get('nonce')!=self.nonce or value.get('pid')!=self.process.pid):
            raise RuntimeError('Guard handshake identity changed')
        if value is not None and time.time()-value['heartbeat_unix']>45:raise RuntimeError('Guard heartbeat stale')
        if self.process.poll() is not None and (value is None or not value.get('terminal')):
            raise RuntimeError('Guard exited without terminal evidence')
        return value

    def close(self):
        if self.process is not None:
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        if self.log is not None:self.log.close()

    def recover(self):
        if not self.lease.is_file() or read(self.lease)['nonce']!=self.nonce:
            raise RuntimeError('Cannot take over without the exact owned lease')
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
        # There is now no other process capable of issuing a later power-on.
        result=emergency_shutdown(self.api,self.nonce,checkpoint=lambda r:write(self.folder/'emergency_shutdown.json',r))
        if result['shutdown_confirmed'] and read(self.lease)['nonce']==self.nonce:self.lease.unlink()
        return result


class RemoteJob:
    def __init__(self,api,folder,run_id,python,model,use_system_ssh_proxy=False):
        self.api,self.folder,self.python,self.model=api,folder,python,model
        self.session=REMOTE_ROOT+'/'+run_id;self.client=None;self.created=False
        self.heartbeat=lambda:None
        self.command_count=0
        self.use_system_ssh_proxy=use_system_ssh_proxy;self.proxy_socket=None;self.operation='not_started'

    def mark(self,operation,**extra):
        self.operation=operation
        write(self.folder/'remote_setup_status.json',dict(operation=operation,observed_unix=time.time(),
            use_system_ssh_proxy=self.use_system_ssh_proxy,**extra))

    def release_source(self):
        return self.folder/'release.zip'

    def restore_assets(self):
        """Optional source restoration after a verified transport archive."""

    def upload(self,source,destination,seconds=45):
        """Bound all SFTP operations, including opening and final acknowledgements.

        Progress callbacks alone do not run while a pipelined upload waits for
        acknowledgements. The owner thread maintains the guard heartbeat and
        closes the transport at the unchanged deadline; it never retries data.
        """
        started=time.monotonic();progress={'sent':0,'total':source.stat().st_size}
        errors=[];finished=threading.Event()
        transfer_client=self.client
        def transfer():
            try:
                with transfer_client.open_sftp() as sftp:
                    sftp.get_channel().settimeout(min(15,seconds))
                    sftp.put(str(source),destination,callback=lambda sent,total:progress.update(sent=sent,total=total))
            except BaseException as error:errors.append(error)
            finally:finished.set()
        thread=threading.Thread(target=transfer,daemon=True);thread.start()
        error=None
        try:
            while not finished.wait(.25):
                self.heartbeat()
                elapsed=time.monotonic()-started
                write(self.folder/'upload_status.json',dict(stage='uploading',**progress,elapsed_seconds=elapsed,observed_unix=time.time()))
                if elapsed>=seconds:raise TimeoutError('SFTP total deadline including acknowledgements')
            if errors:raise errors[0]
        except BaseException as exc:
            error=type(exc).__name__;self.close();thread.join(timeout=2);raise
        finally:
            write(self.folder/'upload_status.json',dict(stage='failed' if error else 'completed',error_type=error,
                **progress,elapsed_seconds=time.monotonic()-started,observed_unix=time.time(),transfer_thread_alive=thread.is_alive()))

    def command(self,command,seconds=30):
        self.heartbeat()
        self.command_count+=1
        transport=None;channel=None;phase='transport_lookup'
        wrapped='timeout -k 2 '+str(seconds)+' bash -lc '+shlex.quote(command)
        deadline=time.monotonic()+seconds+5;parts=[];size=0;returncode=None
        last_heartbeat=time.monotonic()
        try:
            import paramiko
            transport=self.client.get_transport() if self.client is not None else None
            if transport is None:
                raise paramiko.SSHException('SSH session not active')
            # Channel creation can fail while is_active() still returns true.
            transport.set_keepalive(15)
            phase='channel_open'
            channel=transport.open_session(timeout=10)
            channel.settimeout(10)
            phase='command_exec'
            channel.exec_command(wrapped)
            phase='command_read'
            while True:
                for ready,get in ((channel.recv_ready,channel.recv),(channel.recv_stderr_ready,channel.recv_stderr)):
                    while ready():
                        block=get(65536);size+=len(block)
                        if size>2*1024**2:raise ValueError('Remote control output exceeded bound')
                        parts.append(block)
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():break
                if time.monotonic()>=deadline:raise TimeoutError('Bounded remote command expired')
                if time.monotonic()-last_heartbeat>=5:
                    self.heartbeat();last_heartbeat=time.monotonic()
                time.sleep(.05)
            text=b''.join(parts).decode('utf-8','replace')
            returncode=channel.recv_exit_status()
            if returncode!=0:
                raise RuntimeError('Remote command failed; see retained runtime/job artifacts')
            return text.strip()
        except BaseException as exc:
            # Preserve bounded partial output even if SSH/timeout interrupts the
            # command before an exit code is available. Do not log credentials
            # or arbitrary exception strings from the connection library.
            write(self.folder/f'remote_command_failure_{self.command_count:04d}.json',
                  dict(command=command,returncode=returncode,error_type=type(exc).__name__,
                       phase=phase,
                       ssh_error_code=str(exc) if str(exc) in ('Timeout opening channel.', 'SSH session not active', 'Channel closed.') else None,
                       ssh_transport_active=bool(transport and transport.is_active()),
                       channel_closed=channel.closed if channel is not None else None,
                       output=b''.join(parts).decode('utf-8','replace'),observed_unix=time.time()))
            raise
        finally:
            if channel is not None:channel.close()

    def setup(self):
        self.heartbeat()
        import paramiko
        self.mark('ssh_connect_and_authenticate')
        if self.use_system_ssh_proxy:
            # reconnect owns old-client cleanup; do not overwrite the client first.
            self.reconnect()
        else:
            self.close()
            self.mark('connection_snapshot')
            snapshot=self.api.connection()  # Never serialize credentials.
            self.client=paramiko.SSHClient()
            hosts=self.api.token_path.parent/'known_hosts'
            self.mark('host_key_load')
            self.client.load_host_keys(str(hosts));self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
            self.client.connect(snapshot['proxy_host'],port=int(snapshot['ssh_port']),username='root',
                            password=snapshot['root_password'],timeout=15,auth_timeout=15,banner_timeout=15,
                            look_for_keys=False,allow_agent=False,sock=self.proxy_socket)
            del snapshot
        # Cheap path gates precede any release transfer. A missing runtime must
        # not spend the upload budget or create a misleading remote run folder.
        self.mark('primary_runtime_paths')
        self.command('test -x '+shlex.quote(self.python)+' && test -d '+shlex.quote(self.model),10)
        # A reused remote run directory is a recovery case, never a new attempt.
        self.mark('create_remote_run')
        self.command('mkdir -p '+shlex.quote(REMOTE_ROOT))
        self.command('mkdir '+shlex.quote(self.session));self.created=True
        self.mark('release_upload')
        source=self.release_source();self.upload(source,self.session+'/release.zip',45)
        expected=hashlib.sha256(source.read_bytes()).hexdigest()
        self.mark('release_verify_and_extract')
        if self.command('sha256sum '+shlex.quote(self.session+'/release.zip')).split()[0]!=expected:
            raise ValueError('Uploaded public release changed')
        self.command('python3 -m zipfile -e '+shlex.quote(self.session+'/release.zip')+' '+shlex.quote(self.session))
        self.restore_assets()
        command=[self.python,self.session+'/probe_public_mcq_cloud_runtime.py','--model',self.model,
                 '--output',self.session+'/runtime_probe.json']
        self.mark('primary_runtime_probe')
        self.command(shlex.join(command),45)
        result=json.loads(self.command('cat '+shlex.quote(self.session+'/runtime_probe.json')))
        if result.get('stage')!='passed' or not result.get('linux_group_cleanup'):
            raise RuntimeError('Primary runtime/process-group probe failed')
        write(self.folder/'runtime_probe.json',result)
        self.mark('ready')

    def launch(self,timeout_seconds=1920):
        if type(timeout_seconds) is not int or not 1<=timeout_seconds<=1920:
            raise ValueError('Outer teacher deadline may only tighten the frozen bound')
        args=[self.python,'-u',self.session+'/supervise_public_mcq_teacher.py',
              '--assets',self.session+'/assets','--code',self.session+'/code','--model',self.model,
              '--backbone','qwen3','--output',self.session+'/supervision']
        script=('set +e\n'+'timeout -k 5 '+str(timeout_seconds)+' '+shlex.join(args)+' >'+shlex.quote(self.session+'/supervisor.log')+' 2>&1\n'
                +'rc=$?\nprintf "%s" "$rc" >'+shlex.quote(self.session+'/exit.code')+'\n')
        (self.folder/'job.sh').write_text(script,encoding='utf-8',newline='\n')
        with self.client.open_sftp() as sftp:
            sftp.get_channel().settimeout(15);sftp.put(str(self.folder/'job.sh'),self.session+'/job.sh')
        self.command('nohup bash '+shlex.quote(self.session+'/job.sh')+' </dev/null >/dev/null 2>&1 &')

    def sample(self):
        code="import json;from pathlib import Path;p=Path("+repr(self.session)+");s=p/'supervision/supervisor_status.json';e=p/'exit.code';print(json.dumps(dict(supervisor=json.loads(s.read_text()) if s.exists() else None,exit_code=int(e.read_text()) if e.exists() else None)))"
        return json.loads(self.read_command('python3 -c '+shlex.quote(code),10))

    def reconnect(self):
        from provision_public_mcq_runtime import connect_ready
        self.close()
        connect_ready(self.api,self,self.folder,self.heartbeat,proxy_factory=system_proxy_socket)

    def read_command(self,command,seconds=10):
        """One retry of an idempotent read after an observed dead transport."""
        try:return self.command(command,seconds)
        except Exception:
            transport=self.client.get_transport() if self.client is not None else None
            if not self.use_system_ssh_proxy or transport is not None and transport.is_active():raise
            self.reconnect()
            return self.command(command,seconds)

    def archive(self):
        if not self.created:return None
        code="import tarfile;from pathlib import Path;p=Path("+repr(self.session)+");t=tarfile.open(p/'artifacts.tar.gz','w:gz');[(t.add(p/n,arcname=n)) for n in ['supervision','runtime_probe.json','supervisor.log','exit.code','job.sh'] if (p/n).exists()];t.close()"
        self.command('python3 -c '+shlex.quote(code),30)
        with self.client.open_sftp() as sftp:
            sftp.get_channel().settimeout(15)
            if sftp.stat(self.session+'/artifacts.tar.gz').st_size>128*1024**2:raise ValueError('Artifact size exceeds transfer bound')
            started=time.monotonic()
            def progress(*_):
                if time.monotonic()-started>30:raise TimeoutError('Artifact download deadline')
            sftp.get(self.session+'/artifacts.tar.gz',str(self.folder/'artifacts.tar.gz'),callback=progress)
        return hashlib.sha256((self.folder/'artifacts.tar.gz').read_bytes()).hexdigest()

    def close(self):
        client,stream=self.client,self.proxy_socket
        self.client=None;self.proxy_socket=None
        try:
            if client is not None:client.close()
        finally:
            if stream is not None:stream.close()


def execute_cycle(api,guard,remote,folder,*,clock=time.monotonic,sleep=time.sleep,
                  max_powered_seconds=2400,continuation=None,profile='frozen_train10_v1'):
    """Injectable lifecycle: tests use no server, credential or real model."""
    if profile == 'native_four_view_v1':
        if continuation is None:
            if max_powered_seconds != 900:raise ValueError('Native diagnostic initial cap must be900s')
        elif (continuation.get('original_seconds')!=900 or continuation.get('prior_teacher_launches')!=0
              or continuation.get('charged_seconds',-900)+max_powered_seconds!=900
              or not 480<=max_powered_seconds<900):
            raise ValueError('Invalid native prelaunch continuation budget')
        worker_timeout, expected_completed, max_calls, prelaunch_seconds = min(600,max_powered_seconds-300), 4, 4, 180
    elif profile == 'frozen_train10_v1':
        worker_timeout, expected_completed, max_calls, prelaunch_seconds = 1920, 10, 116, 360
    elif profile == 'complete_dev231_v1':
        contract=read(folder/'batch_contract.json')
        if (max_powered_seconds!=7200 or continuation is not None
                or contract.get('schema')!='aircop_complete_dev_baseline_v1'
                or contract.get('powered_seconds')!=7200 or contract.get('model_calls')!=231
                or len(set(contract.get('episode_ids',[])))!=231):
            raise ValueError('Complete development batch contract mismatch')
        worker_timeout, expected_completed, max_calls, prelaunch_seconds = 6600, 231, 231, 480
    else:
        raise ValueError('Unknown frozen execution profile')
    if type(max_powered_seconds) is not int or (profile == 'frozen_train10_v1' and not 2040<=max_powered_seconds<=2400):
        raise ValueError('Insufficient or expanded cycle budget')
    started=clock();state=dict(stage='checking_node',teacher_launches=0,automatic_model_retries=0,
        actual_teacher_calls=None,shutdown_confirmed=False,started_unix=time.time(),
        max_powered_seconds=max_powered_seconds,worker_timeout_seconds=worker_timeout,unattended_supervision_available=False,
        execution_profile=profile,record_unit='single_view_observation' if profile=='native_four_view_v1' else 'question',
        shutdown_request_ceiling=6,
        notification_path='foreground JSON events and cloud_status.json; no off-machine notification')
    if continuation:state['continuation']=continuation
    guard_started=False;remote_started=False;power_requested=False
    def save(**values):
        state.update(values,heartbeat_unix=time.time(),elapsed_seconds=clock()-started)
        write(folder/'cloud_status.json',state)
        print(json.dumps({k:state.get(k) for k in ('stage','actual_teacher_calls','shutdown_confirmed','error_type')}),flush=True)
    def sample_guard():
        value=guard.sample()
        if value and (value.get('terminal') or value.get('state') in ('stopping','guard_failed_before_power')):
            raise RuntimeError('Power guard is stopping or unavailable')
        return value
    try:
        save()
        if api.status()!='shutdown':raise RuntimeError('Owned node occupied; no power action permitted')
        guard.start();guard_started=True;save(stage='awaiting_guard')
        while True:
            value=sample_guard()
            if value and value['state']=='armed':break
            if clock()-started>30:raise TimeoutError('Guard readiness deadline')
            sleep(.2)
        power_requested=True;guard.request('run');save(stage='starting_node')
        boot=clock()
        while True:
            guard.request('run');value=sample_guard()
            if value and value.get('node_status')=='running':break
            if clock()-boot>=180:raise TimeoutError('Node startup deadline')
            sleep(2)
        def remote_heartbeat():
            guard.request('run');sample_guard()
            if state['stage']=='runtime_preflight' and clock()-started>prelaunch_seconds:
                raise TimeoutError('Prelaunch preparation budget exhausted')
        remote.heartbeat=remote_heartbeat
        guard.request('run');save(stage='runtime_preflight');remote.setup()
        guard.request('run');sample_guard()
        if clock()-started+worker_timeout+120>max_powered_seconds:raise TimeoutError('Insufficient remaining powered budget; no teacher launch')
        # Persist launch intent first. A lost SSH reply must not cause relaunch.
        state['teacher_launches']=1;save(stage='launching_teacher');remote_started=True;remote.launch()
        save(stage='observing_teacher');last_calls=0;last_signature=None
        heartbeat=None;last_remote_change=clock();observing_started=clock()
        while True:
            guard.request('run');sample_guard()
            if clock()-started>=max_powered_seconds-120:raise TimeoutError('Shutdown reserve reached')
            sample=remote.sample();supervisor=sample.get('supervisor')
            if supervisor:
                received_heartbeat=supervisor.get('heartbeat_unix')
                if type(received_heartbeat) not in (int,float) or not math.isfinite(received_heartbeat):
                    raise ValueError('Remote supervisor heartbeat invalid')
                if heartbeat is not None and received_heartbeat<heartbeat:raise ValueError('Remote heartbeat reversed')
                if heartbeat!=received_heartbeat:heartbeat=received_heartbeat;last_remote_change=clock()
                if clock()-last_remote_change>=45:raise TimeoutError('Remote supervisor heartbeat stopped')
                worker=supervisor.get('worker_status') or {}
                calls=worker.get('actual_model_calls',last_calls)
                if type(calls) is not int or not last_calls<=calls<=max_calls:raise ValueError('Invalid/reversed teacher call count')
                last_calls=calls;state['actual_teacher_calls']=calls
                state['remote_snapshot']=sample;state['snapshot_observed_unix']=time.time()
                signature=(supervisor.get('state'),worker.get('current_episode'),worker.get('completed'))
                if signature!=last_signature:save();last_signature=signature
                if supervisor.get('state')=='failed' or worker.get('stage')=='failed':
                    raise RuntimeError('Remote teacher failure; no automatic retry')
            if sample.get('exit_code') is not None:
                if (sample['exit_code']!=0 or not supervisor or supervisor.get('state')!='completed'
                        or not supervisor.get('worker_reaped') or supervisor.get('worker_returncode')!=0
                        or supervisor.get('worker_status',{}).get('completed')!=expected_completed
                        or supervisor.get('worker_status',{}).get('stage')!='completed'
                        or supervisor.get('worker_status',{}).get('failed')!=0):
                    raise RuntimeError('Remote job lacks complete frozen-profile evidence')
                save(stage='teacher_completed');break
            if not supervisor and clock()-observing_started>=30:raise TimeoutError('Remote supervisor did not start')
            sleep(5)
    except BaseException as exc:
        state['error_type']=type(exc).__name__;state['failure_phase']=state['stage']
        state['failure_operation']=getattr(remote,'operation',None)
        state['failure_errno']=getattr(exc,'errno',None)
        if not remote_started:state['actual_teacher_calls']=0
        # Avoid persisting exception strings containing connection details.
        try:save(stage='failed')
        except OSError:pass
    finally:
        if guard_started:
            try:
                if power_requested and getattr(remote,'created',False):
                    guard.request('run');state['artifact_sha256']=remote.archive()
            except Exception as exc:state['archive_error_type']=type(exc).__name__
            finally:
                try:remote.close()
                except Exception:pass
                try:
                    guard.request('stop');stop_started=clock()
                    while clock()-stop_started<120:
                        value=guard.sample()
                        if value and value.get('terminal'):
                            state['guard_terminal']=value;state['shutdown_confirmed']=value.get('shutdown_confirmed',False);break
                        sleep(2)
                except Exception as exc:state['shutdown_error_type']=type(exc).__name__
                if power_requested and not state['shutdown_confirmed'] and hasattr(guard,'recover'):
                    try:
                        recovery=guard.recover();state['emergency_shutdown']=recovery
                        state['shutdown_confirmed']=recovery['shutdown_confirmed']
                    except Exception as exc:state['emergency_shutdown_error_type']=type(exc).__name__
                guard.close()
        state['remote_launch_intent']=remote_started
        if state['stage']=='teacher_completed' and state['shutdown_confirmed'] and state.get('artifact_sha256'):
            state['stage']='completed'
        elif state['stage']=='teacher_completed':state['stage']='delivery_or_shutdown_incomplete'
        if clock()-started>max_powered_seconds:
            state.update(stage='powered_deadline_breached',powered_deadline_breached=True)
        save(ended_unix=time.time())
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('control-script','assets','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--remote-python',default='/path/to/aihcir/envs/qwen3/bin/python')
    parser.add_argument('--remote-model',default='/path/to/aihcir/models/Qwen3-VL-8B-Instruct_0c351dd_modelscope')
    parser.add_argument('--use-system-ssh-proxy',action='store_true')
    parser.add_argument('--continue-prelaunch',type=Path)
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]{1,64}',args.run_id):raise ValueError('Invalid unique run ID')
    args.output.mkdir(parents=True,exist_ok=False)
    preflight=dict(stage='preflight',power_actions=0,model_calls=0)
    cap=2400;continuation=None
    try:
        api=OwnedControl(args.control_script)
        if api.status()!='shutdown':raise RuntimeError('Owned node not available')
        if args.continue_prelaunch:
            cap,continuation=continuation_budget(args.continue_prelaunch,api.node_hash)
            # Exclusive claim is consumed even if later preparation fails.
            with (args.continue_prelaunch/'continuation_claim.json').open('x',encoding='utf-8') as stream:
                json.dump(dict(output=str(args.output.resolve()),**continuation),stream,indent=2)
        prepare(args.assets,args.protocol,args.output)
        preflight['stage']='ready_for_guard'
    except Exception as exc:
        preflight.update(stage='blocked',error_type=type(exc).__name__,errno=getattr(exc,'errno',None))
        write(args.output/'preflight.json',preflight);print(json.dumps(preflight));raise SystemExit(1)
    write(args.output/'preflight.json',preflight)
    guard=LocalGuard(args.output,args.control_script,api,cap)
    remote=RemoteJob(api,args.output,args.run_id,args.remote_python,args.remote_model,args.use_system_ssh_proxy)
    result=execute_cycle(api,guard,remote,args.output,max_powered_seconds=cap,continuation=continuation)
    raise SystemExit(0 if result['stage']=='completed' else 1)


if __name__=='__main__':main()
