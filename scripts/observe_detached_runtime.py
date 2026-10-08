"""Observe one deadline-bound remote setup command without replaying its launch."""
import base64
import json
from pathlib import PurePosixPath
import re
import shlex
import time

from public_mcq_cloud_guard import write


def observe(remote,folder,heartbeat,reconnect,request,*,clock=time.monotonic,sleep=time.sleep):
    seconds=request['seconds'];job=request['job_id'];status_path=request['status_path']
    if type(seconds) is not int or not 60<=seconds<=1850:raise ValueError('Detached command bound')
    if not re.fullmatch(r'[a-z0-9_]{5,80}',job):raise ValueError('Invalid detached job identity')
    if not status_path.startswith('/path/to/aihcir/') or '..' in PurePosixPath(status_path).parts:
        raise ValueError('Status path outside project')
    root='/path/to/aihcir/sessions/'+job;started=clock()
    script=('set +e\n'+'timeout -k 5 '+str(seconds-20)+' bash -lc '+shlex.quote(request['command'])+
        ' >'+shlex.quote(root+'/output.log')+' 2>&1\nrc=$?\nprintf "%s" "$rc" >'+shlex.quote(root+'/exit.code')+'\n')
    encoded=base64.b64encode(script.encode()).decode()
    create="import base64;from pathlib import Path;p=Path("+repr(root)+");p.mkdir(exist_ok=False);(p/'job.sh').write_bytes(base64.b64decode("+repr(encoded)+"))"
    remote.command('python3 -c '+shlex.quote(create),15)
    state=dict(stage='launching',job_id=job,launches=1,launch_intent=True,started_unix=time.time(),
        seconds=seconds,actual_model_calls=0,automatic_command_retries=0,read_reconnections=0)
    def save(**kw):
        state.update(kw,observed_unix=time.time(),elapsed_seconds=clock()-started)
        write(folder/'detached_status.json',state)
    def fail(reason,error_type=RuntimeError):
        save(stage='failed',failure_reason=reason)
        raise error_type(reason)
    save()
    try:
        remote.command('nohup bash '+shlex.quote(root+'/job.sh')+' </dev/null >/dev/null 2>&1 &',10)
    except Exception as error:
        # Ambiguous launch reply: read the unique job, never submit it again.
        save(launch_reply_error_type=type(error).__name__)
        remote.close();reconnect();state['read_reconnections']+=1
    code="import json;from pathlib import Path;p=Path("+repr(root)+");s=Path("+repr(status_path)+");e=p/'exit.code';print(json.dumps(dict(progress=json.loads(s.read_text()) if s.exists() else None,exit_code=int(e.read_text()) if e.exists() else None)))"
    last_progress=None;last_change=clock()
    while True:
        heartbeat()
        if clock()-started>=seconds:fail('Detached setup deadline',TimeoutError)
        try:sample=json.loads(remote.command('python3 -c '+shlex.quote(code),10))
        except Exception:
            remote.close();reconnect();state['read_reconnections']+=1
            sample=json.loads(remote.command('python3 -c '+shlex.quote(code),10))
        progress=sample.get('progress');save(stage='observing',sample=sample)
        if progress:
            stamp=progress.get('updated_unix')
            if stamp!=last_progress:last_progress=stamp;last_change=clock()
            if progress.get('stage')=='failed':fail('Remote setup failed; no command retry')
        elif clock()-started>60:fail('Detached command did not produce status',TimeoutError)
        if clock()-last_change>90:fail('Remote setup heartbeat stalled',TimeoutError)
        if sample['exit_code'] is not None:
            if sample['exit_code']!=0 or not progress or progress.get('stage')!='completed':
                fail('Detached setup did not complete')
            save(stage='completed');return state
        for _ in range(10):heartbeat();sleep(1)
