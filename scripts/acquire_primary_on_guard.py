"""One fixed teacher acquisition inside an already supervised powered lease."""
import math
import time

from public_mcq_cloud_guard import write


def acquire(remote,folder,heartbeat,seconds,*,clock=time.monotonic,sleep=time.sleep):
    if type(seconds) is not int or not 900<=seconds<=1920:
        raise ValueError('Teacher requires900..1920 seconds inside the existing lease')
    started=clock();deadline=started+seconds
    state=dict(stage='runtime_preflight',started_unix=time.time(),teacher_launches=0,
               automatic_model_retries=0,actual_teacher_calls=0,shutdown_confirmed=False,
               outer_seconds=seconds,unattended_supervision_available=False)
    def save(**kw):
        state.update(kw,heartbeat_unix=time.time(),elapsed_seconds=clock()-started)
        write(folder/'cloud_status.json',state)
        print({k:state.get(k) for k in ('stage','actual_teacher_calls','completed_questions')},flush=True)
    try:
        save();remote.heartbeat=heartbeat;remote.setup()
        outer=min(1920,math.floor(deadline-clock()-45))
        if outer<600:raise TimeoutError('Insufficient lease after real runtime preflight')
        # Persist intent once, including the tighter wall bound, before launch.
        save(stage='launching_teacher',teacher_launches=1,remote_launch_intent=True,worker_outer_seconds=outer)
        remote.launch(timeout_seconds=outer)
        seen=None;changed=clock();observing=clock();calls=0;signature=None
        save(stage='observing_teacher')
        while True:
            heartbeat()
            if clock()>=deadline-40:raise TimeoutError('Artifact reserve reached')
            sample=remote.sample();supervisor=sample.get('supervisor')
            if supervisor:
                stamp=supervisor.get('heartbeat_unix')
                if type(stamp) not in (float,int) or not math.isfinite(stamp) or seen is not None and stamp<seen:
                    raise ValueError('Invalid supervisor heartbeat')
                if stamp!=seen:seen=stamp;changed=clock()
                if clock()-changed>=45:raise TimeoutError('Supervisor heartbeat stale')
                worker=supervisor.get('worker_status') or {};value=worker.get('actual_model_calls',calls)
                if type(value) is not int or not calls<=value<=116:raise ValueError('Invalid model call count')
                calls=value;state.update(actual_teacher_calls=calls,remote_snapshot=sample,completed_questions=worker.get('completed'))
                current=(supervisor.get('state'),worker.get('completed'))
                if current!=signature:save();signature=current
                if supervisor.get('state')=='failed' or worker.get('stage')=='failed':raise RuntimeError('Teacher failure retained; no retry')
            elif clock()-observing>=30:raise TimeoutError('Supervisor startup deadline')
            if sample.get('exit_code') is not None:
                if (sample['exit_code']!=0 or not supervisor or supervisor.get('state')!='completed'
                        or not supervisor.get('worker_reaped') or supervisor.get('worker_returncode')!=0
                        or worker.get('completed')!=10 or worker.get('failed')!=0 or worker.get('stage')!='completed'):
                    raise RuntimeError('Ten-question acquisition incomplete')
                save(stage='teacher_completed');break
            sleep(2)
    except BaseException as exc:
        save(stage='failed',error_type=type(exc).__name__)
    finally:
        try:
            if remote.created:state['artifact_sha256']=remote.archive()
        except Exception as exc:state['archive_error_type']=type(exc).__name__
        save(ended_unix=time.time())
    return state
