"""Bound one teacher process; cloud power supervision must be provided separately.

The production CLI uses the frozen batch and has no timeout or retry overrides.
It runs on Linux. Local tests use explicit fixture mode and never load a model.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def file_operation(operation):
    """Allow at most 30ms for Windows atomic-replacement sharing conflicts.

    Permission errors on other platforms/codes fail immediately. This changes
    neither scientific retry policy nor any monotonic supervision deadline.
    """
    for attempt in range(4):
        try:
            return operation()
        except PermissionError as exc:
            if os.name!='nt' or getattr(exc,'winerror',None) not in (5,32) or attempt==3:
                raise
            time.sleep(.01)


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    file_operation(lambda:temporary.replace(path))


@dataclass(frozen=True)
class Bounds:
    process_seconds: float = 1920
    startup_seconds: float = 30
    loading_seconds: float = 300
    heartbeat_seconds: float = 200
    episode_seconds: float = 180
    terminate_grace_seconds: float = 5
    poll_seconds: float = 1

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('All supervision bounds must be positive and finite')


class Monitor:
    """Monotonic limits do not extend when a worker rewrites its heartbeat."""
    def __init__(self, bounds, started, episode_ids, max_calls):
        self.bounds, self.started = bounds, started
        self.ids, self.max_calls = list(episode_ids), max_calls
        self.stage, self.stage_started = None, started
        self.episode, self.episode_started = None, None
        self.last_heartbeat, self.last_calls, self.last_completed = None, 0, 0

    def inspect(self, value, now, wall_now):
        if now-self.started >= self.bounds.process_seconds:
            return 'process_deadline'
        if value is None:
            return 'missing_status' if now-self.started >= self.bounds.startup_seconds else None
        stage = value.get('stage')
        if stage not in ('preflight', 'loading_model', 'acquiring', 'completed', 'failed'):
            return 'invalid_status_stage'
        heartbeat = value.get('heartbeat_unix')
        if (type(heartbeat) not in (float, int) or not math.isfinite(heartbeat)
                or heartbeat > wall_now+5):
            return 'invalid_heartbeat'
        if self.last_heartbeat is not None and heartbeat < self.last_heartbeat:
            return 'heartbeat_reversed'
        self.last_heartbeat = heartbeat
        if stage != self.stage:
            permitted = {None: {'preflight','loading_model','acquiring','completed','failed'},
                         'preflight': {'loading_model','acquiring','completed','failed'},
                         'loading_model': {'acquiring','completed','failed'},
                         'acquiring': {'completed','failed'}, 'completed': set(), 'failed': set()}
            if stage not in permitted[self.stage]:
                return 'stage_reversed'
            self.stage, self.stage_started = stage, now
        for field, ceiling in (('actual_model_calls', self.max_calls), ('completed',len(self.ids)), ('failed',len(self.ids))):
            if type(value.get(field)) is not int or not 0 <= value[field] <= ceiling:
                return 'invalid_count_'+field
        if value.get('planned_questions') != len(self.ids) or value.get('max_model_calls') != self.max_calls:
            return 'changed_batch_limits'
        if value['actual_model_calls'] < self.last_calls or value['completed'] < self.last_completed:
            return 'count_reversed'
        self.last_calls, self.last_completed = value['actual_model_calls'], value['completed']
        if value['completed']+value['failed'] > len(self.ids):
            return 'invalid_question_total'
        if stage == 'failed' or value['failed']:
            return 'worker_reported_failure'
        if stage == 'preflight' and now-self.started >= self.bounds.startup_seconds:
            return 'preflight_deadline'
        if stage == 'loading_model' and now-self.stage_started >= self.bounds.loading_seconds:
            return 'loading_deadline'
        if stage != 'loading_model' and wall_now-heartbeat > self.bounds.heartbeat_seconds:
            return 'stale_heartbeat'
        if stage == 'acquiring':
            episode = value.get('current_episode')
            if episode not in self.ids:
                return 'unknown_episode'
            if episode != self.episode:
                if self.episode is not None and self.ids.index(episode) <= self.ids.index(self.episode):
                    return 'episode_repeated_or_reversed'
                self.episode, self.episode_started = episode, now
            if now-self.episode_started >= self.bounds.episode_seconds:
                return 'episode_deadline'
        if stage == 'completed' and (value['completed'] != len(self.ids) or value['failed']):
            return 'incomplete_terminal_status'
        return None


def stop_process(process, grace, fixture):
    if os.name == 'posix':
        # Kill the whole session's process group, including surviving descendants.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        until = time.monotonic()+grace
        while time.monotonic() < until:
            process.poll()  # Reap the root before checking for other group members.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(min(.05, grace))
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif fixture:
        # Windows fixtures create no children. Production Linux group cleanup is
        # a separate deployment check, never inferred from this branch.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                process.kill()
    else:
        raise RuntimeError('Production supervision requires Linux process groups')
    process.wait(timeout=max(1, grace))


def supervise(command, output, episode_ids, max_calls, *, bounds=Bounds(), fixture=False):
    if not fixture and not sys.platform.startswith('linux'):
        raise RuntimeError('Production supervision requires Linux; no worker launched')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, started_wall = time.monotonic(), time.time()
    monitor = Monitor(bounds, started, episode_ids, max_calls)
    record = dict(schema='public_mcq_process_supervision_v1', mode='fixture' if fixture else 'real_worker',
                  bounds=asdict(bounds), started_unix=started_wall, supervisor_pid=os.getpid(),
                  automatic_retries=0, launches=0, state='starting', terminal=False,
                  power_control_available=False, unattended_supervision_available=False,
                  notification_path='foreground stdout plus events.jsonl and supervisor_status.json',
                  command_sha256=hashlib.sha256(json.dumps(command).encode()).hexdigest())
    process, sample, reason = None, None, None
    operation = 'initializing'
    events = (output/'events.jsonl').open('x', encoding='utf-8')

    def save(event=None):
        record['heartbeat_unix'] = time.time()
        write(output/'supervisor_status.json', record)
        if event:
            item = dict(event=event, observed_unix=time.time(), state=record['state'], reason=record.get('reason'))
            events.write(json.dumps(item)+'\n');events.flush()
            print(json.dumps(item), flush=True)

    save('supervisor_started')
    try:
        with (output/'worker.log').open('xb') as log:
            operation = 'launch_worker'
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name=='posix',
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            record.update(launches=1, worker_pid=process.pid, state='observing')
            operation = 'publish_worker_started'
            save('worker_started')
            last_signature = None
            while True:
                status = output/'result/status.json'
                operation = 'read_worker_status'
                sample = json.loads(file_operation(lambda:status.read_text(encoding='utf-8'))) if status.exists() else None
                operation = 'inspect_worker_status'
                reason = monitor.inspect(sample, time.monotonic(), time.time())
                if sample is not None:
                    record['worker_status'] = sample
                signature = None if sample is None else (sample.get('stage'), sample.get('current_episode'), sample.get('completed'))
                operation = 'publish_supervisor_status'
                save('worker_progress' if signature != last_signature else None)
                last_signature = signature
                operation = 'poll_worker'
                returncode = process.poll()
                if reason:
                    break
                if returncode is not None:
                    reason = 'worker_nonzero_exit' if returncode else ('completed' if sample and sample.get('stage')=='completed' else 'missing_completed_status')
                    break
                time.sleep(bounds.poll_seconds)
    except BaseException as exc:
        reason = 'supervisor_interrupted' if isinstance(exc, (KeyboardInterrupt,SystemExit)) else 'supervisor_error'
        record['error_type'] = type(exc).__name__
        record.update(error_operation=operation,error_errno=getattr(exc,'errno',None),
                      error_winerror=getattr(exc,'winerror',None))
    finally:
        record.update(state='stopping', reason=reason)
        try:
            save('worker_stopping')
        finally:
            if process is not None:
                try:
                    remaining = max(0, bounds.process_seconds-(time.monotonic()-started))
                    stop_process(process, min(bounds.terminate_grace_seconds, remaining), fixture)
                    record['worker_reaped'] = process.poll() is not None
                    record['worker_returncode'] = process.returncode
                except Exception as exc:
                    record.update(cleanup_error=type(exc).__name__, worker_reaped=False)
            record.update(state='completed' if reason=='completed' and record.get('worker_reaped') else 'failed',
                          terminal=True, ended_unix=time.time(), elapsed_seconds=time.monotonic()-started)
            artifacts = {}
            for file in sorted(output.rglob('*')):
                if file.is_file() and not file.is_symlink() and file.name not in ('supervisor_status.json','artifact_manifest.json','events.jsonl'):
                    artifacts[file.relative_to(output).as_posix()] = dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            write(output/'artifact_manifest.json',dict(schema='public_mcq_supervised_artifacts_v1',files=artifacts,
                    source='fixture_process' if fixture else 'real_worker',scientific_effect_validated=False))
            try:
                save('supervisor_terminal')
            finally:
                events.close()
    return record


def load_frozen_batch(assets):
    batch = json.loads((assets/'batch.json').read_text(encoding='utf-8'))
    required = dict(schema='public_mcq_teacher_batch_v1', episode_count=10, max_model_calls=116,
                    max_batch_seconds=1800, external_timeout_seconds=1920, max_powered_seconds=2400,
                    max_episode_seconds=180, automatic_model_retries=0)
    ids = batch.get('episode_ids')
    if (any(batch.get(k) != v for k,v in required.items()) or not isinstance(ids,list)
            or len(ids)!=10 or any(not isinstance(v,str) for v in ids) or len(set(ids))!=10):
        raise ValueError('Frozen batch limits changed; no worker launched')
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets','code','model','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--backbone', required=True, choices=['qwen3','qwen25'])
    args = parser.parse_args()
    def interrupted(*_):
        raise KeyboardInterrupt('Supervisor termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    batch = load_frozen_batch(args.assets)
    worker = Path(__file__).with_name('run_public_mcq_teacher.py')
    command = [sys.executable,'-u',str(worker),'--assets',str(args.assets.resolve()),'--code',str(args.code.resolve()),
               '--model',str(args.model.resolve()),'--backbone',args.backbone,'--output',str(args.output.resolve()/'result')]
    record = supervise(command,args.output,batch['episode_ids'],batch['max_model_calls'])
    raise SystemExit(0 if record['state']=='completed' else 1)


if __name__ == '__main__':
    main()
