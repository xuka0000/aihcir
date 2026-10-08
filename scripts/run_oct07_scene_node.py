"""Owned native scene jobs using the verified existing CARLA environment."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import threading
import time

def native_inventory_schedule(powered_seconds):
    """Budget the zero-RGB lane once, including measured cold-start overhead.

    Attempt24 needed53.781s from guard creation to measurement. Allow60s
    startup and5s child imports before its unchanged30s minimum measurement.
    The controller's240s reserve includes the guardian's180s stop reserve.
    This schedule is engineering preparation, not runtime/source admission.
    """
    import math
    if isinstance(powered_seconds,bool) or not isinstance(powered_seconds,(int,float)) or not math.isfinite(powered_seconds):
        raise ValueError('Native inventory powered window must be finite')
    minimum=240+60+5+30
    if powered_seconds<minimum:
        raise ValueError(f'Native inventory reserve/startup conflict before payment: needs{minimum}s window')
    return dict(minimum_window_seconds=minimum,maximum_powered_seconds=powered_seconds,
                outer_preservation_seconds=240,guardian_stop_reserve_seconds=180,
                startup_upper_seconds=60,child_import_allowance_seconds=5,
                minimum_measurement_seconds=30,work_seconds=powered_seconds-240,
                measurement_seconds_after_startup_and_imports=powered_seconds-240-60-5,
                scientific_admission=False)


def native_inner_deadline(work_deadline,inventory):
    # Acquisition keeps its original120s child reserve.
    return work_deadline if inventory else work_deadline-120


def inventory_measurement_deadline(work_deadline,now):
    if work_deadline-now<35:
        raise TimeoutError('Insufficient bounded native measurement time before child launch')
    return min(now+90,work_deadline)


def tonight_execution_window(phase, attempt):
    """Read the packaged user-authorized window without changing scene science."""
    import hashlib
    phase=Path(phase)
    from oct07_qualification_budget import continuation_batch, operational_window
    batch=continuation_batch(phase,attempt)
    if batch is not None and operational_window(phase,batch) is not None:
        return batch['absolute_deadline']
    path=phase/'tonight_native_execution_window.json'
    if not path.exists():
        return None
    value=json.loads(path.read_text(encoding='utf-8-sig'))
    if attempt not in value.get('execution_ids',[]):
        return None
    source=phase/'tonight_core_authorization_20261007.json'
    approval=json.loads(source.read_text(encoding='utf-8-sig'))
    if (hashlib.sha256(source.read_bytes()).hexdigest()!=value.get('authorization_sha256')
            or approval.get('approved') is not True
            or approval.get('authorization_id') not in ('USER-AIHCIR-TONIGHT-CORE-20261007','USER-AIHCIR-48H-CONTINUATION-20261007')
            or approval.get('prior_attempts_costs_and_elapsed_retained') is not True
            or approval.get('TEST_sealed') is not True
            or value.get('scientific_parameters_changed') is not False
            or value.get('new_scientific_RGB_frames',0)!=0
            or value.get('new_model_calls',0)!=0
            or value.get('deadline')!=datetime.fromisoformat(approval['native_repair_stop']).timestamp()):
        raise ValueError('Packaged tonight engineering authorization differs')
    return value['deadline']


def main(root,mode,attempt,scope='parent',lease_deadline=None):
    import fcntl
    root=Path(root);phase=root/'oct07_scene_repair';code=root/'code_scene_repair'
    contract=json.loads((phase/'contract.json').read_text());prepared=json.loads((root/'runtime_preparation.json').read_text())
    if (phase/'PAUSE').exists() or (phase/'STOP').exists():raise ValueError('Existing native campaign pause/stop is preserved')
    from oct07_capture_scope import selected_ids,verify_existing_capture
    selected=selected_ids(phase,scope)
    from oct07_qualification_budget import continuation_batch
    continuation=continuation_batch(phase,attempt)
    if continuation and continuation['mode']!=mode:raise ValueError('Native mode differs from approved batch')
    if continuation:selected=set(continuation['scene_ids'])
    if continuation:
        from oct07_qualification_budget import verify_source_freeze
        verify_source_freeze(phase,attempt,code)
    if root.resolve()!=Path('/path/to/aihcir/benchmarks/carlaair_v1'):raise ValueError('Unapproved native dependency root')
    lock=(root/'oct07/capture.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();limit={'material':1800,'qualification':10800,'capture':40000}[mode]
    tonight_cutoff=tonight_execution_window(phase,attempt)
    cutoff=continuation['absolute_deadline'] if continuation else (tonight_cutoff if tonight_cutoff is not None else datetime.fromisoformat(contract['compute_stop']).timestamp())
    deadline=min(started+limit,cutoff)
    if lease_deadline is not None:
        if lease_deadline<=started+240:raise TimeoutError('Shared lease cutoff reached')
        deadline=min(deadline,lease_deadline-240)
    if mode=='qualification':
        from oct07_qualification_budget import deadline as qualification_deadline
        deadline=min(deadline,(tonight_cutoff if tonight_cutoff is not None else qualification_deadline(phase,attempt))-240)
    qualification_budget=json.loads((phase/'qualification_execution_budget.json').read_text()) if mode=='qualification' else {}
    geometry_diagnostic=attempt in qualification_budget.get('diagnostic_only_execution_ids',[])
    native_visual_inventory=attempt in qualification_budget.get('native_visual_inventory_execution_ids',[])
    if native_visual_inventory and (not geometry_diagnostic or attempt not in qualification_budget.get('native_component_execution_ids',[])):
        raise ValueError('Visual inventory must retain original engineering-only admission')
    if native_visual_inventory:
        inventory_batch=continuation or json.loads((phase/'priority_recovery_authorization.json').read_text())['batches'][attempt]
        if inventory_batch.get('native_visual_inventory_only') is not True or inventory_batch.get('new_engineering_RGB_limit')!=0:
            raise ValueError('Native inventory has no new RGB authorization')
    if geometry_diagnostic:deadline=min(deadline,started+600)
    if mode=='material' and (phase/'material_budget.json').exists():
        from oct07_material_budget import deadline as material_deadline
        reserve=120 if attempt=='scene_material4_detailfix1_config1' else (240 if attempt=='scene_material4_complete7' else 360)
        deadline=min(deadline,material_deadline(phase,attempt)-reserve)
    state=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],campaign=contract['campaign_id'],
        run_class='scene_'+mode,attempt=attempt,stage='preparing',started_at=started,deadline=deadline,
        reporting_scope=scope,planned_material_frames=14 if mode=='material' else 0,planned_scenes=24 if mode=='qualification' else ((len(selected) if selected is not None else 384) if mode=='capture' else 0),
        completed_scenes=0,admitted_scenes=0,scientific_results=0)
    def save(path,value):
        path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
    mu=threading.RLock();stop=threading.Event();cancel=threading.Event()
    def update(**values):
        with mu:state.update(values,heartbeat=time.time());save(phase/(attempt+'_status.json'),dict(state))
    def beat():
        while not stop.wait(15):update()
    heartbeat=threading.Thread(target=beat,daemon=True);heartbeat.start()
    def interrupted(*a):
        cancel.set();raise TimeoutError('Exact owned shutdown signal')
    signal.signal(signal.SIGTERM,interrupted)
    from make_settings import settings
    for team,port in ((2,41451),(4,41452)):
        config=settings(team,port)
        correction=phase/'native_source_correction.json'
        if correction.exists():
            receipt=json.loads(correction.read_text())
            if receipt.get('frozen') is not True or receipt.get('profile')!='ue426_extended_EV100_camera_and_scene_v1':
                raise ValueError('Native source correction identity differs')
            from oct07_native_exposure_profile import apply
            config=apply(config)
        else:
            for cam in [config['CameraDefaults'],*[camera for v in config['Vehicles'].values() for camera in v['Cameras'].values()]]:
                for capture in cam['CaptureSettings']:
                    if capture['ImageType']==0:capture.update(TargetGamma=1.4,AutoExposureSpeed=100,AutoExposureBias=0,
                        AutoExposureMaxBrightness=.64,AutoExposureMinBrightness=.03)
        save(code/f'settings_{team}uav.json',config)
    snapshot=json.loads((code/'snapshot_manifest.json').read_text())
    import hashlib
    for team in (2,4):snapshot['files'][f'settings_{team}uav.json']=hashlib.sha256((code/f'settings_{team}uav.json').read_bytes()).hexdigest()
    save(code/'snapshot_manifest.json',snapshot)
    lib=root/'compat_glibc/lib/x86_64-linux-gnu'
    python=[str(lib/'ld-linux-x86-64.so.2'),'--library-path',':'.join((str(lib),str(root/'venv/lib'),str(root/'venv/lib/python3.10/site-packages/carla.libs'))),str(root/'venv/bin/python')]
    env=dict(os.environ,AIHCIR_BENCHMARK_ROOT=str(root))
    output_allowance=(1024**3 if mode=='material' else 6*1024**3)
    priority=qualification_budget.get('priority_recovery_allocation',{})
    if attempt in priority.get('execution_ids',[]):
        recovery=json.loads((phase/'priority_recovery_authorization.json').read_text())
        output_allowance=int(recovery['batches'][attempt].get('maximum_effective_source_bytes',output_allowance))
        if not 0<output_allowance<=6*1024**3:raise ValueError('Invalid bounded native sensor allowance')
    free=shutil.disk_usage(root).free
    if free<5*1024**3+output_allowance*2:raise ValueError('Native outputs and temporary preservation would breach storage floor')
    env['AIHCIR_NATIVE_MIN_FREE_BYTES']=str(max(5*1024**3,free-output_allowance))
    update(native_storage=dict(free_bytes=free,floor_bytes=5*1024**3,
        output_write_limit_bytes=output_allowance,temporary_archive_write_limit_bytes=output_allowance,peak_new_bytes=2*output_allowance))
    groups=[];inventory=json.loads((phase/'capture_inventory.json').read_text()) if (phase/'capture_inventory.json').exists() else {}
    qualified=set()
    if mode=='qualification' and not geometry_diagnostic:
        allocation=qualification_budget.get('explicit_completion_allocation',{})
        priority=qualification_budget.get('priority_recovery_allocation',{})
        if continuation:
            state['current_batch_ids']=continuation['scene_ids']
            replaced={}
            # Preserve complete rejected sources in their original folders;
            # only detach their old inventory references for targeted repair.
            for eid in continuation.get('replace_rejected',[]):
                if eid in inventory:replaced[eid]=inventory.pop(eid)
            if replaced:
                save(phase/(attempt+'_rejected_inventory_retained.json'),replaced)
                save(phase/'capture_inventory.json',inventory)
        elif attempt in priority.get('execution_ids',[]):
            recovery=json.loads((phase/'priority_recovery_authorization.json').read_text())
            batch=recovery['batches'][attempt]
            if batch.get('replace_rejected'):
                from oct07_qualification_resume import apply_plan
                state['replaced_rejected_capture_ids']=apply_plan(root,inventory,attempt)
                save(phase/'capture_inventory.json',inventory)
            state['current_batch_ids']=batch['scene_ids']
        elif allocation.get('execution_id')==attempt or attempt in allocation.get('diagnosed_repair_execution_ids',[]):
            from oct07_qualification_resume import apply_plan
            state['replaced_rejected_capture_ids']=apply_plan(root,inventory,attempt)
            save(phase/'capture_inventory.json',inventory)
        allowed=set(contract['scene_qualification']['qualification_scene_ids'])
        reused=[]
        for eid,item in inventory.items():
            if eid in allowed:
                verify_existing_capture(root,eid,item);reused.append(eid)
        state['completed_scenes']=len(reused)
        state['reused_completed_scene_ids']=sorted(reused)
    if mode=='capture':
        gate=json.loads((phase/'qualification_gate.json').read_text());qualified=set(gate['qualified'])
        if gate.get('passed') is not True or qualified!=set(contract['scene_qualification']['qualification_scene_ids']):raise ValueError('TRAIN24 qualification/reuse gate is closed')
        if hashlib.sha256((phase/'native_material_selection.json').read_bytes()).hexdigest()!=gate['material_selection_sha256']:raise ValueError('Qualified material changed')
        for name,h in gate['capture_code_sha256'].items():
            if hashlib.sha256((code/name).read_bytes()).hexdigest()!=h:raise ValueError('Qualified capture code changed')
        for eid,item in gate['qualified'].items():
            if eid not in inventory:raise ValueError('Native qualified capture inventory is missing')
            # Admission may run after transfer on another host; this native
            # inventory binds the original paths, not an assumed local mirror.
            record=Path(inventory[eid]['probe'])/'training_views'/eid/'observations.json'
            scene=root/'releases/oct07_scenes_rgb_v4/train'/eid/'scene.json'
            if hashlib.sha256(record.read_bytes()).hexdigest()!=inventory[eid]['observations_sha256'] or hashlib.sha256(scene.read_bytes()).hexdigest()!=inventory[eid]['source_sha256']:raise ValueError('Native qualified capture changed')
            for file,h in inventory[eid]['files_sha256'].items():
                if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=h:raise ValueError('Qualified native RGB/mask changed')
        state['completed_scenes']=len(qualified);state['admitted_scenes']=len(qualified)
        for eid,item in inventory.items():
            if eid not in qualified and (selected is None or eid in selected):
                verify_existing_capture(root,eid,item)
                state['completed_scenes']+=1
    if geometry_diagnostic:groups=[('geometry','Town01',2,[])]
    elif mode=='material':groups=[('material','Town01',2,[])]
    else:
        for split in ('train','validation','test'):
            specs=[json.loads(line) for line in (phase/'specifications'/(split+'.specs.jsonl')).read_text().splitlines()]
            if selected is not None:specs=[s for s in specs if s['episode_id'] in selected]
            if mode=='qualification':specs=[s for s in specs if split=='train' and s['episode_id'] in contract['scene_qualification']['qualification_scene_ids'] and s['episode_id'] not in inventory]
            if mode=='qualification' and state.get('current_batch_ids') is not None:
                specs=[s for s in specs if s['episode_id'] in state['current_batch_ids']]
            elif mode=='capture':specs=[s for s in specs if s['episode_id'] not in inventory]
            for map_name in sorted({s['map'] for s in specs}):
                for team in (2,4):
                    ids=[s['episode_id'] for s in specs if s['map']==map_name and s['team_size']==team]
                    if ids:groups.append((split,map_name,team,ids))
    def register_completed(folder,split,episodes):
        for eid in episodes:
            with mu:
                if eid in inventory:continue
            record=folder/'probe/training_views'/eid/'observations.json'
            if not record.is_file():continue
            try:data=json.loads(record.read_text())
            except json.JSONDecodeError:continue
            if data.get('stage')!='actual_views_collected_not_validated':continue
            entry=dict(probe=str(folder/'probe'),attempt=attempt,split=split,
                observations_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),source_sha256=data['source_sha256'])
            assets={str(record.parent/o['image']):o['sha256'] for o in data['observations']}
            for measures in data['private_pixel_measurements'].values():
                for m in measures.values():
                    assets[str(record.parent/m['visible_mask'])]=m['visible_mask_sha256']
                    assets[str(record.parent/m['isolated_candidate_mask'])]=m['isolated_mask_sha256']
            entry['files_sha256']=assets
            with mu:
                if eid in inventory:raise ValueError('Capture registration duplicated')
                inventory[eid]=entry;save(phase/'capture_inventory.json',inventory)
                state['completed_scenes']+=1;update(last_completed_episode=eid)
    def run_group(group):
        split,map_name,team,episodes=group
        if (phase/'PAUSE').exists() or (phase/'STOP').exists():raise ValueError('Existing native campaign pause/stop is preserved')
        folder=root/'sessions'/('scene_'+split+'_'+map_name+'_'+str(team)+'_'+attempt);folder.mkdir(exist_ok=False)
        port=41451 if team==2 else 41452
        carla_port=2000 if team==2 else 2100
        def group_update(**values):
            with mu:
                state.setdefault("groups",{}).setdefault(folder.name,{}).update(values,heartbeat=time.time())
                update(stage="capturing_actual_views",active_groups=sum(not g.get("ended_at") for g in state["groups"].values()))
        for check_port in (carla_port,carla_port+1,carla_port+2,port):
            try:
                with socket.create_connection(('127.0.0.1',check_port),timeout=.5):raise RuntimeError('Native API occupied; do not replace a live simulator')
            except (ConnectionRefusedError,TimeoutError):pass
        args=['runuser','-u',prepared['service_user'],'--','env','-u','DISPLAY','-u','WAYLAND_DISPLAY',
            'XDG_RUNTIME_DIR='+str(root/'service_home/xdg'),'VK_ICD_FILENAMES='+str(code/'nvidia_egl_icd.json'),
            prepared['binary'],'/Game/Carla/Maps/'+map_name,'-RenderOffScreen','-unattended','-nosound','-nosplash',
            '-stdout','-FullStdOutLogOutput','-ResX=640','-ResY=480','-quality-level=Epic','-carla-rpc-port='+str(carla_port),
            '-graphicsadapter=0','--settings',str(code/f'settings_{team}uav.json')]
        detail=phase/'native_detail_correction.json'
        if detail.exists():
            receipt=json.loads(detail.read_text())
            from oct07_native_detail_profile import PROFILE,launch_argument
            if receipt.get('frozen') is not True or receipt.get('profile')!=PROFILE:
                raise ValueError('Native detail correction identity differs')
            from oct07_qualification_budget import verify_source_freeze
            priority_source=verify_source_freeze(phase,attempt,code)
            effective_hashes=priority_source['code_sha256'] if priority_source else receipt['code_sha256']
            for name,h in effective_hashes.items():
                if hashlib.sha256((code/name).read_bytes()).hexdigest()!=h:
                    raise ValueError('Frozen detail scientific source changed')
            # Pinned SimHUD treats everything after --settings as its path.
            # Use the native RPC setter below, leaving the settings path last.
        with (folder/'simulator.log').open('xb') as log:
            sim=subprocess.Popen(args,cwd=prepared['runtime'],stdout=log,stderr=log,start_new_session=True)
            group_update(stage='native_start',simulator_pid=sim.pid,simulator_start_ticks=Path('/proc/'+str(sim.pid)+'/stat').read_text().split()[21],carla_port=carla_port,airsim_port=port)
            child=None
            try:
                inner_deadline=native_inner_deadline(deadline,native_visual_inventory)
                until=min(time.time()+600,inner_deadline)
                while True:
                    if cancel.is_set():raise TimeoutError('Owned controller cancellation')
                    if time.time()>=until:raise TimeoutError('Native startup deadline')
                    try:
                        with socket.create_connection(('127.0.0.1',carla_port),timeout=.5):pass
                        with socket.create_connection(('127.0.0.1',port),timeout=.5):pass
                        break
                    except (ConnectionRefusedError,TimeoutError):
                        if sim.poll() is not None:raise RuntimeError('Owned simulator exited; inspect its preserved log')
                        if time.time()>=until:raise TimeoutError('Native startup deadline')
                        time.sleep(2)
                if detail.exists():
                    from oct07_native_detail_profile import verify_startup_log
                    log_offset=(folder/'simulator.log').stat().st_size
                    command_receipt=json.loads(subprocess.check_output(
                        python+[str(code/'oct07_native_detail_profile.py'),'--port',str(port)],
                        env=env,stderr=subprocess.PIPE,timeout=min(25,max(.1,inner_deadline-time.time())) if native_visual_inventory else 25))
                    save(folder/'native_detail_transport.json',command_receipt)
                    detail_until=min(time.time()+30,inner_deadline)
                    while True:
                        text=(folder/'simulator.log').read_bytes()[log_offset:].decode(errors='replace')
                        try:
                            verified=verify_startup_log(text);break
                        except ValueError:
                            if time.time()>=detail_until:raise
                            time.sleep(1)
                    verified.update(simulator_pid=sim.pid,observed_at=time.time(),
                        simulator_log_sha256=hashlib.sha256(text.encode()).hexdigest())
                    save(folder/'native_detail_readback.json',verified)
                    group_update(detail_render_readback=verified)
                command=python+[str(code/'oct07_material_probe_v4.py'),'--output',str(folder/'probe'),'--port',str(port)] if mode=='material' else python+[str(code/'oct07_capture_probe.py'),'--port',str(port),'--carla-port',str(carla_port),'--agents',str(team),'--expected-map',map_name,'--split',split,'--output',str(folder/'probe'),'--episode-ids',','.join(episodes),'--collect-scenes',str(len(episodes)),'--construction-version','v4','--phase-directory','oct07_scene_repair']
                if geometry_diagnostic:command=python+[str(code/'oct07_native_geometry_diagnosis.py'),'--root',str(root),'--output',str(folder/'geometry_diagnosis.json'),'--expected-map',map_name]
                if attempt in qualification_budget.get('native_component_execution_ids',[]):
                    if not geometry_diagnostic:raise ValueError('Component metrology must remain outside benchmark acquisition')
                    command=python+[str(code/'oct07_native_component_probe.py'),'--root',str(root),'--output',str(folder/'component_metrology.json'),'--expected-map',map_name,'--carla-port',str(carla_port),'--port',str(port)]
                if native_visual_inventory:
                    command=python+[str(code/'oct07_native_visual_inventory.py'),'--root',str(root),'--output',str(folder/'component_metrology.json'),'--expected-map',map_name,'--carla-port',str(carla_port),'--measurement-deadline',str(inventory_measurement_deadline(deadline,time.time()))]
                if state.get('current_batch_ids') is not None and not geometry_diagnostic:
                    command+=['--native-bounds-diagnosis']
                if attempt in qualification_budget.get('template_only_execution_ids',[]):command+=['--templates-only']
                if mode=='material' and attempt=='scene_material4_complete7':
                    command+=['--resume-receipt',str(phase/'native_material_completion_protocol.json')]
                with (folder/'probe.log').open('xb') as stream:
                    child=subprocess.Popen(command,stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
                    group_update(stage='measuring_material' if mode=='material' else 'capturing_actual_views',child_pid=child.pid,child_start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21])
                    child_reserve=0 if native_visual_inventory else (30 if attempt=='scene_material4_detailfix1_config1' else (60 if attempt=='scene_material4_complete7' else 120))
                    while child.poll() is None:
                        register_completed(folder,split,episodes)
                        if cancel.is_set() or time.time()>=deadline-child_reserve:raise TimeoutError('Original native execution deadline')
                        time.sleep(2)
                    register_completed(folder,split,episodes)
                    if child.returncode:raise RuntimeError('Native scene child failed; inspect preserved evidence, no replay')
                register_completed(folder,split,episodes)
                if episodes and any(eid not in inventory for eid in episodes):raise ValueError('Group ended with incomplete or failed scenes; completed prefix is preserved')
                group_update(stage='group_complete_independent_admission_pending',ended_at=time.time())
            finally:
                for process in (child,sim):
                    if process is not None and process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(15)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(10)
                group_update(ended_at=time.time(),child_live=False,simulator_live=False)
    try:
        from concurrent.futures import ThreadPoolExecutor,as_completed
        workers=2 if mode in ('qualification','capture') and not geometry_diagnostic else 1
        from oct07_native_group_lanes import lanes
        grouped=lanes(groups) if workers==2 else [[g for g in groups]]
        def run_lane(lane):
            failures=[]
            for group in lane:
                # Only the admitted zero-RGB lane avoids a duplicate inner
                # reserve;240s outer preservation and the guardian stay intact.
                minimum_lane_seconds=60 if native_visual_inventory else 300
                if cancel.is_set() or time.time()>=deadline-minimum_lane_seconds:
                    failures.append(dict(group=list(group[:3]),error='Existing execution cutoff reached'));break
                try:run_group(group)
                except Exception as exc:
                    failures.append(dict(group=list(group[:3]),error=repr(exc)))
                    update(group_failures=failures)
            return failures
        errors=[]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures={executor.submit(run_lane,lane):lane for lane in grouped}
            try:
                for future in as_completed(futures):
                    try:errors.extend(future.result())
                    except Exception as exc:
                        errors.append(dict(groups=[list(g[:3]) for g in futures[future]],error=repr(exc)))
                        update(group_failures=errors)
            except BaseException:
                cancel.set();raise
        if errors:raise RuntimeError('Native groups failed; healthy independent groups were allowed to finish: '+repr(errors))
        update(stage='geometry_diagnosis_complete' if geometry_diagnostic else 'complete_independent_admission_pending',ended_at=time.time())
    except BaseException as exc:
        import traceback
        update(stage='failed',error_type=type(exc).__name__,traceback=traceback.format_exc(),ended_at=time.time());raise
    finally:stop.set();heartbeat.join(2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=('material','qualification','capture'),required=True);p.add_argument('--attempt',required=True);p.add_argument('--scope',choices=('parent','first','tonight'),default='parent');p.add_argument('--lease-deadline',type=float);a=p.parse_args()
    if not a.attempt.replace('_','').isalnum():raise ValueError('Unsafe unique attempt')
    main(a.root,a.mode,a.attempt,a.scope,a.lease_deadline)
