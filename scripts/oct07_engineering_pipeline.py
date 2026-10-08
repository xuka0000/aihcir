"""Bounded real TRAIN engineering outputs, isolated from scientific admission.

Uses the existing public detector, author controllers and independent scorer.
No fitted TRACE, TRAIN120 gate, official demonstration or formal result is made.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from prepare_oct07_image_campaign import sha, write

CASE_IDS=('train_100000','train_100001','train_100006','train_100019')


def materialize(source,target,digest=None):
    source=Path(source);target=Path(target)
    # Linked destinations verify the same physical bytes, so their final hash
    # also verifies the source. Independent copies keep a final source hash.
    stamp=source.stat();source_digest=sha(source)
    if digest and source_digest!=digest:raise ValueError('Preserved source hash changed: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        try:os.link(source,target)
        except OSError:shutil.copyfile(source,target)
    # Creating a hard link legitimately changes ctime/link count. Content,
    # size, mtime and identity still have to agree; verify the linked target.
    linked=source.samefile(target);after_link=source.stat()
    before=(stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns)
    if before!=(after_link.st_dev,after_link.st_ino,after_link.st_size,after_link.st_mtime_ns) or (not linked and stamp.st_ctime_ns!=after_link.st_ctime_ns):
        raise ValueError('Preserved source changed while creating destination')
    stamp=after_link
    if sha(target)!=source_digest:raise ValueError('Engineering destination differs from preserved source')
    if not linked and sha(source)!=source_digest:
        raise ValueError('Preserved source changed during copy verification')
    current=source.stat()
    if (stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns,stamp.st_ctime_ns)!=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns):
        raise ValueError('Preserved source changed during materialization')


def prepare(root):
    from assemble_oct07_public_data import OBS
    from oct07_runtime_scope import verify_runtime
    root=Path(root).resolve();verify_runtime(root);parent=root.parent
    admitted=json.loads((parent/'qualification_admission.json').read_text(encoding='utf-8'))
    rows={r['case_id']:r for r in admitted['rows'] if r.get('passed') is True}
    if not set(CASE_IDS)<=set(rows):raise ValueError('Exact preserved TRAIN sources not independently qualified')
    private=json.loads((parent/'private_qualification_manifest.json').read_text(encoding='utf-8'))
    out=root/'engineering_actual_train';out.mkdir(exist_ok=True)
    c=copy.deepcopy(json.loads((root/'contract.json').read_text(encoding='utf-8')))
    c.pop('reporting_scope',None)
    c['campaign_id']+='/engineering-only';c['scientific_admission']=False
    # Runtime access accounting uses the unchanged original32/1024 call budget.
    c['engineering_only']=True;c['engineering_units_limit']=24
    write(out/'contract.json',c)
    public=[];private_index={};sources={}
    for eid in CASE_IDS:
        item=private[eid];scene=Path(item['scene']);record=Path(item['observations'])
        if sha(scene)!=item['scene_sha256'] or sha(record)!=item['observations_sha256']:
            raise ValueError('Qualified scene/observations changed')
        actual=json.loads(record.read_text(encoding='utf-8'))
        if actual.get('stage')!='actual_views_collected_not_validated':raise ValueError('Preserved observations incomplete')
        observations=[];folder=out/'public_rgb'/eid
        for obs in actual['observations']:
            if not OBS<=set(obs):raise ValueError('Public observation fields missing')
            o={k:obs[k] for k in OBS};observations.append(o)
            materialize(record.parent/o['image'],folder/o['image'],o['sha256'])
        expected=2*rows[eid]['team_size']
        if len(observations)!=expected:raise ValueError('TRAIN two-round frame coverage differs')
        path=folder/'public_observations.json'
        write(path,dict(episode_id=eid,instruction=actual['instruction'],observations=observations))
        s=json.loads(scene.read_text(encoding='utf-8'))
        public.append(dict(case_id=eid,episode_id=eid,team_size=rows[eid]['team_size'],split='train',
            family=rows[eid]['task_family'],condition=rows[eid]['visual_condition_requested'],
            run_class='engineering',public_observations=path.relative_to(out).as_posix(),
            public_manifest_sha256=sha(path),physical_layout_sha256=s['geometry_sha256'],
            cluster_tile=s.get('construction_tile')))
        # Private truth is staged separately and read ONLY by the score command.
        sp=out/'private_source'/eid/'scene.json';rp=sp.parent/'observations.json'
        materialize(scene,sp,item['scene_sha256']);materialize(record,rp,item['observations_sha256'])
        private_index[eid]=dict(scene=sp.relative_to(out).as_posix(),observations=rp.relative_to(out).as_posix(),
            scene_sha256=sha(sp),observations_sha256=sha(rp))
        sources[eid]=dict(scene_sha256=sha(scene),observations_sha256=sha(record),
            qualification_admission_sha256=sha(parent/'qualification_admission.json'))
    for name in ('author_source_registry.json','processor_manifest.json','feature_encoder.json',
                 'retrieval_model_source.json','native_assets_manifest.json'):
        materialize(root/name,out/name)
    for name in ('author_sources','task_prompts'):
        for path in (root/name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                materialize(path,out/path.relative_to(root))
    write(out/'public_new_capture_manifest.json',dict(cases=public,private_labels_present=False,
        engineering_only=True,TRAIN120_admission=False))
    write(out/'engineering_private_manifest.json',private_index)
    receipt=dict(observed_at=time.time(),case_ids=list(CASE_IDS),frames=sum(2*r['team_size'] for r in public),
        sources=sources,engineering_only=True,formal_algorithm_rows=0,scientific_acceptance=False,
        public_manifest_sha256=sha(out/'public_new_capture_manifest.json'),
        next_command='execute_oct07_admitted_pipeline.py --stage engineering',
        dependency='Actual pinned GroundingDINO public proposals, native assets and GPU operators')
    write(out/'engineering_input_receipt.json',receipt)
    return receipt


def schedule(contract):
    methods=[m for m in contract['sources']['author_methods'] if m!='llm_planner']
    if len(methods)!=11:raise ValueError('The original twelve-method registry changed')
    jobs=[(eid,method,'flow') for eid in CASE_IDS for method in ('react','fifo','no_comm')]
    jobs += [(CASE_IDS[0],method,'author_interface') for method in methods]
    jobs += [(CASE_IDS[1],'all_view','memory_reference')]
    if len(jobs)!=24:raise ValueError('Engineering scope differs')
    return jobs


def frozen_rgb_inputs(out,public,data):
    """Bind proposal-side paths to exact qualified public TRAIN image hashes."""
    expected={}
    for case in public:
        obs_path=out/case['public_observations']
        if sha(obs_path)!=case['public_manifest_sha256']:raise ValueError('Frozen public RGB manifest changed')
        declared={obs['image']:obs['sha256'] for obs in json.loads(obs_path.read_text(encoding='utf-8'))['observations']}
        bundle_path=out/data[case['case_id']]['bundle']
        bundle=json.loads(bundle_path.read_text(encoding='utf-8'))
        views={v['id']:v['sha256'] for v in bundle['views']}
        if len(views)!=len(bundle['views']) or views!=declared:
            raise ValueError('Proposal RGB differs from the original qualified TRAIN frames')
        for name,digest in views.items():expected[bundle_path.parent/name]=digest
    if len(expected)!=24:raise ValueError('Original 24 TRAIN frames changed')
    return expected


def run(root,deadline):
    import fcntl
    from assemble_oct07_public_data import assemble_proposals
    from oct07_image_core import FrozenRGBCache,NativeVLM,Packet,PublicEpisode
    from run_oct07_image_campaign import run_row,episode_timeout
    root=Path(root).resolve();out=root/'engineering_actual_train'
    c=json.loads((out/'contract.json').read_text(encoding='utf-8'))
    if c.get('engineering_only') is not True:raise ValueError('Only isolated engineering inputs accepted')
    deadline=min(deadline,datetime.fromisoformat(c['compute_stop']).timestamp())
    if deadline-time.time()<300:raise TimeoutError('No protected time for real model work and preservation')
    lock=(out/'engineering.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    public=json.loads((out/'public_new_capture_manifest.json').read_text(encoding='utf-8'))['cases']
    if [r['case_id'] for r in public]!=list(CASE_IDS) or any(r['split']!='train' for r in public):
        raise ValueError('Engineering selection changed or TEST entered')
    # Never manufacture candidates from private masks or annotations.
    assemble_proposals(out,public)
    data={r['case_id']:r for r in json.loads((out/'data_manifest.json').read_text(encoding='utf-8'))['cases']}
    rgb_cache=FrozenRGBCache(frozen_rgb_inputs(out,public,data));vlm=None
    try:
        sizes=[]
        for case in data.values():
            path=out/case['bundle'];ep=PublicEpisode(json.loads(path.read_text()),path.parent,2,'original','none',17,rgb_cache=rgb_cache)
            for candidate in ep.candidates:
                view=ep.views[ep.candidates[candidate]['entity']['view_id']]
                sender=view['agent'];recipient=next(a for a in ep.agents if a!=sender)
                sizes.append(len(Packet.proposal(ep,sender,recipient,candidate,view['round']).wire))
        if not sizes:raise ValueError('Actual common frontend returned no packets; diagnose without private replacements')
        b0=4*max(sizes) # Engineering-only cap; NOT the formal TRAIN120 P95 freeze.
        for name in ('calls','predictions','failures'):(out/name).mkdir(exist_ok=True)
        processor=json.loads((out/'processor_manifest.json').read_text(encoding='utf-8'))
        vlm=NativeVLM('qwen3',c['models']['qwen3'],processor);status=[]
        for index,(eid,method,purpose) in enumerate(schedule(c)):
            task=f'engineering_{index:02d}_{eid}_{method}'
            if (out/'predictions'/(task+'.json')).exists() or (out/'failures'/(task+'.json')).exists():
                raise ValueError('Unreviewed engineering attempt already exists; no replay')
            if time.time()+30>=deadline:break
            row=dict(task_id=task,case_id=eid,run_class='engineering',kind=purpose,
                team_size=data[eid]['team_size'],backbone='qwen3',method=method,seed=17,
                presentation='original',cap_multiple=1,horizon=2,perturbation='none')
            try:
                with episode_timeout(min(180,deadline-time.time()-30)):
                    result=run_row(out,row,vlm,data,b0,rgb_cache=rgb_cache)
                write(out/'predictions'/(task+'.json'),result,True)
                status.append(dict(task_id=task,executed=True,status=result['status'],
                    generated_samples=result['generated_samples'],source_pixels=result['source_pixels'],
                    seconds=result['seconds'],sha256=sha(out/'predictions'/(task+'.json'))))
            except BaseException as exc:
                write(out/'failures'/(task+'.json'),dict(row,error_type=type(exc).__name__,
                    traceback=traceback.format_exc(),observed_at=time.time()),True)
                status.append(dict(task_id=task,executed=True,status='failed',error_type=type(exc).__name__))
                # An implementation/runtime fault closes this branch; wrong
                # predictions remain measured output and do not trigger tuning.
                if not isinstance(exc,TimeoutError):raise
            finally:
                write(out/'engineering_status.json',dict(observed_at=time.time(),planned=24,
                    executed=len(status),rows=status,formal_algorithm_rows=0,scientific_acceptance=False))
        complete=len(status)==24
        rgb_verification=rgb_cache.verify()
        write(out/('engineering_execution_complete.json' if complete else 'engineering_execution_partial.json'),dict(observed_at=time.time(),planned=24,
            executed=len(status),rows=status,model_revision=c['models']['qwen3']['revision'],
            engineering_packet_cap_bytes=b0,formal_TRAIN_B0=False,formal_algorithm_rows=0,
            rgb_verification=rgb_verification,
            complete=complete,termination='all_original_units_terminal' if complete else 'protected_compute_boundary'))
    finally:
        rgb_cache.close();del vlm;lock.close()


def score(root):
    # This command is a separate process from model/controller execution.
    from oct07_independent_score import score as independent_score
    out=Path(root).resolve()/'engineering_actual_train'
    complete_path=out/'engineering_execution_complete.json'
    partial_path=out/'engineering_execution_partial.json'
    if complete_path.exists() and partial_path.exists():raise ValueError('Ambiguous engineering attempt; inspect preserved identity')
    status=json.loads((complete_path if complete_path.exists() else partial_path).read_text(encoding='utf-8'))
    if status['planned']!=24 or status['executed']!=len(status['rows']) or not 0<=status['executed']<=24:
        raise ValueError('Engineering fixed denominator changed')
    contract=json.loads((out/'contract.json').read_text(encoding='utf-8'))
    expected=[f'engineering_{i:02d}_{eid}_{method}' for i,(eid,method,_) in enumerate(schedule(contract))]
    if [r['task_id'] for r in status['rows']]!=expected[:status['executed']]:
        raise ValueError('Engineering original task order/identity changed')
    private=json.loads((out/'engineering_private_manifest.json').read_text(encoding='utf-8'))
    data={r['case_id']:r for r in json.loads((out/'data_manifest.json').read_text(encoding='utf-8'))['cases']}
    # New decoded-cache batches retain an independently rechecked RGB boundary;
    # legacy manifests remain readable under their original source identity.
    if 'rgb_verification' in status:
        proof=status['rgb_verification']
        if proof.get('entry_and_completion_verified') is not True or proof.get('files')!=24:
            raise ValueError('Engineering RGB completion verification is absent')
        public=json.loads((out/'public_new_capture_manifest.json').read_text(encoding='utf-8'))['cases']
        for image,digest in frozen_rgb_inputs(out,public,data).items():
            if sha(image)!=digest:raise ValueError('Engineering RGB changed before independent score')
    scored=[]
    for row in status['rows']:
        path=out/'predictions'/(row['task_id']+'.json')
        if not path.exists():scored.append(dict(row,quality=None,reason='Preserved engineering execution fault'));continue
        if sha(path)!=row['sha256']:raise ValueError('Engineering prediction changed before independent score')
        p=json.loads(path.read_text(encoding='utf-8'));case=data[p['case_id']];item=private[p['case_id']]
        if (sha(out/case['bundle'])!=p['bundle_sha256'] or
                sha(out/'contract.json')!=p['contract_sha256']):
            raise ValueError('Engineering public source/contract changed before score')
        for key in ('scene','observations'):
            if sha(out/item[key])!=item[key+'_sha256']:raise ValueError('Private score source changed')
        result=independent_score(p['prediction'],json.loads((out/case['bundle']).read_text(encoding='utf-8')),
            json.loads((out/item['observations']).read_text(encoding='utf-8')),
            json.loads((out/item['scene']).read_text(encoding='utf-8')),p['wire_events'],
            horizon=p['horizon'],all_view=p['method']=='all_view',id_map=p['candidate_id_map'])
        if result['wire_bytes']!=p['communication_bytes']:raise ValueError('Independent wire sum differs')
        scored.append(dict(row,independent_score=result))
    passed=(status['executed']==24 and all(r.get('generated_samples',0)>0 and r.get('source_pixels',0)>0 for r in status['rows']))
    report=dict(observed_at=time.time(),planned=24,executed=status['executed'],rows=scored,
        real_RGB_and_scoring_chain_verified=passed,method_performance_supported=False,
        full_twelve_author_source_gate_passed=False,TRAIN120_admission=False,TRACEfits=0,
        formal_algorithm_rows=0,scientific_acceptance=False)
    write(out/'engineering_independent_review.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=('prepare','run','score'),required=True);p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(a.root)))
    elif a.mode=='score':print(json.dumps(score(a.root)))
    else:
        if a.deadline is None:p.error('--deadline is required for model execution')
        run(a.root,a.deadline)
