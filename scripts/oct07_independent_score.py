"""Private scoring only. Never imports an algorithm or changes its output."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

IDENTITIES={'target':'target','van':'reference','engine':'second_reference','marker':'direction_marker'}

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/union if union>0 else 0.

def private_identity(candidate,record,allowed_crops):
    if candidate is None:return None
    view=candidate['entity']['view_id'];box=candidate['entity']['bbox_xyxy']
    if view not in allowed_crops:return None
    # A received crop must actually include the claimed detector box; a view
    # ID alone never grants access to all other objects in that original RGB.
    if not any(c[0]<=box[0] and c[1]<=box[1] and c[2]>=box[2] and c[3]>=box[3] for c in allowed_crops[view]):return None
    scores=[]
    for key,measure in record['private_pixel_measurements'].get(view,{}).items():
        if measure.get('visible_pixels',0)<16 or measure.get('instance_mask_verified') is not True:continue
        if not all(measure.get(k) is True for k in ('visible_subset_of_isolated','visible_aba_exact','isolated_aba_exact')):continue
        b=measure['projected']['bbox_xyxy'];b=[max(0,b[0]),max(0,b[1]),min(640,b[2]),min(480,b[3])]
        scores.append((iou(box,b),key))
    scores.sort(reverse=True)
    if not scores or scores[0][0]<.5 or len(scores)>1 and scores[0][0]-scores[1][0]<=1e-6:return None
    return scores[0][1]

def score(prediction,bundle,record,scene,wire_events,agent='UAV_0',horizon=2,all_view=False,id_map=None):
    truth=scene['private_truth'];candidates={c['id']:c for c in bundle['candidates']};id_map=id_map or {k:k for k in candidates}
    allowed={}
    for v in bundle['views']:
        if v['round']<horizon and (v['agent']==agent or all_view):allowed.setdefault(v['id'],[]).append([0,0,640,480])
    actual_bytes=0
    for event in wire_events:
        h=event['header']
        if event['accepted']:actual_bytes+=event['wire_bytes']
        if event['delivered'] and h['kind']=='image' and h['recipient']==agent and h['round']<horizon:
            allowed.setdefault(h['source_view'],[]).append(h['crop_xyxy'])
    if wire_events and actual_bytes!=wire_events[-1]['total_bytes']:raise ValueError('Wire ledger sum mismatch')
    assignment=prediction.get('assignment',{});roles=list(assignment);grounded={}
    for role,cid in assignment.items():
        original=id_map.get(cid) if cid else None
        grounded[role]=private_identity(candidates.get(original),record,allowed)
    required={'target'}|{r[k] for r in truth['required_relations'] for k in ('subject','object')}
    needed=[r for r,key in IDENTITIES.items() if key in required]
    correct={r:grounded.get(r)==IDENTITIES[r] for r in needed}
    target=correct.get('target',False);direction=prediction.get('observed_direction')
    predicted_relations=set();true_relations={(e['subject'],e['relation'],e['object']) for e in truth['required_relations']}
    # Public query graph defines the demanded fixed relations. A latent value
    # must come from the method output and is never filled from private truth.
    instruction=scene['instruction']
    if 'matches' in instruction:
        edges=[('target','near','van'),('van',direction,'engine'),('marker',direction,'engine')]
    elif 'fire engine' in instruction:
        explicit=next((d for d in ('east','west','north','south') if d in instruction),None)
        nested='van that is' in instruction or 'van whose' in instruction
        edges=[('target','near','van'),('van' if nested else 'target',explicit,'engine')]
    else:
        explicit=next((d for d in ('east','west','north','south') if d in instruction),None)
        edges=[('target','near','van'),('target',explicit,'van')]
    invalid=0
    for a,p,b in edges:
        if grounded.get(a) is None or grounded.get(b) is None or p is None:invalid+=1
        else:predicted_relations.add((grounded[a],p,grounded[b]))
    hit=len(predicted_relations&true_relations);precision=hit/max(1,len(predicted_relations)+invalid);recall=hit/max(1,len(true_relations))
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    exact=bool(prediction.get('valid') and all(correct.values()) and not invalid and predicted_relations==true_relations)
    role_fraction=sum(correct.values())/len(needed)
    return dict(target_correct=target,relation_f1=f1,joint_exact=exact,role_fraction=role_fraction,
        utility=int(exact)+role_fraction/(len(needed)+1),grounded=grounded,wire_bytes=actual_bytes,
        predicted_relations=sorted(predicted_relations),true_relation_count=len(true_relations),allowed_crops=allowed)

def score_sealed(root,private_manifest,run_class=None):
    root=Path(root)
    from oct07_baseline_freeze import require_test_release
    require_test_release(root,run_class)
    manifest=json.loads(Path(private_manifest).read_text(encoding='utf-8'))
    public=json.loads((root/'data_manifest.json').read_text());cases={r['case_id']:r for r in public['cases']}
    import csv,sqlite3
    contract=json.loads((root/'contract.json').read_text())
    if hashlib.sha256((root/'matrix.csv').read_bytes()).hexdigest()!=contract['matrix_sha256']:raise ValueError('Fixed scoring matrix changed')
    with (root/'matrix.csv').open(newline='') as f:fixed={r['task_id']:r for r in csv.DictReader(f)}
    from oct07_runtime_scope import queue_path
    ledger=queue_path(root)
    db=sqlite3.connect('file:'+str(ledger.resolve())+'?mode=ro',uri=True) if ledger.exists() else None
    seals=sorted((root/'predictions').glob('*.json'));rows=[];seen=set()
    for path in seals:
        if run_class is not None and path.stem in fixed and fixed[path.stem]['run_class']!=run_class:continue
        result=json.loads(path.read_text());cid=result['case_id']
        task=result['task_id'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if task not in fixed or task in seen or path.stem!=task:raise ValueError('Unknown/duplicate prediction identity')
        if any(str(result.get(k))!=str(v) for k,v in fixed[task].items()):raise ValueError('Prediction configuration differs from fixed row')
        if db:
            seal=db.execute('SELECT result_sha256 FROM jobs WHERE id=?',(task,)).fetchone()
            if not seal or seal[0]!=digest:raise ValueError('Sealed raw prediction changed')
        seen.add(task)
        private=manifest[cid];pub=cases[cid]
        bundle=json.loads((root/pub['bundle']).read_text());scene=json.loads(Path(private['scene']).read_text());record=json.loads(Path(private['observations']).read_text())
        if hashlib.sha256(Path(private['scene']).read_bytes()).hexdigest()!=record['source_sha256']:raise ValueError('Private source mismatch')
        if hashlib.sha256((root/pub['bundle']).read_bytes()).hexdigest()!=result['bundle_sha256']:raise ValueError('Public source changed after prediction')
        if result['status']=='completed':
            metrics=score(result['prediction'],bundle,record,scene,result['wire_events'],horizon=result['horizon'],all_view=result['method']=='all_view',id_map=result['candidate_id_map'])
        else:metrics=dict(target_correct=False,relation_f1=0.,joint_exact=False,utility=0.,failure=True)
        rows.append(dict(fixed[task],status=result['status'],physical_scene=pub['physical_layout_sha256'],cluster_tile=pub.get('cluster_tile'),
            prediction_sha256=digest,seconds=result['seconds'],generated_samples=result['generated_samples'],
            input_tokens=result['input_tokens'],output_tokens=result['output_tokens'],source_pixels=result['source_pixels'],**metrics))
    for path in sorted((root/'failures').glob('*.json')):
        if run_class is not None and path.stem in fixed and fixed[path.stem]['run_class']!=run_class:continue
        result=json.loads(path.read_text());task=result['task_id']
        if task in seen or task not in fixed or path.stem!=task:raise ValueError('Duplicate/unknown failed row')
        seen.add(task);pub=cases.get(result['case_id'],{})
        rows.append(dict(fixed[task],status='failed',physical_scene=pub.get('physical_layout_sha256'),cluster_tile=pub.get('cluster_tile'),
            target_correct=False,relation_f1=0.,joint_exact=False,utility=0.,failure=True,
            seconds=None,generated_samples=None,input_tokens=None,output_tokens=None,source_pixels=None,wire_bytes=None,
            failure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),error_type=result['type']))
    if db:db.close()
    return rows
