"""Bind bounded native material measurements and explicit original-RGB review."""
import argparse
import json
from pathlib import Path
import time
from prepare_oct07_image_campaign import sha,write
from oct07_material_probe_v4 import CANDIDATES

def freeze(root,measurement,review):
    root=Path(root);measurement=Path(measurement);review=Path(review)
    data=json.loads(measurement.read_text());assessment=json.loads(review.read_text())
    if data.get('stage')!='measured' or data.get('completed_frames')!=14 or data.get('planned_frames')!=14:raise ValueError('Bounded fourteen-frame native diagnostic is incomplete')
    if not all(data.get(k) is True for k in ('native_uav_camera','exact_actor_mask')):raise ValueError('Measurement is not the actual UAV/mask interface')
    rows=data['attempts'];passing=[]
    plan={(model,color,z) for model in CANDIDATES for color in ('red','blue','white') for z in (26.,70.)}|{('vehicle.mercedes.sprinter','yellow',z) for z in (26.,70.)}
    if len(rows)!=14 or {(r['blueprint'],r['color'],r['height']) for r in rows}!=plan:raise ValueError('Native diagnostic coverage changed')
    for model in CANDIDATES:
        if all(r.get('strict_majority_passed') is True for r in rows if r['blueprint']==model):passing.append(model)
    if not passing or not all(r.get('strict_majority_passed') is True for r in rows if r['color']=='yellow'):raise ValueError('No native material passes unchanged cues')
    selected=passing[0];chosen=[r for r in rows if r['blueprint']==selected or r['color']=='yellow'];files={}
    if assessment.get('recognizable_required_cues') is not True or assessment.get('measurement_sha256')!=sha(measurement):raise ValueError('Original RGB review is absent or stale')
    if set(assessment.get('reviewed_rgb_sha256',[]))!={r['sha256'] for r in chosen}:raise ValueError('Review must cover all six candidate and two yellow-reference frames')
    for r in chosen:
        if r.get('status')!='measured' or r.get('visible_actor_pixels',0)<16 or 2*r.get('cue_actor_pixels',0)<=r['visible_actor_pixels']:raise ValueError('Frozen strict-majority/16-pixel rule failed')
        m=r['measurement']
        if not all(m.get(k) is True for k in ('visible_aba_exact','isolated_aba_exact','instance_mask_verified','visible_subset_of_isolated')):raise ValueError('Exact instance masks are unverified')
        for name,digest in [(r['file'],r['sha256']),(m['visible_mask'],m['visible_mask_sha256']),(m['isolated_candidate_mask'],m['isolated_mask_sha256'])]:
            p=(measurement.parent/name).resolve()
            if measurement.parent.resolve() not in p.parents or sha(p)!=digest:raise ValueError('Native raw asset changed')
            files[str(p)]=digest
    value=dict(candidate_blueprint=selected,native_rgb_review_passed=True,measurement_sha256=sha(measurement),
        measurement=str(measurement.resolve()),review=str(review.resolve()),review_sha256=sha(review),
        files_sha256=files,first_passing_candidate=True,actor_mask_rule='strict majority, at least16 pixels',
        benchmark_admitted=False,algorithm_scores_used=False,user_acceptance=False,observed_at=time.time())
    write(root/'native_material_selection.json',value,True);return value

def verify(root):
    root=Path(root);s=json.loads((root/'native_material_selection.json').read_text(encoding='utf-8'))
    if s.get('native_rgb_review_passed') is not True or s.get('candidate_blueprint') not in CANDIDATES:raise ValueError('Native selection is not qualified')
    if sha(Path(s['measurement']))!=s['measurement_sha256'] or sha(Path(s['review']))!=s['review_sha256']:raise ValueError('Material evidence changed')
    for p,digest in s['files_sha256'].items():
        if sha(Path(p))!=digest:raise ValueError('Reviewed native material asset changed')
    return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--measurement',type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
    print(json.dumps(freeze(a.root,a.measurement,a.review) if a.measurement and a.review else verify(a.root)))
