"""Independent native target-box scoring; never creates relation predictions."""
import argparse
import json
import math
from pathlib import Path
import time
from prepare_oct07_image_campaign import sha,write
from oct07_independent_score import iou

def evaluate(root,private_manifest,manifest_name='public_capture_manifest.json',split=None):
    root=Path(root);public=json.loads((root/manifest_name).read_text())
    cases=[c for c in public['cases'] if split is None or c['split']==split];planned=0
    if any(c['split']=='test' for c in cases):
        from oct07_baseline_freeze import require_test_release
        require_test_release(root,'formal')
    private=json.loads(Path(private_manifest).read_text())
    seals=json.loads((root/'native_output_seals.json').read_text()) if (root/'native_output_seals.json').exists() else {}
    rows=[]
    for c in cases:
        cid=c['case_id'];truth=private[cid];record=json.loads(Path(truth['observations']).read_text())
        if sha(Path(truth['scene']))!=record['source_sha256'] or sha(Path(truth['observations']))!=truth['observations_sha256']:raise ValueError('Private bbox source changed')
        obs=json.loads((root/c['public_observations']).read_text())['observations']
        planned+=2*len(obs)
        for method in ('groundingdino','aerialvg'):
            for view in obs:
                p=root/'native_bbox'/method/cid/(view['image']+'.json')
                if not p.exists():continue
                raw=json.loads(p.read_text());key=p.relative_to(root).as_posix()
                if not seals or seals.get(key,seals.get('.sealed_test/'+key))!=sha(p):raise ValueError('Native output lacks an unchanged completion seal')
                if raw['image_sha256']!=view['sha256'] or raw['case_id']!=cid or raw['view_id']!=view['image']:raise ValueError('Native output/image identity changed')
                target=record['private_pixel_measurements'][view['image']]['target'];expected=target['projected']['bbox_xyxy'];pred=raw['selected_bbox']
                visible=target.get('visible_pixels',0)>=16 and all(target.get(k) is True for k in ('visible_aba_exact','isolated_aba_exact','visible_subset_of_isolated'))
                valid=pred is not None and len(pred)==4 and all(math.isfinite(v) for v in pred) and pred[2]>pred[0] and pred[3]>pred[1]
                overlap=iou(pred,expected) if valid and expected else 0.
                rows.append(dict(case_id=cid,method=method,view_id=view['image'],visible_target=visible,bbox_iou=overlap,
                    target_correct=bool(visible and overlap>=.5),valid_prediction=bool(valid),prediction_sha256=sha(p),
                    seconds=raw['seconds'],source_pixels=raw['source_pixels'],peak_cuda_bytes=raw['peak_cuda_bytes']))
    write(root/'native_bbox_scores.json',dict(planned=planned,observed=len(rows),rows=rows,
        evaluation_unit='method_by_captured_frame; invisible targets retained and separately flagged',
        reference_box='private projected simulator bbox, not human AerialVG annotation',relation_metric=None,observed_at=time.time()))
    return dict(observed=len(rows),planned=planned)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--private-manifest',type=Path,required=True);p.add_argument('--manifest',default='public_capture_manifest.json');p.add_argument('--split',choices=('train','test','validation'));a=p.parse_args();print(json.dumps(evaluate(a.root,a.private_manifest,a.manifest,a.split)))
