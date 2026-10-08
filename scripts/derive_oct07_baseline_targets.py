"""Derive evidence targets only after all twelve DEV baselines terminate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from oct07_independent_score import score_sealed
from prepare_oct07_image_campaign import write,sha
from run_oct07_image_campaign import matrix

def derive(root):
    root=Path(root);contract,matrix_rows=matrix(root);methods=set(contract['sources']['author_methods'])
    expected={r['task_id']:r for r in matrix_rows if r['run_class']=='development' and r['method'] in methods}
    private=root/'private_development_manifest.json';manifest=json.loads(private.read_text(encoding='utf-8'))
    if set(manifest)!={r['case_id'] for r in expected.values()}:raise ValueError('Only frozen DEV private observations may enter target derivation')
    rows=[r for r in score_sealed(root,private,run_class='development') if r['task_id'] in expected]
    if len(rows)!=contract['units']['public_baseline_development'] or set(r['task_id'] for r in rows)!=set(expected):raise ValueError('All frozen public DEV baseline rows are required; no partial target selection')
    groups=defaultdict(list)
    for r in rows:groups[(r['backbone'],int(r['team_size']),int(r['cap_multiple']),r['method'])].append(r)
    summaries=[]
    for (backbone,team,cap,method),values in sorted(groups.items()):
        # Average seeds/presentation within each physical layout before the
        # scene mean; method eligibility differs for two-agent CoELA.
        layouts=defaultdict(list)
        for r in values:
            if r.get('physical_scene') is None:raise ValueError('Physical layout identity missing; cannot fabricate independent sample units')
            layouts[r['physical_scene']].append(r)
        layout_means=[sum(float(r['joint_exact']) for r in v)/len(v) for v in layouts.values()]
        measured_times=[r['seconds'] for r in values if r.get('seconds') is not None]
        summaries.append(dict(backbone=backbone,team_size=team,cap_multiple=cap,method=method,
            joint_exact=sum(layout_means)/len(layout_means),independent_layouts=len(layouts),matrix_rows=len(values),
            failures=sum(r['status']!='completed' for r in values),
            measured_seconds_mean=sum(measured_times)/len(measured_times) if measured_times else None,
            timing_coverage=len(measured_times),training_exposure='100 standard TRAIN demonstrations' if method=='llm_planner' else 'No new task-specific fitting'))
    cohorts=defaultdict(list)
    for row in summaries:cohorts[(row['backbone'],row['team_size'],row['cap_multiple'])].append(row)
    targets=[dict(backbone=key[0],team_size=key[1],cap_multiple=key[2],
        applicable_methods=sorted(v['method'] for v in values),
        best_quality=max(v['joint_exact'] for v in values),
        tied_methods=sorted(v['method'] for v in values if v['joint_exact']==max(x['joint_exact'] for x in values))) for key,values in sorted(cohorts.items())]
    value=dict(selection_split='development',public_methods=12,baseline_rows_reviewed=len(rows),
        matrix_sha256=sha(root/'matrix.csv'),private_DEV_manifest_sha256=sha(private),summaries=summaries,targets=targets,
        observed_at=time.time(),final_test_results_opened=False,algorithm_selected=False,
        interpretation='Quality targets and measured costs support bounded mechanism diagnosis; no winner or significance claim from a mean alone')
    write(root/'baseline_targets.json',value,True);return dict(rows=len(rows),cohorts=len(targets))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(derive(a.root)))
