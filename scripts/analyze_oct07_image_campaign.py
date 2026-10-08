"""Independent matrix coverage and paired physical-scene/tile statistics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from oct07_independent_score import score_sealed
from prepare_oct07_image_campaign import write,sha

GROUP=('run_class','kind','backbone','cap_multiple','horizon','perturbation')
METRICS=('target_correct','relation_f1','joint_exact','utility','wire_bytes','generated_samples','source_pixels','seconds')

def complete_case_values(rows,fixed):
    """Average every frozen seed/order row within a case, preserving failed rows."""
    expected=defaultdict(set)
    for p in fixed:expected[tuple(str(p[k]) for k in GROUP)+(p['method'],p['case_id'])].add(p['task_id'])
    averaged=defaultdict(lambda:defaultdict(list))
    for r in rows:averaged[tuple(str(r[k]) for k in GROUP)+(r['method'],)][r['case_id']].append(r)
    summary={}
    for group,cases in averaged.items():
        summary[group]={}
        for case,items in cases.items():
            if expected[group+(case,)]!={r['task_id'] for r in items}:continue
            value=dict(physical_scene=items[0]['physical_scene'],cluster_tile=items[0]['cluster_tile'],n_episode_rows=len(items))
            for m in METRICS:
                vals=[r.get(m) for r in items];value[m]=sum(float(v) for v in vals)/len(vals) if all(v is not None for v in vals) else None
            summary[group][case]=value
    return summary

def paired_ci(a,b,metric,cluster_key,seed=17,repetitions=5000):
    clusters=defaultdict(list);common=sorted(set(a)&set(b))
    for case in common:
        if a[case].get(metric) is None or b[case].get(metric) is None:continue
        cluster=a[case].get(cluster_key)
        if cluster is None:continue
        clusters[cluster].append(float(a[case][metric])-float(b[case][metric]))
    if not clusters:return dict(mean_difference=None,ci95=None,n_paired_cases=0,n_clusters=0)
    arrays=[np.array(clusters[k],dtype=float) for k in sorted(clusters)];rng=np.random.default_rng(seed);means=[]
    for _ in range(repetitions):
        sample=rng.integers(0,len(arrays),size=len(arrays));means.append(float(np.concatenate([arrays[i] for i in sample]).mean()))
    return dict(mean_difference=float(np.concatenate(arrays).mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        n_paired_cases=sum(map(len,arrays)),n_clusters=len(arrays),bootstrap_repetitions=repetitions,
        cluster_unit=cluster_key,seed=seed,algorithm='Resample clusters with replacement, retain all paired cases per sampled cluster; percentile CI')

def analyze(root,private_manifest,run_class=None):
    rows=score_sealed(root,private_manifest,run_class=run_class)
    with (root/'matrix.csv').open(newline='') as f:fixed=list(csv.DictReader(f))
    contract=json.loads((root/'contract.json').read_text(encoding='utf-8'))
    if len(fixed)!=contract['units']['image_episode_rows'] or sha(root/'matrix.csv')!=contract['matrix_sha256']:raise ValueError('Statistical denominator or frozen matrix differs')
    if run_class is not None:fixed=[r for r in fixed if r['run_class']==run_class]
    planned=len(fixed)
    index={r['task_id']:r for r in rows};counts=Counter(r['status'] for r in rows)
    coverage=dict(planned=planned,executed=len(rows),completed_valid=counts['completed'],invalid_output=counts['invalid_output'],
        failed=counts['failed'],not_executed=planned-len(rows),complete=len(rows)==planned,
        immutable_matrix_sha256=sha(root/'matrix.csv'),comparison_acceptance=False)
    summary=complete_case_values(rows,fixed)
    comparisons=[]
    for group,cases in summary.items():
        if group[-1]!='trace_full':continue
        for method in ('fifo','repeat_vote',*contract['sources']['author_methods'],'trace_no_task','trace_no_receiver','trace_no_visual_features','no_comm','all_view'):
            peer_group=list(group[:-1])
            if method.startswith('trace_no_'):
                if group[1]!='main':continue
                peer_group[1]='ablation'
            if method=='all_view':
                if group[1]!='main':continue
                peer_group[1]='all_view_reference'
            peer=summary.get(tuple(peer_group)+(method,))
            if peer is None:continue
            for metric in METRICS:
                for unit in ('physical_scene','cluster_tile'):
                    comparisons.append(dict(group=dict(zip(GROUP,group)),method='trace_full',comparator=method,metric=metric,
                        comparison_role='information_access_reference' if method=='all_view' else 'matched_matrix_comparison',
                        **paired_ci(cases,peer,metric,unit)))
    prefix=(run_class+'_') if run_class else ''
    write(root/(prefix+'independent_coverage.json'),coverage)
    write(root/(prefix+'independent_scores.json'),dict(rows=rows,coverage=coverage,scorer_sha256=sha(Path(__file__).with_name('oct07_independent_score.py')),
        analysis_code_sha256=sha(Path(__file__))))
    write(root/(prefix+'paired_statistics.json'),dict(comparisons=comparisons,coverage=coverage,
        source_sha256=sha(root/(prefix+'independent_scores.json')),complete_matrix_required_for_final_claim=True))
    return coverage

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--private-manifest',type=Path,required=True);p.add_argument('--run-class',choices=('development','formal','all'),default='development');a=p.parse_args();print(json.dumps(analyze(a.root,a.private_manifest,None if a.run_class=='all' else a.run_class)))
