"""Export complete independently scored data for the original owner's paper review."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import time

from analyze_oct07_image_campaign import GROUP,METRICS,complete_case_values
from prepare_oct07_image_campaign import sha,write


def checked_outputs(root,contract,fixed):
    sources={};all_rows=[];comparisons=[]
    for split,key in [('development','development_rows'),('formal','formal_rows')]:
        names=[split+'_'+name+'.json' for name in ('independent_coverage','independent_scores','paired_statistics')]
        coverage,scores,paired=[json.loads((root/name).read_text(encoding='utf-8-sig')) for name in names]
        expected={r['task_id'] for r in fixed if r['run_class']==split}
        actual=scores['rows']
        if (coverage.get('complete') is not True or coverage.get('planned')!=contract['units'][key]
            or coverage.get('executed')!=len(expected) or len(actual)!=len(expected)
            or {r['task_id'] for r in actual}!=expected
            or scores.get('coverage')!=coverage or paired.get('coverage')!=coverage
            or coverage.get('immutable_matrix_sha256')!=sha(root/'matrix.csv')
            or paired.get('source_sha256')!=sha(root/(split+'_independent_scores.json'))):
            raise ValueError('Complete hash-bound fixed accounting is required; partial results stay partial')
        if (scores.get('scorer_sha256')!=sha(Path(__file__).with_name('oct07_independent_score.py'))
            or scores.get('analysis_code_sha256')!=sha(Path(__file__).with_name('analyze_oct07_image_campaign.py'))):
            raise ValueError('Independent scorer/statistical source version changed')
        statuses=Counter(r['status'] for r in actual)
        if (set(statuses)-{'completed','failed','invalid_output'}
            or statuses['completed']!=coverage['completed_valid'] or statuses['failed']!=coverage['failed']
            or statuses['invalid_output']!=coverage['invalid_output']):
            raise ValueError('Failure/invalid-output accounting differs')
        for row in actual:
            for metric in METRICS:
                value=row.get(metric)
                if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value)):
                    raise ValueError('Non-finite/non-numeric statistic')
            if row['status']!='completed' and any(row.get(m)!=0 for m in ('target_correct','relation_f1','joint_exact','utility')):
                raise ValueError('Failed scientific rows must retain zero quality and their original denominator')
        all_rows.extend(actual);comparisons.extend(paired['comparisons'])
        sources.update({name:sha(root/name) for name in names})
    return all_rows,comparisons,sources


def csv_output(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='raise');writer.writeheader();writer.writerows(rows)


def aggregate_cases(summary,case_metadata=None):
    output=[]
    for group,cases in sorted(summary.items()):
        buckets=defaultdict(dict)
        for case,value in cases.items():
            metadata=case_metadata[case] if case_metadata is not None else {}
            condition=(metadata['task_family'],metadata['visual_condition_requested'],metadata['team_size']) if metadata else ()
            buckets[condition][case]=value
        for condition,values in sorted(buckets.items()):
            row=dict(zip((*GROUP,'method'),group))
            if condition:row.update(zip(('task_family','visual_condition','team_size'),condition))
            row.update(n_cases=len(values),n_physical_layouts=len({v['physical_scene'] for v in values.values()}),
                n_tiles=len({v['cluster_tile'] for v in values.values() if v['cluster_tile'] is not None}),
                episode_rows=sum(v['n_episode_rows'] for v in values.values()))
            for metric in METRICS:
                measurements=[v[metric] for v in values.values()]
                row[metric]=sum(measurements)/len(measurements) if all(v is not None for v in measurements) else None
            row['role']='information_access_reference' if row['method']=='all_view' else 'frozen_budget_comparison'
            output.append(row)
    return output


def export(root):
    root=Path(root).resolve()
    from prepare_oct07_tonight_core import verify
    verify(root)
    from oct07_baseline_freeze import require_test_release,require_first_native_complete
    from oct07_fit_identity import verify_fits
    contract=json.loads((root/'contract.json').read_text(encoding='utf-8-sig'))
    require_test_release(root,'formal');require_first_native_complete(root,contract)
    fit_identity=verify_fits(root)
    fixed=list(csv.DictReader((root/'matrix.csv').open(encoding='utf-8-sig')))
    raw,comparisons,sources=checked_outputs(root,contract,fixed)
    summary=complete_case_values(raw,fixed)
    metadata={row['case_id']:row for split in ('validation','test') for row in
        map(json.loads,(root/'specifications'/f'{split}.specs.jsonl').read_text(encoding='utf-8').splitlines())}
    folder=root/'analysis_exports';folder.mkdir(exist_ok=True)
    columns=[*GROUP,'method','n_cases','n_physical_layouts','n_tiles','episode_rows',*METRICS,'role']
    csv_output(folder/'method_quality_cost.csv',aggregate_cases(summary),columns)
    csv_output(folder/'conditions_sensitivity_robustness.csv',aggregate_cases(summary,metadata),
        [*columns,'task_family','visual_condition','team_size'])
    paired=[]
    for value in comparisons:
        interval=value.get('ci95')
        paired.append(dict(**value['group'],method=value['method'],comparator=value['comparator'],metric=value['metric'],
            mean_difference=value['mean_difference'],ci_low=interval[0] if interval else None,
            ci_high=interval[1] if interval else None,n_paired_cases=value['n_paired_cases'],n_clusters=value['n_clusters'],
            cluster_unit=value.get('cluster_unit'),bootstrap_repetitions=value.get('bootstrap_repetitions'),
            comparison_role=value.get('comparison_role','matched_matrix_comparison')))
    csv_output(folder/'paired_effects.csv',paired,[*GROUP,'method','comparator','metric','mean_difference','ci_low','ci_high',
        'n_paired_cases','n_clusters','cluster_unit','bootstrap_repetitions','comparison_role'])
    accounting=[]
    counts=Counter((r['run_class'],r['kind'],r['method'],r['status']) for r in raw)
    for group,n in sorted(counts.items()):accounting.append(dict(zip(('run_class','kind','method','status'),group),rows=n))
    csv_output(folder/'all_status_accounting.csv',accounting,['run_class','kind','method','status','rows'])
    native=json.loads((root/'native_bbox_scores.json').read_text())
    expected_native=sum(4*metadata[r['case_id']]['team_size'] for r in metadata.values() if r['split']=='test')
    if (native['planned']!=expected_native or native['observed']!=native['planned'] or len(native['rows'])!=native['planned']
        or len({(r['case_id'],r['method'],r['view_id']) for r in native['rows']})!=native['planned']):
        raise ValueError('Actual native visual comparison is incomplete')
    csv_output(folder/'native_visual_frontends.csv',native['rows'],list(native['rows'][0]))
    sources.update({'contract.json':sha(root/'contract.json'),'matrix.csv':sha(root/'matrix.csv'),
        'native_bbox_scores.json':sha(root/'native_bbox_scores.json')})
    trace=[]
    for row in raw:
        if row['method'] not in ('trace_full','trace_no_task','trace_no_receiver','trace_no_visual_features','fifo'):continue
        if row['status']=='failed':
            trace.append(dict(task_id=row['task_id'],status='failed',failure_sha256=row['failure_sha256']));continue
        path=root/'predictions'/(row['task_id']+'.json')
        if sha(path)!=row['prediction_sha256']:raise ValueError('Raw process trace changed after independent scoring')
        value=json.loads(path.read_text())
        trace.append(dict(task_id=row['task_id'],case_id=row['case_id'],method=row['method'],status=row['status'],
            source_path=path.relative_to(root).as_posix(),sha256=sha(path),
            wire_events=len(value.get('wire_events',[])),mechanism_support='awaits_actual_packet_receiver_interpretation'))
    write(folder/'mechanism_trace_index.json',dict(rows=trace,case_selection_by_best_reward=False,
        negative_traces_retained=True,scientific_acceptance=False))
    outputs={path.name:sha(path) for path in folder.iterdir() if path.is_file() and path.name!='export_manifest.json'}
    manifest=dict(scope_id=contract['reporting_scope']['id'],observed_at=time.time(),sources_sha256=sources,
        exporter_sha256=sha(Path(__file__)),fit_checkpoint_sha256=fit_identity,outputs_sha256=outputs,
        algorithm_rows=len(raw),fixed_denominator=5124,all_valid_adverse_rows_retained=True,
        statistic_unit='Case averages over complete frozen seed/order rows; layout and tile clustered paired intervals',
        effect_direction_not_precommitted=True,scientific_acceptance=False,
        semantic_state='awaits_original_owner_interpretation_selection_and_reference_guided_figures',
        required_review='manuscript_evidence_selection_review.json')
    write(folder/'export_manifest.json',manifest);return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(export(parser.parse_args().root),ensure_ascii=False,indent=2))
