"""Freeze the approved scene repair without replaying an existing experiment."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import prepare_oct07_image_campaign as original
from prepare_oct07_image_campaign import ROOT, sha, write

PREVIOUS = ROOT / 'evidence/oct07_image_campaign_20261003_v2'
OUT = ROOT / 'evidence/oct07_image_campaign_20261003_v3_scene_repair'

def specifications(contract):
    for index, (family, condition, team) in enumerate(original.CELLS):
        key = contract['campaign_id'] + '/validation/' + str(index)
        yield dict(episode_id=f'validation_{100000+index:06d}', case_id=f'validation_{100000+index:06d}',
                   split='validation', map='Town02', task_family=family, family=family,
                   visual_condition_requested=condition, condition=condition, team_size=team,
                   scene_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'),
                   recipe_key=hashlib.sha256(key.encode()).hexdigest(), replicate=0, stratum=index,
                   status='pending_scene_instantiation', agent_names=[f'UAV_{a}' for a in range(team)],
                   budgets=dict(decision_rounds=2, communication_bytes=None, per_agent_path_m=1200, simulated_seconds=600))

def prepare(out=OUT, previous=PREVIOUS):
    out, previous = Path(out), Path(previous)
    if (out/'contract.json').exists() or (out/'matrix.csv').exists(): raise FileExistsError('Inspect existing repair identity; never regenerate')
    old = json.loads((previous/'contract.json').read_text(encoding='utf-8'))
    status = json.loads((previous/'status.json').read_text(encoding='utf-8'))
    if status.get('running') or status.get('completed'): raise ValueError('Inspect the executed predecessor before changing a matrix')
    c = copy.deepcopy(old)
    c.update(campaign_id=out.name, schema='oct07_image_campaign_v3_scene_repair', created_at=time.time(),
             session=os.environ.get('CODEX_THREAD_ID'), predecessor_contract_sha256=sha(previous/'contract.json'))
    c['scene_repair_authorization'] = dict(source='User accepted full DEV24, baseline TEST sealed, and instructed Implement the plan.',
        old_failed_attempts_reset=False, color_thresholds_changed=False, algorithm_score_based_selection=False)
    c['data'].update(development_scenes=24, development_map='Town02', capture_scenes=384,
                     construction_version='oct07_two_round_geometry_rgb_v4', formal_model_results_opened=False)
    dev = list(specifications(c))
    cohorts = [(dev, 'development')]
    fields = ('task_id','run_class','kind','case_id','team_size','backbone','method','seed','presentation','cap_multiple','horizon','perturbation')
    counts, rows = {}, []
    saved = original.AUTHORS
    original.AUTHORS = tuple(c['sources']['author_methods'])
    try:
        for cases, run_class in cohorts:
            for case in cases:
                for kind in ('main','ablation','sensitivity','robustness','all_view_reference'):
                    for method, seed, cap, horizon, perturbation in original.configurations(kind,case['team_size']):
                        for backbone in original.BACKBONES:
                            for presentation in ('original','reverse'):
                                row=dict(run_class=run_class,kind=kind,case_id=case['case_id'],team_size=case['team_size'],
                                    backbone=backbone,method=method,seed=seed,presentation=presentation,
                                    cap_multiple=cap,horizon=horizon,perturbation=perturbation)
                                row['task_id']=hashlib.sha256((out.name+json.dumps(row,sort_keys=True,separators=(',',':'))).encode()).hexdigest()[:24]
                                rows.append(row);key=run_class+'/'+kind;counts[key]=counts.get(key,0)+1
    finally: original.AUTHORS = saved
    with (previous/'matrix.csv').open(encoding='utf-8',newline='') as f:
        for old_row in csv.DictReader(f):
            if old_row['run_class']!='formal':continue
            row={k:v for k,v in old_row.items() if k!='task_id'}
            row['task_id']=hashlib.sha256((out.name+json.dumps(row,sort_keys=True,separators=(',',':'))).encode()).hexdigest()[:24]
            rows.append(row);key='formal/'+row['kind'];counts[key]=counts.get(key,0)+1
    if len(rows)!=60816 or len({r['task_id'] for r in rows})!=60816: raise ValueError('Approved full matrix differs')
    c['units'].update(image_episode_rows=60816,development_rows=7728,formal_rows=53088,
        public_baseline_development=3312,public_baseline_formal=33120,native_bbox_frame_rows=3168)
    c['counts']=counts
    c['baseline_first'].update(public_rows=3312,baseline_stage_rows=4272,formal_independent_of_trace=True)
    c['test_seal']=dict(required=True,quality_access_before_release=False,
        release_requires=['algorithm_selection_frozen','all_formal_predictions_terminal'],
        operational_metadata_only=True)
    c['scene_qualification']=dict(task_id=out.name+'/scene_qualification1',material_frames_max=14,
        material_seconds_max=1800,qualification_scene_ids=[f'train_{100000+i:06d}' for i in range(24)],
        qualification_attempt_limit=1,qualification_seconds_max=10800,qualified_train_reused=True,
        after_pass='Continue remaining TRAIN96, DEV24 and TEST240 capture; then full public baseline stages')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'matrix.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    c['matrix_sha256']=sha(out/'matrix.csv')
    (out/'specifications').mkdir()
    for split in ('train','test'):
        os.link(previous/'specifications'/(split+'.specs.jsonl'),out/'specifications'/(split+'.specs.jsonl'))
    with (out/'specifications/validation.specs.jsonl').open('x',encoding='utf-8') as f:
        for r in dev:f.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
    for directory in ('author_sources','task_prompts'):
        for p in (previous/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                dst=out/directory/p.relative_to(previous/directory);dst.parent.mkdir(parents=True,exist_ok=True);os.link(p,dst)
    for name in ('author_source_registry.json','public_baseline_audit.json','manuscript_baseline_citations.json','retrieval_model_source.json'):
        if (previous/name).is_file():
            value=json.loads((previous/name).read_text(encoding='utf-8'));write(out/name,value,True)
    write(out/'contract.json',c,True)
    write(out/'status.json',dict(phase='scene_repair_implementation',observed_at=time.time(),total=60816,
        completed=0,percent=0,succeeded=0,failed=0,running=0,queued=0,not_submitted=60816,eta_seconds=None),True)
    write(out/'dependency_protocol.json',dict(stages=['scene_material','scene_qualification','all_capture','native_frontends',
        'train_budget_and_examples','new_six_interfaces','development_baselines','formal_baselines',
        'trace_development','selected_version','formal_trace','unseal_and_score'],
        public_formal_requires_trace=False,scientific_ready=False),True)
    return c

if __name__=='__main__':
    c=prepare();print(json.dumps(dict(campaign=c['campaign_id'],units=c['units'])))
