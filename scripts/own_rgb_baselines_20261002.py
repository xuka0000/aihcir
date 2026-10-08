"""Frozen RGB-conditioned public-proposal VLM baselines; no private labels on server.

This evaluates the complete retained development cohort. It is not a fresh
detector benchmark or the full communication-learning TRACE system.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/own_rgb_baselines_20261002_v1'
OLD = ROOT / 'evidence/own_bundle_campaign_20261002_v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def write(p, v, exclusive=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with p.open('x', encoding='utf-8') as f: json.dump(v, f, indent=2, allow_nan=False)
    else:
        t = p.with_name(p.name + '.tmp')
        t.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8'); os.replace(t, p)


def public_input(bundle, mode, presentation):
    assert set(bundle) == {'episode_id', 'instruction', 'cutoff_round', 'views', 'candidates'}
    views = [v for v in bundle['views'] if mode != 'local_rgb' or v['agent'] == 'UAV_0']
    available = {v['id'] for v in views}
    candidates = [c for c in bundle['candidates'] if c['entity']['view_id'] in available]
    # Both list position and opaque output ID are counterbalanced without labels.
    candidates = sorted(candidates, key=lambda c: c['id'])
    if presentation == 1: candidates = list(reversed(candidates))
    mapping = {f'c{i:03d}': c for i, c in enumerate(candidates)}
    roles = ['target', 'van'] + (['engine'] if 'fire engine' in bundle['instruction'] else [])
    options = [[k for k, c in mapping.items() if c['role'] == role] + ['none'] for role in roles]
    table = [dict(id=k, role=c['role'], view=c['entity']['view_id'],
                  box=[round(x, 2) for x in c['entity']['bbox_xyxy']],
                  east_north_m=[round(x, 3) for x in c['xy']],
                  proposal_score=round(c['score'], 5)) for k, c in mapping.items()]
    question = (
        'Ground the instruction in the aerial observations. Detector proposals may be false positives; '
        'different proposals can show the same object in different images. Use RGB appearance when supplied '
        'and the provided estimated world east/north coordinates for spatial relations. '
        'Do not equate image left/right with world east/west. Boxes use original 640x480 pixel coordinates.\n'
        'Instruction: ' + bundle['instruction'] + '\n'
        'Select one proposal ID for each role, or none if the available observations do not support that role. '
        'Return ONLY comma-separated IDs, in this exact role order: ' + ','.join(roles) + '.\n'
        'Candidates: ' + json.dumps(table, separators=(',', ':')))
    return views, mapping, roles, options, question


class RoleGrammar:
    """A concatenation of small token tries: no Cartesian assignment expansion."""
    def __init__(self, tokenizer, options, eos):
        self.eos = eos; self.parts = []; self.tokenizer = tokenizer
        for i, choices in enumerate(options):
            # The tokenizer merges ',c'. Prefix delimiters preserve that token;
            # suffix delimiters incorrectly forced ',' followed by a separate 'c'.
            prefix = ',' if i else ''
            sequences = []
            for value in choices:
                text = prefix + value
                ids = tokenizer.encode(text, add_special_tokens=False)
                assert ids and eos not in ids and tokenizer.decode(ids, skip_special_tokens=False) == text
                sequences.append(tuple(ids))
            assert len(set(sequences)) == len(sequences)
            # Each segment is prefix-free; allows exact stage identification.
            assert all(not (len(a) < len(b) and b[:len(a)] == a) for a in sequences for b in sequences)
            self.parts.append(sequences)
        self.max_tokens = sum(max(map(len, p)) for p in self.parts) + 1

    def allowed(self, suffix):
        suffix = tuple(suffix)
        for sequences in self.parts:
            completed = next((s for s in sequences if suffix[:len(s)] == s), None)
            if completed is not None:
                suffix = suffix[len(completed):]; continue
            values = sorted({s[len(suffix)] for s in sequences if len(suffix) < len(s) and s[:len(suffix)] == suffix})
            if not values: raise ValueError('Output outside role grammar')
            return values
        if suffix not in ((), (self.eos,)): raise ValueError('Trailing grammar tokens')
        return [self.eos]

    def callback(self, width):
        return lambda _, ids: self.allowed(ids.tolist()[width:])


def prepare():
    if (OUT/'contract.json').exists(): raise FileExistsError('Frozen campaign exists')
    OUT.mkdir(exist_ok=True)
    image_sources=read(OUT/'local_image_sources.json')
    old = read(OLD/'contract.json')
    cases = [c for c in old['cases'] if c['split'] == 'validation']
    assert len(cases) == 13
    manifests = []
    for c in cases:
        source = OLD/c['bundle']; assert sha(source) == c['bundle_sha256']
        b = read(source)
        dest = OUT/c['bundle']; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source, dest)
        for view in b['views']:
            if view['sha256'] not in image_sources['sources']: continue
            src = ROOT/image_sources['sources'][view['sha256']]; assert sha(src) == view['sha256']
            shutil.copyfile(src, dest.parent/view['id'])
        for mode, presentation in itertools.product(('local_rgb','all_rgb','all_metadata'), (0,1)):
            views, mapping, roles, options, _ = public_input(b, mode, presentation)
            assert len(views) in (4,8,16) and len(mapping) > 0
            assert all(c['entity']['view_id'] in {v['id'] for v in views} for c in mapping.values())
        manifests.extend(dict(path=p.relative_to(OUT).as_posix(), size=p.stat().st_size, sha256=sha(p))
                         for p in dest.parent.iterdir() if p.is_file())
    prior = read(ROOT/'evidence/oct07_experiment_attempt4_20260930_v1/batch_contract.json')
    jobs = [dict(model=model, case_id=c['case_id'], mode=mode, presentation=order)
            for model in prior['models'] for mode in ('local_rgb','all_rgb','all_metadata')
            for c in cases for order in (0,1)]
    for j in jobs: j['id'] = digest(j)[:24]
    for name in ('own_rgb_baselines_20261002.py','run_symmetric_full_campaign_20260930.py'):
        shutil.copyfile(ROOT/'scripts'/name, OUT/name)
    shutil.copyfile(ROOT/'evidence/non_gpu_maintenance_20260930_v1/processor_manifest.json', OUT/'processor_manifest.json')
    enabled_modes=['local_rgb','all_rgb','all_metadata'] if image_sources['found']==image_sources['requested'] else ['all_metadata']
    contract = dict(schema='own_rgb_baselines_v1', run_class='development', created=time.time(),
        deadline=read(OUT/'packet.json')['deadline'], planned_units=len(jobs), unit='model_mode_presentation_scene_rows',
        enabled_modes=enabled_modes,ready_units=sum(j['mode'] in enabled_modes for j in jobs),
        image_coverage=dict(verified=image_sources['found'],required=image_sources['requested'],missing=len(image_sources['missing'])),
        rgb_admission='Requires all168 original-image SHA256 matches; never shrink to one available scene',
        cases=cases, matrix=jobs, models=prior['models'], runtime_qualification=prior['runtime_qualification'],
        files=manifests, reserved_cny=15, parent_ceiling_cny=500, max_powered_seconds=7200,
        max_model_calls=sum(j['mode'] in enabled_modes for j in jobs), max_seconds_per_call=120, automatic_scientific_retries=0,
        seed=17, eos_id=151645, source_contract_sha256=sha(OLD/'contract.json'),
        authorizations=['Current user: resume experiments', 'Existing500CNY and Oct7 campaign', 'Use own retained dataset first'],
        hypothesis='RGB evidence improves public-proposal role assignment beyond estimated geometry and detector metadata; extra views resolve local omissions.',
        reference_experiment_design='evidence/r09_v12_adoption_20261002_v1/reference_experiment_design.json',
        reference_mapping=dict(inherit=['AerialVG grounding/relations and paired method comparison','Where2comm local versus shared observations'],
          adjust=['Complete retained13-scene development cohort, not original-scale replication',
                  'Pinned Qwen3 and InternVL publicly released VLMs with identical public proposals',
                  'Two candidate ID/list orders; no repeat-seed independence claim',
                  'Same raw640x480 source images; native model-specific preprocessing measured, not equated token costs']),
        scope='Frozen zero-shot VLM readout of detector proposals and actual RGB; full learned TRACE remains pending.',
        statistics='Average two presentations per scene; paired scene-level differences across13 scene contents; retain failures and negative results.',
        metrics=['target accuracy','exact required-role grounding','role-order sensitivity','RGB-minus-metadata paired change','input/output tokens','native visual tensor shapes','latency','peak CUDA memory'],
        full_matrix_context=dict(inherited_candidate_rows=715, inherited_fits=9,
            next_dependencies=['RGB baseline error analysis','Full TRACE image encoder and communication mechanism','Independent runtime-certified formal test']),
        prohibited=['Private labels in prompts/server package','Candidate filtering using truth','Resume977image transfer','Reuse closed budgets','Claim formal independent test'],
        checks=dict(reference_mapping=True, all_13_retained=True, private_data_in_public_package=False),
        code_hashes={n:sha(OUT/n) for n in ['own_rgb_baselines_20261002.py','run_symmetric_full_campaign_20260930.py','processor_manifest.json']})
    write(OUT/'contract.json',contract,True)
    write(OUT/'status.json',dict(phase='prepared',completed=0,total=len(jobs),running=0,failed=0,observed_at=time.time()))
    print(json.dumps(dict(planned_rows=len(jobs),ready=contract['ready_units'],scenes=len(cases),source_images=len(manifests)-13)))


def run(root):
    c = read(root/'contract.json'); binding = sha(root/'contract.json')
    assert c['planned_units'] == len(c['matrix']) == 156 and time.time() < c['deadline']
    assert not (root/'private').exists()
    lock=root/'runner.lock'
    write(lock,dict(pid=os.getpid(),started=time.time(),contract_sha256=binding),True)
    status = dict(phase='preflight',completed=0,total=156,ready_total=c['ready_units'],dependency_waiting=156-c['ready_units'],running=1,failed=0,heartbeat=time.time(),eta_seconds=None)
    mutex=threading.RLock();done=threading.Event();begin=time.time()
    def save(**kwargs):
        with mutex:
            status.update(kwargs,heartbeat=time.time(),observed_at=time.time());write(root/'status.json',status)
    def beat():
        while not done.wait(15): save()
    def check():
        if time.time() >= c['deadline'] or any((root/n).exists() for n in ('STOP','PAUSE')):raise TimeoutError('Campaign stop/deadline')
    observer=threading.Thread(target=beat,daemon=True);observer.start()
    completed=0
    try:
        for r in c['files']:
            p=root/r['path'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
        for p,h in c['code_hashes'].items():assert sha(root/p)==h
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor,AutoModelForImageTextToText,Qwen3VLForConditionalGeneration
        from run_symmetric_full_campaign_20260930 import verify_model_files,PROCESSOR_KWARGS
        assert transformers.__version__=='4.57.1' and torch.__version__.startswith('2.8.0') and torch.cuda.is_available()
        torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        write(root/'environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(),total_memory=torch.cuda.get_device_properties(0).total_memory,
            python=sys.version,platform=sys.platform),True)
        cases={v['case_id']:v for v in c['cases']}
        for key, model_spec in c['models'].items():
            check();save(phase='loading_model',model=key)
            verify_model_files(key,model_spec,read(root/'processor_manifest.json'))
            processor=AutoProcessor.from_pretrained(model_spec['directory'],local_files_only=True,trust_remote_code=False,
                **(dict(min_pixels=256*28*28,max_pixels=1024*28*28) if key=='qwen3' else {}))
            cls=Qwen3VLForConditionalGeneration if key=='qwen3' else AutoModelForImageTextToText
            model=cls.from_pretrained(model_spec['directory'],local_files_only=True,trust_remote_code=False,
                torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
            model.requires_grad_(False)
            for job in [j for j in c['matrix'] if j['model']==key and j['mode'] in c['enabled_modes']]:
                check();path=root/'predictions'/(job['id']+'.json')
                if path.exists():raise ValueError('Existing unreviewed attempt; no automatic replay')
                b=read(root/cases[job['case_id']]['bundle'])
                views,mapping,roles,options,question=public_input(b,job['mode'],job['presentation'])
                content=[dict(type='text',text=question)];images=[]
                if job['mode']!='all_metadata':
                    for v in views:
                        content.extend([dict(type='text',text='Image '+v['id']),dict(type='image')])
                        images.append(Image.open((root/cases[job['case_id']]['bundle']).parent/v['id']).convert('RGB'))
                prompt=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
                start=time.time();torch.cuda.reset_peak_memory_stats()
                inputs=processor(text=[prompt],**(dict(images=images) if images else {}),return_tensors='pt',**PROCESSOR_KWARGS[key])
                inputs={k:v.to('cuda:0') if hasattr(v,'to') else v for k,v in inputs.items()}
                if 'pixel_values' in inputs and key=='internvl35_hf':inputs['pixel_values']=inputs['pixel_values'].to(torch.bfloat16)
                width=int(inputs['input_ids'].shape[1]);grammar=RoleGrammar(processor.tokenizer,options,c['eos_id'])
                meta=dict(job,contract_sha256=binding,roles=roles)
                write(root/'attempts'/(job['id']+'.json'),dict(meta,started=start,call_number=completed+1),True)
                save(phase='running',current=job,completed=completed,actual_model_calls=completed+1)
                with torch.inference_mode():
                    output=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=grammar.max_tokens,
                        max_time=min(c['max_seconds_per_call'],c['deadline']-time.time()),
                        eos_token_id=c['eos_id'],pad_token_id=processor.tokenizer.pad_token_id,
                        prefix_allowed_tokens_fn=grammar.callback(width))
                torch.cuda.synchronize();suffix=output[0,width:].tolist()
                text=processor.tokenizer.decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                fields=text.split(',');valid=len(fields)==len(roles) and all(v in opts for v,opts in zip(fields,options)) and bool(suffix) and suffix[-1]==c['eos_id']
                assignment={role:(mapping[v]['id'] if v!='none' else None) for role,v in zip(roles,fields)} if valid else {}
                row=dict(meta,output_text=text,output_ids=suffix,grammar_valid=valid,assignment=assignment,
                    prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_ids=inputs['input_ids'][0].tolist(),
                    model_revision=model_spec['revision'],selected_view_ids=[v['id'] for v in views],
                    image_hashes={v['id']:v['sha256'] for v in views} if images else {},
                    presented_candidate_mapping={k:v['id'] for k,v in mapping.items()},
                    input_tokens=width,output_tokens=len(suffix),image_count=len(images),raw_pixel_count=len(images)*640*480,
                    tensor_shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},
                    processor_kwargs=PROCESSOR_KWARGS[key],peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    seconds=time.time()-start,finished=time.time(),status='completed' if valid else 'invalid_output')
                write(path,row,True);completed+=1
                save(completed=completed,failed=status['failed']+int(not valid),eta_seconds=(time.time()-begin)/completed*(c['ready_units']-completed),eta_scope='Ready52metadata rows only; full RGB ETA unknown')
                print(json.dumps(dict(completed=completed,total=156,model=key,mode=job['mode'],valid=valid,seconds=row['seconds'])),flush=True)
                for im in images:im.close()
                del inputs,output
            del model,processor;torch.cuda.empty_cache()
        write(root/'prediction_seal.json',dict(contract_sha256=binding,files={p.name:sha(p) for p in sorted((root/'predictions').glob('*.json'))}),True)
        write(root/'complete.json',dict(completed=completed,total=156,ready_total=c['ready_units'],dependency_waiting=156-c['ready_units'],invalid_output=status['failed'],at=time.time(),full_matrix_complete=completed==156),True)
        save(phase='completed' if completed==156 else 'ready_matrix_complete_rgb_dependency_wait',running=0,eta_seconds=0)
    except BaseException as exc:
        write(root/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),at=time.time()),True)
        save(phase='failed',running=0,error_type=type(exc).__name__);raise
    finally:
        done.set();observer.join(2)


def exact_required_roles(grounded, required):
    # Public linguistic role names differ from stable private object identities.
    # This mapping belongs only to scoring; it never changes model inputs.
    role_for_identity = {'target': 'target', 'reference': 'van', 'second_reference': 'engine'}
    needed = {'target'} | {edge[k] for edge in required for k in ('subject', 'object')}
    if not needed <= role_for_identity.keys():
        raise ValueError('Undefined private identity in required relation')
    return all(grounded.get(role_for_identity[obj]) == obj for obj in needed)


def score(root):
    c=read(root/'contract.json');seal=read(root/'prediction_seal.json')
    assert sha(root/'contract.json')==seal['contract_sha256'] and len(seal['files'])==c['ready_units']
    sys.path.insert(0,str(OLD/'code'))
    from independent_metric import ground
    private={r['case_id']:r for r in read(OLD/'private_manifest.json')}
    cases={r['case_id']:r for r in c['cases']};rows=[]
    for job in c['matrix']:
        if job['mode'] not in c['enabled_modes']:continue
        p=root/'predictions'/(job['id']+'.json');assert sha(p)==seal['files'][p.name]
        r=read(p);assert r['contract_sha256']==seal['contract_sha256']
        b=read(root/cases[job['case_id']]['bundle']);cand={x['id']:x for x in b['candidates']}
        pr=private[job['case_id']];obs=read(OLD/pr['observations']);scene=read(OLD/pr['scene'])
        assert sha(OLD/pr['observations'])==pr['observations_sha256'] and sha(OLD/pr['scene'])==pr['scene_sha256']
        allowed=set(r['selected_view_ids']);a=r['assignment']
        grounded={role:ground(cand[cid]['entity'],obs,allowed) if cid else None for role,cid in a.items()}
        target=grounded.get('target')=='target'
        required=scene['private_truth']['required_relations']
        exact=bool(r['grammar_valid'] and exact_required_roles(grounded, required))
        rows.append(dict(job,target_correct=target,exact_role_success=exact,grounded_assignment=grounded,
            grammar_valid=r['grammar_valid'],seconds=r['seconds'],input_tokens=r['input_tokens'],output_tokens=r['output_tokens'],
            image_count=r['image_count'],raw_pixel_count=r['raw_pixel_count'],peak_cuda_bytes=r['peak_cuda_bytes']))
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['mode'])].append(r)
    summary=[]
    for (model,mode),group in groups.items():
        summary.append(dict(model=model,mode=mode,n_rows=len(group),n_scenes=13,
            target_pct=sum(x['target_correct'] for x in group)/len(group)*100,
            exact_role_pct=sum(x['exact_role_success'] for x in group)/len(group)*100,
            invalid_outputs=sum(not x['grammar_valid'] for x in group),
            mean_seconds=sum(x['seconds'] for x in group)/len(group),
            mean_input_tokens=sum(x['input_tokens'] for x in group)/len(group),
            mean_images=sum(x['image_count'] for x in group)/len(group)))
    paired=[]
    if len(c['enabled_modes'])==3:
        for model in c['models']:
            for case in cases:
                d={mode:sum(r['exact_role_success'] for r in rows if r['model']==model and r['case_id']==case and r['mode']==mode)/2 for mode in ['local_rgb','all_rgb','all_metadata']}
                paired.append(dict(model=model,case_id=case,**d,rgb_minus_metadata=d['all_rgb']-d['all_metadata'],all_minus_local=d['all_rgb']-d['local_rgb']))
    write(root/'results_rolemap_v2.json',dict(at=time.time(),completed=len(rows),total=156,summary=summary,rows=rows,paired_by_scene=paired,
        metric_version='private-identity-to-public-role-v2',scoring_code_sha256=sha(Path(__file__)),
        supersedes='results.json; original exact-role lookup used incompatible naming; raw predictions unchanged',
        interpretation='Exact role assignments for the fixed query relation graph. Relation existence is supplied by the query; no free relation discovery score.',
        statistical_unit='13 previously inspected scene contents; two ID orders averaged within scene',formal_test_rows=0,
        dependency_waiting=156-c['ready_units'],original_rows_repeated=0,source_contract_sha256=c['source_contract_sha256']),True)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','score']);p.add_argument('--root',type=Path,default=OUT);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='run':run(a.root)
    else:score(a.root)
