"""Separate constrained DEV runner. Default: CPU processors, no weight loading.

Inference is deliberately unavailable in this build; a new explicit resource
contract and reviewed launch implementation are required. Originals untouched.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/symmetric_runner_20260929_v1'
QUAL = ROOT/'evidence/output_grammar_cpu_20260929_v1'
TASK = 'multimodal-runner-builder-20260929'
ASSETS = ROOT/'evidence/baseline_first_20260928_v1/assets'
from qualify_output_grammar_cpu_20260929 import SuffixTrie, verify

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def report(n,step,failed=0):
    p=OUT/('progress_%d.json'%n)
    write(p,dict(total=3,completed=n,succeeded=n-failed,failed=failed,running=int(n<3),pending=max(0,2-n),accepted=0,unit='bounded deliverable milestone'))
    r=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/team_runtime.py'),'report','--task',TASK,'--progress',str(p),'--current-step',step],capture_output=True,encoding='utf-8')
    (OUT/('report_%d_%d.log'%(n,time.time_ns()))).write_text(r.stdout+r.stderr,encoding='utf-8')
    return r.returncode==0 and 'queued' not in r.stdout

def verify_qualification():
    q=read(QUAL/'qualification.json')
    for name,expected in q['evidence'].items():
        if sha(QUAL/name)!=expected: raise ValueError('Frozen qualification changed: '+name)
    if sha(ROOT/'scripts/qualify_output_grammar_cpu_20260929.py')!=q['script_sha256']: raise ValueError('Grammar source changed')
    protocol=read(QUAL/'protocol.json')
    for item in protocol['model_sources'].values():
        for source in item['files'].values():
            if sha(ROOT/source['path'])!=source['sha256']: raise ValueError('Pinned source changed')
    return protocol

def prompt_for(episode,processor):
    if set(episode)!={'schema','id','question','options','views'}: raise ValueError('Unexpected private input')
    question=episode['question']+'\n'+'\n'.join(k+'. '+v for k,v in episode['options'].items())
    question+="\nAnswer with the option's letter from the given choices directly."
    content=[dict(type='text',text=question)]+[dict(type='image') for _ in episode['views']]
    return processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)

def generate_one(model,inputs,processor,grammar,metadata,deadline,max_seconds=120,*,launch_contract=None):
    """Inject already loaded model only after caller's launch gate; single episode."""
    launch_gate(launch_contract or {})
    if deadline!=launch_contract['absolute_deadline_unix']: raise ValueError('Call deadline must equal authorized absolute deadline')
    binding_fields=['prompt_sha256','source_hashes','image_hashes','cohort_sha256','model_revision']
    if any(not metadata.get(k) for k in binding_fields): raise ValueError('Prompt, source, image, cohort and model revision bindings required')
    calls=launch_contract.get('actual_model_calls',0)
    if calls>=launch_contract['new_budget']['max_model_calls']: raise ValueError('New call budget exhausted')
    started=time.monotonic(); remaining=deadline-time.time()
    if remaining<=0: raise TimeoutError('Absolute inference deadline expired')
    width=int(inputs['input_ids'].shape[1])
    if int(inputs['input_ids'].shape[0])!=1: raise ValueError('One episode per call required')
    trie=SuffixTrie(grammar['answer_suffix_ids'].values(),grammar['eos_id'])
    cap=max(len(x) for x in trie.sequences)+1
    kwargs=dict(do_sample=False,num_beams=1,max_new_tokens=cap,max_time=min(max_seconds,remaining),eos_token_id=grammar['eos_id'],pad_token_id=processor.tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=False)
    launch_contract['actual_model_calls']=calls+1
    output=model.generate(**inputs,**kwargs,prefix_allowed_tokens_fn=trie.callback(width))
    raw=output.sequences[0].tolist()
    prefix=inputs['input_ids'][0].tolist()
    if raw[:width]!=prefix: raise ValueError('Generated prefix changed')
    suffix=raw[width:]; elapsed=time.monotonic()-started
    text=processor.tokenizer.decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    accepted=trie.accepts(suffix) and text in 'ABCD' and len(text)==1
    row=dict(metadata,raw_output_ids=raw,input_ids=prefix,generated_suffix_ids=suffix,input_width=width,response=text,parsed_answer=text if accepted else None,grammar_valid=accepted,eos_observed=bool(suffix and suffix[-1]==grammar['eos_id']),token_cap_reached=len(suffix)>=cap,time_limit_observed=elapsed>=kwargs['max_time'],finish_reason='eos' if suffix and suffix[-1]==grammar['eos_id'] else 'unverified_non_eos_stop',generation_config=kwargs,generation_seconds=elapsed,observed_unix=time.time())
    # Flag, retain and abort invalid output; a cap may coincide with valid EOS.
    row['status']='completed' if accepted else 'failed'
    return row

def launch_gate(contract):
    required=['new_budget','absolute_deadline_unix','launch_authorized','launch_id']
    if any(not contract.get(k) for k in required): raise ValueError('NEW budget, absolute deadline and launch authorization required')
    b=contract['new_budget']
    if not isinstance(b,dict) or not all(isinstance(b.get(k),(int,float)) and not isinstance(b[k],bool) and b[k]>0 for k in ['max_model_calls','max_gpu_seconds','max_cost']): raise ValueError('Positive explicit new budget required')
    if contract['launch_authorized'] is not True or contract['absolute_deadline_unix']<=time.time(): raise ValueError('Launch gate closed')
    if contract.get('reuses_closed_contract',True): raise ValueError('Closed contract cannot be reused')
    if contract.get('execution_class')!='fake_generate_model_interface_only':
        runtime=contract.get('runtime_qualification',{})
        if runtime.get('transformers_version')!='4.57.1' or not str(runtime.get('torch_version','')).startswith('2.8.0') or runtime.get('both_exact_processors_validated') is not True or runtime.get('weights_hashes_verified') is not True:
            raise ValueError('Exact pinned processors, Torch2.8 runtime and verified weights gate closed')
    return True

def prepare_sources(fetch=False,keys=None):
    protocol=verify_qualification(); receipts=read(OUT/'fetch_receipts.json') if (OUT/'fetch_receipts.json').exists() else []
    deadline=read(ROOT/('.codex/team_runtime/'+TASK+'.json'))['deadline']
    for key,model in protocol['model_sources'].items():
        if keys and key not in keys: continue
        directory=OUT/'sources'/key; directory.mkdir(parents=True,exist_ok=True)
        metadata=read(ROOT/model['official_metadata_source']); entries={r['rfilename']:r for r in metadata['siblings']}
        for name,item in model['files'].items():
            target=directory/name
            if not target.exists(): shutil.copyfile(ROOT/item['path'],target)
            verify(target,entries[name])
        for name in ['config.json','preprocessor_config.json','processor_config.json','video_preprocessor_config.json']:
            if name not in entries: continue
            target=directory/name
            if target.exists(): verify(target,entries[name]); continue
            retained=ROOT/'evidence/internvl_dev231_20260928_v1/sources'/name
            if key=='internvl35_hf' and retained.exists():
                verify(retained,entries[name]);shutil.copyfile(retained,target);continue
            if not fetch: raise FileNotFoundError('CPU source missing: '+str(target))
            for attempt in range(sum(r['model']==key and r['file']==name for r in receipts),2):
                if time.time()+31>deadline: raise TimeoutError('Task fetch deadline')
                began=time.monotonic(); url='https://huggingface.co/'+model['repository']+'/resolve/'+model['revision']+'/'+name
                if attempt==1:
                    from urllib.parse import urlencode
                    mirror_path='evidence/internvl_dev231_20260928_v1/sources/modelscope_metadata.json' if key=='internvl35_hf' else 'evidence/qwen3_backbone_20260918_v1/modelscope_metadata.json'
                    mirror={r['Path']:r for r in read(ROOT/mirror_path)['Data']['Files']}[name]
                    url='https://modelscope.cn/api/v1/models/'+model['repository']+'/repo?'+urlencode(dict(Revision=mirror['Revision'],FilePath=name))
                partial=directory/(name+'.partial')
                try:
                    result=subprocess.run(['curl.exe','--fail','--location','--silent','--show-error','--connect-timeout','8','--max-time','30','--max-filesize','15000000','--output',str(partial),url],capture_output=True,timeout=30)
                    if result.returncode: raise RuntimeError('curl exit '+str(result.returncode))
                    verified=verify(partial,entries[name]); partial.replace(target)
                    row=dict(status='verified',**verified)
                except Exception as e: row=dict(status='failed',error_type=type(e).__name__,error=str(e))
                receipts.append(dict(model=key,file=name,attempt=attempt+1,elapsed_seconds=time.monotonic()-began,**row));write(OUT/'fetch_receipts.json',receipts)
                if target.exists(): break
            if not target.exists(): raise RuntimeError('Bounded attempts exhausted '+key+'/'+name)
    return protocol

def cpu_check(fetch=False,keys=None,cpu_compat=False,native_qwen=False):
    protocol=verify_qualification() if native_qwen else prepare_sources(fetch,keys)
    dep=OUT/'dependencies'
    if dep.exists(): sys.path.insert(0,str(dep))
    import transformers
    if transformers.__version__!='4.57.1': raise ValueError('Transformers4.57.1 required; installed '+transformers.__version__)
    from transformers import AutoProcessor
    import torch
    if cpu_compat:
        import inspect
        function=torch._dynamo.is_compiling
        if torch.__version__!='2.2.1+cpu' or function() is not False or hasattr(torch.compiler,'is_compiling'): raise ValueError('Unexpected compatibility environment')
        receipt=dict(scope='CPU eager image preprocessing only, never inference or compile',torch_version=torch.__version__,aliased_function='torch._dynamo.is_compiling',actual_function_source=inspect.getsource(function),source_file=inspect.getfile(function),source_file_sha256=sha(inspect.getfile(function)),reason='Native Transformers4.57.1 fast resize requires newer compiler API',exact_Torch28_equivalence=None,processor_mocked=False,observed_unix=time.time())
        torch.compiler.is_compiling=function
        write(OUT/'cpu_compatibility_alias.json',receipt)
    from PIL import Image
    episodes=read(ASSETS/'public_episodes.json');manifest=read(ASSETS/'image_manifest.json')
    if len(episodes)!=231 or [e['id'] for e in episodes]!=manifest['episode_ids']: raise ValueError('Frozen cohort mismatch')
    index={r['path']:r for r in manifest['files']};results={}
    for key in protocol['model_sources']:
        if keys and key not in keys: continue
        started=time.monotonic();directory=OUT/'sources'/key
        options=dict(min_pixels=256*28*28,max_pixels=1024*28*28) if key=='qwen3' else {}
        if native_qwen:
            if key!='qwen3': raise ValueError('Native default proof scoped to Qwen')
            from transformers import AutoTokenizer,Qwen3VLProcessor,Qwen2VLImageProcessor,Qwen3VLVideoProcessor
            tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False)
            processor=Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(**options),tokenizer=tokenizer,video_processor=Qwen3VLVideoProcessor(),chat_template=read(directory/'chat_template.json')['chat_template'])
        else: processor=AutoProcessor.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False,**options)
        factory=dict(status='actual_native_default_processor' if native_qwen else 'actual_AutoProcessor_instantiated',processor_class=type(processor).__name__,image_processor_class=type(processor.image_processor).__name__,image_processor_config=processor.image_processor.to_dict(),factory_options=options,version=transformers.__version__,source_hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file()},expanded_inputs_validated=None,cpu_compatibility_alias=cpu_compat,exact_pinned_AutoProcessor=not native_qwen,missing_pinned_config=native_qwen,observed_unix=time.time())
        if native_qwen: factory['default_config_gap']='Native Qwen3VLProcessor uses actual Qwen2VLImageProcessor class (Transformers auto mapping); image defaults including patch_size=14 are NOT verified against pinned Qwen model config. Video defaults from native Qwen3VLVideoProcessor. This proves callback boundary plumbing only and remains an inference launch blocker.'
        write(OUT/(key+'_processor_factory.json'),factory)
        rows=[]
        # Both retained 3-view and 5-view DEV layouts, one episode each.
        selected=[episodes[0],next(e for e in episodes if len(e['views'])==5)]
        for episode in selected:
            images=[];image_bindings=[]
            for view in episode['views']:
                path=ASSETS/'images'/view['path'];item=index[view['path']]
                if sha(path)!=item['sha256']: raise ValueError('Retained image changed')
                image=Image.open(path).convert('RGB')
                if list(image.size)!=[item['width'],item['height']]: raise ValueError('Image dimensions changed')
                images.append(image);image_bindings.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            prompt=prompt_for(episode,processor);began=time.monotonic()
            kwargs=dict(padding=True) if key=='qwen3' else dict(crop_to_patches=True,min_patches=1,max_patches=3)
            attempt=dict(episode_id=episode['id'],prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),images=image_bindings,processor_kwargs=kwargs,status='processing',expanded_inputs_validated=None,observed_unix=time.time())
            try: inputs=processor(text=[prompt],images=images,return_tensors='pt',**kwargs)
            except Exception as error:
                attempt.update(status='failed',error_type=type(error).__name__,error=str(error),processor_seconds=time.monotonic()-began)
                write(OUT/(key+'_processor_attempt_'+episode['id']+'.json'),attempt)
                raise
            width=int(inputs['input_ids'].shape[1]);text_width=len(processor.tokenizer.encode(prompt,add_special_tokens=False))
            if width<=text_width: raise ValueError('Actual image expansion absent')
            grammar=protocol['grammar_parameters'][key];trie=SuffixTrie(grammar['answer_suffix_ids'].values(),grammar['eos_id']);callback=trie.callback(width)
            ids=inputs['input_ids'][0].tolist()
            if callback(0,ids)!=[32,33,34,35] or grammar['eos_id'] in callback(0,ids): raise ValueError('Expanded boundary callback failed')
            for letter,suffix in grammar['answer_suffix_ids'].items():
                if processor.tokenizer.decode(suffix)!=letter or callback(0,ids+suffix)!=[151645]: raise ValueError('Suffix mismatch')
            rows.append(dict(episode_id=episode['id'],prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),images=image_bindings,input_ids=ids,input_width=width,text_only_width=text_width,tensor_shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},tensor_devices={k:str(v.device) for k,v in inputs.items() if hasattr(v,'device')},processor_seconds=time.monotonic()-began,processor_kwargs=kwargs,callback_verified=True))
        results[key]=dict(status='scoped_cpu_expansion_passed',processor_factory=factory,processor_class=type(processor).__name__,rows=rows,seconds=time.monotonic()-started,source_hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file()},version=transformers.__version__,exact_historical_runtime_equivalence=None)
        write(OUT/(key+'_cpu_expansion.json'),results[key])
    return results

def freeze():
    protocol=verify_qualification()
    contract=dict(schema='blocked_symmetric_dev_contract_v1',run_class='development_constrained_separate',planned_units=462,unit='model-question generate call',models=protocol['model_sources'],questions_per_model=231,new_budget=None,absolute_deadline_unix=None,launch_authorized=False,launch_id=None,reuses_closed_contract=False,launch_status='blocked',optional_smoke=dict(planned_units=4,unit='model-question call',max_executions=1,accounting='separate from 462; never scientific evidence',executed=0),cohort_sha256=sha(ASSETS/'public_episodes.json'),image_manifest_sha256=sha(ASSETS/'image_manifest.json'),grammar_protocol_sha256=sha(QUAL/'protocol.json'),strict_parser='single ASCII A/B/C/D exactly; terminal verified EOS mandatory',failure_policy='retain failure; stop; no blind rerun',inference_implementation='injected constrained interface; CLI weight loading deliberately unavailable',scientific_acceptance=None,user_acceptance=None)
    write(OUT/'batch_contract.json',contract)
    contract['runtime_qualification']=None
    contract['required_runtime_gate']=['Transformers4.57.1','Torch2.8.0 original runtime','both exact pinned AutoProcessors validated without compatibility alias','verified pinned weight hashes','new resource budget and absolute deadline','separately reviewed launch implementation']
    write(OUT/'batch_contract.json',contract)
    preserved=read(QUAL/'original_artifact_preservation.json')['after']
    current={p:sha(ROOT/p) for p in preserved}
    if current!=preserved: raise ValueError('Original artifacts changed')
    write(OUT/'original_artifact_preservation.json',dict(reference=preserved,current=current,byte_identity_preserved=True,observed_unix=time.time()))
    write(OUT/'source_bindings.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/test_symmetric_development_20260929.py',QUAL/'qualification.json',QUAL/'protocol.json',ROOT/'scripts/run_aircop_complete_development.py',ROOT/'scripts/run_internvl_complete_development.py']})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fetch',action='store_true');p.add_argument('--model',choices=['qwen3','internvl35_hf']);p.add_argument('--cpu-compat',action='store_true');p.add_argument('--native-qwen',action='store_true');p.add_argument('--inference-contract',type=Path);args=p.parse_args()
    if args.native_qwen and args.model!='qwen3': p.error('--native-qwen requires --model qwen3')
    if args.inference_contract:
        launch_gate(read(args.inference_contract));raise RuntimeError('Weight loading requires separately reviewed launch implementation; no inference started')
    started=time.time();result=dict(observed_unix=started,gpu_used=False,weights_loaded=False,scientific_acceptance=None,user_acceptance=None)
    try: result.update(cpu_processors=cpu_check(args.fetch,[args.model] if args.model else None,args.cpu_compat,args.native_qwen),status='scoped_cpu_processor_passed' if args.cpu_compat or args.native_qwen else 'cpu_processor_passed' if not args.model else 'cpu_processor_partial_passed')
    except Exception as e:
        result.update(status='cpu_processor_incomplete',cpu_processor_validated=None,error_type=type(e).__name__,error=str(e))
        write(OUT/('failure_%d.json'%time.time_ns()),result)
    result['seconds']=time.time()-started;write(OUT/'validation.json',result);freeze();print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
