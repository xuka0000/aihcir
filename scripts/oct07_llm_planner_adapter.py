"""Official LLM-Planner retrieval/prompt/bias generator with grounding actions."""
import ast
import inspect
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from prepare_oct07_image_campaign import sha

_ENCODERS={}

def retrieval_encoder(root):
    """Verify the pinned assets before an offline, portable CPU load."""
    root=Path(root)
    retrieval=json.loads((root/'retrieval_model_source.json').read_text(encoding='utf-8'))
    override=os.environ.get('AIHCIR_RETRIEVAL_ENCODER_DIR')
    model_dir=Path(override) if override else root/retrieval.get('deployment_directory','retrieval_encoder')
    if not override and not model_dir.is_dir():
        # Reuse the sole existing Windows checkpoint during local validation.
        # A deployed Linux release uses its explicitly named asset directory.
        model_dir=Path(retrieval['directory'])
    if not model_dir.is_dir():
        raise FileNotFoundError('Pinned retrieval assets are absent; provision retrieval_encoder or AIHCIR_RETRIEVAL_ENCODER_DIR')
    for item in retrieval['verified_files']:
        if sha(model_dir/item['path'])!=item['sha256']:
            raise ValueError('Frozen retrieval encoder changed')
    encoder_key=(str(model_dir.resolve()),retrieval['revision'])
    if encoder_key not in _ENCODERS:
        from sentence_transformers import SentenceTransformer
        kwargs=dict(device='cpu')
        if 'local_files_only' in inspect.signature(SentenceTransformer).parameters:
            kwargs['local_files_only']=True
        else:
            from huggingface_hub import constants
            if not constants.HF_HUB_OFFLINE:
                raise RuntimeError('This CPU retrieval version requires HF_HUB_OFFLINE=1 before importing dependencies')
        _ENCODERS[encoder_key]=SentenceTransformer(str(model_dir),**kwargs)
    return _ENCODERS[encoder_key],retrieval

def llm_planner(root,calls):
    from oct07_author_adapters import definitions,verify_sources,output_spec,final_output,receive_view
    import pandas as pd
    from sentence_transformers.util import cos_sim
    from ast import literal_eval
    root=Path(root);folder,source=verify_sources(root,'llm_planner');access=calls.access
    examples=json.loads((root/'llm_planner_train_examples.json').read_text(encoding='utf-8'))
    if examples['source_split']!='train' or examples['source_scenes']!=120 or len(examples['examples'])!=100:raise ValueError('Official few-shot training exposure is incomplete')
    encoder,retrieval=retrieval_encoder(root)
    events=[]
    def create(prompt,engine=None,max_tokens=200,temperature=0.,logit_bias=None):
        # The source formats household plans; this replaces only the domain
        # action vocabulary/header and completion API tokenizer IDs.
        prompt=prompt.replace('household task','aerial visual grounding task')
        prompt+= '\n'+output_spec(access,'actions')+'\nUse comma-separated Search view_id and Finish compact_JSON actions.\n'+access.prompt()
        text=calls(prompt,max_tokens=max_tokens,temperature=temperature,logit_bias=logit_bias)[0]
        events.append(dict(kind='generated_plan',raw=text));return dict(choices=[dict(text=text)])
    ns=dict(random=random,pd=pd,literal_eval=literal_eval,cos_sim=cos_sim,
        ACT_TO_STR={'Search':'Search','Finish':'Finish'},openai=SimpleNamespace(Completion=SimpleNamespace(create=create)))
    bound=definitions(folder/'hlp/hlp_planner.py',['LLM_HLP_Generator'],ns)
    generator=ns['LLM_HLP_Generator'].__new__(ns['LLM_HLP_Generator'])
    class TokenizerPort:
        def encode(self,value):return calls.vlm.processor.tokenizer.encode(','.join(value) if isinstance(value,list) else value)
    generator.sentence_embedder=encoder;generator.tokenizer=TokenizerPort()
    generator.knn_set=pd.DataFrame(examples['examples']);generator.debug=False
    completed=[];seen=set();selected='';replans=0
    # Original Algorithm 1 p5: observations/completed subgoals enter planning;
    # failed action or a new observation step triggers grounded replanning.
    for observation_round in range(access.episode.horizon):
        access.advance(observation_round)
        while replans<10:
            task=dict(task_instr=[access.episode.instruction],step_instr=[],completed_plans=completed,
                      vis_objs=sorted(access.candidates()))
            plan=generator.generate_hlp(task,k=9);replans+=1;pending=plan.strip();failed=False
            while pending:
                if pending.lower().startswith('finish '):
                    payload=pending[7:].strip()
                    try:value,end=json.JSONDecoder().raw_decode(payload)
                    except ValueError:selected=payload;failed=True;break
                    selected=payload[:end];pending=payload[end:].lstrip(' ,\n')
                    prediction=final_output(selected,access)
                    if not prediction.get('valid'):failed=True;break
                    completed.append(('Finish',selected));return prediction,dict(author_functions=bound,plans=events,completed=completed,train_examples_sha256=sha(root/'llm_planner_train_examples.json'),
                        driver='Algorithm1 grounded replanning port; max10 from author e2e defaults',tokenizer_port='Native tokenizer IDs preserve original +0.1 visible-object logit bias',original_benchmark_reproduction=False)
                action,separator,remainder=pending.partition(',');parts=action.strip().split(maxsplit=1)
                if len(parts)!=2 or parts[0].lower()!='search':failed=True;break
                result=receive_view(access,parts[1]);events.append(dict(kind='execution',action=action,result=result))
                if 'error' in result:failed=True;break
                completed.append(('Search',parts[1]));seen.update(access.candidates());pending=remainder.strip()
            if not failed:break
    return final_output(selected,access),dict(author_functions=bound,plans=events,completed=completed,
        train_examples_sha256=sha(root/'llm_planner_train_examples.json'),driver='Algorithm1 grounded replanning task interface',original_benchmark_reproduction=False)
