"""Author-bound controllers for the six added public algorithms.

Only task actions, model transport, output representation and logging are ports.
Source manifests bind every executed author definition to its original hash.
"""
from __future__ import annotations
import ast
import asyncio
import copy
from datetime import datetime
import io
import json
import math
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
from oct07_author_adapters import definitions,verify_sources,output_spec,final_output,parse_output,receive_view,fixed_paid_input


class GroundingTools:
    """Public observation access; completion rewards never expose private truth."""
    def __init__(self,access):self.access=access;self.final=None;self.events=[]
    def step(self,action):
        action=action.strip();observation='Could not execute: use inventory, Search[view_id], or Finish[JSON].';done=False
        if action.rstrip(':').lower()=='inventory':observation=self.access.prompt()
        elif action.lower().startswith('search[') and action.endswith(']'):
            observation=json.dumps(receive_view(self.access,action[action.index('[')+1:-1]))
        elif action.lower().startswith('finish[') and action.endswith(']'):
            raw=action[action.index('[')+1:-1];self.final=final_output(raw,self.access)
            observation='Submitted output; correctness is evaluated independently after sealing.';done=True
        self.events.append(dict(action=action,observation=observation,done=done))
        return observation,0,done,{},{}


def adapt(root,calls):
    folder,manifest=verify_sources(root,'adapt');access=calls.access;tools=GroundingTools(access)
    # The executor explicitly asks its own success heuristic; no gold reward.
    description=output_spec(access,'tool')+'\nActions: inventory; Search[view_id]; Finish[JSON]. Use think: task completed or think: task failed to assess your own subtask progress.\n'
    def llm(prompt,stop=None,max_tokens=150):
        return calls(description+prompt+'\nCurrent public observations: '+access.prompt(),stop=stop,max_tokens=max_tokens)[0]
    def plan_llm(prompt,stop=None):
        return calls(description+prompt,max_tokens=800)[0]
    ns=dict(re=re,copy=copy,sys=SimpleNamespace(stdout=SimpleNamespace(flush=lambda:None)),env=tools,max_runs=40,max_depth=4,
        LM='native-instruct',llm=llm,plan_llm=plan_llm,
        plan_prompt='Decompose the requested visual grounding task into short subtasks. Return Step 1: ..., Step 2: ... and Execution Order: (Step 1 AND Step 2), allowing OR alternatives when appropriate.')
    bound=definitions(folder/'run_textcraft.py',['parse_expression','plan_to_args','fetch_args','print_completion','textcraft_run','plan_and_run'],ns)
    result=ns['plan_and_run']('Available views: '+json.dumps(list(access.episode.views)),access.episode.instruction,0,tools,'',past_action_checkpoint=[])
    return tools.final or dict(valid=False,assignment={},raw=''),dict(author_functions=bound,plan_history=result[7],tool_events=tools.events,
        adaptation='Official TextCraft executor and recursive AND/OR controller; public RGB observation tools and self-generated completion, zero private rewards')


def critic(root,calls):
    folder,manifest=verify_sources(root,'critic');access=calls.access;fixed_paid_input(access);events=[]
    interface=output_spec(access)
    first=calls(interface+'\n'+access.prompt(),max_tokens=300)[0]
    def llm(prompt,model=None,stop=None,max_tokens=300,temperature=0.,verbose=False):
        text=calls(interface+'\n'+prompt+'\nPublic observation state: '+access.prompt(),stop=stop,max_tokens=max_tokens,temperature=temperature)[0]
        events.append(dict(kind='critique',raw=text));return dict(choices=[dict(text=text)])
    def google(query,topk=1):
        # Search returns public views/detections in catalog order, never labels.
        views=list(access.episode.views);view=query.strip()
        if view not in views:
            view=next((v for v in views if v in query),views[min(topk-1,len(views)-1)])
        evidence=json.dumps(receive_view(access,view),sort_keys=True)
        events.append(dict(kind='public_search',query=query,view=view,topk=topk));return dict(title=view,page=evidence)
    class Writer(io.StringIO):
        def close(self):self.sealed=self.getvalue();super().close()
    writer=Writer();args=SimpleNamespace(data='aerial_grounding',critic_type='critic',model=calls.vlm.key,max_tokens=300,
        temperature=0.,use_tool=True,max_interaction=7,max_iter=3,prompt_type='cot',split='public',num_test_sample=1,seed=calls.seed,start=0,end=1)
    sample=dict(question=access.episode.instruction,answer=None,prediction=dict(greedy=dict(text=first)))
    ns=dict(datetime=datetime,json=json,load_prompt=lambda *a:'Verify the proposed grounding using > Search Query: view_id. After tool evidence return most possible answer: followed by the grounding JSON.\n',
        llm=llm,google=google,arg=args,open=lambda *a:writer,load_jsonl=lambda *a:[sample],
        extract_cot_answer=lambda text:text,is_null_answer=lambda text:not text or text=='None',multi_ref_score=lambda *a:(float('nan'),float('nan')),
        print=lambda *a,**kw:None)
    bound=definitions(folder/'CRITIC/src/qa/critic.py',['critic_iter','critic'],ns)
    ns['critic'](args);sealed=json.loads(writer.sealed)
    return final_output(sealed['pred'][-1],access),dict(author_functions=bound,iterations=sealed['cot'],tool_events=events,
        adaptation='Original all-sample CRITIC QA loop, not oracle CRITIC*; public scene search replaces web search; original unused metrics logging receives no private answer')


def dylan(root,calls):
    from sacrebleu import sentence_bleu
    folder,manifest=verify_sources(root,'dylan');base=folder/'code/demo';access=calls.access;fixed_paid_input(access)
    def generate_answer(messages,model):
        prompt='\n'.join(m['role']+': '+m['content'] for m in messages)
        text=calls(prompt,temperature=1.,max_tokens=1024)[0];record=calls.records[-1]
        return text,record['input_tokens'],record['output_tokens']
    ns=dict(random=random,re=re,math=math,sentence_bleu=sentence_bleu,GEN_THRESHOLD=.9,ACTIVATION_MAP={'listwise':0},
        generate_answer=generate_answer,parse_single_choice=lambda text:text,extract_math_answer=lambda text:text,is_equiv=lambda x,y:x==y)
    # Preserve the official open-ended demo's roles, ranking, consensus and graph.
    source=(base/'prompt_lib.py').read_text(encoding='utf-8')
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('ROLE_MAP','ROLE_MAP_MATH','SYSTEM_PROMPT_MMLU','SYSTEM_PROMPT_MATH') for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(base/'prompt_lib.py'),'exec'),ns)
    bound=definitions(base/'prompt_lib.py',['construct_message','construct_ranking_message'],ns)
    bound.update(definitions(base/'utils.py',['most_frequent'],ns))
    bound.update(definitions(base/'LLM_Neuron.py',['LLMNeuron','LLMEdge','parse_ranks','listwise_ranker_2'],ns))
    bound.update(definitions(base/'LLMLP.py',['LLMLP'],ns))
    model=ns['LLMLP'](calls.vlm.key,4,['Assistant']*4,3,'listwise','open-ended',calls.vlm.key)
    model.zero_grad();selected,count,completions,prompt_tokens,completion_tokens=model.forward(output_spec(access)+'\n'+access.prompt())
    # The author's released open-ended demo also back-propagates influence.
    # Keep that path and report this precise variant rather than silently
    # claiming the paper's separate team-optimization experiments.
    importance=model.backward(selected)
    importance=[sum(importance[4*round_id+agent] for round_id in range(3)) for agent in range(4)]
    # The source requests influence weights after the answer. Decode that
    # documented envelope; never replace or change the selected grounding.
    selected_grounding=selected
    try:
        value,end=json.JSONDecoder().raw_decode(selected.strip())
        suffix=selected.strip()[end:].strip()
        if suffix and not re.fullmatch(r'\[\[\s*[0-9,\s]+\]\]',suffix):raise ValueError('Unexpected answer envelope')
        selected_grounding=json.dumps(value)
    except (ValueError,TypeError):pass
    return final_output(selected_grounding,access),dict(author_functions=bound,responses=completions,selected_raw=selected,author_count=count,agent_importance=importance,
        activation='listwise',consensus='original open-ended sentence BLEU >=0.9; not task correctness',
        variant='Author open-ended demo with forward/backward; separate task-team optimization is not run by the released demo',
        adaptation='Original four-agent/three-round demo graph; model transport and common 1024-token ceiling; final selected text preserved')


def autogen(root,calls):
    folder,manifest=verify_sources(root,'autogen');access=calls.access;fixed_paid_input(access)
    sys.path.insert(0,str(folder))
    import autogen as official
    import autogen.agentchat.conversable_agent as conversation
    if Path(official.__file__).resolve()!= (folder/'autogen/__init__.py').resolve():raise ValueError('Wrong AutoGen module version')
    class NativeTransport:
        def __init__(self,**kwargs):self.total_usage_summary={};self.actual_usage_summary={}
        def create(self,messages=None,context=None,**kwargs):
            prompt='\n'.join(m.get('role','user')+': '+str(m.get('content','')) for m in (messages or []))
            text=calls(prompt,max_tokens=min(1024,kwargs.get('max_tokens',1024)),temperature=kwargs.get('temperature',0.))[0]
            r=calls.records[-1];usage=dict(prompt_tokens=r['input_tokens'],completion_tokens=r['output_tokens'],total_tokens=r['input_tokens']+r['output_tokens'])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text,function_call=None,tool_calls=None))],usage=SimpleNamespace(**usage),model=calls.vlm.key,cost=0.)
        @staticmethod
        def extract_text_or_completion_object(response):return [c.message.content for c in response.choices]
        def print_usage_summary(self,*a,**kw):pass
        def clear_usage_summary(self):self.total_usage_summary={};self.actual_usage_summary={}
    previous=conversation.OpenAIWrapper;conversation.OpenAIWrapper=NativeTransport
    try:
        config=dict(model=calls.vlm.key,temperature=0.,max_tokens=1024,cache_seed=None)
        agents=[official.ConversableAgent(name,system_message=role+'\n'+output_spec(access),llm_config=config,
                  human_input_mode='NEVER',code_execution_config=False) for name,role in (
                  ('Planner','Plan the grounding decision from observed evidence.'),
                  ('Reviewer','Review proposed bindings and spatial relations against the observed evidence.'),
                  ('Grounder','Provide the complete final grounding JSON using the discussion and observations.'))]
        group=official.GroupChat(agents=agents,messages=[],max_round=10,speaker_selection_method='auto',allow_repeat_speaker=False)
        manager=official.GroupChatManager(groupchat=group,llm_config=config,human_input_mode='NEVER',code_execution_config=False,
             is_termination_msg=lambda m:False)
        agents[0].initiate_chat(manager,message=output_spec(access)+'\n'+access.prompt(),clear_history=True,silent=True)
        selected=group.messages[-1]['content']
        return final_output(selected,access),dict(messages=group.messages,selected_raw=selected,
            author_file_sha256=next(r['sha256'] for r in manifest['files'] if r['path']=='autogen/agentchat/groupchat.py'),
            speaker_selection='Official auto selector/validator nested chat; all selector calls charged',
            adaptation='Official 0.2.35 ConversableAgent, GroupChat and Manager unchanged; only OpenAI transport bound to native RGB VLM; internal API cost zero excludes measured GPU rental')
    finally:
        conversation.OpenAIWrapper=previous;sys.path.remove(str(folder))


from oct07_agentverse_adapter import agentverse
from oct07_llm_planner_adapter import llm_planner
ADDITIONAL_ADAPTERS=dict(adapt=adapt,critic=critic,dylan=dylan,autogen=autogen,agentverse=agentverse,llm_planner=llm_planner)
