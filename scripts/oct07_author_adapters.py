"""Author controllers executed from hashed source with aerial task interfaces.

Every adapter returns the author's selected output. Invalid output stays
invalid. No TRACE selector, geometry ranking or correctness repair is used.
"""
from __future__ import annotations
import ast
import collections
from functools import partial
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import time
import traceback
from types import SimpleNamespace

from oct07_image_core import BudgetExhausted, Packet, ViewAccess, canonical

def output_spec(access, representation='json'):
    schema=dict(assignment={r:'observed ID or null' for r in access.episode.roles},observed_direction='east/west or null')
    description='Ground the instruction using only the images and fallible observed proposals. The grounding answer has exactly this schema: '+json.dumps(schema)+'. '
    if representation=='json':description+='Return one JSON object, without commentary. '
    elif representation=='python':description+='Return one Python function named solution whose sole statement returns this literal dictionary. Use None for null. '
    elif representation=='tool':description+='Use this schema only as the argument of Finish. Follow the controller tool format for each step. '
    elif representation=='plan':description+='Follow the requested Plan/Answer stage format. The final Answer is one JSON object. '
    elif representation=='actions':description+='Choose from the provided actions; the controller builds the final bindings from executed actions. '
    return description+"For a 'matches' instruction determine the east/west direction from the observed blue marker and fire engine. Otherwise observed_direction is null. Do not assume detection role names certify identity. Never invent an ID."

def final_output(text, access):
    """Keep malformed selected answers as scientific failures, without repair."""
    try:return parse_output(text,access)
    except (ValueError,KeyError,TypeError) as exc:
        return dict(valid=False,raw=text,assignment={},output_error=type(exc).__name__+': '+str(exc))

def parse_output(text, access):
    value=text.strip()
    if value.startswith('```'):value=value.split('\n',1)[1].rsplit('```',1)[0].strip()
    obj=json.loads(value)
    if not isinstance(obj,dict) or set(obj)-{'assignment','observed_direction'} or 'assignment' not in obj:raise ValueError('Output schema')
    roles=access.episode.roles
    if set(obj['assignment'])!=set(roles):raise ValueError('Output roles')
    if any(v is not None and v not in access.candidates() for v in obj['assignment'].values()):raise ValueError('Unobserved output ID')
    if obj.get('observed_direction') not in (None,'east','west'):raise ValueError('Output direction')
    return dict(valid=True,raw=text,**obj)

def definitions(path,names,namespace):
    source=Path(path).read_text(encoding='utf-8');tree=ast.parse(source);bound={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
            body='\n'.join(source.splitlines()[start:node.end_lineno])
            # A valid whole-file definition may end in an escaped newline.
            # Compile the parsed author node, preserving its exact semantics.
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
            bound[node.name]=hashlib.sha256(body.encode()).hexdigest()
    if set(bound)!=set(names):raise ValueError('Missing author definitions: '+str(set(names)-set(bound)))
    return bound

def verify_sources(root,method):
    folder=Path(root)/'author_sources'/method
    m=json.loads((folder/'source_manifest.json').read_text())
    for row in m['files']:
        if hashlib.sha256((folder/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Author source changed')
    return folder,m

def receive_view(access,view):
    episode=access.episode
    if view not in episode.views or episode.views[view]['round']>access.round:return dict(error='Unavailable/future observation')
    sender=episode.views[view]['agent']
    if sender==access.agent:return dict(observation=access.prompt())
    for cid,c in episode.local_candidates(sender,access.round).items():
        if c['entity']['view_id']!=view:continue
        p=Packet.proposal(episode,sender,access.agent,cid,access.round)
        if not access.ledger.send(p):return dict(error='Packet unavailable under communication budget')
    return dict(observation=access.prompt())

def fixed_paid_input(access):
    for t in range(access.episode.horizon):
        access.advance(t)
        for view in sorted(access.episode.views):
            if access.episode.views[view]['round']==t and access.episode.views[view]['agent']!=access.agent:
                receive_view(access,view)

def react(root,calls):
    import requests
    folder,manifest=verify_sources(root,'react');notebook=folder/'hotpotqa.ipynb';access=calls.access
    class Env:
        def __init__(self):self.final=None;self.events=[]
        def reset(self,idx=None):
            return json.dumps(dict(instruction=access.episode.instruction,observation=access.prompt(),
                catalog=[dict(id=k,agent=v['agent'],round=v['round']) for k,v in access.episode.views.items()]),sort_keys=True)
        def step(self,text):
            m=re.fullmatch(r'(search|lookup|finish)\[(.*)\]',text.strip(),re.S)
            if not m:result=dict(error='Use Search[view], Lookup[id,id], Finish[JSON output]')
            else:
                name,payload=m.groups()
                try:
                    if name=='search':result=receive_view(access,payload.strip())
                    elif name=='lookup':
                        a,b=(access.candidates()[k.strip()] for k in payload.split(','))
                        result=dict(east_delta_m=a['xy'][0]-b['xy'][0],north_delta_m=a['xy'][1]-b['xy'][1],source='fallible projected coordinates')
                    else:self.final=parse_output(payload,access);result=dict(submitted=True)
                except (ValueError,KeyError,TypeError):result=dict(error='Invalid observed argument or output schema')
            self.events.append(dict(action=text,feedback=result))
            return json.dumps(result,sort_keys=True),0.,self.final is not None,{'correctness_feedback_available':False}
    env=Env()
    class Transport:
        @staticmethod
        def create(**request):
            prompt=request['prompt'];m=re.search(r'(Thought \d+:|Action \d+:)\s*$',prompt)
            if m is None:raise ValueError('Author completion prefix unavailable')
            out=calls(prompt[:m.start()],assistant_prefix=m.group(1),temperature=request['temperature'],max_tokens=request['max_tokens'],stop=request.get('stop'))
            return dict(choices=[dict(text=s) for s in out])
    task_prompt=output_spec(access,'tool')+'\nUse Thought N: / Action N: / Observation N:. Tools: Search[view_id], Lookup[observed_id,observed_id], Finish[JSON output]. Tool output has no correctness reward. Each action is one complete tool call.\n'
    ns=dict(env=env,openai=SimpleNamespace(Completion=Transport),requests=requests,webthink_prompt=task_prompt)
    bound={}
    for cell in json.loads(notebook.read_text())['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node,ast.FunctionDef) and node.name in ('llm','step','webthink'):
                body=ast.get_source_segment(source,node);exec(compile(body,str(notebook),'exec'),ns)
                bound[node.name]=hashlib.sha256(body.encode()).hexdigest()
    if set(bound)!={'llm','step','webthink'}:raise ValueError('ReAct author functions unavailable')
    # Original webthink uses env.reset once; physical observations advance only
    # along the frozen acquisition horizon, never from an uncharged file.
    access.advance(access.episode.horizon-1)
    reward,info=ns['webthink'](idx=None,to_print=False)
    return env.final or dict(valid=False,assignment={},raw=info.get('traj','')),dict(author_functions=bound,trajectory=info,tools=env.events)

def tot(root,calls):
    import numpy as np
    folder,manifest=verify_sources(root,'tot');access=calls.access;fixed_paid_input(access)
    ns=dict(itertools=itertools,np=np,partial=partial,
        gpt=lambda prompt,n=1,temperature=1.,stop=None,**kw:calls(prompt,n=n,temperature=temperature,max_tokens=1000,stop=stop))
    names=[n.name for n in ast.parse((folder/'src/tot/methods/bfs.py').read_text()).body if isinstance(n,ast.FunctionDef)]
    bound=definitions(folder/'src/tot/methods/bfs.py',names,ns)
    text=ast.parse((folder/'src/tot/tasks/text.py').read_text());cls=next(n for n in text.body if isinstance(n,ast.ClassDef) and n.name=='TextTask')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    unwrap=ast.get_source_segment((folder/'src/tot/tasks/text.py').read_text(),methods['vote_outputs_unwrap'])
    vote_ns={'re':re};exec(compile(unwrap,'author:vote_outputs_unwrap','exec'),vote_ns)
    class Task:
        steps=2;stops=['\nAnswer:\n',None]
        def get_input(self,idx):return access.prompt()
        def cot_prompt_wrap(self,x,y=''):
            return output_spec(access,'plan')+'\nINPUT\n'+x+('\nPlan:\nGive a plan then the exact delimiter Answer:.' if not y else '\nContinue this selected plan. Start with Answer: then JSON.\n'+y)
        def vote_prompt_wrap(self,x,ys):
            return 'Task and observed evidence\n'+x+'\nChoose the best grounded continuation. End with "the best choice is N".\n'+''.join(f'Choice {i}:\n{y}\n' for i,y in enumerate(ys,1))
        vote_outputs_unwrap=staticmethod(vote_ns['vote_outputs_unwrap'])
    args=SimpleNamespace(backend=calls.vlm.key,temperature=1.,method_generate='sample',method_evaluate='vote',method_select='greedy',n_generate_sample=5,n_evaluate_sample=5,n_select_sample=1,prompt_sample='cot')
    selected,trace=ns['solve'](args,Task(),0,to_print=False)
    if len(selected)!=1:raise ValueError('Unselected author frontier')
    parent=trace['steps'][-1]['ys'][0];last=selected[0][len(parent):]
    if 'Answer:' in last:last=last.rsplit('Answer:',1)[1]
    return final_output(last.strip(),access),dict(author_functions=bound,selected=selected,trajectory=trace)

def self_refine(root,calls):
    from official_self_refine_task import definition
    folder,manifest=verify_sources(root,'self_refine');access=calls.access;fixed_paid_input(access)
    class Transport:
        @staticmethod
        def call(*,prompt,engine,max_tokens,stop_token,temperature):
            if max_tokens==600:
                prompt+='\nThe output format required by this interface is feedback followed by def solution(): and a literal return dictionary. Include the function even when the prior solution is correct; in that case return the same bindings. Do not omit or rename requested roles.\n'
            texts=calls(prompt,max_tokens=max_tokens,stop=stop_token,temperature=temperature)
            return dict(choices=[dict(message=dict(content=t)) for t in texts])
        @staticmethod
        def get_first_response(result):return result['choices'][0]['message']['content']
    ns=dict(traceback=traceback,openai_api=SimpleNamespace(OpenaiAPIWrapper=Transport),ENGINE='code-davinci-002');bound={}
    for file,name in [('src/utils.py','Prompt'),('src/utils.py','retry_parse_fail_prone_cmd'),('src/gsm/task_init.py','GSMInit'),('src/gsm/feedback.py','GSMFeedback'),('src/gsm/run.py','iterative_gsm')]:
        body=definition((folder/file).read_text(),name);exec(compile(body,str(folder/file),'exec'),ns);bound[name]=hashlib.sha256(body.encode()).hexdigest()
    prompts=Path(root)/'task_prompts/self_refine'
    init,feedback=ns['GSMInit'],ns['GSMFeedback']
    ns['GSMInit']=lambda engine,prompt_examples,temperature:init(engine=engine,prompt_examples=str(prompts/'init.txt'),temperature=temperature)
    ns['GSMFeedback']=lambda engine,prompt_examples,temperature:feedback(engine=engine,prompt_examples=str(prompts/'feedback.txt'),temperature=temperature)
    trace=ns['iterative_gsm'](question=output_spec(access,'python')+'\n'+access.prompt(),max_attempts=4,feedback_type='rich',temperature=0.)
    if not trace:
        return dict(valid=False,assignment={},raw='',output_error='Author parsing retries exhausted; returned None'),dict(author_functions=bound,trajectory=trace)
    # The author's representation is a literal-return Python function. Read it
    # without executing generated code and without choosing another attempt.
    text=trace[-1]['solution_fixed'].strip()
    if text.startswith('```'):text=text.split('\n',1)[1]
    text=text.split('\n```',1)[0]
    try:
        tree=ast.parse(text)
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('Literal function required')
        body=tree.body[0].body
        if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant):body=body[1:]
        if len(body)!=1 or not isinstance(body[0],ast.Return):raise ValueError('Nonliteral author output')
        value=ast.literal_eval(body[0].value);prediction=final_output(json.dumps(value),access)
    except (SyntaxError,ValueError,TypeError) as exc:
        prediction=dict(valid=False,assignment={},raw=text,output_error=type(exc).__name__+': '+str(exc))
    return prediction,dict(author_functions=bound,trajectory=trace,selected_field='last.solution_fixed')

def coela(root,calls):
    folder,manifest=verify_sources(root,'coela');ep=calls.access.episode
    if len(ep.agents)!=2:raise ValueError('Author CoELA supports two agents in this protocol')
    source=(folder/'tdw_mat/LLM/LLM.py').read_text();cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='LLM')
    ns=dict(random=random,re=re,json=json);methods=[];bound={}
    for node in cls.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('run','parse_answer'):
            body=ast.get_source_segment(source,node);methods.append(body);bound[node.name]=hashlib.sha256(body.encode()).hexdigest()
    exec(compile('class AuthorController:\n'+'\n'.join('\n'.join('    '+s for s in b.splitlines()) for b in methods),'author:CoELA','exec'),ns)
    controllers=[];histories={a:['start'] for a in ep.agents};dialogs={a:[] for a in ep.agents};trace=[];selected=None
    assignments={a:{r:None for r in ep.roles} for a in ep.agents};directions={a:None for a in ep.agents}
    base_access=calls.access
    for i,agent in enumerate(ep.agents):
        obj=ns['AuthorController']();obj.agent_name=('Alice','Bob')[i];obj.oppo_name=('Bob','Alice')[i]
        obj.debug=False;obj.communication=True;obj.cot=True;obj.chat=True;obj.total_cost=0
        obj.goal_desc=ep.instruction;obj.prompt_template=output_spec(base_access,'actions')+'\n$GOAL$\n$PROGRESS$\nActions:\n$AVAILABLE_ACTIONS$\nRecent actions: $ACTION_HISTORY$\nDialogue: $DIALOGUE_HISTORY$'
        obj.generator_prompt_template='Send a concise helpful message to your partner using observed evidence. Enclose the message in quotes.\n$GOAL$\n$PROGRESS$\n$ACTION_HISTORY$\n$DIALOGUE_HISTORY$'
        obj.sampling_params=dict(max_tokens=512,temperature=.7,top_p=1.,n=1)
        obj.progress2text=lambda *a,**kw:calls.access.prompt()+'\nSelected bindings: '+json.dumps(assignments[calls.access.agent])
        def available(message,obj=obj):
            plans=[]
            if message is not None:plans.append('send a message: '+str(message))
            plans+=['Assign '+c['role']+'='+cid for cid,c in calls.access.candidates().items() if c['role'] in ep.roles]
            if 'matches' in ep.instruction:plans+=['Set observed direction=east','Set observed direction=west']
            plans.append('Finalize current bindings')
            return ''.join(f'{chr(65+i)}. {s}\n' for i,s in enumerate(plans)),len(plans),plans
        obj.get_available_plans=available
        obj.generator=lambda prompt,sampling_params:(calls(json.dumps(prompt),n=sampling_params['n'],temperature=sampling_params['temperature'],max_tokens=sampling_params['max_tokens']),0)
        controllers.append(obj)
    for step in range(10):
        t=min(step,ep.horizon-1)
        for i,agent in enumerate(ep.agents):
            calls.access=ViewAccess(ep,base_access.ledger,agent,t)
            other=ep.agents[1-i];obj=controllers[i]
            plan,info=obj.run(t,'observation',{},[],[],[[],[],[]],{},histories[agent],dialogs[agent])
            histories[agent].append(str(plan));trace.append(dict(agent=agent,round=t,plan=plan,info=info))
            if plan and plan.startswith('send a message:'):
                text=plan.split(':',1)[1].strip()
                if base_access.ledger.control('message',agent,other,text=text):dialogs[other].append(obj.agent_name+': '+text)
            elif plan and plan.startswith('Assign '):
                role,cid=plan[7:].split('=',1);assignments[agent][role]=cid
                if agent!='UAV_0':
                    # Executing a task action exports only the selected observed
                    # crop and binding, charging the same wire as other methods.
                    candidate=calls.access.candidates()[cid]
                    packet=Packet.proposal(ep,agent,'UAV_0',cid,t)
                    if base_access.ledger.send(packet) and base_access.ledger.control('binding',agent,'UAV_0',role=role,candidate_id=cid):
                        assignments['UAV_0'][role]=cid
            elif plan=='Finalize current bindings' and agent=='UAV_0':
                selected=dict(valid=True,raw=plan,assignment=dict(assignments[agent]),observed_direction=directions[agent])
                break
            elif plan and plan.startswith('Set observed direction='):
                directions[agent]=plan.split('=',1)[1]
                if agent!='UAV_0' and base_access.ledger.control('direction',agent,'UAV_0',value=directions[agent]):directions['UAV_0']=directions[agent]
        if selected is not None:break
    calls.access=base_access;calls.access.advance(ep.horizon-1)
    return selected or dict(valid=False,raw='',assignment={}),dict(author_functions=bound,trajectory=trace,adaptation='run/parse_answer unchanged; aerial progress and action interfaces; author two-agent controller')

def debate(root,calls):
    folder,manifest=verify_sources(root,'debate');access=calls.access;fixed_paid_input(access)
    ns=dict(time=time);bound=definitions(folder/'mmlu/gen_mmlu.py',['construct_message','construct_assistant_message'],ns)
    # Author agent/round loop, extracted verbatim. The dataset question/answer
    # generator and benchmark scoring are excluded from the policy namespace.
    tree=ast.parse((folder/'mmlu/gen_mmlu.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.If))
    loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i')
    context_node=next(n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='agent_contexts' for x in n.targets))
    round_node=next(n for n in loop.body if isinstance(n,ast.For) and n.target.id=='round')
    # Task-format substitution only: original peer wording requires '(X)';
    # append the aerial output specification to each completion request.
    def answer(context):
        prompt=json.dumps(context)+'\nThe final answer for this task is the required JSON object.\n'+output_spec(access)
        return dict(choices=[dict(message=dict(content=calls(prompt,temperature=1.,max_tokens=1024)[0]))])
    ns.update(agents=3,rounds=2,question=output_spec(access)+'\n'+access.prompt(),generate_answer=answer)
    for node in (context_node,round_node):exec(compile(ast.Module(body=[node],type_ignores=[]),'author:debate-loop','exec'),ns)
    finals=[ctx[-1]['content'] for ctx in ns['agent_contexts']]
    parsed=[]
    for text in finals:
        try:
            output=parse_output(text,access);parsed.append(canonical(dict(assignment=output['assignment'],observed_direction=output.get('observed_direction'))).decode())
        except (ValueError,KeyError):parsed.append('INVALID:'+text)
    eval_ns={};definitions(folder/'mmlu/eval_mmlu.py',['most_frequent'],eval_ns)
    selected=eval_ns['most_frequent'](parsed);index=parsed.index(selected)
    return final_output(finals[index],access),dict(author_functions=bound,contexts=ns['agent_contexts'],selected_index=index,logical_agents=3,debate_rounds=2,temperature_basis='Original OpenAI default 1.0, explicitly declared for native backend')

def repeat_vote(root,calls,count):
    folder,manifest=verify_sources(root,'repeat_vote');fixed_paid_input(calls.access)
    texts=calls(output_spec(calls.access)+'\n'+calls.access.prompt(),n=count,temperature=1.,max_tokens=1024)
    parsed=[]
    for text in texts:
        try:
            out=parse_output(text,calls.access);parsed.append(canonical(dict(assignment=out['assignment'],observed_direction=out.get('observed_direction'))).decode())
        except (ValueError,KeyError):parsed.append('')
    src=(folder/'src/evaluator.py').read_text();fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='base_evaluate_mcq')
    # Pure author Counter/max/tie-selection block. Only MCQ extraction is
    # replaced by canonical task-output extraction. No answer key is passed.
    nodes=[n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='counter' for t in n.targets)]
    nodes+= [n for n in fn.body if isinstance(n,ast.Try)]
    ns=dict(collections=collections,random=random,final_answers=parsed)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'author:majority-vote','exec'),ns)
    selected=ns['debate_answer']
    if not selected:return dict(valid=False,raw='',assignment={}),dict(responses=texts,matched_generated_samples=count)
    return final_output(texts[parsed.index(selected)],calls.access),dict(responses=texts,matched_generated_samples=count,selection='author Counter/max/random tie block unchanged')

def llamar(root,calls):
    folder,manifest=verify_sources(root,'llamar');access=calls.access;fixed_paid_input(access)
    agents=access.episode.agents;trace=[]
    class Env:
        def __init__(self):
            self.step_num=0;self.closed_subtasks=None;self.open_subtasks=[];self.input_dict={};self.final=None
            self.checker=SimpleNamespace(get_coverage=lambda:None,get_transport_rate=lambda:None,check_success=lambda:self.final is not None,subtasks_completed=[])
        def update_subtask(self,value,agent):self.input_dict['subtask']=value
        def update_memory(self,value,agent):self.input_dict['memory']=value
        def get_planner_llm_input(self):return dict(self.input_dict,observation=access.prompt())
        def step(self,actions):
            successes=[]
            for agent,action in zip(agents,actions):
                try:
                    if action.startswith('Finish[') and action.endswith(']'):
                        # Author central actor may submit one complete joint
                        # decision. No auxiliary final model is introduced.
                        result=parse_output(action[7:-1],access)
                        if self.final is None:self.final=result
                    elif action.startswith('Search[') and action.endswith(']'):receive_view(access,action[7:-1])
                    elif action not in ('Done','Idle'):raise ValueError('Unknown task action')
                    successes.append(True)
                except (ValueError,KeyError,TypeError):successes.append(False)
            self.step_num+=1;self.input_dict['previous_actions']=actions;self.input_dict['successes']=successes
            return dict(self.input_dict),successes
    env=Env();ns=dict(json=json,re=re,AGENT_NAMES=agents)
    utils=folder/'SAR/baselines/llamar_utils_multiagent.py'
    bound=definitions(utils,['get_action','process_action_llm_output','update_plan'],ns)
    def response(env,config,action_or_planner,addendum=''):
        if action_or_planner=='planner':schema='Return {"reason":"...","plan":["subtask",...]}'
        elif action_or_planner=='verifier':schema='Return {"reason":"...","completed subtasks":[...]} based on observed execution feedback, never hidden task success.'
        else:
            keys={f"{a}'s action":'Search[view_id], Finish[complete JSON grounding], Idle or Done' for a in agents}
            keys.update(reason='...',subtask='...',memory='...',**{'failure reason':'...'})
            schema='Return the following actor JSON schema: '+json.dumps(keys)+'. A Finish argument is the grounding JSON required below.'
        interface=output_spec(access,'tool') if action_or_planner=='action' else 'Use only the observed evidence. Output exactly the requested module schema, without a grounding answer or additional text.'
        text=calls(schema+'\n'+interface+'\n'+access.prompt()+'\nState: '+json.dumps(env.input_dict)+addendum,
            temperature=.7,max_tokens=1000)[0]
        trace.append(dict(module=action_or_planner,output=text))
        return SimpleNamespace(json=lambda:dict(choices=[dict(message=dict(content=text))]))
    class Logger:
        def log_agent_mem(self,*a,**kw):pass
        def log_step(self,*a,**kw):trace.append(dict(kind='execution',**kw))
    ns.update(env=env,config=SimpleNamespace(temperature=.7,model=calls.vlm.key),get_gpt_response=response,
        action_mapping=lambda env,actions:actions,logger=Logger(),args=SimpleNamespace(verbose=False,action_verbose=False),
        tqdm=SimpleNamespace(trange=range),timeout=10,previous_action=[],previous_success=[True]*len(agents))
    source=(folder/'SAR/baselines/llamar.py').read_text();tree=ast.parse(source)
    initial=[];main=None;active=False
    for node in tree.body:
        if isinstance(node,ast.Assign) and node.lineno>80 and any(isinstance(t,ast.Name) and t.id=='success' for t in node.targets):active=True
        if active:
            if isinstance(node,ast.For):main=node;break
            initial.append(node)
    if not main or not any(isinstance(n,ast.While) for n in initial):raise ValueError('Author planner/actor loop boundary changed')
    try:
        for node in initial+[main]:exec(compile(ast.Module(body=[node],type_ignores=[]),'author:LLaMAR-loop','exec'),ns)
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        trace.append(dict(kind='author_invalid_module_output',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
    return env.final or dict(valid=False,assignment={},raw=''),dict(author_functions=bound,
        author_loop_sha256=hashlib.sha256(ast.get_source_segment(source,main).encode()).hexdigest(),trajectory=trace,
        adaptation='Original SAR planner/actor/verifier/replan loop; task tools and prompts; common native image backend; no semantic action correction')

ADAPTERS=dict(react=react,tot=tot,self_refine=self_refine,coela=coela,llamar=llamar,debate=debate)
from oct07_additional_author_adapters import ADDITIONAL_ADAPTERS
ADAPTERS.update(ADDITIONAL_ADAPTERS)
