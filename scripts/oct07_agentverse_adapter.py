"""Bind official AgentVerse tasksolving methods to public native RGB transport."""
from __future__ import annotations
import __future__
import ast
import asyncio
import bdb
import copy
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from string import Template
from types import SimpleNamespace
import typing
from oct07_author_adapters import verify_sources,output_spec,final_output,fixed_paid_input


class InterfaceRecord:
    """Schema plumbing for native backend objects, with separate mutable fields."""
    def __init__(self,**kwargs):
        for cls in reversed(type(self).__mro__):
            for name in cls.__dict__.get('__annotations__',{}):
                setattr(self,name,copy.deepcopy(getattr(cls,name,None)))
        self.__dict__.update(kwargs)


class Registry:
    def __init__(self):self.classes={}
    def register(self,key):
        def add(cls):self.classes[key]=cls;return cls
        return add
    def build(self,key,**kwargs):return self.classes[key](**kwargs)


def agentverse(root,calls):
    folder,manifest=verify_sources(root,'agentverse');base=folder/'agentverse';access=calls.access;fixed_paid_input(access)
    logger=SimpleNamespace(**{k:(lambda *a,**kw:None) for k in ('info','warn','debug','error')})
    ns=dict(vars(typing),asyncio=asyncio,bdb=bdb,json=json,re=re,Template=Template,Enum=Enum,
        BaseModel=InterfaceRecord,BaseLLM=typing.Any,BaseMemory=typing.Any,OutputParser=InterfaceRecord,
        BaseMemoryManipulator=lambda:None,abstractmethod=lambda f:f,
        Field=lambda default=None,default_factory=None,**kw:default_factory() if default_factory else default,
        logger=logger,Fore=SimpleNamespace(CYAN='',YELLOW='',GREEN='',RED='',BLUE='',MAGENTA=''))
    for key in ('agent_registry','memory_registry','output_parser_registry','EnvironmentRegistry','decision_maker_registry','role_assigner_registry','executor_registry','evaluator_registry'):
        ns[key]=Registry()
    bound={}
    def bind(path,names):
        source=(base/path).read_text(encoding='utf-8')
        for node in ast.parse(source).body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
                # Future annotations preserve the author's deferred types. The
                # registry retains identically named classes under distinct keys.
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(base/path),'exec',flags=__future__.annotations.compiler_flag),ns)
                key=path+':'+node.name+':'+str(node.lineno)
                bound[key]=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()
        missing=set(names)-set(ns)
        if missing:raise ValueError('Missing AgentVerse controller '+str(missing))
    bind('utils.py',['AgentAction','AgentFinish','AgentCriticism','AGENT_TYPES'])
    bind('message.py',['Message','SolverMessage','CriticMessage','ExecutorMessage','EvaluatorMessage','RoleAssignerMessage'])
    ns['LLMResult']=SimpleNamespace
    bind('memory/base.py',['BaseMemory'])
    bind('memory/chat_history.py',['ChatHistoryMemory'])
    tokenizer=calls.vlm.processor.tokenizer
    ns['count_string_tokens']=lambda text,model:len(tokenizer.encode(str(text)))
    bind('agents/base.py',['BaseAgent'])
    bind('output_parser/output_parser.py',['OutputParserError','RoleAssignerParser','HumanevalEvaluatorParser','MGSMCriticAgreeParser'])
    for file,name in [('role_assigner','RoleAssignerAgent'),('solver','SolverAgent'),('critic','CriticAgent'),('evaluator','EvaluatorAgent')]:
        bind('agents/tasksolving_agent/'+file+'.py',[name])
    bind('environments/base.py',['BaseRule','BaseEnvironment'])
    components=[('decision_maker/base.py',['BaseDecisionMaker']),('decision_maker/vertical_solver_first.py',['VerticalSolverFirstDecisionMaker']),
        ('role_assigner/base.py',['BaseRoleAssigner']),('role_assigner/role_description.py',['DescriptionAssigner']),
        ('executor/base.py',['BaseExecutor','DummyExecutor']),('evaluator/base.py',['BaseEvaluator']),('evaluator/basic.py',['BasicEvaluator'])]
    for file,names in components:bind('environments/tasksolving_env/rules/'+file,names)
    bind('environments/tasksolving_env/rules/base.py',['TasksolvingRule'])
    bind('environments/tasksolving_env/basic.py',['BasicEnvironment'])
    trace=[]
    class NativeLLM:
        def __init__(self,temperature=0.,max_tokens=1024):self.args=SimpleNamespace(model=calls.vlm.key);self.temperature=temperature;self.max_tokens=max_tokens
        def send_token_limit(self,model):return 32768
        async def agenerate_response(self,prepend,history,append):
            prompt=prepend+'\n'+json.dumps(history)+'\n'+append+'\nObserved evidence: '+access.prompt()
            raw=calls(prompt,temperature=self.temperature,max_tokens=self.max_tokens)[0]
            trace.append(dict(prompt=prompt,raw=raw));return SimpleNamespace(content=raw)
    class GroundingParser:
        def parse(self,output):
            # Representation only. Invalid selected output remains invalid.
            return ns['AgentFinish']({'output':output.content},output.content)
    memory=lambda:ns['ChatHistoryMemory']()
    common=dict(max_retry=1000,llm=NativeLLM(),memory=memory())
    role=ns['RoleAssignerAgent'](name='Role Assigner',**common,output_parser=ns['output_parser_registry'].build('role_assigner',cnt_critic_agents=2),
         append_prompt_template='Recruit ${cnt_critic_agents} experts for ${task_description}. Feedback: ${advice}. Return numbered role descriptions: 1. ... 2. ...')
    solver=ns['SolverAgent'](name='Solver',llm=NativeLLM(),memory=memory(),max_retry=1000,output_parser=GroundingParser(),
         prepend_prompt_template='You are ${role_description}. '+output_spec(access)+'\nTask: ${task_description}\nPrevious solution: ${former_solution}\nFeedback: ${advice}')
    critic=ns['CriticAgent'](name='Critic',llm=NativeLLM(),memory=memory(),max_retry=1000,
         output_parser=ns['output_parser_registry'].build('mgsm-critic-agree'),
         prepend_prompt_template='Review the proposed grounding and relations from public observations. Task: ${task_description}.',
         append_prompt_template='You are ${role_description}. Check the latest solution; if you agree, end with [Agree]. Otherwise explain the errors and give a better solution.')
    evaluator=ns['EvaluatorAgent'](name='Evaluator',llm=NativeLLM(.3),memory=memory(),max_retry=1000,
         output_parser=ns['output_parser_registry'].build('humaneval-evaluator',dimensions=['Score']),
         append_prompt_template='Evaluate the grounding from observed evidence. Task: ${task_description}\nSolution: ${solution}\nExecution: ${result}\nReturn Score: 0 or 1, then Advice: a specific observation-based assessment.')
    types=ns['AGENT_TYPES'];agents={types.ROLE_ASSIGNMENT:role,types.SOLVER:solver,types.CRITIC:[critic],types.EVALUATION:evaluator,types.EXECUTION:None}
    env=ns['BasicEnvironment'](agents=agents,task_description=access.episode.instruction,max_turn=3,
        rule=dict(role_assigner=dict(type='role_description'),decision_maker=dict(type='vertical-solver-first',max_inner_turns=3),executor=dict(type='dummy'),evaluator=dict(type='basic')))
    env.reset();advice='No advice yet.';previous='No solution yet.';logs=[]
    while not env.is_done():
        result,advice,previous,iteration,success=asyncio.run(env.step(advice,previous));logs.extend(iteration)
    return final_output(previous,access),dict(author_functions=bound,logs=logs,model_modules=trace,
        adaptation='Official recruitment, solver-first vertical discussion, execution and evaluation control methods; common native RGB transport and task prompts; dummy public grounding submission; no gold feedback',
        interface_record_port='Pydantic field plumbing replaced to accept native model/tokenizer objects; controller methods and registry dispatch unchanged')
