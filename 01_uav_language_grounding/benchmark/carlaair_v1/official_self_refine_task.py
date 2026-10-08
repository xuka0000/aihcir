"""Execute pinned author Self-Refine GSM control and prompt classes unchanged.

Only task example files and LLM transport are replaced. Generated Python is
NEVER executed. A literal-return function is the task output representation.
This is a task adaptation, not the author's GSM benchmark reproduction.
"""
import ast
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import urllib.request

COMMIT='9a206d41e5d2d0c241bb441f41eeadb945afaa55'
HASHES={
    'src__utils.py':'8c89d51373374b963ebe69319456a557ff782a3f49881c0649a2e677361d6c96',
    'src__gsm__run.py':'7b381b420a2e7b5471def9c8893d22534b22a9260dc5c1225148a1efdac8c604',
    'src__gsm__task_init.py':'b8a8e1f30716f9c9ad639ddd3db45f118e7ae2ee72329ba445b0bfbd802a5d4d',
    'src__gsm__feedback.py':'ff5d7035147b92630f44f276f4531e2ec5ab4157089c057b7948889ecd9f0806'}


class AuthorTransport:
    def __init__(self, url, deadline, path):
        import os
        port=int(os.environ.get('AIHCIR_SELF_REFINE_PORT','18553'))
        if not 1024<=port<=65535 or url!=f'http://127.0.0.1:{port}/v1/chat/completions': raise ValueError('Only owned service allowed')
        self.url,self.deadline,self.path=url,deadline,path
        self.records=[]

    def call(self, *, prompt, engine, max_tokens, stop_token, temperature):
        if engine!='code-davinci-002' or (max_tokens,temperature) not in ((300,0.),(600,.7)):
            raise ValueError('Author GSM generation settings changed')
        if len(self.records)>=15: raise RuntimeError('Author retry and call bound exceeded')
        if time.time()>=self.deadline: raise TimeoutError('Self-Refine deadline')
        payload=dict(model='Qwen2.5-VL-7B-Instruct',messages=[dict(role='user',content=prompt)],
                     max_tokens=max_tokens,temperature=temperature,n=1,stop=stop_token)
        req=urllib.request.Request(self.url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=min(180,self.deadline-time.time())) as response:
            result=json.load(response)
        record=dict(request=payload,original_engine=engine,response=result,seconds=time.perf_counter()-start)
        self.records.append(record)
        with self.path.open('a') as log: log.write(json.dumps(record)+'\n')
        return result

    @staticmethod
    def get_first_response(result):
        return result['choices'][0]['message']['content']


def definition(source,name):
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    if len(nodes)!=1: raise ValueError('Missing or duplicate author definition')
    node=nodes[0]
    # Include the actual author retry decorator, not just the loop body.
    first=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
    return '\n'.join(source.splitlines()[first:node.end_lineno])


def load_author(upstream, prompts, transport):
    sources={}
    for name,digest in HASHES.items():
        data=(Path(upstream)/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest: raise ValueError('Author source changed: '+name)
        sources[name]=data.decode()
    ns=dict(traceback=traceback,openai_api=SimpleNamespace(OpenaiAPIWrapper=transport),ENGINE='code-davinci-002')
    bound={}
    for file,name in [('src__utils.py','Prompt'),('src__utils.py','retry_parse_fail_prone_cmd'),
                      ('src__gsm__task_init.py','GSMInit'),('src__gsm__feedback.py','GSMFeedback'),
                      ('src__gsm__run.py','iterative_gsm')]:
        body=definition(sources[file],name)
        exec(compile(body,str(Path(upstream)/file)+':'+name,'exec'),ns)
        bound[name]=hashlib.sha256(body.encode()).hexdigest()
    init,feedback=ns['GSMInit'],ns['GSMFeedback']
    # File-location adaptation only. Original class methods, including feedback
    # history accumulation, parsing, retry and stopping condition are preserved.
    ns['GSMInit']=lambda engine,prompt_examples,temperature:init(
        engine=engine,prompt_examples=str(Path(prompts)/'init.txt'),temperature=temperature)
    ns['GSMFeedback']=lambda engine,prompt_examples,temperature:feedback(
        engine=engine,prompt_examples=str(Path(prompts)/'feedback.txt'),temperature=temperature)
    return ns['iterative_gsm'],bound


def decode_solution(text):
    """Read the author's selected code block, never choose or revise an answer.

    The unchanged GSM feedback parser removes the opening code fence when it
    splits on ``def solution():``, but may retain the closing fence. Support
    that presentation boundary explicitly. Prose outside the code block is
    not code; the selected block must still contain one literal-only function.
    """
    if len(text)>100000: raise ValueError('Oversized generated solution')
    code=text.strip();formatting=[]
    if code.startswith('```python\n') or code.startswith('```\n'):
        code=code.split('\n',1)[1];formatting.append('opening_markdown_fence')
    if '\n```' in code:
        code,tail=code.split('\n```',1)
        if tail.startswith(('python','json')) or '\n```' in tail or 'def solution' in tail:
            raise ValueError('Multiple or ambiguous code blocks')
        formatting.append('closing_markdown_fence')
        if tail.strip():formatting.append('prose_outside_selected_code_block')
    elif formatting:raise ValueError('Unclosed code fence')
    tree=ast.parse(code)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('Expected exactly one solution function, no executable statements')
    fn=tree.body[0]
    if fn.name!='solution' or fn.args.args or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.decorator_list:
        raise ValueError('Invalid solution signature')
    body=list(fn.body)
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str): body=body[1:]
    if len(body)!=1 or not isinstance(body[0],ast.Return): raise ValueError('Only a literal return is allowed')
    value=ast.literal_eval(body[0].value)
    if not isinstance(value,dict) or set(value)!={'target_id','relations','confidence','uncertainty'}:
        raise ValueError('Unexpected answer fields')
    return value,formatting


def literal_solution(text):
    return decode_solution(text)[0]
