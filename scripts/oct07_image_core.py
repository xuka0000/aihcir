"""Public-only image transport and measured VLM access for the Oct7 campaign.

No scoring or private scene imports belong in this module. Wire envelopes are
the only route by which a receiver gains remote evidence.
"""
from __future__ import annotations
import copy
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import random
import struct
import time

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode('utf-8')

class BudgetExhausted(BaseException):
    """External hard stop must survive authors' broad Exception retry handlers."""

class FrozenRGBCache:
    """One engineering batch only; never share decoded data between runs.

    Verify every declared file on entry and again before sealing. Each access
    still verifies SHA, including cache hits: Windows ctime is creation time,
    so restored mtime cannot safely prove immutability. Callers own copies.
    This cache changes I/O, not which views a receiver is permitted to access.
    """
    def __init__(self, expected, max_bytes=24*1024**2):
        if max_bytes < 0:raise ValueError('Negative RGB cache bound')
        self.max_bytes=max_bytes;self.bytes=0;self.peak_bytes=0
        self.expected={Path(p).resolve():digest for p,digest in expected.items()}
        self.stamps={};self.cached=OrderedDict();self.closed=False
        self.hash_calls=0;self.hash_bytes=0;self.decodes=0;self.hits=0
        for path,digest in self.expected.items():
            stamp=self.stamp(path)
            if self.digest(path)!=digest or self.stamp(path)!=stamp:
                raise ValueError('Frozen engineering RGB changed at entry')
            self.stamps[path]=stamp
    @staticmethod
    def stamp(path):
        s=path.stat()
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def digest(self,path):
        from prepare_oct07_image_campaign import sha
        self.hash_calls+=1;self.hash_bytes+=path.stat().st_size
        return sha(path)
    def image(self,path,digest):
        from PIL import Image
        path=Path(path).resolve()
        if self.closed or self.expected.get(path)!=digest:
            raise ValueError('RGB cache is closed or outside frozen batch')
        if self.stamp(path)!=self.stamps[path]:raise ValueError('Frozen RGB metadata changed')
        if path in self.cached:
            if self.digest(path)!=digest or self.stamp(path)!=self.stamps[path]:
                raise ValueError('Source RGB hash changed')
            self.hits+=1;self.cached.move_to_end(path)
            return self.cached[path].copy()
        raw=path.read_bytes();self.hash_calls+=1;self.hash_bytes+=len(raw)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Source RGB hash changed')
        with Image.open(io.BytesIO(raw)) as source:im=source.convert('RGB')
        self.decodes+=1
        if im.size!=(640,480) or self.stamp(path)!=self.stamps[path]:
            im.close();raise ValueError('Unfrozen source size or RGB changed during decode')
        size=im.width*im.height*3
        if size>self.max_bytes:return im
        while self.bytes+size>self.max_bytes:
            _,old=self.cached.popitem(last=False);self.bytes-=old.width*old.height*3;old.close()
        self.cached[path]=im;self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
        return im.copy()
    def verify(self):
        if self.closed:raise ValueError('Closed RGB batch')
        for path,digest in self.expected.items():
            if self.stamp(path)!=self.stamps[path] or self.digest(path)!=digest or self.stamp(path)!=self.stamps[path]:
                raise ValueError('Frozen RGB changed before completion')
        return dict(files=len(self.expected),hash_calls=self.hash_calls,hash_bytes=self.hash_bytes,
                    decodes=self.decodes,cache_hits=self.hits,peak_cached_rgb_bytes=self.peak_bytes,
                    cache_limit_bytes=self.max_bytes,entry_and_completion_verified=True)
    def close(self):
        for im in self.cached.values():im.close()
        self.cached.clear();self.bytes=0;self.closed=True

class PublicEpisode:
    KEYS=frozenset(('episode_id','instruction','cutoff_round','views','candidates'))
    def __init__(self,bundle,folder,horizon=2,presentation='original',perturbation='none',seed=17,rgb_cache=None):
        if set(bundle)!=self.KEYS:raise ValueError('Non-whitelisted public bundle')
        if horizon not in (1,2) or presentation not in ('original','reverse'):raise ValueError('Unfrozen presentation/horizon')
        self.id=bundle['episode_id'];self.instruction=bundle['instruction'];self.folder=Path(folder)
        self.rgb_cache=rgb_cache
        self.views={v['id']:copy.deepcopy(v) for v in bundle['views'] if v['round']<horizon}
        if any(set(v)-{'id','agent','round','sha256','camera_world_position'} for v in self.views.values()):raise ValueError('View contains private fields')
        self.agents=sorted({v['agent'] for v in self.views.values()})
        items=[copy.deepcopy(c) for c in bundle['candidates'] if c['entity']['view_id'] in self.views]
        if presentation=='reverse':items.reverse()
        self.original_ids={f'c{i:04d}':c['id'] for i,c in enumerate(items)}
        self.candidates={}
        rng=random.Random(seed)
        for i,c in enumerate(items):
            if set(c)-{'id','role','xy','score','span_m','entity','sources','detector_evidence'}:raise ValueError('Candidate has unapproved fields')
            c['id']=f'c{i:04d}'
            if perturbation=='pose_1m':
                # Shared source-wise measured-position error, not truth correction.
                offsets=getattr(self,'_pose_offsets',{})
                view=c['entity']['view_id']
                if view not in offsets:offsets[view]=(rng.gauss(0,1),rng.gauss(0,1))
                self._pose_offsets=offsets;c['xy']=[x+z for x,z in zip(c['xy'],offsets[view])]
            elif perturbation not in ('none','drop_0.1'):raise ValueError('Unknown perturbation')
            self.candidates[c['id']]=c
        self.roles=['target','van']+(['engine'] if 'fire engine' in self.instruction else [])+(['marker'] if 'blue car' in self.instruction and 'matches' in self.instruction else [])
        self.horizon=horizon
    def local_views(self,agent,round_id):
        if agent not in self.agents:raise ValueError('Unknown sender')
        return {k for k,v in self.views.items() if v['agent']==agent and v['round']<=round_id}
    def local_candidates(self,agent,round_id):
        ids=self.local_views(agent,round_id)
        return {k:c for k,c in self.candidates.items() if c['entity']['view_id'] in ids}
    def image(self,view):
        from PIL import Image
        row=self.views[view];p=self.folder/view
        if self.rgb_cache is not None:return self.rgb_cache.image(p,row['sha256'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source RGB hash changed')
        im=Image.open(p).convert('RGB')
        if im.size!=(640,480):raise ValueError('Unfrozen source size')
        return im

def crop_jpeg(image,box):
    from PIL import Image
    if len(box)!=4 or not all(math.isfinite(x) for x in box):raise ValueError('Invalid detector box')
    x1,y1,x2,y2=box
    if x2<=x1 or y2<=y1:raise ValueError('Empty detector box')
    cx,cy=(x1+x2)/2,(y1+y2)/2;w,h=1.5*(x2-x1),1.5*(y2-y1)
    box=(max(0,math.floor(cx-w/2)),max(0,math.floor(cy-h/2)),min(image.width,math.ceil(cx+w/2)),min(image.height,math.ceil(cy+h/2)))
    crop=image.crop(box).resize((224,224),Image.Resampling.BICUBIC)
    buf=io.BytesIO();crop.save(buf,format='JPEG',quality=85,subsampling=2,optimize=False,progressive=False)
    return buf.getvalue(),list(box)

class Packet:
    def __init__(self,header,payload=b''):
        self.header=copy.deepcopy(header);self.payload=bytes(payload)
        raw=canonical(self.header);self.wire=struct.pack('!II',len(raw),len(self.payload))+raw+self.payload
        self.sha256=hashlib.sha256(self.wire).hexdigest()
    @classmethod
    def proposal(cls,episode,sender,receiver,candidate,round_id):
        local=episode.local_candidates(sender,round_id)
        if candidate not in local or sender==receiver:raise ValueError('Sender may export only its observed local proposal')
        c=local[candidate];view=c['entity']['view_id']
        image=episode.image(view)
        try:raw,box=crop_jpeg(image,c['entity']['bbox_xyxy'])
        finally:image.close()
        return cls(dict(kind='image',episode_id=episode.id,sender=sender,recipient=receiver,round=round_id,
            source_view=view,source_sha256=episode.views[view]['sha256'],crop_xyxy=box,candidate=c,
            encoding='JPEG',size=[224,224],quality=85,payload_sha256=hashlib.sha256(raw).hexdigest()),raw)

class WireLedger:
    def __init__(self,agents,cap,seed=17,drop_probability=0):
        if type(cap)!=int or cap<0:raise ValueError('Invalid byte cap')
        self.agents=tuple(agents);self.cap=cap;self.bytes=0;self.events=[];self.received={a:[] for a in agents}
        self.rng=random.Random(seed);self.drop=drop_probability
    def send(self,packet):
        h=packet.header;a,b=h['sender'],h['recipient']
        if a not in self.agents or b not in self.agents or a==b:raise ValueError('Invalid wire route')
        allowed=self.bytes+len(packet.wire)<=self.cap
        delivered=allowed and self.rng.random()>=self.drop
        if allowed:self.bytes+=len(packet.wire)
        self.events.append(dict(header=h,wire_sha256=packet.sha256,wire_bytes=len(packet.wire),
            payload_bytes=len(packet.payload),accepted=allowed,delivered=delivered,total_bytes=self.bytes))
        if delivered:self.received[b].append(packet)
        return delivered
    def control(self,kind,sender,recipient,**fields):
        return self.send(Packet(dict(kind=kind,sender=sender,recipient=recipient,**fields)))

class ViewAccess:
    def __init__(self,episode,ledger,agent='UAV_0',round_id=0,all_view_reference=False):
        self.episode=episode;self.ledger=ledger;self.agent=agent;self.round=round_id;self.all_view=all_view_reference
    def advance(self,round_id):
        if not self.round<=round_id<self.episode.horizon:raise ValueError('Invalid or future round')
        self.round=round_id
    def candidates(self):
        local=self.episode.local_candidates(self.agent,self.round)
        if self.all_view:return {k:v for k,v in self.episode.candidates.items() if self.episode.views[v['entity']['view_id']]['round']<=self.round}
        out=dict(local)
        for p in self.ledger.received[self.agent]:
            if p.header['kind']=='image' and p.header['round']<=self.round:out[p.header['candidate']['id']]=p.header['candidate']
        return out
    def images(self):
        from PIL import Image
        ids=(set(self.episode.views) if self.all_view else self.episode.local_views(self.agent,self.round))
        images=[];names=[]
        for view in sorted(ids):
            if self.episode.views[view]['round']<=self.round:images.append(self.episode.image(view));names.append(view)
        if not self.all_view:
            for p in self.ledger.received[self.agent]:
                if p.header['kind']=='image' and p.header['round']<=self.round:
                    images.append(Image.open(io.BytesIO(p.payload)).convert('RGB'));names.append('paid_crop:'+p.sha256)
        return names,images
    def prompt(self):
        return json.dumps(dict(instruction=self.episode.instruction,round=self.round,roles=self.episode.roles,
            candidates=list(self.candidates().values()),note='Fallible detector proposals. Crops retain original full-frame boxes and ENU coordinates. Do not infer identity from query role.'),sort_keys=True)
    def options(self):
        cs=self.candidates()
        options=[[k for k,c in cs.items() if c['role']==role]+['none'] for role in self.episode.roles]
        if 'matches' in self.episode.instruction:options.append(['east','west','none'])
        return options

class NativeVLM:
    def __init__(self,key,spec,processor_manifest):
        import torch
        torch.set_num_threads(4)
        from transformers import AutoProcessor,AutoModelForImageTextToText,Qwen3VLForConditionalGeneration
        from run_symmetric_full_campaign_20260930 import verify_model_files,PROCESSOR_KWARGS
        verify_model_files(key,spec,processor_manifest)
        self.torch=torch;self.key=key;self.spec=spec;self.processor_kwargs=PROCESSOR_KWARGS[key]
        self.processor=AutoProcessor.from_pretrained(spec['directory'],local_files_only=True,trust_remote_code=False,
            **(dict(min_pixels=256*28*28,max_pixels=1024*28*28) if key=='qwen3' else {}))
        cls=Qwen3VLForConditionalGeneration if key=='qwen3' else AutoModelForImageTextToText
        self.model=cls.from_pretrained(spec['directory'],local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval().requires_grad_(False)
    def generate(self,prompt,access,*,temperature=0.,max_tokens=1024,n=1,grammar=False,stop=None,seed=17,assistant_prefix=None,logit_bias=None):
        from own_rgb_baselines_20261002 import RoleGrammar
        names,images=access.images(); t=self.torch;results=[]
        content=[dict(type='text',text=prompt)]
        for name in names:content.extend([dict(type='text',text='Image '+name),dict(type='image')])
        try:
            chat=self.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            # Completion APIs continue the author's final Thought/Action prefix.
            # Put that prefix AFTER the native assistant start token; moving it
            # inside a user message changes the original completion contract.
            if assistant_prefix is not None:chat+=assistant_prefix
            begin=time.perf_counter();inputs=self.processor(text=[chat],images=images,return_tensors='pt',**self.processor_kwargs)
            inputs={k:v.to('cuda:0') if hasattr(v,'to') else v for k,v in inputs.items()}
            if self.key=='internvl35_hf':inputs['pixel_values']=inputs['pixel_values'].to(t.bfloat16)
            prep=time.perf_counter()-begin;width=inputs['input_ids'].shape[1];eos=self.model.generation_config.eos_token_id
            if isinstance(eos,list):eos=eos[0]
            constrained=RoleGrammar(self.processor.tokenizer,access.options(),eos) if grammar else None
            for i in range(n):
                t.manual_seed(seed+i);t.cuda.manual_seed_all(seed+i);t.cuda.reset_peak_memory_stats();start=time.perf_counter()
                args=dict(do_sample=temperature>0,num_beams=1,max_new_tokens=min(max_tokens,constrained.max_tokens) if constrained else max_tokens,
                    eos_token_id=eos,pad_token_id=self.processor.tokenizer.pad_token_id,max_time=180)
                if temperature>0:args.update(temperature=temperature,top_p=1.)
                if constrained:args['prefix_allowed_tokens_fn']=constrained.callback(width)
                if logit_bias:
                    from transformers import LogitsProcessor,LogitsProcessorList
                    class NativeTokenBias(LogitsProcessor):
                        def __call__(self,input_ids,scores):
                            for token,bias in logit_bias.items():scores[:,int(token)]+=float(bias)
                            return scores
                    args['logits_processor']=LogitsProcessorList([NativeTokenBias()])
                with t.inference_mode():out=self.model.generate(**inputs,**args)
                t.cuda.synchronize();ids=out[0,width:].tolist();text=self.processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                if stop and not constrained:
                    stops=[stop] if isinstance(stop,str) else stop
                    for term in stops:
                        if term in text:text=text[:text.index(term)]
                results.append(dict(text=text,generated_ids=ids,input_tokens=int(width),output_tokens=len(ids),
                    preprocessing_seconds=prep if i==0 else 0,generation_seconds=time.perf_counter()-start,
                    image_names=names,source_pixels=sum(im.width*im.height for im in images),
                    tensor_shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},
                    peak_cuda_bytes=t.cuda.max_memory_allocated(),model_revision=self.spec['revision'],assistant_prefix=assistant_prefix,logit_bias=logit_bias))
            return results
        finally:
            for im in images:im.close()

class EpisodeCalls:
    def __init__(self,vlm,access,output,seed=17,deadline=None):
        self.vlm=vlm;self.access=access;self.output=Path(output);self.output.parent.mkdir(parents=True,exist_ok=True)
        self.seed=seed;self.used=0;self.records=[];self.deadline=deadline or time.time()+3600
    def __call__(self,prompt,*,n=1,temperature=0.,max_tokens=1024,grammar=False,stop=None,assistant_prefix=None,logit_bias=None):
        if type(n)!=int or n<1 or self.used+n>32 or max_tokens>1024 or time.time()>=self.deadline:raise BudgetExhausted('Generated sample/time budget exhausted')
        attempt=dict(index=self.used,n=n,temperature=temperature,max_tokens=max_tokens,grammar=grammar,
            prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),started_at=time.time(),assistant_prefix=assistant_prefix,logit_bias=logit_bias)
        self.used+=n;self._log(dict(kind='attempt',**attempt))
        try:
            rows=self.vlm.generate(prompt,self.access,n=n,temperature=temperature,max_tokens=max_tokens,
                grammar=grammar,stop=stop,seed=self.seed+attempt['index']*1009,assistant_prefix=assistant_prefix,logit_bias=logit_bias)
            if len(rows)!=n:raise ValueError('Missing generated samples')
            for row in rows:self.records.append(row)
            self._log(dict(kind='result',index=attempt['index'],samples=rows,finished_at=time.time()))
            return [r['text'] for r in rows]
        except BaseException as exc:
            self._log(dict(kind='failed_attempt',index=attempt['index'],charged_samples=n,error_type=type(exc).__name__,finished_at=time.time()));raise
    def _log(self,row):
        with self.output.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    def readout(self):
        latent='matches' in self.access.episode.instruction
        instruction='Select one observed proposal per role '+','.join(self.access.episode.roles)+'. Return only comma-separated IDs, or none for unresolved roles.'
        if latent:instruction+=' Append one more comma-separated field: the observed east/west side of the blue marker relative to the fire engine, or none if unresolved.'
        instruction+='\n'+self.access.prompt()
        text=self(instruction,grammar=True)[0];fields=text.split(',');roles=self.access.episode.roles
        if len(fields)!=len(roles)+int(latent) or any(x not in opts for x,opts in zip(fields,self.access.options())):return dict(valid=False,raw=text,assignment={})
        return dict(valid=True,raw=text,assignment={r:(None if x=='none' else x) for r,x in zip(roles,fields)},
            observed_direction=fields[-1] if latent else None)
