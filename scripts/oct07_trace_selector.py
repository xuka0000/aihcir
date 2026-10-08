"""Frozen image/text encoder and TRAIN-only utility selector."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import random
import time

from oct07_image_core import Packet, canonical

class FrozenCLIP:
    def __init__(self,directory,revision):
        import torch
        from transformers import CLIPModel,CLIPProcessor
        self.torch=torch;self.revision=revision
        self.processor=CLIPProcessor.from_pretrained(directory,local_files_only=True)
        self.model=CLIPModel.from_pretrained(directory,local_files_only=True).to('cpu').eval().requires_grad_(False)
    def encode(self,packet,instruction,receiver_public_state):
        from PIL import Image
        t=self.torch
        with Image.open(io.BytesIO(packet.payload)).convert('RGB') as im:
            image=self.processor(images=im,return_tensors='pt')
        text=self.processor(text=[instruction,receiver_public_state],return_tensors='pt',padding=True,truncation=True,max_length=77)
        with t.inference_mode():
            visual=self.model.get_image_features(**image);words=self.model.get_text_features(**text)
            visual=visual/visual.norm(dim=-1,keepdim=True);words=words/words.norm(dim=-1,keepdim=True)
        # Explicit separate feature groups make the ablations auditable. No
        # receiver-dependent similarity or feature survives no_receiver.
        return dict(visual=visual[0].float().tolist(),task=words[0].float().tolist(),receiver=words[1].float().tolist(),
            provenance=dict(packet_sha256=packet.sha256,image_payload_sha256=hashlib.sha256(packet.payload).hexdigest(),encoder_revision=self.revision))

def vector(feature,condition):
    groups=[]
    for name in ('visual','task','receiver'):
        disabled=(condition=='no_visual_features' and name=='visual' or condition=='no_task' and name=='task' or condition=='no_receiver' and name=='receiver')
        groups+=([0.]*len(feature[name]) if disabled else feature[name])
    return groups

def make_head():
    from torch import nn
    return nn.Sequential(nn.Linear(1536,96),nn.GELU(),nn.Linear(96,1))

def train(labels,output,condition,seed,backbone):
    import torch
    import torch.nn.functional as F
    rows=[r for r in labels if r['backbone']==backbone]
    if not rows or any(r['split']!='train' for r in rows):raise ValueError('Training requires source-bound TRAIN labels only')
    groups={}
    for row in rows:groups.setdefault((row['physical_scene'],row['snapshot_id']),[]).append(row)
    scenes={}
    for key,group in groups.items():scenes.setdefault(key[0],[]).append(group)
    torch.manual_seed(seed);head=make_head();optim=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    rng=random.Random(seed);history=[]
    # One snapshot per scene per optimizer step, rotated over epochs. Every
    # scene contributes equally; correlated snapshots are not extra scenes.
    for epoch in range(20):
        order=list(scenes);rng.shuffle(order);losses=[]
        for scene in order:
            snapshots=scenes[scene];batch=snapshots[epoch%len(snapshots)]
            x=torch.tensor([vector(r['features'],condition) for r in batch],dtype=torch.float32)
            y=torch.tensor([r['delta_utility'] for r in batch],dtype=torch.float32)
            pred=head(x).flatten();loss=F.smooth_l1_loss(pred,y)
            indices=(y[:,None]>y[None,:]).nonzero(as_tuple=False)
            if len(indices):loss=loss+F.softplus(-(pred[indices[:,0]]-pred[indices[:,1]])).mean()
            optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optim.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,scene_mean_loss=sum(losses)/len(losses)))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state_dict=head.state_dict(),condition=condition,seed=seed,backbone=backbone,epoch=20,
        training_rows_sha256=hashlib.sha256(canonical(rows)).hexdigest()),out/'last.pt')
    (out/'fit.json').write_text(json.dumps(dict(condition=condition,seed=seed,backbone=backbone,
        n_scenes=len(scenes),n_snapshots=len(groups),n_labels=len(rows),history=history,checkpoint_selection='final_epoch',
        training_rows_sha256=hashlib.sha256(canonical(rows)).hexdigest(),checkpoint_sha256=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest(),
        training_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    return out/'last.pt'

def select_and_transfer(calls,encoder,checkpoint,condition):
    import torch
    access=calls.access;ep=access.episode;ledger=access.ledger
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['condition']!=condition or state['backbone']!=calls.vlm.key:raise ValueError('Selector identity mismatch')
    head=make_head().eval();head.load_state_dict(state['state_dict']);decisions=[];prediction=None
    # Terminal observations are real native VLM calls. No geometric decoder is
    # substituted, and all payload/control bytes consume the same cap.
    for round_id in range(ep.horizon):
        access.advance(round_id);prediction=calls.readout()
        receiver_state=json.dumps(dict(prediction=prediction,observed=list(access.candidates())),sort_keys=True)
        offers=[]
        for sender in ep.agents:
            if sender==access.agent:continue
            if not ledger.control('advertisement',access.agent,sender,round=round_id,state=receiver_state):continue
            for cid in ep.local_candidates(sender,round_id):
                if cid in access.candidates():continue
                packet=Packet.proposal(ep,sender,access.agent,cid,round_id)
                feature=encoder.encode(packet,ep.instruction,receiver_state)
                with torch.inference_mode():gain=float(head(torch.tensor(vector(feature,condition))).item())
                bid=Packet(dict(kind='bid',sender=sender,recipient=access.agent,round=round_id,packet_sha256=packet.sha256,score=gain,wire_bytes=len(packet.wire)))
                grant=Packet(dict(kind='grant',sender=access.agent,recipient=sender,packet_sha256=packet.sha256))
                ack=Packet(dict(kind='ack',sender=access.agent,recipient=sender,packet_sha256=packet.sha256))
                # Do not spend the last feasible complete-packet reservation
                # on bids. This rule uses only actual public wire sizes.
                if ledger.bytes+len(bid.wire)+len(grant.wire)+len(ack.wire)+len(packet.wire)>ledger.cap:continue
                if not ledger.send(bid):continue
                offers.append((gain/max(1,len(packet.wire)),packet,gain,feature['provenance']))
        for _,packet,gain,provenance in sorted(offers,key=lambda v:(-v[0],v[1].sha256)):
            if gain<=0:continue
            sender=packet.header['sender']
            if not ledger.control('grant',access.agent,sender,packet_sha256=packet.sha256):continue
            if not ledger.send(packet):continue
            ledger.control('ack',access.agent,sender,packet_sha256=packet.sha256)
            decisions.append(dict(round=round_id,selected_packet=packet.sha256,predicted_gain=gain,provenance=provenance))
        if offers:prediction=calls.readout()
    return prediction,dict(decisions=decisions,condition=condition,selector_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())

def fifo(calls,communicate=True):
    access=calls.access;ep=access.episode;decisions=[];prediction=None
    for t in range(ep.horizon):
        access.advance(t)
        if communicate:
            for sender in ep.agents:
                if sender==access.agent:continue
                for cid in ep.local_candidates(sender,t):
                    if cid in access.candidates():continue
                    p=Packet.proposal(ep,sender,access.agent,cid,t)
                    if access.ledger.send(p):decisions.append(p.sha256)
        prediction=calls.readout()
    return prediction,dict(order='public sender/candidate FIFO',decisions=decisions)
