"""Hash-bound public-only image assembly; never consults private annotations."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
from prepare_oct07_image_campaign import ROOT,OUT,sha,write

VIEW=re.compile(r'round(\d+)_UAV_(\d+)\.png')
OBS={'agent','source_id','image','sha256','timestamp','width','height','camera_world_position','camera_orientation_ned_wxyz'}

def assemble_dev(root):
    root=Path(root);old=ROOT/'evidence/own_bundle_campaign_20261002_v1'
    if json.loads((root/'contract.json').read_text()).get('schema')=='oct07_image_campaign_v3_scene_repair':
        raise ValueError('DEV24 requires its own native admission; historical DEV13 is diagnostic only')
    c=json.loads((ROOT/'evidence/own_rgb_baselines_20261002_v3/contract.json').read_text())
    restored=json.loads((ROOT/'evidence/own_dataset_rgb_restored_20261002_v1/manifest.json').read_text());cases=[];frames=0
    for row in c['cases']:
        src=old/row['public_observations']
        if sha(src)!=row['public_manifest_sha256']:raise ValueError('Inherited public source changed')
        public=json.loads(src.read_text());obs=[o for o in public['observations'] if int(VIEW.fullmatch(o['image'])[1])<2]
        if len(obs)!=2*row['team_size']:raise ValueError('DEV two-round coverage changed')
        folder=root/'public_rgb'/row['case_id'];folder.mkdir(parents=True,exist_ok=True)
        for o in obs:
            if set(o)!=OBS:raise ValueError('Public observation whitelist changed')
            source=ROOT/restored['sources'][o['sha256']]['path'];dst=folder/o['image']
            if sha(source)!=o['sha256']:raise ValueError('Restored original RGB changed')
            if not dst.exists():
                try:os.link(source,dst)
                except OSError:shutil.copyfile(source,dst)
            if sha(dst)!=o['sha256']:raise ValueError('Existing assembled RGB changed')
        data=dict(episode_id=public['episode_id'],instruction=public['instruction'],observations=obs)
        p=folder/'public_observations.json'
        if p.exists():
            if json.loads(p.read_text())!=data:raise ValueError('Existing public observations changed')
        else:write(p,data,True)
        cases.append(dict(row,public_observations=p.relative_to(root).as_posix(),public_manifest_sha256=sha(p),run_class='development'))
        frames+=len(obs)
    if len(cases)!=13 or frames!=84:raise ValueError('Fixed native DEV denominator changed')
    manifest=dict(cases=cases,case_count=13,frames=84,native_bbox_units=168,run_class='development',horizon=2,
        source_contract_sha256=sha(ROOT/'evidence/own_rgb_baselines_20261002_v3/contract.json'),private_labels_present=False)
    p=root/'public_capture_manifest.json'
    if p.exists():
        if json.loads(p.read_text())!=manifest:raise ValueError('Public assembly manifest changed')
    else:write(p,manifest,True)
    return manifest

def assemble_proposals(root,metadata):
    root=Path(root);cases=[]
    for row in metadata:
        p=root/'proposal_outputs'/row['case_id']/'bundle.json';bundle=json.loads(p.read_text())
        if set(bundle)!={'episode_id','instruction','cutoff_round','views','candidates'}:raise ValueError('Algorithm public boundary changed')
        for v in bundle['views']:
            if sha(p.parent/v['id'])!=v['sha256']:raise ValueError('Missing proposal RGB')
        cases.append(dict(row,bundle=p.relative_to(root).as_posix(),bundle_sha256=sha(p)))
    write(root/'data_manifest.json',dict(cases=cases))
    return len(cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT);a=p.parse_args();print(json.dumps(assemble_dev(a.root)))
