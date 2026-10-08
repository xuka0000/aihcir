"""Reuse six inherited interfaces and seal only the six new method gates."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from prepare_oct07_image_campaign import ROOT,sha,write

NEW={'llm_planner','adapt','critic','autogen','agentverse','dylan'}
def merge(root):
    root=Path(root);old=ROOT/'evidence/oct07_image_campaign_20261003_v1/cloud_repair2/effective_outputs.zip'
    contract=json.loads((root/'contract.json').read_text(encoding='utf-8'))
    backbones=tuple(contract['models'])
    with zipfile.ZipFile(old) as z:
        prior=json.loads(z.read('smoke_results.json'))
        if prior.get('passed') is not True or prior['total']!=14:raise ValueError('Inherited six-method image interface evidence differs')
        for row in prior['results']:
            filename='smoke_outputs/'+next(n.rsplit('/',1)[1] for n in z.namelist() if n.startswith('smoke_outputs/') and n.endswith('_'+row['method']+'.json') and row['backbone'] in n)
            if hashlib.sha256(z.read(filename)).hexdigest()!=row['output_sha256']:raise ValueError('Inherited raw interface output changed')
    new=json.loads((root/'smoke_additional1_results.json').read_text(encoding='utf-8'))
    if new.get('passed') is not True or new['total']!=len(NEW)*len(backbones) or set(new['methods'])!=NEW:raise ValueError('Complete six-new-method native RGB gate is required')
    required={(method,backbone) for method in NEW for backbone in backbones}
    if {(r['method'],r['backbone']) for r in new['results']}!=required:raise ValueError('New method/backbone coverage differs')
    for row in new['results']:
        p=root/'smoke_outputs'/('smoke_additional1_'+row['backbone']+'_'+row['method']+'.json')
        if sha(p)!=row['output_sha256'] or row['calls']>32:raise ValueError('New native interface output/budget identity differs')
    inherited=[r for r in prior['results'] if r['backbone'] in backbones]
    result=dict(passed=True,total=len(inherited)+len(new['results']),public_method_backbone_units=12*len(backbones),diagnostic_fifo_units=len(backbones),
        inherited_archive_sha256=sha(old),additional_sha256=sha(root/'smoke_additional1_results.json'),
        results=inherited+new['results'],real_native_RGB=True,algorithm_utility_validated=False,
        old_closed_repair_budget_not_reset=True)
    write(root/'smoke_results.json',result,True);return dict(native_interfaces=result['total'],public_method_backbone_units=result['public_method_backbone_units'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(merge(a.root)))
