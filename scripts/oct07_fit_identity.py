"""Validate the exact 24-fit source/seed/condition matrix without loading GPUs."""
import hashlib
import json
from pathlib import Path
from oct07_image_core import canonical
from prepare_oct07_image_campaign import sha

def fit_backbones(backbones=None):
    values=tuple(('qwen3','internvl35_hf') if backbones is None else backbones)
    if not values or len(set(values))!=len(values) or any(v not in ('qwen3','internvl35_hf') for v in values):
        raise ValueError('Invalid frozen backbone subset')
    return values

def verify_fits(root,backbones=None):
    root=Path(root);verified={}
    if backbones is None and (root/'contract.json').exists():
        c=json.loads((root/'contract.json').read_text(encoding='utf-8'))
        if 'reporting_scope' in c:
            from oct07_runtime_scope import verify_runtime
            verify_runtime(root);backbones=tuple(c['models'])
    for backbone in fit_backbones(backbones):
        folder=root/'training_labels'/backbone
        complete=json.loads((folder/'complete.json').read_text())
        rows=[json.loads(p.read_text()) for p in sorted(folder.glob('*.label.json'))]
        if complete['scenes']!=120 or complete['labels']!=len(rows) or not rows or any(r['split']!='train' or r['backbone']!=backbone for r in rows):raise ValueError('Frozen TRAIN-label coverage differs')
        source=hashlib.sha256(canonical(rows)).hexdigest()
        for condition in ('full','no_task','no_receiver','no_visual_features'):
            for seed in (17,29,43):
                path=root/'fits'/backbone/condition/str(seed);meta=json.loads((path/'fit.json').read_text())
                expected=dict(backbone=backbone,condition=condition,seed=seed,n_labels=len(rows),training_rows_sha256=source,
                              checkpoint_sha256=sha(path/'last.pt'),training_code_sha256=sha(Path(__file__).with_name('oct07_trace_selector.py')),checkpoint_selection='final_epoch')
                if any(meta.get(k)!=v for k,v in expected.items()) or [r['epoch'] for r in meta['history']]!=list(range(1,21)):raise ValueError('Frozen fit source/checkpoint identity differs')
                verified[(path/'last.pt').relative_to(root).as_posix()]=expected['checkpoint_sha256']
    return verified
