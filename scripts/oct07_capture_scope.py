"""Select native capture IDs before acquisition, never by score or image quality."""
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def selected_ids(phase, scope='parent'):
    if scope=='parent':return None
    if scope=='tonight':
        from prepare_oct07_tonight_core import selection
        phase=Path(phase)
        plan=json.loads((phase/'tonight_core_protocol_20261007.json').read_text(encoding='utf-8'))
        if sha(phase/'contract.json')!=plan['parent']['contract_sha256']:
            raise ValueError('Prospective tonight parent identity changed')
        expected=selection(phase)
        with (phase/'tonight_runtime/scene_ids.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        if rows!=expected or len(rows)!=156 or len({r['episode_id'] for r in rows})!=156:
            raise ValueError('Prospective tonight capture selection changed')
        for split,count in (('train',120),('validation',12),('test',24)):
            path=phase/'tonight_runtime/specifications'/(split+'.specs.jsonl')
            if sha(path)!=plan['specification_sha256'][split]:
                raise ValueError('Tonight capture specifications changed')
            specs=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(specs)!=count or {s['episode_id'] for s in specs}!={r['episode_id'] for r in rows if r['split']==split}:
                raise ValueError('Tonight capture split changed')
        return {r['episode_id'] for r in rows}
    if scope!='first':raise ValueError('Unknown capture reporting scope')
    phase=Path(phase)
    plan=json.loads((phase/'first_manuscript_protocol.json').read_text(encoding='utf-8'))
    if sha(phase/'contract.json')!=plan['parent']['contract_sha256'] or sha(phase/'first_manuscript_scene_ids.csv')!=plan['scene_ids_sha256']:
        raise ValueError('Prospective first capture identity changed')
    with (phase/'first_manuscript_scene_ids.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=192 or len({r['episode_id'] for r in rows})!=192 or {s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}!={'train':120,'validation':24,'test':48}:
        raise ValueError('First native capture denominator changed')
    groups={}
    for split in ('train','validation','test'):
        p=phase/'specifications'/(split+'.specs.jsonl')
        if sha(p)!=plan['selection']['scene_specification_sha256'][split]:raise ValueError('Frozen parent capture specifications changed')
        specs=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
        if split=='test':
            for s in specs:groups.setdefault((s['task_family'],s['visual_condition_requested'],s['team_size']),[]).append(s)
            if len(groups)!=24 or any(len(v)!=10 for v in groups.values()):raise ValueError('Parent TEST strata changed')
            expected={s['episode_id'] for v in groups.values() for s in sorted(v,key=lambda s:s['episode_id'])[:2]}
        else:expected={s['episode_id'] for s in specs}
        if {r['episode_id'] for r in rows if r['split']==split}!=expected:raise ValueError('First capture ID selection changed')
    return {r['episode_id'] for r in rows}


def verify_existing_capture(benchmark,eid,item):
    benchmark=Path(benchmark).resolve()
    probe=Path(item['probe']).resolve()
    if benchmark/'sessions' not in probe.parents:raise ValueError('Native inventory path escaped the owned sessions')
    record=probe/'training_views'/eid/'observations.json'
    scene=benchmark/'releases/oct07_scenes_rgb_v4'/item['split']/eid/'scene.json'
    if sha(record)!=item['observations_sha256'] or sha(scene)!=item['source_sha256']:
        raise ValueError('Previously completed native capture changed; do not recapture')
    data=json.loads(record.read_text(encoding='utf-8'))
    if data.get('stage')!='actual_views_collected_not_validated':raise ValueError('Existing native capture is incomplete')
    for name,digest in item['files_sha256'].items():
        path=Path(name).resolve()
        if benchmark/'sessions' not in path.parents or sha(path)!=digest:
            raise ValueError('Previously captured RGB/mask identity changed')
    return True
