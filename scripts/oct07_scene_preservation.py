"""Bounded effective sensor exports; never copies native engines/environments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

PART_BYTES=256*1024**2
MAX_BYTES=6*1024**3

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def preserve(root,attempt,export_id=None,max_bytes=MAX_BYTES):
    root=Path(root);phase=root/'oct07_scene_repair'
    export_id=export_id or attempt
    if any(not n.replace('_','').isalnum() for n in (attempt,export_id)):
        raise ValueError('Unsafe export identity')
    if not 0<max_bytes<=MAX_BYTES:raise ValueError('Invalid effective export allowance')
    files=[f for f in phase.glob(attempt+'*') if f.is_file() and f.suffix in ('.json','.log')]
    files.extend(phase/n for n in ('capture_inventory.json','native_material_selection.json','material_budget.json',
        'material_renewal_budget.json','material_sourcefix_budget.json','native_source_correction.json',
        'material_detail_budget.json','native_detail_correction.json','detail_interface_recovery.json','detail_launch_recovery.json','material_completion_budget.json','native_material_completion_protocol.json','qualification_gate.json') if (phase/n).is_file())
    for correction in (phase/'native_source_correction.json',phase/'native_detail_correction.json'):
        if not correction.is_file():continue
        frozen=json.loads(correction.read_text(encoding='utf-8'))
        if frozen.get('attempt')==attempt:
            for name,digest in frozen['code_sha256'].items():
                if Path(name).name!=name:raise ValueError('Unsafe frozen native source identity')
                source=root/'code_scene_repair'/name
                if sha(source)!=digest:raise ValueError('Executed native source was overwritten before preservation')
                files.append(source)
            files.extend(root/'code_scene_repair'/n for n in ('settings_2uav.json','snapshot_manifest.json') if (root/'code_scene_repair'/n).is_file())
    files.extend(f for d in (root/'sessions').glob('scene_*_'+attempt) for f in d.rglob('*') if f.is_file())
    # Preserve sources for partial scenes too, before the completed inventory
    # can register them. Only exact current-attempt observation paths qualify.
    release=(root/'releases/oct07_scenes_rgb_v4').resolve()
    for session in (root/'sessions').glob('scene_*_'+attempt):
        for record in session.glob('probe/training_views/*/observations.json'):
            source=json.loads(record.read_text()).get('source_scene_path')
            if source:
                path=Path(source).resolve()
                if release not in path.parents or path.name!='scene.json':raise ValueError('Unexpected partial scene source')
                if path.is_file():files.append(path)
    inventory=json.loads((phase/'capture_inventory.json').read_text()) if (phase/'capture_inventory.json').is_file() else {}
    for eid,item in inventory.items():
        if item.get('attempt')==attempt:
            f=root/'releases/oct07_scenes_rgb_v4'/item['split']/eid/'scene.json'
            if f.is_file():files.append(f)
    files=sorted(set(files));total=sum(f.stat().st_size for f in files)
    if total>max_bytes:raise ValueError('Effective sensor export exceeds registered write budget')
    batches=[];batch=[];size=0
    for f in files:
        n=f.stat().st_size
        if n>PART_BYTES:raise ValueError('Unexpected large sensor file; diagnose before preservation')
        if batch and size+n>PART_BYTES:batches.append(batch);batch=[];size=0
        batch.append(f);size+=n
    if batch:batches.append(batch)
    parts=[]
    for index,batch in enumerate(batches):
        if shutil.disk_usage(root).free<5*1024**3+PART_BYTES:raise ValueError('Remote floor/peak prevents effective asset preservation')
        out=phase/(export_id+f'_effective_{index:03d}.zip')
        rows=[dict(path=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in batch]
        with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
            for f in batch:z.write(f,f.relative_to(root).as_posix())
            z.writestr('preservation_manifest.json',json.dumps(dict(files=rows)))
        parts.append(dict(path=out.relative_to(root).as_posix(),bytes=out.stat().st_size,sha256=sha(out),files=len(batch)))
    receipt=dict(parts=parts,total_source_bytes=total,source_files=len(files),maximum_part_bytes=PART_BYTES,
        maximum_archive_bytes=max_bytes,maximum_native_output_and_archive_peak_bytes=2*max_bytes,
        max_effective_source_bytes=max_bytes,source_attempt=attempt,export_id=export_id)
    (phase/(export_id+'_preservation.json')).write_text(json.dumps(receipt,indent=2));return receipt

def extract(part,destination):
    part=Path(part);destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(part) as z:
        manifest=json.loads(z.read('preservation_manifest.json'));rows=manifest['files']
        if set(z.namelist())!={r['path'] for r in rows}|{'preservation_manifest.json'}:raise ValueError('Preserved native inventory mismatch')
        if sum(r['bytes'] for r in rows)>PART_BYTES:raise ValueError('Unexpected extraction write peak')
        for r in rows:
            path=destination/r['path']
            if Path(r['path']).is_absolute() or '..' in Path(r['path']).parts or destination not in path.resolve().parents:raise ValueError('Unsafe native asset path')
            if z.getinfo(r['path']).file_size!=r['bytes']:raise ValueError('Uncompressed native asset size changed')
            raw=z.read(r['path'])
            if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Preserved native asset changed')
            if path.exists():
                if sha(path)!=r['sha256']:raise ValueError('Existing effective native file differs')
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:f.write(raw)
    return dict(verified_files=len(rows),destination=str(destination))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--attempt',required=True);p.add_argument('--export-id');p.add_argument('--max-bytes',type=int,default=MAX_BYTES);a=p.parse_args()
    if not a.attempt.replace('_','').isalnum():raise ValueError('Unsafe attempt')
    print(json.dumps(preserve(a.root,a.attempt,a.export_id,a.max_bytes)))
