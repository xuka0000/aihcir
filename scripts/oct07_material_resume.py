"""Reuse a verified seven-row prefix; measure only its missing seven conditions."""
import copy,hashlib,json,os
from pathlib import Path

def inherited_prefix(receipt,plans):
    receipt=Path(receipt);binding=json.loads(receipt.read_text())
    source=Path(binding['remote_source_directory'])
    report=source/'result.json'
    if hashlib.sha256(report.read_bytes()).hexdigest()!=binding['source_result_sha256']:
        raise ValueError('Partial measurement identity changed')
    data=json.loads(report.read_text());rows=[r for r in data['attempts'] if r.get('status')=='measured']
    if (len(rows)!=7 or data.get('completed_frames')!=7 or data.get('planned_frames')!=14
        or binding.get('new_measurement_limit')!=7 or binding.get('original_renderer_profile_unchanged') is not True
        or not data.get('native_uav_camera') or not data.get('exact_actor_mask')):
        raise ValueError('No exact seven-row material prefix')
    files={}
    for index,r in enumerate(rows):
        if (r['index']!=index or (r['blueprint'],r['color'],r['height'])!=plans[index]):
            raise ValueError('Original prefix plan changed')
        m=r['measurement']
        for name,digest in [(r['file'],r['sha256']),(m['visible_mask'],m['visible_mask_sha256']),
                            (m['isolated_candidate_mask'],m['isolated_mask_sha256'])]:
            path=(source/name).resolve()
            if source.resolve() not in path.parents or path.is_symlink():raise ValueError('Unsafe inherited raw asset')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Inherited raw asset changed')
            files[name]=(path,digest)
    if {name:h for name,(p,h) in files.items()}!=binding['reused_files_sha256']:
        raise ValueError('Inherited raw inventory differs')
    return copy.deepcopy(rows),files,data,binding

def link_prefix(files,output):
    output=Path(output).resolve()
    for name,(source,digest) in files.items():
        target=(output/name).resolve()
        if output not in target.parents or target.exists():raise ValueError('Reuse would overwrite a raw asset')
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,target)

