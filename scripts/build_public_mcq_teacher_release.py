"""Build only complete public acquisition assets; never include private labels."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'01_uav_language_grounding/benchmark'))
from public_crossview.aircop import canonical,relative_image_path,sha
from public_crossview.receiver_binding import batch_bindings

PUBLIC_MODULES=('__init__.py','aircop.py','budget.py','mcq_requests.py',
                'teacher_observations.py','teacher_collection.py','teacher_qwen.py',
                'receiver_binding.py')


def assemble(assets, protocol):
    batch=json.loads((assets/'batch.json').read_text(encoding='utf-8'))
    episodes=json.loads((assets/'public_episodes.json').read_text(encoding='utf-8'))
    grouped=json.loads((protocol/'grouped_development.json').read_text(encoding='utf-8'))
    if sha(grouped)!=batch['grouped_protocol_sha256'] or sha(episodes)!=batch['public_episodes_sha256']:
        raise ValueError('Frozen sources changed')
    batch_bindings(batch, episodes)
    if any(grouped['assignments'][e['id']]['split']!='train' for e in episodes):
        raise ValueError('Non-training question in release')
    source={e['id']:e for e in json.loads((protocol/'public/train.json').read_text(encoding='utf-8'))}
    if any(e!=source[e['id']] for e in episodes):
        raise ValueError('Question or image list changed')
    needed={relative_image_path(v['path']) for e in episodes for v in e['views']}
    missing=sorted(name for name in needed if not (assets/'images'/name).is_file())
    if missing or not (assets/'image_manifest.json').is_file():
        return None,dict(state='blocked_missing_source_images',missing_image_count=len(missing),missing_images=missing,
                         manifest_available=(assets/'image_manifest.json').is_file(),actual_model_calls=0)
    manifest=json.loads((assets/'image_manifest.json').read_text(encoding='utf-8'))
    if manifest['dataset_revision']!=batch['dataset_revision'] or manifest['episode_ids']!=batch['episode_ids']:
        raise ValueError('Image manifest revision or episode IDs changed')
    index={r['path']:r for r in manifest['files']}
    if len(index)!=len(manifest['files']) or set(index)!=needed:
        raise ValueError('Image manifest is not an exact source inventory')
    # Selection strata stay local. The worker receives no question-type labels,
    # answer keys, scene annotations, offline supervision code or raw dataset.
    public_batch={k:v for k,v in batch.items() if k!='selection'}
    files={'assets/batch.json':(canonical(public_batch)+'\n').encode('utf-8'),
           'assets/public_episodes.json':(canonical(episodes)+'\n').encode('utf-8'),
           'assets/image_manifest.json':(canonical(manifest)+'\n').encode('utf-8'),
           'run_public_mcq_teacher.py':(ROOT/'scripts/run_public_mcq_teacher.py').read_bytes(),
           'probe_public_mcq_cloud_runtime.py':(ROOT/'scripts/probe_public_mcq_cloud_runtime.py').read_bytes(),
           'supervise_public_mcq_teacher.py':(ROOT/'scripts/supervise_public_mcq_teacher.py').read_bytes()}
    for name in PUBLIC_MODULES:
        files['code/public_crossview/'+name]=(ROOT/'01_uav_language_grounding/benchmark/public_crossview'/name).read_bytes()
    for name,item in index.items():
        content=(assets/'images'/relative_image_path(name)).read_bytes()
        if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:
            raise ValueError('Source image byte mismatch')
        files['assets/images/'+name]=content
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in sorted(files.items()):
            entry=zipfile.ZipInfo(name,date_time=(2026,9,26,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,content)
    data=output.getvalue()
    return data,dict(state='release_built_not_launched',sha256=hashlib.sha256(data).hexdigest(),
                     files={name:hashlib.sha256(content).hexdigest() for name,content in files.items()},
                     private_keys_included=False,private_question_types_included=False,actual_model_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    data,report=assemble(args.assets,args.protocol)
    (args.output/'readiness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if data is not None:(args.output/'release.zip').write_bytes(data)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','missing_images')}))


if __name__=='__main__':main()
