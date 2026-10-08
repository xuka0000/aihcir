"""Freeze a ten-question grouped-TRAIN acquisition batch before model output."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '01_uav_language_grounding/benchmark'))
from public_crossview.aircop import canonical, relative_image_path, sha

REVISION = 'c82dc24d185f841f2854f6efc37fc67b3e62f22b'
GROUPS = ('Sim3/scene_001', 'Sim3/scene_002', 'Sim3/scene_003', 'Sim3/scene_004', 'Sim5/3')
KINDS = ('Object Matching', 'Object Grounding')


def immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Frozen file differs: ' + str(path))
    else:
        with path.open('xb') as handle:
            handle.write(data)


def freeze(protocol, public):
    strata = defaultdict(list)
    for episode_id, assignment in protocol['assignments'].items():
        if assignment['split'] == 'train' and assignment['question_type'] in KINDS:
            strata[(assignment['source_group'], assignment['question_type'])].append(episode_id)
    ids, selection = [], []
    for group in GROUPS:
        for kind in KINDS:
            eligible = strata[(group, kind)]
            if not eligible:
                raise ValueError('An intended training stratum is empty')
            chosen = min(eligible, key=lambda value: hashlib.sha256(('17/' + value).encode()).hexdigest())
            ids.append(chosen)
            selection.append(dict(id=chosen, source_group=group, question_type=kind, eligible_count=len(eligible)))
    episodes = [public[i] for i in ids]
    if len(set(ids)) != 10 or set(ids) & set(protocol['smoke_ids']):
        raise ValueError('Training batch overlaps development smoke or duplicates IDs')
    max_calls = sum(len(e['views']) + 1 + 3*(len(e['views'])-1) for e in episodes)
    record = dict(schema='public_mcq_teacher_batch_v1', dataset_revision=REVISION,
                  grouped_protocol_sha256=sha(protocol), public_episodes_sha256=sha(episodes), episode_ids=ids,
                  selection=selection, selection_seed=17, selection_rule='Minimum SHA256 of 17/ID in each fixed TRAIN group/task stratum; no answer access',
                  episode_count=10, max_model_calls=max_calls, max_new_tokens_per_observation=256,
                  max_regions_per_view=2, max_image_pixels=802816, max_episode_seconds=180,
                  max_batch_seconds=1800, external_timeout_seconds=1920, max_powered_seconds=2400,
                  automatic_model_retries=0, phase='offline_teacher_interface_and_target_acquisition',
                  scope='Grouped training only; no heldout or official-test inference; not a formal efficacy comparison',
                  communication_scope='Counterfactual packet lengths recorded; no actual distributed transfer claimed',
                  primary_backbone='Qwen3-VL-8B-Instruct',
                  regression_backbone='Qwen2.5-VL-7B-Instruct only if explicitly selected; never relabeled primary')
    return record, episodes


def fetch_images(episodes, output, deadline):
    from PIL import Image
    files = []
    for name in sorted({v['path'] for e in episodes for v in e['views']}):
        name = relative_image_path(name)
        path = output/'images'/name
        if path.exists():
            data = path.read_bytes()
        else:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Image acquisition deadline reached')
            url = f'https://huggingface.co/datasets/EasonFan/AirCopBench/resolve/{REVISION}/' + quote(name, safe='/')
            chunks, size = [], 0
            with urlopen(url, timeout=min(20, remaining)) as response:
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Image acquisition deadline reached')
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 20*1024**2:
                        raise ValueError('Unexpectedly large source image')
                    chunks.append(chunk)
            data = b''.join(chunks)
        with Image.open(io.BytesIO(data)) as image:
            image.load(); width, height = image.size
        immutable(path, data)
        files.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), width=width, height=height))
    manifest = dict(schema='public_mcq_teacher_assets_v1', dataset_revision=REVISION,
                    episode_ids=[e['id'] for e in episodes], files=files, official_test_used=False)
    immutable(output/'image_manifest.json', (canonical(manifest)+'\n').encode('utf-8'))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fetch-images', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((args.protocol/'grouped_development.json').read_text(encoding='utf-8'))
    public = {e['id']:e for e in json.loads((args.protocol/'public/train.json').read_text(encoding='utf-8'))}
    record, episodes = freeze(protocol, public)
    immutable(args.output/'batch.json', (canonical(record)+'\n').encode('utf-8'))
    immutable(args.output/'public_episodes.json', (canonical(episodes)+'\n').encode('utf-8'))
    if args.fetch_images:
        try:
            manifest = fetch_images(episodes, args.output, time.monotonic()+900)
            print(json.dumps(dict(images=len(manifest['files']), acquisition='completed')))
        except Exception as exc:
            failure = dict(state='failed',error_type=type(exc).__name__,observed_unix=time.time(),
                           model_calls=0,retained_images=[str(p.relative_to(args.output)) for p in (args.output/'images').rglob('*') if p.is_file()])
            immutable(args.output/f'acquisition_failure_{time.time_ns()}.json', canonical(failure).encode())
            raise
    print(json.dumps(dict(episodes=len(episodes), max_model_calls=record['max_model_calls'],
                         unique_images=len({v['path'] for e in episodes for v in e['views']}), image_acquisition_requested=args.fetch_images)))


if __name__ == '__main__':
    main()
