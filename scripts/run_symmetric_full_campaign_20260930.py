"""Offline, bounded, resumable 2 x 231 AirCop DEV constrained inference.

Validate is the default. --run requires a new explicit launch contract. Never
read answer keys. An interrupted GPU session needs manual accounting before
resume; the runner does not replay an uncertain generate call.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'evidence/baseline_first_20260928_v1/assets'
GRAMMAR = ROOT / 'evidence/output_grammar_cpu_20260929_v1/protocol.json'
QUAL = ROOT / 'evidence/non_gpu_maintenance_20260930_v1'
MODELS = {
    'qwen3': ('/path/to/aihcir/models/Qwen3-VL-8B-Instruct_0c351dd_modelscope', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
    'internvl35_hf': ('/path/to/aihcir/models/InternVL3_5-8B-HF_741a7d0', '741a7d03020411e666c6109218ab71e08151ef86'),
}
PROCESSOR_KWARGS = {'qwen3': {'padding': True}, 'internvl35_hf': {'crop_to_patches': True, 'min_patches': 1, 'max_patches': 3}}
QUALIFIED_FILES = {'qwen3': {'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'chat_template.json', 'config.json', 'preprocessor_config.json', 'video_preprocessor_config.json'},
                   'internvl35_hf': {'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja', 'config.json', 'preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json'}}
EXPECTED = 462


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def event(out, item):
    path = out / 'events.jsonl'
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(item, observed_unix=time.time()), ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_contract(contract, now=None):
    now = time.time() if now is None else now
    if contract.get('schema') != 'symmetric_full_campaign_contract_v1' or contract.get('run_class') != 'development_constrained_separate':
        raise ValueError('New full campaign contract required; old blocked contract rejected')
    if contract.get('launch_authorized') is not True or contract.get('reuses_closed_contract') is not False or not contract.get('launch_id'):
        raise ValueError('Explicit new launch authorization required')
    if not positive(contract.get('absolute_deadline_unix')) or contract['absolute_deadline_unix'] <= now:
        raise ValueError('Absolute deadline expired')
    budget = contract.get('new_budget') or {}
    for key in ('max_model_calls', 'max_gpu_seconds', 'max_cost'):
        if not positive(budget.get(key)):
            raise ValueError('Positive new budget required: ' + key)
    if type(budget['max_model_calls']) is not int or budget['max_model_calls'] != EXPECTED or not positive(contract.get('cost_per_gpu_second')):
        raise ValueError('462-call capacity and positive cost rate required')
    if budget['max_gpu_seconds'] * contract['cost_per_gpu_second'] > budget['max_cost']:
        raise ValueError('Cost budget does not cover GPU seconds budget')
    runtime = contract.get('runtime_qualification') or {}
    if runtime.get('transformers_version') != '4.57.1' or not str(runtime.get('torch_version', '')).startswith('2.8.0') or runtime.get('both_exact_processors_validated') is not True or runtime.get('cpu_compatibility_alias') is not False:
        raise ValueError('Exact native processor runtime gate closed')
    if contract.get('planned_units') != EXPECTED or contract.get('seed') != 17:
        raise ValueError('Fixed 462 matrix and seed 17 required')
    if set(contract.get('models', {})) != set(MODELS):
        raise ValueError('Both frozen models required')
    for key, (path, revision) in MODELS.items():
        model = contract['models'][key]
        if model.get('directory') != path or model.get('revision') != revision or not isinstance(model.get('files'), dict) or not model['files']:
            raise ValueError('Pinned local model map and file hashes required: ' + key)
    paths = contract.get('asset_paths') or {}
    if set(paths) != {'assets_dir', 'grammar', 'processor_qualification', 'processor_manifest'}:
        raise ValueError('Explicit offline asset paths required')
    return True


def validate_assets(contract):
    bindings = {'episodes': ASSETS / 'public_episodes.json', 'image_manifest': ASSETS / 'image_manifest.json',
                'grammar': GRAMMAR, 'processor_qualification': QUAL / 'exact_processor_qualification.json',
                'processor_manifest': QUAL / 'processor_manifest.json',
                'generator_source': ROOT / 'scripts/run_symmetric_development_20260929.py',
                'grammar_source': ROOT / 'scripts/qualify_output_grammar_cpu_20260929.py'}
    for name, path in bindings.items():
        if sha(path) != contract.get('asset_hashes', {}).get(name):
            raise ValueError('Frozen asset hash mismatch: ' + name)
    episodes = read(bindings['episodes']); manifest = read(bindings['image_manifest'])
    protocol = read(GRAMMAR); qualification = read(bindings['processor_qualification'])
    if len(episodes) != 231 or [x['id'] for x in episodes] != manifest['episode_ids'] or len(set(manifest['episode_ids'])) != 231:
        raise ValueError('Frozen DEV cohort mismatch')
    if any(set(x) != {'schema', 'id', 'question', 'options', 'views'} or set(x['options']) != set('ABCD') for x in episodes):
        raise ValueError('Public question schema changed or answer leakage')
    if qualification.get('both_exact_processors_validated') is not True or qualification.get('manifest_sha256') != sha(bindings['processor_manifest']):
        raise ValueError('Native processor qualification mismatch')
    if protocol.get('grammar_parameters') != read(bindings['processor_manifest'])['grammar_parameters']:
        raise ValueError('Grammar binding mismatch')
    return episodes, manifest, protocol


def verify_model_files(key, model_binding, processor_manifest):
    directory = Path(model_binding['directory'])
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError('Local model directory absent or symlink: ' + key)
    declared = model_binding['files']
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != set(declared) or any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('Model file inventory changed or symlink present: ' + key)
    if not any(name.endswith('.safetensors') for name in actual):
        raise ValueError('Weight shards absent: ' + key)
    if not QUALIFIED_FILES[key].issubset(actual):
        raise ValueError('Qualified processor files absent: ' + key)
    verified = {}
    for name in sorted(actual):
        path = directory / name
        expected = declared[name]
        if not isinstance(expected, dict) or expected.get('size') != path.stat().st_size or expected.get('sha256') != sha(path):
            raise ValueError('Model file hash/size mismatch: ' + key + '/' + name)
        verified[name] = {'size': expected['size'], 'sha256': expected['sha256']}
        pinned = processor_manifest['files'].get('sources/' + key + '/' + name)
        if name in QUALIFIED_FILES[key] and pinned != expected['sha256']:
            raise ValueError('Processor source changed: ' + key + '/' + name)
    return verified


def matrix(episodes):
    return [(key, episode) for key in MODELS for episode in episodes]


def replay(out, binding, expected_pairs):
    path = out / 'events.jsonl'
    rows = {}; reserved = {}; sessions = []
    if not path.exists():
        return rows, reserved, sessions
    for line in path.read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        kind = item.get('type'); pair = (item.get('model'), item.get('episode_id'))
        if kind == 'session_start':
            if item.get('binding_sha256') != binding or (sessions and sessions[-1].get('type') != 'session_end'):
                raise ValueError('Unclosed or mismatched GPU session; manual diagnosis required')
            sessions.append(item)
        elif kind == 'session_end':
            if not sessions or sessions[-1].get('type') != 'session_start':
                raise ValueError('Malformed session journal')
            sessions.append(item)
        elif kind == 'reserve':
            if pair not in expected_pairs or pair in reserved or pair in rows or item.get('binding_sha256') != binding:
                raise ValueError('Duplicate/foreign reservation')
            reserved[pair] = item
        elif kind == 'result':
            if pair not in reserved or pair in rows or item.get('binding_sha256') != binding:
                raise ValueError('Result without matching reservation')
            p = out / 'rows' / pair[0] / (pair[1] + '.json')
            if not p.exists() or sha(p) != item.get('file_sha256'):
                raise ValueError('Row file missing or changed')
            row = read(p)
            if row.get('status') != 'completed' or row.get('grammar_valid') is not True or row.get('binding_sha256') != binding:
                raise ValueError('Only verified completed rows may resume')
            rows[pair] = row
        elif kind in ('start', 'failure'):
            if pair not in reserved:
                raise ValueError('Start/failure without reservation')
        else:
            raise ValueError('Unknown journal event')
    if sessions and sessions[-1]['type'] != 'session_end':
        raise ValueError('Unclosed GPU session; manual accounting required')
    if set(reserved) != set(rows):
        raise ValueError('Incomplete charged attempt; manual diagnosis required')
    return rows, reserved, sessions


def progress(out, completed, calls, gpu_seconds, budget, state, current=None):
    stage = {'preflight': 'preflight', 'loading_model': 'loading_model', 'running': 'acquiring',
             'computed': 'completed', 'partial': 'failed', 'failed': 'failed'}[state]
    previous = read(out/'status.json') if (out/'status.json').exists() else {}
    if current is None and stage == 'acquiring':
        current = previous.get('current')
    failures = max(0, calls-completed) if state in ('failed','partial') else 0
    inflight = max(0, calls-completed-failures)
    atomic_json(out / 'status.json', {'stage': stage, 'phase': state, 'planned_questions': EXPECTED,
        'max_model_calls': budget['max_model_calls'], 'actual_model_calls': calls,
        'planned_units': EXPECTED, 'unit': 'model-question generate call', 'completed': completed,
        'succeeded': completed, 'failed': failures, 'running': inflight,
        'pending': EXPECTED - completed - failures - inflight, 'charged_model_calls': calls,
        'charged_gpu_seconds': gpu_seconds, 'charged_cost_upper_bound': gpu_seconds * budget['cost_per_gpu_second'],
        'state': state, 'current': current, 'current_episode': (current['model'] + ':' + current['episode_id']) if current else None,
        'heartbeat_unix': time.time(), 'observed_unix': time.time(), 'scientific_acceptance': None, 'user_acceptance': None})


def launch(contract, out):
    validate_contract(contract)
    episodes, manifest, protocol = validate_assets(contract)
    expected_pairs = {(key, e['id']) for key, e in matrix(episodes)}
    binding = digest({k: contract[k] for k in ('schema', 'launch_id', 'new_budget', 'absolute_deadline_unix', 'asset_hashes', 'models', 'seed')})
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'runner.lock'
    with lock.open('x') as f:
        f.write(json.dumps({'pid': os.getpid(), 'start_unix': time.time(), 'binding_sha256': binding}))
        f.flush(); os.fsync(f.fileno())
    clean = False
    try:
        rows, reservations, sessions = replay(out, binding, expected_pairs)
        used_seconds = sum(end['gpu_seconds'] for start, end in zip(sessions[::2], sessions[1::2]))
        calls = len(reservations)
        budget = dict(contract['new_budget'], cost_per_gpu_second=contract['cost_per_gpu_second'])
        progress(out, len(rows), calls, used_seconds, budget, 'preflight')
        processor_manifest = read(QUAL / 'processor_manifest.json')
        source_hashes = {key: verify_model_files(key, contract['models'][key], processor_manifest) for key in MODELS}
        # Imported only after offline and hash preflight, so default validation never imports torch.
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration
        from run_symmetric_development_20260929 import generate_one, prompt_for
        if transformers.__version__ != '4.57.1' or not torch.__version__.startswith('2.8.0') or not torch.cuda.is_available():
            raise ValueError('Qualified CUDA runtime unavailable')
        torch.set_num_threads(4); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        index = {item['path']: item for item in manifest['files']}
        for key in MODELS:
            remaining = [e for e in episodes if (key, e['id']) not in rows]
            if not remaining:
                continue
            if time.time() >= contract['absolute_deadline_unix'] or used_seconds >= budget['max_gpu_seconds']:
                raise TimeoutError('Total deadline or GPU budget reached')
            progress(out, len(rows), calls, used_seconds, budget, 'loading_model' if key == 'qwen3' else 'running')
            session_start = time.monotonic()
            event(out, {'type': 'session_start', 'model': key, 'binding_sha256': binding})
            directory = contract['models'][key]['directory']
            processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                **({'min_pixels': 256 * 28 * 28, 'max_pixels': 1024 * 28 * 28} if key == 'qwen3' else {}))
            model_class = Qwen3VLForConditionalGeneration if key == 'qwen3' else AutoModelForImageTextToText
            model = model_class.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
            model.requires_grad_(False)
            for episode in remaining:
                spent = time.monotonic() - session_start
                total = used_seconds + spent
                if calls >= budget['max_model_calls'] or total >= budget['max_gpu_seconds'] or total * budget['cost_per_gpu_second'] >= budget['max_cost'] or time.time() >= contract['absolute_deadline_unix']:
                    raise TimeoutError('Call, GPU, cost or deadline limit reached')
                images = []; image_hashes = {}
                for view in episode['views']:
                    item = index[view['path']]; path = ASSETS / 'images' / view['path']
                    if sha(path) != item['sha256']:
                        raise ValueError('Input image changed: ' + view['path'])
                    im = Image.open(path).convert('RGB')
                    if im.size != (item['width'], item['height']):
                        raise ValueError('Image dimensions changed')
                    images.append(im); image_hashes[view['path']] = item['sha256']
                prompt = prompt_for(episode, processor)
                inputs = processor(text=[prompt], images=images, return_tensors='pt', **PROCESSOR_KWARGS[key])
                if int(inputs['input_ids'].shape[0]) != 1:
                    raise ValueError('Batch size must be one')
                inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
                if key == 'internvl35_hf':
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
                metadata = {'model': key, 'episode_id': episode['id'], 'binding_sha256': binding,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'source_hashes': source_hashes[key],
                    'image_hashes': image_hashes, 'cohort_sha256': contract['asset_hashes']['episodes'],
                    'model_revision': MODELS[key][1], 'processor_kwargs': PROCESSOR_KWARGS[key], 'seed': 17}
                pair = (key, episode['id'])
                event(out, {'type': 'reserve', 'model': key, 'episode_id': episode['id'], 'binding_sha256': binding, 'call_number': calls + 1})
                calls += 1
                progress(out, len(rows), calls, used_seconds + time.monotonic() - session_start, budget, 'running', {'model': key, 'episode_id': episode['id']})
                event(out, {'type': 'start', 'model': key, 'episode_id': episode['id']})
                try:
                    remaining_seconds = min(120, budget['max_gpu_seconds'] - (used_seconds + time.monotonic() - session_start),
                                            (budget['max_cost'] / budget['cost_per_gpu_second']) - (used_seconds + time.monotonic() - session_start))
                    if remaining_seconds <= 0:
                        raise TimeoutError('No generation budget remains')
                    gate = {'new_budget': contract['new_budget'], 'absolute_deadline_unix': contract['absolute_deadline_unix'],
                            'launch_authorized': True, 'launch_id': contract['launch_id'], 'reuses_closed_contract': False,
                            'runtime_qualification': dict(contract['runtime_qualification'], weights_hashes_verified=True),
                            'actual_model_calls': calls - 1}
                    with torch.inference_mode():
                        row = generate_one(model, inputs, processor, protocol['grammar_parameters'][key], metadata,
                            contract['absolute_deadline_unix'], max_seconds=remaining_seconds, launch_contract=gate)
                    row['generation_config']['prefix_allowed_tokens_fn'] = 'SuffixTrie.callback(actual_input_width)'
                    row['charged_call_number'] = calls
                    path = out / 'rows' / key / (episode['id'] + '.json')
                    atomic_json(path, row)
                    if row['status'] != 'completed':
                        raise ValueError('Invalid constrained output; no automatic replay')
                    event(out, {'type': 'result', 'model': key, 'episode_id': episode['id'], 'binding_sha256': binding, 'file_sha256': sha(path)})
                    rows[pair] = row
                except Exception as exc:
                    event(out, {'type': 'failure', 'model': key, 'episode_id': episode['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
                    raise
                progress(out, len(rows), calls, used_seconds + time.monotonic() - session_start, budget, 'running')
            del model; del processor
            torch.cuda.empty_cache()
            elapsed = time.monotonic() - session_start
            used_seconds += elapsed
            event(out, {'type': 'session_end', 'model': key, 'gpu_seconds': elapsed, 'binding_sha256': binding})
        progress(out, len(rows), calls, used_seconds, budget, 'computed' if len(rows) == EXPECTED else 'partial')
        clean = True
    except Exception:
        if 'budget' in locals():
            progress(out, len(rows) if 'rows' in locals() else 0, calls if 'calls' in locals() else 0,
                     used_seconds + (time.monotonic() - session_start if 'session_start' in locals() else 0) if 'used_seconds' in locals() else 0,
                     budget, 'failed')
        raise
    finally:
        if clean:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Perform authorized offline GPU inference; default is validation only')
    args = parser.parse_args()
    contract = read(args.contract)
    validate_contract(contract)
    global ASSETS, GRAMMAR, QUAL
    ASSETS = Path(contract['asset_paths']['assets_dir'])
    GRAMMAR = Path(contract['asset_paths']['grammar'])
    QUAL = Path(contract['asset_paths']['processor_manifest']).parent
    if QUAL / 'exact_processor_qualification.json' != Path(contract['asset_paths']['processor_qualification']) or QUAL / 'processor_manifest.json' != Path(contract['asset_paths']['processor_manifest']):
        raise ValueError('Qualification and manifest paths must share their standard directory')
    episodes, _, _ = validate_assets(contract)
    assert len(matrix(episodes)) == EXPECTED
    if args.run:
        launch(contract, args.output)
    else:
        print(json.dumps({'status': 'validated_no_model_load', 'planned_units': EXPECTED, 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
