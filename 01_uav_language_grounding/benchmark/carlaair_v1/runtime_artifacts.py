"""Immutable job inputs and independent, fail-closed collection accounting."""
import hashlib
import json
from pathlib import Path


def freeze_code(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(exist_ok=False)
    files = {}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix not in ('.py', '.json', '.sh'):
            continue
        if path.name == 'snapshot_manifest.json':
            # This file describes an earlier snapshot. A new snapshot writes its
            # own manifest at the same destination path.
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink not permitted in runtime code: {path.name}')
        data = path.read_bytes()
        target = destination / path.name
        target.write_bytes(data)
        target.chmod(0o444)
        files[path.name] = hashlib.sha256(data).hexdigest()
    if not files:
        raise ValueError('Empty runtime snapshot')
    manifest = destination / 'snapshot_manifest.json'
    manifest.write_text(json.dumps({'source': str(source), 'files': files}, indent=2))
    manifest.chmod(0o444)
    destination.chmod(0o555)
    return verify_code(destination)


def verify_code(directory):
    directory = Path(directory)
    raw = (directory / 'snapshot_manifest.json').read_bytes()
    manifest = json.loads(raw)
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe snapshot filename')
        path = directory / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Job source drift: {name}')
    return {'directory': str(directory), 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'file_count': len(manifest['files'])}


def audit_collection(directory, assigned, team, rounds, pixel_audit=False,
                     static_visual_wind=False, fast_instance_screen=False):
    """Do not trust a child's return code or its aggregate counters alone."""
    from PIL import Image
    directory = Path(directory)
    status = json.loads((directory / 'status.json').read_text())
    if assigned < 1 or rounds < 1:
        raise ValueError('Expected nonempty collection')
    if (status.get('stage') != 'collection_finished_not_validated'
            or status.get('assigned_scenes') != assigned
            or status.get('completed_scenes') != assigned or status.get('failed_scenes') != 0):
        raise RuntimeError('Independent parent audit rejected partial/failed collection')
    records = sorted(directory.glob('*/observations.json'))
    if len(records) != assigned:
        raise RuntimeError('Episode record count does not match assignment')
    count = 0
    for path in records:
        record = json.loads(path.read_text())
        if (record.get('stage') != 'actual_views_collected_not_validated'
                or record.get('team_size') != team or any(record.get('new_collisions', {}).values())):
            raise RuntimeError(f'Incomplete/colliding scene: {path.parent.name}')
        expected = {f'round{r:03d}_UAV_{i}.png' for r in range(rounds) for i in range(team)}
        observations = record['observations']
        if len(observations) != len(expected) or {o['image'] for o in observations} != expected:
            raise RuntimeError('Missing or duplicate observations')
        public = json.loads((path.parent / 'public_observations.json').read_text())
        from collect_runtime_views import public_observation
        if public['observations'] != [public_observation(o) for o in observations]:
            raise RuntimeError('Public/private observation mismatch or privileged fields leaked')
        for obs in observations:
            if static_visual_wind:
                for key in ('visual_weather_profile_before_rgb', 'visual_weather_profile_after_mask'):
                    profile = obs.get(key, {})
                    if (profile.get('wind_intensity') != 0
                            or profile.get('dynamic_weather_annotation_validated') is not False
                            or profile.get('scope') != 'visual_weather_not_flight_wind'):
                        raise RuntimeError('Missing or inconsistent static visual weather evidence')
            image_path = path.parent / obs['image']
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != obs['sha256']:
                raise RuntimeError('RGB content hash mismatch')
            with Image.open(image_path) as image:
                image.load()
                if image.size != (640, 480) or image.format != 'PNG':
                    raise RuntimeError('Unexpected RGB shape/format')
            if pixel_audit:
                measurements = record.get('private_pixel_measurements', {}).get(obs['image'], {})
                if set(measurements) != set(record['private_native_instances']):
                    raise RuntimeError('Missing instance pixel annotations')
                for values in measurements.values():
                    if not values.get('visible_aba_exact') or not values.get('isolated_aba_exact'):
                        raise RuntimeError('Missing exact stencil restoration controls')
                    for mask_key, hash_key in (('visible_mask', 'visible_mask_sha256'),
                                               ('isolated_candidate_mask', 'isolated_mask_sha256')):
                        mask_path = (path.parent / values[mask_key]).resolve()
                        if path.parent.resolve() not in mask_path.parents:
                            raise RuntimeError('Mask path escaped episode')
                        if hashlib.sha256(mask_path.read_bytes()).hexdigest() != values[hash_key]:
                            raise RuntimeError('Mask content hash mismatch')
            if fast_instance_screen:
                measurements = record.get('private_fast_instance_measurements', {}).get(obs['image'], {})
                if set(measurements) != set(record['private_native_instances']):
                    raise RuntimeError('Missing fixed-ID instance screen annotations')
                for values in measurements.values():
                    if values.get('visible_aba_exact') or values.get('scientific_visibility_certified'):
                        raise RuntimeError('A screening mask was mislabeled as certified')
                    mask_path = (path.parent / values['visible_mask_screen']).resolve()
                    if path.parent.resolve() not in mask_path.parents:
                        raise RuntimeError('Screen mask path escaped episode')
                    if hashlib.sha256(mask_path.read_bytes()).hexdigest() != values['visible_mask_screen_sha256']:
                        raise RuntimeError('Screen mask content hash mismatch')
                    with Image.open(mask_path) as mask_image:
                        import numpy as np
                        mask_pixels = np.asarray(mask_image.convert('L'))
                    if (mask_pixels.shape != (480, 640)
                            or not np.isin(mask_pixels, (0, 255)).all()
                            or int(np.count_nonzero(mask_pixels)) != values['visible_pixels_screen']):
                        raise RuntimeError('Screen mask shape, binary values or count mismatch')
                    if values['mask_pixels_inside_projected_envelope'] != values['visible_pixels_screen']:
                        raise RuntimeError('Fixed-ID pixels escaped the native actor projected envelope')
            count += 1
    if count != status.get('observations'):
        raise RuntimeError('Annotated observation count mismatch')
    return {'complete_scenes': assigned, 'verified_rgb': count,
            'static_visual_weather_checks': bool(static_visual_wind),
            'dataset_ready': False, 'runtime_validated_episodes': 0,
            'scope': 'sensor collection integrity only; not benchmark/algorithm validation'}
