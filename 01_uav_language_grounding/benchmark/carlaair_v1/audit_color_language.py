"""Private RGB color-cue precheck after exact actor-mask collection.

An actor's blueprint color attribute is only a request. This audit tests the
actual UAV RGB pixels belonging to that actor. A strict majority in one visible
view is a conservative candidate screen, not human recognizability or episode
certification. No private color measurement is exported to policy inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import cv2
import numpy as np


COLOR_RGB = {'red': (180, 20, 20), 'blue': (20, 40, 180),
             'white': (240, 240, 240), 'yellow': (200, 160, 20)}
MIN_VISIBLE_PIXELS = 16  # Same observed-instance floor as the private scorer.


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or not path.parts or '..' in path.parts or
            '\\' in name or ':' in name):
        raise ValueError('Unsafe relative actor-mask or image path')
    return path


def cue_mask(hsv, color):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if color == 'red':
        return ((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 50)
    if color == 'blue':
        return (h >= 100) & (h <= 135) & (s >= 80) & (v >= 50)
    if color == 'yellow':
        return (h >= 15) & (h <= 40) & (s >= 80) & (v >= 50)
    if color == 'white':
        return (s <= 45) & (v >= 180)
    raise ValueError('Unsupported color cue')


def required_roles(scene):
    instruction = scene['instruction'].lower()
    target_match = re.search(r'\b(red|blue|white) car\b', instruction)
    if (scene.get('task_family')=='cross_view_disambiguation'
            and re.fullmatch(r'find the car near the van (?:north|south|east|west) of the fire engine\. use world directions\.', instruction)):
        # This frozen task specifies spatial relations and actor classes, with
        # no color predicate. Keep all RGB/hash/physical gates; do not invent a
        # color requirement from the construction actor's paint attribute.
        return {}
    if not target_match or 'yellow van' not in instruction:
        raise ValueError('Unsupported explicit color-language task')
    roles={'target': target_match.group(1), 'reference': 'yellow'}
    if scene.get('task_family')=='progressive_discovery':
        if 'blue car' not in instruction:raise ValueError('Progressive instruction omitted marker color')
        roles.update(direction_marker='blue',alternative_reference='yellow')
    if scene.get('construction_version')=='oct07_two_round_geometry_rgb_v4':
        # Same-color alternatives are part of the ambiguity, not decorative
        # actors. Verify their requested cues as well as the true target.
        for obj in scene['objects']:
            if obj['key'].startswith('distractor_') and obj.get('color'):
                roles[obj['key']]=target_match.group(1)
    return roles


def audit_payloads(scene_raw, record_raw, public_raw, read_image, read_mask):
    scene, record, public = [json.loads(raw) for raw in
                             (scene_raw, record_raw, public_raw)]
    if (sha(scene_raw) != record['source_sha256'] or
            not (scene['episode_id'] == record['episode_id'] == public['episode_id'])):
        raise ValueError('Scene and observation source binding differs')
    if not (scene['instruction'] == record['instruction'] == public['instruction']):
        raise ValueError('Private/public instruction differs')
    roles = required_roles(scene)
    source_objects = {obj['key']: obj for obj in scene['objects']}
    if not set(roles).issubset(source_objects):
        raise ValueError('Missing color-cued source actor')
    for role, color in roles.items():
        color_text = source_objects[role].get('color')
        if color_text is None:
            raise ValueError(f'Source omitted {role} color')
        if tuple(int(value.strip()) for value in color_text.split(',')) != COLOR_RGB[color]:
            raise ValueError(f'Source color differs from instruction for {role}')
    public_obs = {row['image']: row for row in public['observations']}
    private_obs = {row['image']: row for row in record['observations']}
    if (not public_obs or len(public_obs) != len(public['observations']) or
            len(private_obs) != len(record['observations']) or
            set(public_obs) != set(private_obs) or
            set(public_obs) != set(record['private_pixel_measurements'])):
        raise ValueError('Public, private and mask-view inventory differs')

    readback = {}
    for role, color in roles.items():
        actor = record['private_native_instances'][role]
        requested = actor.get('color_requested_rgb')
        runtime = actor.get('color_runtime_readback_rgb')
        expected = list(COLOR_RGB[color])
        readback[role] = {'expected_rgb': expected,
                          'source_request_matches': True,
                          'requested_rgb': requested,
                          'runtime_readback_rgb': runtime,
                          'verified': requested == runtime == expected}

    rows = []
    for view in sorted(public_obs):
        public_row, private_row = public_obs[view], private_obs[view]
        if (public_row['sha256'] != private_row['sha256'] or
                public_row['source_id'] != private_row['source_id']):
            raise ValueError(f'Public/private image identity differs: {view}')
        raw = read_image(str(safe_relative(view)))
        if sha(raw) != public_row['sha256']:
            raise ValueError(f'Public RGB hash differs: {view}')
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (480, 640, 3):
            raise ValueError(f'Unexpected RGB image dimensions: {view}')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for role, color in roles.items():
            measure = record['private_pixel_measurements'][view][role]
            if (measure.get('visible_aba_exact') is not True or
                    measure.get('instance_mask_verified') is not True and
                    measure['visible_pixels'] > 0):
                raise ValueError(f'Exact visible-mask check failed: {view}/{role}')
            mask_name = str(safe_relative(measure['visible_mask']))
            mask_raw = read_mask(mask_name)
            if sha(mask_raw) != measure['visible_mask_sha256']:
                raise ValueError(f'Actor-mask hash differs: {view}/{role}')
            mask = cv2.imdecode(np.frombuffer(mask_raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (480, 640):
                raise ValueError(f'Unexpected actor-mask dimensions: {view}/{role}')
            pixels = int(np.count_nonzero(mask))
            if pixels != measure['visible_pixels']:
                raise ValueError(f'Actor-mask pixel count differs: {view}/{role}')
            cue_pixels = int(np.count_nonzero(cue_mask(hsv, color) & (mask > 0)))
            rows.append({'view_id': view, 'role': role, 'cue': color,
                         'visible_actor_pixels': pixels, 'cue_actor_pixels': cue_pixels,
                         'cue_fraction': cue_pixels / pixels if pixels else None,
                         'evaluable': pixels >= MIN_VISIBLE_PIXELS,
                         'strict_majority_cue': pixels >= MIN_VISIBLE_PIXELS and 2 * cue_pixels > pixels,
                         'image_sha256': sha(raw), 'exact_mask_sha256': sha(mask_raw)})
    per_role = {}
    for role in roles:
        subset = [row for row in rows if row['role'] == role]
        per_role[role] = {'cue': roles[role],
                          'evaluable_views': sum(row['evaluable'] for row in subset),
                          'strict_majority_views': sum(row['strict_majority_cue'] for row in subset),
                          'cue_pixels': sum(row['cue_actor_pixels'] for row in subset),
                          'visible_actor_pixels': sum(row['visible_actor_pixels'] for row in subset),
                          'runtime_color_readback_verified': readback[role]['verified']}
    cue_screen = all(row['strict_majority_views'] >= 1 for row in per_role.values())
    readback_screen = all(row['runtime_color_readback_verified'] for row in per_role.values())
    status = ('blocked_observed_color_cue' if not cue_screen else
              'pending_runtime_attribute_readback' if not readback_screen else
              'candidate_requires_native_rgb_review')
    return {'schema': 'private_color_language_precheck_v1',
            'color_predicates_requested': bool(roles),
            'episode_id': scene['episode_id'], 'state': status,
            'source_scene_sha256': sha(scene_raw),
            'private_record_sha256': sha(record_raw),
            'public_manifest_sha256': sha(public_raw),
            'role_cues': roles, 'runtime_color_attributes': readback,
            'rows': rows, 'per_role': per_role,
            'cue_majority_screen_passed': cue_screen,
            'runtime_attribute_screen_passed': readback_screen,
            'native_rgb_review_required': True,
            'benchmark_admitted': False,
            'formal_test_episodes_added': 0,
            'rule': ('For every explicit actor color, at least one exact-mask view with >=16 '
                     'visible actor pixels must have a strict majority of pixels in the '
                     'predeclared color cue. This is a conservative precheck only.'),
            'limits': ('A passed cue screen does not establish human recognition, full '
                       'background labeling, runtime certification or method accuracy.')}


def audit_files(scene_path, record_path, output_path=None):
    scene_path, record_path = Path(scene_path), Path(record_path)
    folder = record_path.parent
    public_path = folder / 'public_observations.json'
    report = audit_payloads(scene_path.read_bytes(), record_path.read_bytes(),
                            public_path.read_bytes(),
                            lambda name: (folder / name).read_bytes(),
                            lambda name: (folder / name).read_bytes())
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(output_path)
        output_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_files(args.scene, args.record, args.output)
    print(json.dumps({'episode_id': result['episode_id'], 'state': result['state'],
                      'cue_majority_screen_passed': result['cue_majority_screen_passed']}))
