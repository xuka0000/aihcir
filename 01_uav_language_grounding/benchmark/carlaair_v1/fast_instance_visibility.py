"""Development-only single-pass instance visibility screen.

Five fixed stencil IDs are assigned once per scene. The color lookup is the
AirSim palette hypothesis and must be checked against same-run ABA masks before
any scientific visibility claim. This module never produces amodal support.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from audit_observation_coverage import projected_support
from pixel_visibility import PixelVisibilityAudit


PALETTE = {
    0: (0, 0, 0),
    1: (153, 108, 6),
    2: (112, 105, 191),
    3: (89, 121, 72),
    4: (190, 225, 64),
    5: (206, 190, 59),
}


class FastInstanceVisibility:
    def __init__(self, client, max_captures=32):
        self.client = client
        self.reader = PixelVisibilityAudit(client, max_captures=max_captures)
        self.assignments = {}

    def register(self, key, new_actor_names):
        stencil_id = len(self.assignments) + 1
        if stencil_id not in PALETTE:
            raise ValueError('Single-pass screen supports at most five scene instances')
        names = sorted(new_actor_names)
        matched = []
        for name in names:
            if self.client.simSetSegmentationObjectID(name, stencil_id, False):
                readback = int(self.client.simGetSegmentationObjectID(name))
                if readback not in (-1, stencil_id):
                    raise RuntimeError(f'Stencil ID readback mismatch for {name}: {readback}')
                matched.append((name, readback))
        if not matched:
            raise RuntimeError(f'No newly spawned mesh owner matches {key}')
        self.assignments[key] = {
            'id': stencil_id,
            'mesh_owners': [name for name, _ in matched],
            'id_readback': {name: readback for name, readback in matched},
            'id_readback_verified': all(readback == stencil_id for _, readback in matched),
        }

    def capture(self, agent, observation, instances, folder):
        if set(instances) != set(self.assignments):
            raise RuntimeError('Every native instance needs one fixed stencil assignment')
        out = Path(folder) / 'private_fast_instance_screen'
        out.mkdir(exist_ok=True)
        stem = Path(observation['image']).stem
        frame = self.reader.render(agent, observation, out / (stem + '_segmentation'))
        observed_colors = {
            tuple(int(x) for x in row)
            for row in np.unique(frame.reshape(-1, 3), axis=0)
        }
        expected_colors = set(PALETTE.values())
        unexpected = sorted(observed_colors - expected_colors)
        if unexpected:
            raise RuntimeError(f'Unknown segmentation colors in fixed-ID screen: {unexpected[:8]}')
        output = {}
        for key, obj in instances.items():
            assignment = self.assignments[key]
            color = np.asarray(PALETTE[assignment['id']], dtype=np.uint8)
            mask = np.all(frame == color, axis=2)
            path = out / (stem + '_' + key + '_visible.png')
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            projected = projected_support(observation, obj['bbox_world_vertices'])
            ys, xs = np.where(mask)
            within = 0
            if projected['bbox_xyxy'] is not None:
                x0, y0, x1, y1 = projected['bbox_xyxy']
                within = int(((xs >= math.floor(x0)-1) & (xs <= math.ceil(x1)+1)
                              & (ys >= math.floor(y0)-1) & (ys <= math.ceil(y1)+1)).sum())
            output[key] = {
                'stencil_id': assignment['id'],
                'palette_rgb_hypothesis': list(PALETTE[assignment['id']]),
                'mesh_owners': assignment['mesh_owners'],
                'id_readback': assignment['id_readback'],
                'id_readback_verified': assignment['id_readback_verified'],
                'visible_pixels_screen': int(mask.sum()),
                'mask_pixels_inside_projected_envelope': within,
                'projected': projected,
                'visible_mask_screen': str(path.relative_to(folder)),
                'visible_mask_screen_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'visible_aba_exact': False,
                'isolated_aba_exact': False,
                'amodal_support_measured': False,
                'same_run_aba_equivalence': 'pending',
                'scientific_visibility_certified': False,
            }
        return output

    def calibrate_target(self, agent, observation, instances, folder, screen_result):
        return self.calibrate_instance('target', agent, observation, instances, folder, screen_result)

    def calibrate_instance(self, key, agent, observation, instances, folder, screen_result,
                           measure_only=False):
        """Compare a fixed-ID mask with ABA; diagnostic mode retains disagreement."""
        if key not in self.assignments or key not in instances:
            raise ValueError(f'Unknown native actor for ABA calibration: {key}')
        out = Path(folder) / 'private_fast_instance_screen'
        report_path = out / (Path(observation['image']).stem + f'_{key}_ABA_comparison.json')
        result = {'agent': agent, 'image': observation['image'],
                  'instance_key': key,
                  'timestamp': observation['timestamp'],
                  'same_run_same_paused_view': True,
                  'pixelwise_equal': False,
                  'scientific_visibility_certified_for_other_frames': False}
        try:
            aba = PixelVisibilityAudit(self.client)
            aba.names[key] = list(self.assignments[key]['mesh_owners'])
            exact = aba.capture(agent, observation, {key: instances[key]}, folder)[key]
            with Image.open(Path(folder) / screen_result[key]['visible_mask_screen']) as image:
                fast_mask = np.asarray(image.convert('L')) > 0
            with Image.open(Path(folder) / exact['visible_mask']) as image:
                aba_mask = np.asarray(image.convert('L')) > 0
            mismatched = int(np.count_nonzero(fast_mask != aba_mask))
            result.update(aba_visible_pixels=exact['visible_pixels'],
                          aba_isolated_candidate_pixels=exact['isolated_candidate_pixels'],
                          fast_visible_pixels=screen_result[key]['visible_pixels_screen'],
                          mismatched_pixels=mismatched,
                          pixelwise_equal=mismatched == 0,
                          aba_visible_mask=exact['visible_mask'],
                          aba_visible_mask_sha256=exact['visible_mask_sha256'],
                          fast_visible_mask=screen_result[key]['visible_mask_screen'],
                          fast_visible_mask_sha256=screen_result[key]['visible_mask_screen_sha256'])
            if mismatched and not measure_only:
                raise RuntimeError(f'Fast mask differs from same-frame ABA by {mismatched} pixels')
            return result
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            restore_failures = []
            for assignment in self.assignments.values():
                for name in assignment['mesh_owners']:
                    if not self.client.simSetSegmentationObjectID(name, assignment['id'], False):
                        restore_failures.append(name)
            if restore_failures:
                result['restore_error'] = f'Failed to restore {restore_failures}'
            report_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            if restore_failures:
                raise RuntimeError(result['restore_error'])
