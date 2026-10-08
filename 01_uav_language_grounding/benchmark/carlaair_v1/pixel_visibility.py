"""Private differential-stencil visibility diagnostics on the actual UAV camera.

The binary mask is obtained by changing ONLY a known actor's stencil 0 -> 1
while the simulator is paused. No target location is supplied to the policy.
Isolated-custom-depth masks remain AMODAL CANDIDATES until an occluder control
validates the compiled shader. They must not certify an occlusion class yet.
Source: CarlaAir 46154110... AirBlueprintLib.{h,cpp}, SetObjectStencilID.
"""
import hashlib
import io
import math
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from audit_observation_coverage import projected_support
from native_sensor_storage import before_write

def store_png(array,path):
    before_write(path)
    Image.fromarray(array).save(path)


def difference_mask(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != (480, 640, 3) or second.shape != first.shape:
        raise ValueError('Unexpected segmentation image dimensions')
    return np.any(first != second, axis=2)


def stable_tail(frames):
    return len(frames) >= 3 and all(np.array_equal(frames[-1], f) for f in frames[-3:-1])


class PixelVisibilityAudit:
    def __init__(self, client, max_captures=32):
        if max_captures < 8 or max_captures > 32:
            raise ValueError('Render capture budget must be between 8 and 32')
        self.client = client
        self.max_captures = max_captures
        self.names = {}
        self.render_traces = []
        from native_stencil_state import NativeStencilState
        self.native_stencils=NativeStencilState(client)

    def reset(self, value):
        if not self.client.simSetSegmentationObjectID('.*', value, True):
            raise RuntimeError('Compiled runtime does not expose mesh stencil control')

    def register(self, key, new_actor_names):
        matched = self.native_stencils.remember(new_actor_names)
        if not matched:
            raise RuntimeError(f'No newly spawned mesh owner matches {key}')
        self.names[key] = matched

    def mark(self, key, value):
        for name in self.names[key]:
            if not self.client.simSetSegmentationObjectID(name, value, False):
                raise RuntimeError('Registered instance disappeared during capture')

    def render(self, agent, observation, trace_prefix, debug=False):
        import airsim
        # v3 raw traces changed all 307200 pixels only on request 7 at an
        # identical paused timestamp. Thus consecutive immediate reads are not
        # a render fence. CarlaAir UnrealImageCapture.cpp lines39-42 uses a
        # 200ms synchronization fallback for capture visibility changes.
        # Real-time settling below does not advance the paused simulation.
        time.sleep(.2)
        request = [airsim.ImageRequest('0', airsim.ImageType.Segmentation, False, True)]
        frames, trace, frame_indices, consecutive = [], [], [], []
        started = time.monotonic()
        diagnostic = {'agent': agent, 'image': observation['image'],
                      'expected_timestamp': observation['timestamp'],
                      'max_captures': self.max_captures, 'frames': trace,
                      'accepted': False, 'eight_capture_gate_passed': None}
        try:
            for index in range(self.max_captures):
                response = self.client.simGetImages(request, vehicle_name=agent)[0]
                row = {'request': index, 'timestamp': int(response.time_stamp),
                       'elapsed_seconds': time.monotonic()-started}
                trace.append(row)
                payload = bytes(response.image_data_uint8)
                row.update(payload_bytes=len(payload), payload_sha256=hashlib.sha256(payload).hexdigest())
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        array = np.array(image.convert('RGB'))
                except (UnidentifiedImageError, OSError) as error:
                    # A failed RPC image may have timestamp0 and an undecodable
                    # payload. It is not an observation and cannot support a
                    # mask. Retry only within the ORIGINAL raw-request budget.
                    row['decode_error'] = f'{type(error).__name__}: {error}'
                    invalid=Path(str(trace_prefix)+f'_invalid_{index:02d}.bin');before_write(invalid,len(payload));invalid.write_bytes(payload)
                    consecutive.clear()
                    if response.time_stamp not in (0, observation['timestamp']):
                        raise RuntimeError('Malformed image also has a different paused timestamp')
                    if index == 7:
                        diagnostic['eight_capture_gate_passed'] = False
                    time.sleep(.05)
                    continue
                difference_mask(array, array)  # reject wrong dimensions even on first read
                frames.append(array)
                frame_indices.append(index)
                consecutive.append(array)
                row.update(sha256_raw_pixels=hashlib.sha256(array.tobytes()).hexdigest(),
                           changed_pixels_from_previous=int(difference_mask(frames[-2], array).sum())
                               if len(frames) > 1 else None)
                if hasattr(response, 'camera_position'):
                    p, q = response.camera_position, response.camera_orientation
                    row['camera_local_position'] = [p.x_val, p.y_val, p.z_val]
                    row['camera_orientation_ned_wxyz'] = [q.w_val, q.x_val, q.y_val, q.z_val]
                if response.time_stamp != observation['timestamp']:
                    raise RuntimeError('Pixel annotation is not at the paused RGB timestamp')
                for field in ('camera_local_position', 'camera_orientation_ned_wxyz'):
                    if field in observation and row.get(field) != observation[field]:
                        raise RuntimeError(f'Paused RGB/segmentation camera mismatch: {field}')
                if index == 7:
                    diagnostic['eight_capture_gate_passed'] = len(frames) >= 8 and stable_tail(consecutive)
                # Keep EXACT whole-frame equality, the old minimum of eight reads,
                # and the same paused timestamp. Extra reads only extend the
                # bounded render wait; they never relax a pixel/class threshold.
                if len(frames) >= 8 and stable_tail(consecutive):
                    diagnostic['accepted'] = True
                    return frames[-1]
                time.sleep(.05)
            raise RuntimeError(f'Segmentation render state did not converge in {self.max_captures} captures')
        except Exception as error:
            diagnostic['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            self.render_traces.append(trace)
            # Failure evidence is unconditional, not hidden behind a debug flag.
            # Also keep all raw reads when the old eight-read gate would fail.
            if debug or not diagnostic['accepted'] or diagnostic['eight_capture_gate_passed'] is False:
                for index, array in zip(frame_indices, frames):
                    store_png(array,str(trace_prefix)+f'_{index:02d}.png')
            if frames:
                store_png(frames[-1],str(trace_prefix)+'_last.png')
            Path(str(trace_prefix)+'_trace.json').write_text(json.dumps(diagnostic, indent=2))

    def capture(self, agent, observation, instances, folder, debug=False):
        out = Path(folder) / 'private_pixel_audit'
        out.mkdir(exist_ok=True)
        output = {}
        try:
            for key, obj in instances.items():
                self.render_traces = []
                prefix = Path(observation['image']).stem + '_' + key
                def render(state):
                    return self.render(agent, observation, out / (prefix+'_'+state), debug=debug)
                self.reset(0)
                base = render('base')
                self.mark(key, 1)
                marked = render('marked')
                visible = difference_mask(base, marked)
                self.mark(key, 0)
                restored = render('restored_base')
                if not np.array_equal(base, restored):
                    raise RuntimeError('Visible stencil A-B-A control changed non-target scene state')
                self.reset(-1)
                self.mark(key, 0)
                isolated_zero = render('isolated_zero')
                self.mark(key, 1)
                isolated_one = render('isolated_one')
                isolated = difference_mask(isolated_zero, isolated_one)
                self.mark(key, 0)
                isolated_restored = render('isolated_restored_zero')
                if not np.array_equal(isolated_zero, isolated_restored):
                    raise RuntimeError('Isolated stencil A-B-A control changed non-target scene state')
                visible_path = out / (prefix+'_visible.png')
                isolated_path = out / (prefix+'_isolated_candidate.png')
                store_png(visible.astype(np.uint8)*255,visible_path)
                store_png(isolated.astype(np.uint8)*255,isolated_path)
                projected = projected_support(observation, obj['bbox_world_vertices'])
                pixel_count = int(visible.sum())
                isolated_count = int(isolated.sum())
                ys, xs = np.where(visible)
                within = 0
                if projected['bbox_xyxy'] is not None:
                    x0, y0, x1, y1 = projected['bbox_xyxy']
                    within = int(((xs >= math.floor(x0)-1) & (xs <= math.ceil(x1)+1)
                                  & (ys >= math.floor(y0)-1) & (ys <= math.ceil(y1)+1)).sum())
                output[key] = {'mesh_owners': self.names[key], 'visible_pixels': pixel_count,
                    'visible_aba_exact': True, 'isolated_aba_exact': True,
                    'render_traces': self.render_traces,
                    'isolated_candidate_pixels': isolated_count, 'projected': projected,
                    'visible_mask_sha256': hashlib.sha256(visible_path.read_bytes()).hexdigest(),
                    'isolated_mask_sha256': hashlib.sha256(isolated_path.read_bytes()).hexdigest(),
                    'visible_mask': str(visible_path.relative_to(folder)),
                    'isolated_candidate_mask': str(isolated_path.relative_to(folder)),
                    'mask_pixels_inside_projected_envelope': within,
                    'visible_subset_of_isolated': bool(np.all(~visible | isolated)),
                    'occlusion_ratio': None, 'amodal_control_validated': False,
                    'occlusion_candidate': 1-pixel_count/isolated_count if isolated_count else None,
                    'instance_mask_verified': pixel_count > 0 and within == pixel_count,
                    'candidate_boundary': 'Stencil-isolated depth is not yet verified amodal support'}
        finally:
            restored=self.native_stencils.restore()
            (out/'native_stencil_restore.json').write_text(json.dumps(restored,indent=2))
        return output
