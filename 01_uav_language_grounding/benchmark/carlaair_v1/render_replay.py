"""Privileged replay of failed cameras for render-switch interventions.

Teleports ONLY a diagnostic camera/vehicle in an owned simulator. These are
not flight trajectories, benchmark episodes, model inputs, or performance data.
Each condition keeps the geometry/pose/timestamp fixed and preserves raw PNGs.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
from pixel_visibility import PixelVisibilityAudit, difference_mask
from runtime_frame_probe import save, xyz
from renderer_weather import set_visual_wind


CONDITIONS = (
    ('runtime_default', ()),
    ('jitter_on', ('r.CustomDepthTemporalAAJitter 1', 'r.PostProcessAAQuality 4')),
    ('depth_jitter_off', ('r.CustomDepthTemporalAAJitter 0',)),
    ('jitter_restored', ('r.CustomDepthTemporalAAJitter 1',)),
    ('all_aa_off', ('r.CustomDepthTemporalAAJitter 0', 'r.PostProcessAAQuality 0')),
    ('all_restored', ('r.CustomDepthTemporalAAJitter 1', 'r.PostProcessAAQuality 4')),
    ('wind_zero', ('weather.wind_intensity 0',)),
    ('wind_restored', ('weather.wind_intensity restore',)),
    ('wind_zero_repeat', ('weather.wind_intensity 0',)),
    ('wind_restored_repeat', ('weather.wind_intensity restore',)),
)


def q_product(first, second):
    w, v = first[0], np.asarray(first[1:])
    x, u = second[0], np.asarray(second[1:])
    return np.r_[w*x-np.dot(v,u), w*u+x*v+np.cross(v,u)]


def vehicle_for_camera(position, quaternion, camera_offset, camera_quaternion):
    """Invert the original body-to-camera extrinsic, not just camera position.

    Moving a zero-offset camera into the aircraft body introduces new occlusion
    despite matching its world pose. All inputs/outputs use NED and wxyz.
    """
    q = np.asarray(quaternion, dtype=float)
    mount = np.asarray(camera_quaternion, dtype=float)
    q = q/np.linalg.norm(q)
    mount = mount/np.linalg.norm(mount)
    body = q_product(q, mount*[1,-1,-1,-1])
    offset = q_product(q_product(body, np.r_[0,camera_offset]), body*[1,-1,-1,-1])[1:]
    return np.asarray(position)-offset, body


def sequence_metrics(arrays):
    if len(arrays) != 40:
        raise ValueError('Eight warmup plus 32 measured images required')
    measured = np.stack(arrays[8:])
    changes = [int(difference_mask(a, b).sum()) for a, b in zip(measured[:-1], measured[1:])]
    varied = np.any(measured != measured[0], axis=(0, 3))
    return {'warmup_frames': 8, 'measured_frames': 32, 'changed_pixels': changes,
            'varying_pixels': int(varied.sum()), 'all_32_identical': not any(changes),
            'last_16_identical': not any(changes[-15:])}


def pose_residual(response, expected_position, expected_quaternion):
    q = response.camera_orientation
    actual_q = np.array([q.w_val, q.x_val, q.y_val, q.z_val])
    return {'position_m': float(np.linalg.norm(np.array(xyz(response.camera_position))-expected_position)),
            'quaternion_l2': float(min(np.linalg.norm(actual_q-expected_quaternion),
                                       np.linalg.norm(actual_q+expected_quaternion)))}


def run_replay(args, ac, cc, calibration, local_origins):
    import airsim
    import carla
    out = args.output/'renderer_replay'
    out.mkdir(exist_ok=False)
    root = args.output.parents[2]
    sources = [(2, '000002', 'round003_UAV_0.png', 'target'),
               (2, '000004', 'round000_UAV_1.png', 'reference')] if args.agents == 2 else [
               (4, '000001', 'round002_UAV_0.png', 'target')]
    status = {'stage': 'replaying', 'scope': 'privileged camera diagnostic, NOT benchmark',
              'dataset_ready': False, 'runtime_validated_episodes': 0, 'cases': []}
    def update():
        save(out/'result.json', status)
    world = cc.get_world()
    owned = []
    paused = False
    original_wind = world.get_weather().wind_intensity
    try:
        for team, suffix, image_name, key in sources:
            old = root/'sessions'/f'runtime_repair_{team}uav_v1/probe/training_views'/f'train_{suffix}'/'observations.json'
            previous = json.loads(old.read_text())
            obs = next(o for o in previous['observations'] if o['image'] == image_name)
            scene_path = root/'scene_assets_v3/train'/f'train_{suffix}'/'scene.json'
            if hashlib.sha256(scene_path.read_bytes()).hexdigest() != previous['source_sha256']:
                raise RuntimeError('Source construction changed since failure')
            scene = json.loads(scene_path.read_text())
            folder = out/f'train_{suffix}'
            folder.mkdir()
            case = {'episode': previous['episode_id'], 'old_observation_sha256': hashlib.sha256(old.read_bytes()).hexdigest(),
                    'source_image': obs, 'marked_key': key, 'conditions': []}
            status['cases'].append(case)
            pixel = PixelVisibilityAudit(ac)
            for obj in scene['objects']:
                before = set(ac.simListSceneObjects())
                bp = world.get_blueprint_library().find(obj['blueprint'])
                if bp.has_attribute('color') and 'color' in obj:
                    bp.set_attribute('color', obj['color'])
                x, y, z, yaw = obj['transform']
                actor = world.spawn_actor(bp, carla.Transform(carla.Location(x=x,y=y,z=z), carla.Rotation(yaw=yaw)))
                owned.append(actor)
                actor.set_simulate_physics(False)
                pixel.register(obj['key'], set(ac.simListSceneObjects())-before)
            for _ in range(3):
                world.wait_for_tick(20)
            agent = obs['agent']
            expected_position = ((np.array(obs['camera_world_position'])-calibration['world_origin_m'])*[1,1,-1]
                                 -local_origins[agent])
            expected_q = np.array(obs['camera_orientation_ned_wxyz'])
            ac.simPause(True)
            paused = True
            # Put the camera at the saved world pose. Diagnostic teleport is
            # explicitly separate from the flight-collection implementation.
            mount_q = airsim.to_quaternion(-math.pi/2, 0, 0)
            mount_offset = np.array([.5,0,.1])
            ac.simSetCameraPose('0', airsim.Pose(airsim.Vector3r(*mount_offset), mount_q), vehicle_name=agent)
            # simSetVehiclePose queues a physics-body change, which did not
            # reach the rendered actor while paused in replay v1. Pinned
            # WorldSimApi::setObjectPose instead calls SetActorLocationAndRotation
            # on the game thread immediately, in GLOBAL NED. Never use this
            # privileged diagnostic operation in actual flight collection.
            global_position = expected_position + local_origins[agent]
            body_position, body_q = vehicle_for_camera(global_position, expected_q, mount_offset,
                [mount_q.w_val,mount_q.x_val,mount_q.y_val,mount_q.z_val])
            case['placement'] = {'api': 'simSetObjectPose', 'frame': 'global_ned',
                                 'body_position': body_position.tolist(), 'body_quaternion_wxyz': body_q.tolist(),
                                 'camera_mount_offset': mount_offset.tolist(),
                                 'camera_mount_pitch_degrees': -90}
            if not ac.simSetObjectPose(agent, airsim.Pose(airsim.Vector3r(*body_position),
                airsim.Quaternionr(body_q[1],body_q[2],body_q[3],body_q[0])),
                teleport=True):
                raise RuntimeError('Diagnostic rendered-actor placement failed')
            request = [airsim.ImageRequest('0', airsim.ImageType.Scene, False, True)]
            for _ in range(8):
                response = ac.simGetImages(request, vehicle_name=agent)[0]
                time.sleep(.05)
            residual = pose_residual(response, expected_position, expected_q)
            case['pose_residual'] = residual
            if residual['position_m'] > .001 or residual['quaternion_l2'] > 1e-5:
                raise RuntimeError(f'Replay camera is not at the recorded pose: {residual}')
            timestamp = int(response.time_stamp)
            pixel.reset(0)
            pixel.mark(key, 1)
            for name, commands in CONDITIONS:
                condition = {'name': name, 'commands': list(commands), 'transport_ack': [],
                             'command_effect_verified_by_api': False, 'frames': [], 'started_unix': time.time()}
                # Pinned AirBlueprintLib.cpp only acknowledges a controller,
                # not command validity. Preserve engine output and infer effects
                # only from measured switch/reversal comparisons, never bool True.
                for command in commands:
                    if command.startswith('weather.wind_intensity '):
                        value = command.split()[1]
                        condition['wind_configuration'] = set_visual_wind(world,
                            original_wind if value == 'restore' else float(value))
                        condition['wind_readback'] = condition['wind_configuration']['readback']
                        continue
                    if not ac.simRunConsoleCommand(command):
                        raise RuntimeError('Unreal console transport unavailable')
                    condition['transport_ack'].append(True)
                    ac.simRunConsoleCommand(command.split()[0])
                for variable in ('r.CustomDepthTemporalAAJitter', 'r.PostProcessAAQuality'):
                    ac.simRunConsoleCommand(variable)
                time.sleep(.2)
                rgb = ac.simGetImages(request, vehicle_name=agent)[0]
                (folder/(name+'_rgb.png')).write_bytes(bytes(rgb.image_data_uint8))
                images=[]
                for index in range(40):
                    response = ac.simGetImages([airsim.ImageRequest('0',airsim.ImageType.Segmentation,False,True)],vehicle_name=agent)[0]
                    if response.time_stamp != timestamp:
                        raise RuntimeError('Paused replay time advanced')
                    residual = pose_residual(response, expected_position, expected_q)
                    if residual['position_m'] > .001 or residual['quaternion_l2'] > 1e-5:
                        raise RuntimeError('Replay camera moved between interventions')
                    data = bytes(response.image_data_uint8)
                    (folder/f'{name}_{index:02d}.png').write_bytes(data)
                    with Image.open(io.BytesIO(data)) as image:
                        array = np.array(image.convert('RGB'))
                    images.append(array)
                    condition['frames'].append({'index': index, 'timestamp': int(response.time_stamp),
                        'pixel_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                        'image_sha256': hashlib.sha256(data).hexdigest()})
                    time.sleep(.05)
                condition.update(sequence_metrics(images))
                condition['ended_unix'] = time.time()
                case['conditions'].append(condition)
                update()
            ac.simPause(False)
            paused=False
            responses=cc.apply_batch_sync([carla.command.DestroyActor(a.id) for a in owned],False)
            if any(r.has_error() for r in responses):
                raise RuntimeError('Failed to clean replay-owned actors')
            owned=[]
        status.update(stage='replay_complete_not_benchmark', ended_unix=time.time())
        update()
    except Exception as error:
        status.update(stage='failed', error=f'{type(error).__name__}: {error}', ended_unix=time.time())
        update()
        raise
    finally:
        # These are explicit intervention reference values, not a claim that
        # unknown prior user settings were discovered. The owned sim exits next.
        ac.simRunConsoleCommand('r.CustomDepthTemporalAAJitter 1')
        ac.simRunConsoleCommand('r.PostProcessAAQuality 4')
        set_visual_wind(world, original_wind)
        if paused:
            ac.simPause(False)
        if owned:
            cc.apply_batch_sync([carla.command.DestroyActor(a.id) for a in owned],False)
