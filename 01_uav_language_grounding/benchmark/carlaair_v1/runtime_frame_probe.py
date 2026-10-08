"""Runtime coordinate/API audit. Privileged calibration, never a policy score.

Camera and vehicle poses are vehicle-local NED in the pinned CarlaAir source.
Object poses are global NED. Derive translations from matched measured landmarks,
not an assumption that the different APIs expose the same origin.
"""
import argparse
from survey_geometry import PROFILES
import json
import math
from pathlib import Path
import re
import time

import numpy as np


def xyz(value):
    return [float(value.x_val), float(value.y_val), float(value.z_val)]


def carla_xyz(value):
    return [float(value.x), float(value.y), float(value.z)]


def pose(value):
    q = value.orientation
    return {'position': xyz(value.position),
            'quaternion_wxyz': [q.w_val, q.x_val, q.y_val, q.z_val]}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def match_landmark_names(objects, environment):
    """CARLA names components; AirSim names actors. Keep only unique matches.

    Name normalization only proposes a correspondence. Independent geometry
    residuals below must still verify the origin before any use.
    """
    candidates = {}
    names = set(objects)
    for record in environment:
        name = record['name']
        actor = name if name in names else re.sub(r'_SM_\d+$', '', name)
        if actor in names:
            candidates.setdefault(actor, []).append(record)
    return {name: rows[0] for name, rows in candidates.items() if len(rows) == 1}


def translation_from_landmarks(pairs, tolerance=.05):
    if len(pairs) < 3:
        raise ValueError('At least three matched landmarks are required')
    world = np.asarray([p['carla_world'] for p in pairs], dtype=float)
    ned = np.asarray([p['airsim_global_ned'] for p in pairs], dtype=float)
    if not np.isfinite(world).all() or not np.isfinite(ned).all():
        raise ValueError('Landmarks must be finite')
    # Source-verified axes: Unreal XYZ in metres differs from NED by Z sign.
    offsets = world - ned * np.array([1., 1., -1.])
    offset = np.median(offsets, axis=0)
    residual = np.linalg.norm(offsets - offset, axis=1)
    if residual.max() > tolerance:
        raise ValueError(f'Landmarks disagree about origin: residual={residual.tolist()}')
    if np.linalg.matrix_rank(world - world[0], tol=.01) < 2:
        raise ValueError('Landmarks must span two spatial directions')
    return {'world_origin_m': offset.tolist(), 'max_residual_m': float(residual.max()),
            'landmark_count': len(pairs), 'ned_to_carla_axes': [1, 1, -1]}


def probe(args):
    import airsim
    import carla
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    result = {'stage': 'starting', 'started_unix': time.time(), 'agents': args.agents,
              'publication_scores': None, 'runtime_validated_episodes': 0,
              'dataset_ready': False, 'scope': 'privileged runtime coordinate calibration only'}
    def update(**values):
        result.update(values, updated_unix=time.time())
        save(out / 'coordinate_audit.json', result)
        print(json.dumps({'stage': result['stage'], 'updated_unix': result['updated_unix']}), flush=True)
    cc = carla.Client('127.0.0.1', args.carla_port)
    cc.set_timeout(90)
    world = cc.get_world()
    if args.expected_map and world.get_map().name.rsplit('/', 1)[-1] != args.expected_map:
        world = cc.load_world(args.expected_map)
        time.sleep(2)
    ac = airsim.MultirotorClient(port=args.port, timeout_value=20)
    paused = False
    try:
        from runtime_artifacts import verify_code
        update(code_snapshot=verify_code(Path(__file__).resolve().parent))
        if getattr(args, 'static_visual_wind', False):
            if getattr(args, 'render_replay', False):
                raise ValueError('Static collection profile would contaminate native wind replay')
            from renderer_weather import set_visual_wind
            update(visual_weather_profile=set_visual_wind(world, 0),
                   weather_validation_boundary='Static visual wind only; dynamic-weather masks not validated')
        if getattr(args,'scene_repair',False):
            from renderer_weather import set_frozen_visual_profile
            update(visual_weather_profile=set_frozen_visual_profile(world))
        ac.confirmConnection()
        names = ac.listVehicles()
        if len(names) != args.agents:
            raise RuntimeError(f'Expected {args.agents} UAVs, got {names}')
        update(stage='inventory', vehicle_names=names, map=world.get_map().name,
               carla_methods=[x for x in dir(world) if any(k in x for k in ('ray', 'object', 'level'))])
        objects = ac.simListSceneObjects('.*')
        save(out / 'scene_object_names.json', objects)
        environment = world.get_environment_objects(carla.CityObjectLabel.Any)
        env_records = [{'id': int(x.id), 'name': x.name, 'position': carla_xyz(x.transform.location),
                        'label': str(x.type)} for x in environment]
        if getattr(args, 'static_box_inventory', False):
            for row, x in zip(env_records, environment):
                row['bounding_box'] = {'location': carla_xyz(x.bounding_box.location),
                                         'extent': carla_xyz(x.bounding_box.extent),
                                         'rotation': [x.bounding_box.rotation.pitch,
                                                      x.bounding_box.rotation.yaw,
                                                      x.bounding_box.rotation.roll],
                                         'vertices_identity_transform': [carla_xyz(v) for v in
                                             x.bounding_box.get_world_vertices(carla.Transform())]}
                row['box_boundary'] = 'Inventory only; coordinate frame and geometric conservatism require audit'
        save(out / 'environment_objects.json', env_records)
        actors = [{'id': x.id, 'type': x.type_id, 'position': carla_xyz(x.get_location())}
                  for x in world.get_actors()]
        save(out / 'carla_actors.json', actors)
        by_name = match_landmark_names(objects, env_records)
        common = sorted(by_name)
        pairs = []
        invalid = []
        for name in common[:40]:
            point = xyz(ac.simGetObjectPose(name).position)
            if all(math.isfinite(x) for x in point):
                pairs.append({'name': name, 'carla_world': by_name[name]['position'],
                              'airsim_global_ned': point})
            else:
                invalid.append(name)
        uav_objects = list(names) + [n for n in objects if any(s in n.lower() for s in ('flying', 'drone'))]
        object_poses = {}
        for name in uav_objects[:40]:
            value = pose(ac.simGetObjectPose(name))
            if all(math.isfinite(x) for x in value['position']):
                object_poses[name] = value
        vehicles = {}
        for name in names:
            vehicles[name] = {'local_pose': pose(ac.simGetVehiclePose(vehicle_name=name)),
                              'camera_local_pose': pose(ac.simGetCameraInfo('0', vehicle_name=name).pose),
                              'collision': bool(ac.simGetCollisionInfo(vehicle_name=name).has_collided)}
        update(stage='landmark_matching', environment_count=len(environment),
               scene_object_count=len(objects), shared_names=len(common),
               landmark_pairs=pairs, invalid_object_poses=invalid,
               uav_object_candidates=object_poses, vehicles=vehicles)
        calibration = translation_from_landmarks(pairs)
        update(stage='global_frame_calibrated', calibration=calibration)
        ac.simPause(True)
        paused = True
        local_origins = {}
        for name in names:
            global_pose = pose(ac.simGetObjectPose(name))
            local_pose = pose(ac.simGetVehiclePose(vehicle_name=name))
            global_position = np.asarray(global_pose['position'])
            local_position = np.asarray(local_pose['position'])
            if not np.isfinite(global_position).all():
                raise RuntimeError(f'Named UAV global pose unavailable: {name}')
            local_origins[name] = (global_position - local_position).tolist()
        ac.simPause(False)
        paused = False
        update(stage='local_frames_calibrated', local_origins_global_ned=local_origins)
        for name in names:
            ac.enableApiControl(True, vehicle_name=name)
            ac.armDisarm(True, vehicle_name=name)
        for name in names:
            ac.takeoffAsync(timeout_sec=15, vehicle_name=name).join()
            ac.moveToZAsync(-15, 3, timeout_sec=30, vehicle_name=name).join()
            ac.hoverAsync(vehicle_name=name).join()
        motion_records = []
        for index, name in enumerate(names):
            initial_collision = ac.simGetCollisionInfo(vehicle_name=name)
            before = pose(ac.simGetVehiclePose(vehicle_name=name))
            ac.moveByVelocityAsync(1.5, .5 if index % 2 else -.5, 0, 2,
                                   vehicle_name=name).join()
            ac.hoverAsync(vehicle_name=name).join()
            ac.simPause(True)
            paused = True
            after = pose(ac.simGetVehiclePose(vehicle_name=name))
            global_after = pose(ac.simGetObjectPose(name))
            residual = float(np.linalg.norm(np.asarray(global_after['position']) -
                             np.asarray(after['position']) - local_origins[name]))
            image = ac.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene,
                                                        False, True)], vehicle_name=name)[0]
            if not image.image_data_uint8:
                raise RuntimeError(f'Empty camera image for {name}')
            (out / f'{name}_calibrated_rgb.png').write_bytes(bytes(image.image_data_uint8))
            collision = ac.simGetCollisionInfo(vehicle_name=name)
            record = {'agent': name, 'before_local': before, 'after_local': after,
                      'after_global': global_after, 'origin_residual_m': residual,
                      'displacement_m': math.dist(before['position'], after['position']),
                      'camera_local_position': xyz(image.camera_position),
                      'camera_global_ned': (np.asarray(xyz(image.camera_position)) + local_origins[name]).tolist(),
                      'new_collision': collision.time_stamp != initial_collision.time_stamp,
                      'image_timestamp': int(image.time_stamp), 'image_resolution': [image.width, image.height]}
            motion_records.append(record)
            ac.simPause(False)
            paused = False
            update(stage='verifying_local_frames_in_motion', motion_records=motion_records)
            if residual > .05 or record['displacement_m'] < .5:
                raise RuntimeError(f'Local frame/motion verification failed: {name}')
            if record['new_collision']:
                raise RuntimeError(f'Collision during measured displacement: {name}')
        update(stage='coordinate_and_motion_passed')
        if getattr(args, 'ground_clamp_diagnostic', False):
            if args.collect_scenes or getattr(args, 'render_replay', False):
                raise ValueError('Ground clamp diagnosis cannot be mixed with collection')
            from diagnose_ground_clamp import run_diagnostic
            run_diagnostic(args, ac, world, calibration, local_origins)
        if getattr(args, 'amodal_intervention', False):
            from amodal_intervention import run_amodal_intervention
            run_amodal_intervention(args, ac, cc, calibration, local_origins)
        if getattr(args, 'render_replay', False):
            if args.collect_scenes or args.stencil_control:
                raise ValueError('Privileged replay cannot be combined with benchmark collection')
            from render_replay import run_replay
            run_replay(args, ac, cc, calibration, local_origins)
        if getattr(args, 'stencil_control', False):
            from stencil_control import run_control
            run_control(args, ac, cc, calibration, local_origins)
            if args.collect_scenes or getattr(args, 'preview_episode', None):
                # The suspended-asset control changes UAV start positions. Reset
                # this owned environment BEFORE any benchmark episode, restoring
                # the same configured start instead of inheriting the unit test.
                ac.reset()
                reset_origins = {}
                for name in names:
                    global_position = np.asarray(xyz(ac.simGetObjectPose(name).position))
                    local_position = np.asarray(xyz(ac.simGetVehiclePose(vehicle_name=name).position))
                    reset_origin = global_position-local_position
                    if not np.isfinite(reset_origin).all() or np.linalg.norm(reset_origin-local_origins[name]) > .05:
                        raise RuntimeError('Simulator reset changed the calibrated UAV frame')
                    reset_origins[name] = reset_origin.tolist()
                    # AirSim reset clears API control and arm state.
                    ac.enableApiControl(True, vehicle_name=name)
                    ac.armDisarm(True, vehicle_name=name)
                    ac.takeoffAsync(timeout_sec=30, vehicle_name=name).join()
                    ac.moveToZAsync(-15, 3, timeout_sec=30, vehicle_name=name).join()
                    ac.hoverAsync(vehicle_name=name).join()
                update(stage='control_passed_environment_reset', reset_local_origins=reset_origins)
        if getattr(args, 'preview_episode', None):
            from condition_candidate_preview import run_preview
            preview = run_preview(args, ac, cc, calibration, local_origins)
            update(stage='condition_preview_complete_not_benchmark', condition_preview=preview)
        if getattr(args,'material_probe_v4',False):
            from oct07_material_probe_v4 import run_uav_material
            run_uav_material(args,ac,cc,calibration,local_origins)
            update(stage='native_uav_material_diagnostic_complete')
        if args.collect_scenes:
            from collect_runtime_views import collect
            collection = collect(args, ac, cc, calibration, local_origins)
            if (collection['completed_scenes'] != collection['assigned_scenes']
                    or collection['failed_scenes'] != 0):
                raise RuntimeError('Collection incomplete or failed; see training_views/status.json')
            update(stage='coordinate_passed_views_collected_not_validated', collection=collection)
        update(ended_unix=time.time())
    except Exception as error:
        update(stage='failed', error=f'{type(error).__name__}: {error}', ended_unix=time.time())
        raise
    finally:
        if paused:
            ac.simPause(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--carla-port', type=int, required=True)
    parser.add_argument('--agents', type=int, required=True)
    parser.add_argument('--expected-map')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ground-clamp-diagnostic', action='store_true')
    parser.add_argument('--collect-scenes', type=int, default=0)
    parser.add_argument('--scene-assets',type=Path,help='Explicit isolated candidate release; never the frozen original scenes')
    parser.add_argument('--observe-rounds', type=int, default=4)
    parser.add_argument('--probe-timeout', type=int, default=600)
    parser.add_argument('--pixel-audit', action='store_true')
    parser.add_argument('--fast-instance-screen', action='store_true',
                        help='Development screen only; no ABA or amodal certification')
    parser.add_argument('--fast-calibrate-first-frame', action='store_true',
                        help='Same-run target screen/ABA comparison on first paused view')
    parser.add_argument('--fast-calibrate-all-actors', action='store_true',
                        help='Same-run ABA comparison once for each visible scene actor')
    parser.add_argument('--fast-calibrate-measure-only', action='store_true',
                        help='Retain exact pixel disagreements as unqualified diagnostic measurements')
    parser.add_argument('--stencil-control', action='store_true')
    parser.add_argument('--render-replay', action='store_true')
    parser.add_argument('--static-visual-wind', action='store_true')
    parser.add_argument('--survey-profile', choices=PROFILES, default='wide70')
    parser.add_argument('--episode-ids', help='Explicit development train scene IDs, comma separated')
    parser.add_argument('--preview-episode')
    parser.add_argument('--preview-attempt-start',type=int,default=1)
    parser.add_argument('--preview-attempt-count',type=int,default=2)
    parser.add_argument('--preview-view-limit',type=int,default=8)
    parser.add_argument('--amodal-intervention', action='store_true')
    parser.add_argument('--separated-transit', action='store_true')
    parser.add_argument('--persistent-view-assignment', action='store_true')
    parser.add_argument('--transit-cruise-z', type=float, choices=(70., 120.), default=70.)
    parser.add_argument('--static-box-inventory', action='store_true')
    parser.add_argument('--static-box-clearance', action='store_true')
    parser.add_argument('--static-box-refinement',action='store_true')
    parser.add_argument('--static-transit-routing', action='store_true')
    probe(parser.parse_args())
