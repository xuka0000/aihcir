"""Actual multi-UAV flights and RGB from immutable constructed training scenes.

This is sensor/trajectory data collection, NOT a TRACE or oracle policy result.
The collector never marks a scene runtime_validated without pixel-level visibility,
occlusion, relation disclosure and full candidate audits, which remain separate.
"""
import hashlib
import json
import math
from pathlib import Path
import time
from benchmark import canonical

import numpy as np

from runtime_frame_probe import xyz, pose, save
from flight_feedback import reach_views
from scene_paths import scene_path,selected_split,validate_collection_source
from survey_geometry import fixed_waypoints, transit_stages, screen_descent_columns, assign_views, clearance_waypoints, lateral_clearance_waypoints


def survey_waypoints(center, team, rounds, profile='wide70'):
    """Task-independent survey geometry inside an advertised 300 m square."""
    return fixed_waypoints(center, team, rounds, profile)


def public_observation(record):
    """Private object geometry and validation flags cannot enter policy inputs."""
    fields = ('agent', 'source_id', 'image', 'sha256', 'timestamp', 'width', 'height',
              'camera_world_position', 'camera_orientation_ned_wxyz')
    return {key: record[key] for key in fields}


def preview_route_binding(scene, profile, waypoints):
    evidence=scene.get('preview_evidence',{})
    expected=evidence.get('route_waypoints_sha256')
    if expected is None:
        return None
    actual=hashlib.sha256(canonical(waypoints).encode()).hexdigest()
    return {'expected_sha256':expected,'actual_sha256':actual,'profile':profile,
            'matched':actual==expected and profile==evidence.get('route_profile')}


def color_triplet(value):
    """Normalize CARLA's comma-separated actor color without guessing a default."""
    if not isinstance(value, str):
        raise ValueError('Actor color must be an RGB string')
    parts = value.split(',')
    if len(parts) != 3:
        raise ValueError('Actor color must contain three RGB channels')
    try:
        rgb = tuple(int(part.strip()) for part in parts)
    except ValueError as error:
        raise ValueError('Actor color channel is not an integer') from error
    if any(not 0 <= channel <= 255 for channel in rgb):
        raise ValueError('Actor color channel outside 0..255')
    return rgb


def collect(args, ac, cc, calibration, local_origins):
    import airsim
    import carla
    from PIL import Image
    import io
    root = args.output.parents[2]
    output = args.output / 'training_views'
    output.mkdir(exist_ok=False)
    world = cc.get_world()
    static_boxes = None
    if getattr(args, 'static_box_clearance', False):
        from static_box_clearance import StaticBoxScreen
        static_boxes = StaticBoxScreen.from_world(world)
    map_name = world.get_map().name.rsplit('/', 1)[-1]
    selected = getattr(args, 'episode_ids', None)
    split=selected_split(selected.split(',')) if selected else 'train'
    specifications = root / 'specifications' / (split+'.specs.jsonl')
    rows = [json.loads(line) for line in specifications.read_text().splitlines()]
    rows = [r for r in rows if r['map'] == map_name and r['team_size'] == args.agents]
    if selected:
        selected = selected.split(',')
        rows = [r for r in rows if r['episode_id'] in selected]
        if len(rows) != len(set(selected)) or len(rows) != args.collect_scenes:
            raise ValueError('Requested development episode IDs/count/map/team do not match')
    rows = rows[:args.collect_scenes]
    if not rows:
        raise RuntimeError('No existing training scenes match the runtime')
    names = [f'UAV_{i}' for i in range(args.agents)]
    origin = np.asarray(calibration['world_origin_m'])
    axis = np.array([1., 1., -1.])
    status = {'stage': 'collecting', 'assigned_scenes': len(rows), 'completed_scenes': 0,
              'failed_scenes': 0, 'observations': 0, 'runtime_validated_episodes': 0,
              'publication_scores': None, 'dataset_ready': False, 'started_unix': time.time(),
              'collection_route': 'fixed survey, not learned exploration',
              'split': split, 'map': map_name, 'team_size': args.agents}
    def update(**values):
        status.update(values, updated_unix=time.time())
        save(output / 'status.json', status)
        print(json.dumps(status), flush=True)
    library = world.get_blueprint_library()
    for name in names:
        # Actual onboard camera, no evaluator-owned overview image as input.
        ac.simSetCameraPose('0', airsim.Pose(airsim.Vector3r(.5, 0, .1),
                                           airsim.to_quaternion(-math.pi / 2, 0, 0)), vehicle_name=name)
    failed_in_row = 0
    for specification in rows:
        if time.time() - status['started_unix'] > args.probe_timeout - 120:
            update(stage='bounded_partial', ended_unix=time.time())
            return status
        episode = specification['episode_id']
        candidate_assets=getattr(args,'scene_assets',None)
        source = scene_path(root,episode,candidate_assets)
        scene = json.loads(source.read_text())
        validate_collection_source(scene,candidate_release=candidate_assets is not None)
        folder = output / episode
        folder.mkdir(exist_ok=False)
        owned = []
        record = {'episode_id': episode, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'source_scene_path':str(source), 'source_construction_status':scene['status'],
                  'task_family': scene['task_family'],
                  'visual_condition_requested': scene['visual_condition_requested'],
                  'team_size': args.agents, 'instruction': scene['instruction'],
                  'stage': 'preparing_scene', 'runtime_validated': False,
                  'validation_pending': ['instance visibility', 'amodal occlusion',
                                         'progressive relation disclosure', 'background candidates'],
                  'observations': [], 'private_native_instances': {},
                  'waypoint_displacement_lower_bound_m': {n: 0. for n in names},
                  'path_accounting_status': 'waypoint chords only; not a final flight-budget measurement',
                  'flight_attempts': [],
                  'new_collisions': {n: False for n in names}}
        try:
            pixel_audit = None
            fast_instance_screen = None
            if getattr(args, 'fast_instance_screen', False) and getattr(args, 'pixel_audit', False):
                raise ValueError('Choose one instance annotation mode per development capture')
            if getattr(args, 'pixel_audit', False):
                from pixel_visibility import PixelVisibilityAudit
                pixel_audit = PixelVisibilityAudit(ac)
            if getattr(args, 'fast_instance_screen', False):
                from fast_instance_visibility import FastInstanceVisibility
                fast_instance_screen = FastInstanceVisibility(ac)
            for obj in scene['objects']:
                prior_names = set(ac.simListSceneObjects()) if (pixel_audit or fast_instance_screen) else set()
                bp = library.find(obj['blueprint'])
                requested_color = obj.get('color')
                if requested_color is not None:
                    color_triplet(requested_color)
                    if not bp.has_attribute('color'):
                        raise RuntimeError(f'Color cue unsupported by blueprint: {obj["key"]}')
                    bp.set_attribute('color', requested_color)
                if bp.has_attribute('role_name'):
                    bp.set_attribute('role_name', 'runtime_view_collection')
                x, y, z, yaw = obj['transform']
                actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=z),
                                                                carla.Rotation(yaw=yaw)))
                if actor is None:
                    raise RuntimeError(f'Unable to restore native object {obj["key"]}')
                owned.append(actor)
                actor.set_simulate_physics(False)
                readback_color = actor.attributes.get('color') if requested_color is not None else None
                if requested_color is not None and (
                        readback_color is None or
                        color_triplet(readback_color) != color_triplet(requested_color)):
                    raise RuntimeError(f'Runtime actor color differs from source: {obj["key"]}')
                if pixel_audit:
                    pixel_audit.register(obj['key'], set(ac.simListSceneObjects())-prior_names)
                if fast_instance_screen:
                    fast_instance_screen.register(obj['key'], set(ac.simListSceneObjects())-prior_names)
                tf = actor.get_transform()
                vertices = actor.bounding_box.get_world_vertices(tf)
                record['private_native_instances'][obj['key']] = {
                    'actor_id': actor.id, 'world_position': [tf.location.x, tf.location.y, tf.location.z],
                    'bbox_world_vertices': [[v.x, v.y, v.z] for v in vertices],
                    'color_requested_rgb': list(color_triplet(requested_color)) if requested_color is not None else None,
                    'color_runtime_readback_rgb': list(color_triplet(readback_color)) if readback_color is not None else None}
            # Scene setup may use truth; the advertised region must not encode an
            # exact target point. All methods later receive this same fixed region.
            locations = np.asarray([o['transform'][:2] for o in scene['objects']])
            if scene.get('construction_version') in ('crossview_two_sector_candidates_v4',
                                                     'crossview_two_sector_candidates_v5',
                                                     'crossview_two_sector_candidates_v6',
                                                     'crossview_two_sector_candidates_v7'):
                center = scene.get('search_region_center_xy')
                if (not isinstance(center, list) or len(center) != 2
                        or not np.isfinite(center).all()
                        or any(np.max(np.abs(locations[:,k] - center[k])) > 150 for k in (0,1))):
                    raise ValueError('Invalid frozen cross-view search region')
                if args.survey_profile != scene.get('survey_profile_required'):
                    raise ValueError('Cross-view scene requires its frozen condition route')
                record['search_region_construction'] = 'frozen_two_sector_v4'
            else:
                center = (np.round(locations.mean(axis=0) / 100) * 100).tolist()
            record['search_region'] = {'center_world_xy': center, 'size_m': [300, 300]}
            record['region_protocol_status'] = 'collection development; final evaluation protocol not frozen'
            profile = getattr(args, 'survey_profile', 'wide70')
            record['survey_profile'] = profile
            record['survey_route_policy_status'] = 'Fixed development acquisition; not a TRACE result'
            from survey_geometry import CONSTRUCTION_PROFILES
            if profile in CONSTRUCTION_PROFILES:
                record['private_construction_route'] = CONSTRUCTION_PROFILES[profile]
                record['survey_route_policy_status'] = 'Offline truth-assisted dataset construction; not an inference or exploration policy'
            waypoints = survey_waypoints(center, args.agents, args.observe_rounds, profile)
            record['survey_waypoints_world'] = waypoints
            separation = 10. if getattr(args, 'separated_transit', False) else 0.
            cruise_z = getattr(args, 'transit_cruise_z', 70.)
            refined_screen = None
            if getattr(args,'static_box_refinement',False):
                if static_boxes is None:raise ValueError('Ray refinement requires recorded static boxes')
                from static_box_clearance import MeasuredRayRefinement
                refined_screen=MeasuredRayRefinement(world,static_boxes,
                    on_progress=lambda progress: update(stage='screening_static_descent',
                        current_episode=episode,static_preflight_progress=progress))
                record['private_static_preflight_mode']='explicit_dense_ray_refinement_with_actual_flight_checks'
            if not math.isfinite(cruise_z) or not 70. <= cruise_z <= 120.:
                raise ValueError('Transit cruise height outside diagnostic range')
            record['transit_cruise_z_m'] = cruise_z
            def descent_screen(points):
                top = cruise_z+(args.agents-1)*separation
                if refined_screen is not None:return refined_screen(points,top)
                if static_boxes is not None:
                    from static_box_clearance import combined_screen
                    return combined_screen(world, points, static_boxes, top)
                return screen_descent_columns(world, points, cruise_top=top)
            record['static_box_clearance_enabled'] = static_boxes is not None
            record['transit_vertical_separation_m'] = separation
            if profile.endswith('_clearance'):
                clearance_function = lateral_clearance_waypoints if profile.endswith('_lateral_clearance') else clearance_waypoints
                if profile.endswith('_independent_clearance'):
                    if static_boxes is None:
                        raise ValueError('Independent offsets require static box clearance')
                    from static_box_clearance import independent_clearance_waypoints
                    clearance_function = independent_clearance_waypoints
                waypoints, clearance = clearance_function(waypoints, descent_screen)
                record['private_clearance_acquisition'] = clearance
                record['survey_waypoints_world'] = waypoints
                save(folder / 'observations.json', record)
                if not clearance['passed']:
                    raise RuntimeError('No clear descent within acquisition altitude ceiling')
            if profile != 'wide70':
                record['private_descent_screen'] = descent_screen(waypoints)
                save(folder / 'observations.json', record)
                if not record['private_descent_screen']['passed']:
                    raise RuntimeError('Fixed low-altitude route intersects sampled static geometry')
            binding=preview_route_binding(scene,profile,waypoints)
            if binding is not None:
                record['private_preview_route_binding']=binding
                save(folder / 'observations.json', record)
                if not binding['matched']:
                    raise RuntimeError('Physical collection route differs from measured construction preview')
            previous = {n: xyz(ac.simGetVehiclePose(vehicle_name=n).position) for n in names}
            collisions = {n: ac.simGetCollisionInfo(vehicle_name=n).time_stamp for n in names}
            episode_start = time.time()
            locked_view_indices = None
            calibrated_instance_keys = set()
            for round_index, targets in enumerate(waypoints):
                if getattr(args, 'static_visual_wind', False):
                    from renderer_weather import verify_static_visual_wind
                    record['visual_weather_profile'] = verify_static_visual_wind(world)
                update(stage='collecting',current_episode=episode, current_round=round_index)
                remaining_time = args.probe_timeout - 60 - (time.time()-status['started_unix'])
                if remaining_time < 30:
                    raise TimeoutError('Insufficient bounded flight time for next viewpoint')
                if profile == 'wide70':
                    phases = [('direct', targets)]
                else:
                    current_world = [(np.asarray(xyz(ac.simGetVehiclePose(vehicle_name=n).position))
                                      + local_origins[n]) * axis + origin for n in names]
                    if separation:
                        targets, assignment = assign_views(current_world, targets,
                            locked_indices=locked_view_indices)
                        if getattr(args,'persistent_view_assignment',False):
                            locked_view_indices=assignment['destination_indices']
                            assignment['assignment_policy']='first_round_permutation_retained_for_full_history'
                        record.setdefault('view_assignments', []).append(dict(
                            round=round_index, targets_world=targets, **assignment))
                    phases = transit_stages(current_world, targets, cruise_z=cruise_z,
                                            vertical_separation_m=separation)
                    if getattr(args, 'static_transit_routing', False):
                        if static_boxes is None or separation < 10.:
                            raise ValueError('Static routing requires box inventory and separated cruise levels')
                        from static_transit import route_cruise_phases
                        try:
                            phases, route = route_cruise_phases(phases, static_boxes)
                        except ValueError as error:
                            # Preserve the actual launch pose and complete intended
                            # path for diagnosis; never bypass the original rejection.
                            from transit_corridor_diagnostic import corridor_segments
                            record['private_static_transit_failure'] = dict(
                                round=round_index, error=str(error),
                                current_world=[list(map(float,p)) for p in current_world],
                                phases=phases, segments=corridor_segments(current_world,phases),
                                gate_overridden=False, flight_authorized=False)
                            save(folder / 'observations.json', record)
                            raise
                        record.setdefault('private_static_transit_routes', []).append(dict(round=round_index, **route))
                for phase, phase_targets in phases:
                    remaining_time = args.probe_timeout - 60 - (time.time()-status['started_unix'])
                    if remaining_time < 10:
                        raise TimeoutError('No remaining bounded transit budget')
                    record['current_transit_phase'] = phase
                    reach_views(ac, names, phase_targets, local_origins, origin, round_index, record,
                                lambda: save(folder / 'observations.json', record), remaining_time)
                ac.simPause(True)
                try:
                    if getattr(args, 'static_visual_wind', False):
                        record['visual_weather_profile'] = verify_static_visual_wind(world)
                    for name, target in zip(names, targets):
                        vehicle = ac.simGetVehiclePose(vehicle_name=name)
                        position = xyz(vehicle.position)
                        record['waypoint_displacement_lower_bound_m'][name] += math.dist(previous[name], position)
                        previous[name] = position
                        world_position = (np.asarray(position) + local_origins[name]) * axis + origin
                        residual = float(np.linalg.norm(world_position - target))
                        if residual > 5:
                            raise RuntimeError(f'Flight did not reach prescribed view for {name}: {residual:.2f} m')
                        if getattr(args, 'static_visual_wind', False):
                            visual_profile = verify_static_visual_wind(world)
                        response = ac.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene,
                                                                       False, True)], vehicle_name=name)[0]
                        data = bytes(response.image_data_uint8)
                        if not data:
                            raise RuntimeError(f'Empty actual UAV image from {name}')
                        with Image.open(io.BytesIO(data)) as img:
                            if float(np.asarray(img).std()) < 1:
                                raise RuntimeError('Nearly constant UAV image')
                        filename = f'round{round_index:03d}_{name}.png'
                        (folder / filename).write_bytes(data)
                        q = response.camera_orientation
                        camera_world = (np.asarray(xyz(response.camera_position)) + local_origins[name]) * axis + origin
                        obs = {'agent': name, 'source_id': f'{episode}/{name}/{response.time_stamp}',
                               'image': filename, 'sha256': hashlib.sha256(data).hexdigest(),
                               'timestamp': int(response.time_stamp), 'width': response.width, 'height': response.height,
                               'camera_world_position': camera_world.tolist(),
                               'camera_local_position': xyz(response.camera_position),
                               'camera_orientation_ned_wxyz': [q.w_val, q.x_val, q.y_val, q.z_val]}
                        record['observations'].append(obs)
                        if getattr(args, 'static_visual_wind', False):
                            obs['visual_weather_profile_before_rgb'] = visual_profile
                        if pixel_audit:
                            measurements = pixel_audit.capture(name, obs,
                                record['private_native_instances'], folder)
                            record.setdefault('private_pixel_measurements', {})[filename] = measurements
                        if fast_instance_screen:
                            measurements = fast_instance_screen.capture(name, obs,
                                record['private_native_instances'], folder)
                            record.setdefault('private_fast_instance_measurements', {})[filename] = measurements
                            if (getattr(args, 'fast_calibrate_first_frame', False)
                                    and round_index == 0 and name == names[0]):
                                record['private_fast_calibration'] = fast_instance_screen.calibrate_target(
                                    name, obs, record['private_native_instances'], folder, measurements)
                                save(folder / 'observations.json', record)
                            if getattr(args, 'fast_calibrate_all_actors', False):
                                for key, value in measurements.items():
                                    if key in calibrated_instance_keys or value['visible_pixels_screen'] < 16:
                                        continue
                                    comparison = fast_instance_screen.calibrate_instance(
                                        key, name, obs, record['private_native_instances'], folder, measurements,
                                        measure_only=getattr(args, 'fast_calibrate_measure_only', False))
                                    record.setdefault('private_fast_all_actor_calibrations', {})[key] = comparison
                                    calibrated_instance_keys.add(key)
                                    save(folder / 'observations.json', record)
                        if getattr(args, 'static_visual_wind', False):
                            obs['visual_weather_profile_after_mask'] = verify_static_visual_wind(world)
                        status['observations'] += 1
                        current_collision = ac.simGetCollisionInfo(vehicle_name=name).time_stamp
                        record['new_collisions'][name] |= current_collision != collisions[name]
                        save(folder / 'observations.json', record)
                finally:
                    ac.simPause(False)
            if (getattr(args, 'fast_calibrate_all_actors', False)
                    and calibrated_instance_keys != set(record['private_native_instances'])):
                raise RuntimeError('Not every scene actor appeared for same-run ABA calibration')
            record.update(stage='actual_views_collected_not_validated', elapsed_wall_seconds=time.time()-episode_start)
            save(folder / 'observations.json', record)
            # This file contains no target/object ground truth.
            save(folder / 'public_observations.json', {
                'episode_id': episode, 'instruction': scene['instruction'],
                'search_region': record['search_region'],
                'observations': [public_observation(o) for o in record['observations']]})
            status['completed_scenes'] += 1
            failed_in_row = 0
        except Exception as error:
            record.update(stage='failed', error=f'{type(error).__name__}: {error}')
            save(folder / 'observations.json', record)
            status['failed_scenes'] += 1
            failed_in_row += 1
        finally:
            if owned:
                responses = cc.apply_batch_sync([carla.command.DestroyActor(x.id) for x in owned], False)
                if any(r.has_error() for r in responses):
                    raise RuntimeError('Failed to clean collector-owned scene actors')
            update()
        if failed_in_row >= 3:
            update(stage='failed', error='Three consecutive scene failures', ended_unix=time.time())
            raise RuntimeError('Three consecutive scene failures; preserve artifacts and diagnose')
    update(stage='collection_finished_not_validated', ended_unix=time.time())
    return status
