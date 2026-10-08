"""Bounded, privileged rendered screening of deterministic validation layouts.

This moves one diagnostic camera to the announced fixed survey viewpoints.
It never flies an episode, changes a frozen specification, scores a model, or
admits a candidate to the benchmark. Every attempted layout and image stays in
the owned session's evidence directory, including rejection and RPC failures.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from audit_semantic_gate import _measurement, validate_renderer_control, LARGE_TARGET_AREA_PX2
from audit_observation_coverage import projected_support
from visual_condition_geometry import full_frame_projection
from benchmark import canonical
from pixel_visibility import PixelVisibilityAudit
from render_replay import vehicle_for_camera, pose_residual
from runtime_frame_probe import save, xyz
from scene_catalog_v2 import recipe
from survey_geometry import fixed_waypoints, clearance_waypoints, screen_descent_columns


class RouteRejected(Exception):
    """Expected candidate attrition, separate from runtime implementation failure."""


def validate_preview_request(root, episode, map_name, team, first, count, view_limit, profile):
    """Check the whole request before the owned simulator is launched."""
    if not (isinstance(episode, str) and episode.startswith('validation_')
            and len(episode) == 17 and episode[11:].isdigit()):
        raise ValueError('Preview requires one fixed validation ID; test is sealed')
    if team not in (2, 4) or profile not in ('wide70','multiscale70_30',
                                             'multiscale70_30_clearance'):
        raise ValueError('Preview supports only the fixed wide/multiscale route and measured clearance')
    if not (isinstance(first, int) and first >= 0 and isinstance(count, int)
            and 1 <= count <= 16 and first + count <= 64 and isinstance(view_limit, int)
            and 1 <= view_limit <= team * 4):
        raise ValueError('Preview attempt/view bounds exceeded')
    path = Path(root) / 'specifications' / 'validation.specs.jsonl'
    matches = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
               if json.loads(line)['episode_id'] == episode]
    if len(matches) != 1:
        raise ValueError('Preview ID absent or duplicated in frozen validation specifications')
    spec = matches[0]
    if (spec['map'], spec['team_size'], spec['split']) != (map_name, team, 'validation'):
        raise ValueError('Preview map/team/split does not match the frozen specification')
    return spec


def assess_target(measurement, requested):
    """Use the existing semantic gate's exact pixel/class rule, not a new cutoff."""
    if requested not in ('clear', 'occluded', 'small_target'):
        raise ValueError('Unknown requested condition')
    normalized = _measurement(measurement, renderer_control_passed=True)
    in_frame = full_frame_projection(measurement.get('projected'))
    return {'requested': requested, 'normalized': normalized,
            'full_frame_projection':in_frame,
            'condition_observed': in_frame and requested in normalized['actual_conditions'],
            'decision_scope': 'privileged single-view preview only; flight and episode audit pending'}


def assess_reference(measurement):
    """Conservative pixel floor for a task reference, not a recognizability test."""
    visible = measurement.get('visible_pixels', 0)
    isolated = measurement.get('isolated_candidate_pixels', 0)
    controlled = (measurement.get('visible_aba_exact') is True
                  and measurement.get('isolated_aba_exact') is True
                  and measurement.get('visible_subset_of_isolated') is True
                  and measurement.get('instance_mask_verified') is True)
    supported = (controlled and full_frame_projection(measurement.get('projected'))
                 and isinstance(visible, int) and isinstance(isolated, int)
                 and visible >= 16 and visible == isolated)
    return {'reference_pixel_floor_observed': supported,
            'criterion': 'full-frame, verified instance mask, >=16 visible pixels, no measured missing isolated pixels',
            'visual_recognizability': 'unreviewed',
            'scope': 'conservative necessary screen; RGB interpretation and physical flight remain pending'}


def reference_view_plan(projection_screen, required_keys, per_reference=2):
    """Deterministically test the two largest full-frame projections per reference."""
    if per_reference != 2:
        raise ValueError('Frozen reference render cap is two views per endpoint')
    plan = {}
    for key in required_keys:
        eligible = []
        for row in projection_screen:
            projected = row['endpoint_projections'][key]
            area = projected.get('projected_area_px2')
            if full_frame_projection(projected) and isinstance(area, (int,float)) and math.isfinite(area) and area > 0:
                eligible.append((-area, row['view_index']))
        for _, view_index in sorted(eligible)[:per_reference]:
            plan.setdefault(view_index, []).append(key)
    return {index: sorted(keys) for index, keys in sorted(plan.items())}


def geometry_rejection(projected, requested):
    """Reject only classes impossible under the frozen semantic area rule."""
    area = projected.get('projected_area_px2')
    if not projected.get('box_intersects_image') or area is None or not math.isfinite(area) or area <= 0:
        return 'target_projection_absent'
    if not full_frame_projection(projected):
        return 'target_projection_cropped'
    if requested == 'clear' and area < LARGE_TARGET_AREA_PX2:
        return 'clear_target_area_below_frozen_threshold'
    if requested == 'small_target' and area >= LARGE_TARGET_AREA_PX2:
        return 'small_target_area_above_frozen_threshold'
    return None


def preview_waypoints(center, team, profile, screen=None):
    """Use the collector's own fixed route and 70 m clearance procedure."""
    requested = fixed_waypoints(center,team,4,profile)
    if profile == 'multiscale70_30_clearance':
        if screen is None:
            raise ValueError('Clearance preview requires a measured static screen')
        actual, report = clearance_waypoints(requested,screen)
    else:
        actual, report = requested, {'kind':'no_static_clearance_requested',
                                     'passed':None,'actual':requested,
                                     'swept_volume_safety_proven':False}
    return requested,actual,report


def run_preview(args, ac, cc, calibration, local_origins):
    import airsim
    import carla
    root = args.output.parents[2]
    spec = validate_preview_request(root, args.preview_episode, args.expected_map,
                                    args.agents, args.preview_attempt_start,
                                    args.preview_attempt_count, args.preview_view_limit,
                                    args.survey_profile)
    control_path = args.output / 'controlled_occluder' / 'result.json'
    control_bytes = control_path.read_bytes()
    control = validate_renderer_control(json.loads(control_bytes))
    if not control['passed']:
        raise RuntimeError('Same-run renderer control did not pass')
    out = args.output / 'condition_preview'
    out.mkdir(exist_ok=False)
    world = cc.get_world()
    if world.get_map().name.rsplit('/', 1)[-1] != spec['map']:
        raise RuntimeError('Preview map changed after calibration')
    world.set_weather(carla.WeatherParameters.ClearNoon)
    points = [[t.location.x, t.location.y, t.location.z, t.rotation.yaw]
              for t in world.get_map().get_spawn_points()]
    blueprints = [bp.id for bp in world.get_blueprint_library()]
    status = {'kind': 'privileged_rendered_condition_candidate_preview_v6',
              'stage': 'running', 'episode_id': spec['episode_id'],
              'spec_sha256': hashlib.sha256(canonical(spec).encode()).hexdigest(),
              'requested': spec['visual_condition_requested'],
              'renderer_control_sha256': hashlib.sha256(control_bytes).hexdigest(),
              'renderer_control_passed': True,
              'attempt_indices': list(range(args.preview_attempt_start,
                                            args.preview_attempt_start+args.preview_attempt_count)),
              'view_limit_per_attempt': args.preview_view_limit,
              'survey_profile': args.survey_profile,
              'attempts': [], 'selected_attempt': None,
              'benchmark_episodes_validated': 0, 'publication_scores': None,
              'dataset_ready': False,
              'reference_rule': 'two largest full-frame projections per reference; one verified unoccluded >=16-pixel instance view required',
              'reference_visual_recognizability': 'requires separate original-RGB review',
              'boundary': 'Diagnostic teleport and truth-assisted masks; full physical flight, collision, task-family and independent instance audits remain pending',
              'started_unix': time.time()}
    def update():
        status['updated_unix'] = time.time()
        save(out / 'result.json', status)
    update()
    origin = np.asarray(calibration['world_origin_m'], dtype=float)
    axis = np.array([1., 1., -1.])
    agent = 'UAV_0'
    mount_q = airsim.to_quaternion(-math.pi/2, 0, 0)
    mount_offset = np.array([.5, 0., .1])
    expected_q = np.array([mount_q.w_val, mount_q.x_val, mount_q.y_val, mount_q.z_val])
    ac.simSetCameraPose('0', airsim.Pose(airsim.Vector3r(*mount_offset), mount_q),
                        vehicle_name=agent)
    deadline = time.monotonic() + min(900, max(60, args.probe_timeout-180))
    try:
        for attempt in status['attempt_indices']:
            row = {'attempt': attempt, 'stage': 'generating', 'views': [],
                   'projection_screen': [],
                   'condition_observed': False, 'accepted_for_collection': False,
                   'started_unix': time.time()}
            status['attempts'].append(row)
            update()
            owned = []
            paused = False
            pixel = None
            folder = out / f'attempt_{attempt:02d}'
            folder.mkdir()
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Preview bound exhausted before candidate generation')
                scene = recipe(spec, points, blueprints, attempt)
                save(folder / 'scene_geometry.json', scene)
                row['geometry_sha256'] = scene['geometry_sha256']
                row['scene_file_sha256'] = hashlib.sha256((folder/'scene_geometry.json').read_bytes()).hexdigest()
                old = root/'scene_assets_v3'/'validation'/spec['episode_id']/'scene.json'
                if old.is_file() and json.loads(old.read_text()).get('geometry_sha256') == scene['geometry_sha256']:
                    raise ValueError('Generated layout duplicates the frozen original')
                pixel = PixelVisibilityAudit(ac)
                instances = {}
                required_references = [obj['key'] for obj in scene['objects']
                                       if obj['key'] in ('reference','second_reference')]
                row['required_references'] = sorted(required_references)
                for obj in scene['objects']:
                    tracked = obj['key'] == 'target' or obj['key'] in required_references
                    before = set(ac.simListSceneObjects()) if tracked else None
                    bp = world.get_blueprint_library().find(obj['blueprint'])
                    if 'color' in obj and bp.has_attribute('color'):
                        bp.set_attribute('color', obj['color'])
                    x, y, z, yaw = obj['transform']
                    actor = world.try_spawn_actor(bp, carla.Transform(
                        carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)))
                    if actor is None:
                        raise RuntimeError(f'Native spawn failed: {obj["key"]}')
                    owned.append(actor)
                    actor.set_simulate_physics(False)
                    if tracked:
                        pixel.register(obj['key'], set(ac.simListSceneObjects())-before)
                        vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
                        instances[obj['key']] = {'bbox_world_vertices':
                            [[v.x,v.y,v.z] for v in vertices]}
                for _ in range(3):
                    world.wait_for_tick(20)
                locations = np.asarray([obj['transform'][:2] for obj in scene['objects']])
                center = (np.round(locations.mean(axis=0)/100)*100).tolist()
                screen = (lambda candidates: screen_descent_columns(world,candidates,cruise_top=70.))
                requested_waypoints,waypoints,route = preview_waypoints(
                    center,args.agents,args.survey_profile,screen)
                row['center_world_xy'] = center
                row['requested_waypoints_sha256'] = hashlib.sha256(canonical(requested_waypoints).encode()).hexdigest()
                row['route_preflight'] = route
                row['waypoints_sha256'] = hashlib.sha256(canonical(waypoints).encode()).hexdigest()
                update()
                if route['passed'] is False:
                    raise RouteRejected('no_clear_descent_within_acquisition_ceiling')
                all_views = [(r,s,p) for r,batch in enumerate(waypoints)
                             for s,p in enumerate(batch)][:args.preview_view_limit]
                for view_index,(round_index,slot,waypoint) in enumerate(all_views):
                    camera = {
                        'camera_world_position':waypoint,
                        'camera_orientation_ned_wxyz':expected_q.tolist()}
                    endpoint_projections = {key:projected_support(camera,obj['bbox_world_vertices'])
                                            for key,obj in instances.items()}
                    projected = endpoint_projections['target']
                    row['projection_screen'].append({'view_index':view_index,
                        'round':round_index,'nominal_agent_slot':slot,
                        'waypoint_world':waypoint,'projected':projected,
                        'endpoint_projections':endpoint_projections,
                        'rejection':geometry_rejection(projected,spec['visual_condition_requested'])})
                eligible = [entry for entry in row['projection_screen'] if entry['rejection'] is None]
                reference_plan = reference_view_plan(row['projection_screen'], required_references)
                row['reference_view_plan'] = reference_plan
                row['reference_pixel_floor_observed'] = {key:False for key in required_references}
                render_indices = sorted({entry['view_index'] for entry in eligible} | set(reference_plan))
                row['stage'] = 'rendering' if render_indices else 'rejected_projection'
                update()
                for view_index in render_indices:
                    candidate_view = row['projection_screen'][view_index]
                    view_index = candidate_view['view_index']
                    reference_keys = [key for key in reference_plan.get(view_index, [])
                                      if not row['reference_pixel_floor_observed'][key]]
                    test_target = candidate_view['rejection'] is None and not row['condition_observed']
                    if not reference_keys and not test_target:
                        continue
                    round_index,slot,waypoint = (candidate_view['round'],
                        candidate_view['nominal_agent_slot'],candidate_view['waypoint_world'])
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Preview bound exhausted during rendered views')
                    expected_global = (np.asarray(waypoint)-origin)*axis
                    body_position, body_q = vehicle_for_camera(
                        expected_global, expected_q, mount_offset, expected_q)
                    ac.simPause(True)
                    paused = True
                    if not ac.simSetObjectPose(agent, airsim.Pose(
                            airsim.Vector3r(*body_position),
                            airsim.Quaternionr(body_q[1],body_q[2],body_q[3],body_q[0])),
                            teleport=True):
                        raise RuntimeError('Diagnostic rendered-camera placement failed')
                    request = [airsim.ImageRequest('0', airsim.ImageType.Scene, False, True)]
                    for _ in range(8):
                        response = ac.simGetImages(request, vehicle_name=agent)[0]
                        time.sleep(.05)
                    expected_local = expected_global-np.asarray(local_origins[agent])
                    residual = pose_residual(response, expected_local, expected_q)
                    if residual['position_m'] > .01 or residual['quaternion_l2'] > 1e-4:
                        raise RuntimeError(f'Preview camera pose mismatch: {residual}')
                    data = bytes(response.image_data_uint8)
                    with Image.open(io.BytesIO(data)) as image:
                        if float(np.asarray(image.convert('RGB')).std()) < 1:
                            raise RuntimeError('Nearly constant diagnostic RGB')
                    filename = f'view_{view_index:02d}.png'
                    (folder/filename).write_bytes(data)
                    q = response.camera_orientation
                    observation = {'agent': agent, 'image': filename,
                        'timestamp': int(response.time_stamp),
                        'camera_local_position': xyz(response.camera_position),
                        'camera_world_position': waypoint,
                        'camera_orientation_ned_wxyz': [q.w_val,q.x_val,q.y_val,q.z_val]}
                    projected = projected_support(observation,instances['target']['bbox_world_vertices'])
                    rejection = geometry_rejection(projected,spec['visual_condition_requested'])
                    if test_target and (rejection != candidate_view['rejection'] or (rejection is None and
                            abs(projected['projected_area_px2']-
                                candidate_view['projected']['projected_area_px2']) > 1.)):
                        raise RuntimeError('Measured camera projection disagrees with pre-screen')
                    view = {'round':round_index,'nominal_agent_slot':slot,
                        'diagnostic_agent':agent,'waypoint_world':waypoint,
                        'rgb_sha256':hashlib.sha256(data).hexdigest(),
                        'camera_pose_residual':residual,'projected':projected,
                        'geometry_rejection':rejection}
                    measured = pixel.capture(agent, observation,
                        {key:instances[key] for key in (['target'] if test_target else []) + reference_keys}, folder)
                    if test_target:
                        measurement = measured['target']
                        assessed = assess_target(measurement, spec['visual_condition_requested'])
                        view.update(measurement=measurement,assessment=assessed)
                        row['condition_observed'] |= assessed['condition_observed']
                    view['endpoint_measurements'] = {key:measured[key] for key in reference_keys}
                    view['reference_assessments'] = {key:assess_reference(measured[key])
                                                     for key in reference_keys}
                    for key in reference_keys:
                        row['reference_pixel_floor_observed'][key] |= view['reference_assessments'][key]['reference_pixel_floor_observed']
                    row['views'].append(view)
                    update()
                    ac.simPause(False)
                    paused = False
                    if row['condition_observed'] and all(row['reference_pixel_floor_observed'].values()):
                        break
                reference_complete = all(row['reference_pixel_floor_observed'].values())
                row['stage'] = ('preview_condition_and_reference_floor_observed'
                                if row['condition_observed'] and reference_complete else
                                'rejected_reference_floor' if not reference_complete else
                                'rejected_condition_absent' if eligible else 'rejected_projection')
                row['rejection_reason'] = (None if row['condition_observed'] and reference_complete else
                    'reference_pixel_floor_not_observed' if not reference_complete else
                    'requested_condition_not_observed_in_rendered_eligible_views' if eligible else
                    'no_view_satisfies_frozen_projection_necessity')
                if row['condition_observed'] and reference_complete:
                    status['selected_attempt'] = attempt
                    status['selection_scope'] = 'preview_only_not_collection_admission'
            except RouteRejected as error:
                row.update(stage='rejected_static_route',rejection_reason=str(error))
            except Exception as error:
                row.update(stage='failed', rejection_reason=f'{type(error).__name__}: {error}')
            finally:
                if paused:
                    ac.simPause(False)
                if pixel is not None:
                    try:
                        pixel.reset(0)
                    except Exception as cleanup_error:
                        row.update(stage='failed', rejection_reason=f'Stencil cleanup failed: {cleanup_error}')
                        status['stage'] = 'failed'
                if owned:
                    replies = cc.apply_batch_sync([carla.command.DestroyActor(a.id) for a in owned],False)
                    if any(reply.has_error() for reply in replies):
                        row.update(stage='failed', rejection_reason='Owned actor cleanup failed')
                        status['stage'] = 'failed'
                row['ended_unix'] = time.time()
                update()
            if status['stage'] == 'failed' or status['selected_attempt'] is not None:
                break
        if status['stage'] != 'failed':
            status['stage'] = ('preview_match_not_benchmark' if status['selected_attempt'] is not None
                               else 'all_attempts_rejected')
        status['ended_unix'] = time.time()
        update()
        return {key:status[key] for key in ('stage','episode_id','requested','selected_attempt',
                    'benchmark_episodes_validated','dataset_ready')}
    except Exception as error:
        status.update(stage='failed',error=f'{type(error).__name__}: {error}',ended_unix=time.time())
        update()
        raise
