"""Measured closed-loop survey controller, not a learned planning algorithm."""
import math
import time
import numpy as np


def bounded_velocity(position, target, max_speed=8., gain=.6):
    error = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    if error.shape != (3,) or not np.isfinite(error).all() or max_speed <= 0 or gain <= 0:
        raise ValueError('Invalid flight feedback')
    velocity = gain * error
    speed = float(np.linalg.norm(velocity))
    return velocity * min(1., max_speed/max(speed, 1e-9))


def reach_views(ac, names, targets, local_origins, origin, round_index,
                record, save_record, remaining_seconds):
    """One-second commands, measured pose/timestamp each cycle, fixed 5m gate.

    Prevents a long asynchronous move timeout from being mistaken for arrival.
    Sampled trajectory lengths are identified as sampled, not mathematically exact.
    """
    import airsim
    axis = np.array([1., 1., -1.])
    desired = {name: (np.asarray(target)-origin)*axis-np.asarray(local_origins[name])
               for name, target in zip(names, targets)}
    start = time.monotonic()
    prior = {}
    trace = record.setdefault('trajectory_samples', [])
    lengths = record.setdefault('sampled_trajectory_length_m', {n: 0. for n in names})
    record['trajectory_sampling'] = 'At each <=1s commanded motion cycle; measured sim timestamps retained'
    collisions = {n: ac.simGetCollisionInfo(vehicle_name=n).time_stamp for n in names}
    initial = {}
    best = float('inf')
    last_improvement = time.monotonic()
    while True:
        distances = {}
        rendered_distances = {}
        measured = {}
        for name in names:
            state = ac.getMultirotorState(vehicle_name=name)
            p = state.kinematics_estimated.position
            point = np.array([p.x_val, p.y_val, p.z_val])
            measured[name] = point
            distances[name] = float(np.linalg.norm(desired[name]-point))
            # An estimated-state arrival is insufficient if the renderer has
            # not reached the same pose. Keep the original velocity feedback,
            # but require both interfaces to agree before pausing for images.
            rendered = ac.simGetVehiclePose(vehicle_name=name).position
            rendered_point = np.array([rendered.x_val,rendered.y_val,rendered.z_val])
            if not np.isfinite(rendered_point).all():
                raise ValueError('Invalid renderer pose during arrival check')
            rendered_distances[name] = float(np.linalg.norm(desired[name]-rendered_point))
            initial.setdefault(name, distances[name])
            if name in prior:
                lengths[name] += float(np.linalg.norm(point-prior[name]))
            prior[name] = point
            collision_info = ac.simGetCollisionInfo(vehicle_name=name)
            collision = collision_info.time_stamp
            trace.append({'round': round_index, 'agent': name, 'timestamp': int(state.timestamp),
                          'transit_phase': record.get('current_transit_phase', 'direct'),
                          'position_local_ned': point.tolist(), 'remaining_m': distances[name],
                          'rendered_position_local_ned':rendered_point.tolist(),
                          'rendered_remaining_m':rendered_distances[name],
                          'estimated_rendered_gap_m':float(np.linalg.norm(point-rendered_point)),
                          'collision_timestamp': int(collision)})
            if collision != collisions[name]:
                record['new_collisions'][name] = True
                record.setdefault('collision_events', []).append({
                    'agent': name, 'round': round_index,
                    'transit_phase': record.get('current_transit_phase', 'direct'),
                    'timestamp': int(collision), 'object_name': collision_info.object_name,
                    'object_id': int(collision_info.object_id),
                    'penetration_depth': float(collision_info.penetration_depth),
                    'impact_point': [collision_info.impact_point.x_val,
                                     collision_info.impact_point.y_val,
                                     collision_info.impact_point.z_val]})
                save_record()
                raise RuntimeError(f'New collision during feedback flight: {name}')
        if record.get('transit_vertical_separation_m', 0) > 0:
            global_points = [measured[n]+np.asarray(local_origins[n]) for n in names]
            minimum = min(float(np.linalg.norm(a-b)) for i,a in enumerate(global_points)
                          for b in global_points[i+1:])
            record['minimum_sampled_team_separation_m'] = min(minimum,
                record.get('minimum_sampled_team_separation_m', float('inf')))
            if minimum < 3.:
                save_record()
                raise RuntimeError(f'Team separation safety stop at {minimum:.3f} m')
        save_record()
        maximum = max(*distances.values(),*rendered_distances.values())
        if maximum < 1.5:
            break
        if maximum < best-1.:
            best = maximum
            last_improvement = time.monotonic()
        # Distance-scaled transit allowance; final mission budget remains unchanged.
        limit = min(remaining_seconds, max(initial.values())/3. + 45.)
        if time.monotonic()-start > limit or time.monotonic()-last_improvement > 30:
            raise TimeoutError(f'Feedback transit budget/stall: estimated={distances}, rendered={rendered_distances}')
        futures = []
        for name in names:
            velocity = bounded_velocity(measured[name], desired[name])
            futures.append(ac.moveByVelocityAsync(*velocity.tolist(), 1.,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.), vehicle_name=name))
        for future in futures:
            future.join()
    for name in names:
        ac.hoverAsync(vehicle_name=name).join()
    record['flight_attempts'].append({'round': round_index, 'controller': 'measured_velocity_feedback',
        'transit_phase': record.get('current_transit_phase', 'direct'),
        'initial_distances_m': initial, 'final_prehover_distances_m': distances,
        'final_prehover_rendered_distances_m':rendered_distances,
        'wall_seconds': time.monotonic()-start})
    save_record()
