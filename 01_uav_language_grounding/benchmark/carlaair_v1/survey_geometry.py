"""Fixed collection geometry. Neither a learned policy nor an oracle planner."""
import math
import itertools
import numpy as np
from construction_routes import load_profiles, waypoints as construction_waypoints

CONSTRUCTION_PROFILES = load_profiles()


STAGED_PARAMETERS = {
    'staged110_20_independent_clearance': (110.,20.,0.,1),
    'staged140_50_independent_clearance': (140.,50.,0.,1),
    'staged110_50_phase45_independent_clearance': (110.,50.,math.pi/4,0),
    'staged110_20_phase45_independent_clearance': (110.,20.,math.pi/4,0),
    'staged110_20_phase45_alternate_independent_clearance': (110.,20.,math.pi/4,1),
}
FIXED_SECTOR_PARAMETERS = {'fixedsector20_50_phase45_independent_clearance':(20.,50.,math.pi/4)}
CROSSVIEW_CENTER_SECTOR_HEIGHTS = {
    'crossview_center_sectors_clear_v1_independent_clearance': 45.,
    'crossview_center_sectors_clear_v2_independent_clearance': 40.,
    'crossview_center_sectors_small_v1_independent_clearance': 70.,
    'crossview_center_sectors_occluded_v1_independent_clearance': 50.,
}
PROFILES = ('wide70', 'multiscale70_30', 'multiscale70_30_clearance', 'multiscale70_30_lateral_clearance',
            'multiscale70_30_independent_clearance','splitsectors_clear_2uav_independent_clearance',
             'splitsectors_small_2uav_independent_clearance')+tuple(STAGED_PARAMETERS)+tuple(FIXED_SECTOR_PARAMETERS)+tuple(CROSSVIEW_CENTER_SECTOR_HEIGHTS)+tuple(CONSTRUCTION_PROFILES)


def assign_views(current, targets, locked_indices=None):
    """Exact small-team distance assignment, never target-object-aware.

    Only permutes the same view locations among UAVs. Exhaustive enumeration
    is bounded by 4!; this is not an implementation of the Hungarian algorithm.
    It reduces crossing transits but is not a collision-freedom certificate.
    """
    current, targets = np.asarray(current,float), np.asarray(targets,float)
    if (current.shape != targets.shape or current.ndim != 2 or current.shape[1] != 3
            or len(current) not in (2,4) or not np.isfinite(current).all()
            or not np.isfinite(targets).all()):
        raise ValueError('Invalid team viewpoint assignment')
    def cost(order):
        return float(np.linalg.norm(current[:,:2]-targets[list(order),:2],axis=1).sum())
    if locked_indices is None:
        order = min(itertools.permutations(range(len(current))),key=lambda p:(cost(p),p))
    else:
        if (not isinstance(locked_indices,(list,tuple))
                or any(type(i) is not int for i in locked_indices)
                or sorted(locked_indices)!=list(range(len(current)))):
            raise ValueError('Locked view assignment must be a complete permutation')
        order=tuple(locked_indices)
    return targets[list(order)].tolist(), {'destination_indices':list(order),
        'total_horizontal_distance_m':cost(order),'identity_assignment_distance_m':cost(tuple(range(len(current)))),
        'uses_object_truth':False,'collision_free_proven':False}


def fixed_waypoints(center, team, rounds, profile='wide70'):
    if team not in (2, 4) or rounds < 1 or profile not in PROFILES:
        raise ValueError('Unsupported survey dimensions/profile')
    if len(center) != 2 or not np.isfinite(center).all():
        raise ValueError('Invalid public region centre')
    if profile in CONSTRUCTION_PROFILES:
        return construction_waypoints(CONSTRUCTION_PROFILES[profile],center,team,rounds)
    if profile in CROSSVIEW_CENTER_SECTOR_HEIGHTS:
        if rounds != 4:
            raise ValueError('Cross-view construction route requires four complete rounds')
        z = CROSSVIEW_CENTER_SECTOR_HEIGHTS[profile]
        result = []
        for t in range(4):
            central_offset = 55 if (profile == 'crossview_center_sectors_clear_v2_independent_clearance'
                                    and team == 2 and t == 3) else (-12 if t % 2 else 12)
            central = [float(center[0]), float(center[1] + central_offset), z]
            side = -100. if t % 2 == 0 else 100.
            mobile = [float(center[0] + side), float(center[1] + (-10 if t < 2 else 10)), z]
            if team == 2:
                result.append([central, mobile])
            else:
                left = [float(center[0] - 100), float(center[1] + (-10 if t % 2 else 10)), z]
                right = [float(center[0] + 100), float(center[1] + (10 if t % 2 else -10)), z]
                outer = [float(center[0]), float(center[1] + 65), z]
                result.append([central, left, right, outer])
        return result
    if profile in ('splitsectors_clear_2uav_independent_clearance','splitsectors_small_2uav_independent_clearance'):
        if team!=2 or rounds!=4:
            raise ValueError('Split-sector construction probe requires two agents and four rounds')
        # Fixed public acquisition geometry, developed on archived geometry only.
        # Actual clearance may change positions; full-history pixel audit remains required.
        if profile=='splitsectors_small_2uav_independent_clearance':
            return [[[float(center[0]+4*(t%2)),float(center[1]-20),50.],
                     [float(center[0]+4*(t%2)),float(center[1]+40),50.]] for t in range(4)]
        return [[[float(center[0]+4*(t%2)),float(center[1]-40),50.],
                 [float(center[0]+20+4*(t%2)),float(center[1]),30.]] for t in range(4)]
    if (profile.startswith('multiscale70_30') or profile.startswith('staged') or profile in FIXED_SECTOR_PARAMETERS) and rounds != 4:
        raise ValueError('The multiscale development profile has exactly four rounds')
    result = []
    for t in range(rounds):
        angle,parity=0.,1
        radius, altitude, phase, count = 75., 70., t, rounds
        if profile.startswith('multiscale70_30'):
            radius, altitude = (75., 70.) if t < 2 else (20., 30.)
            phase, count = t % 2, 2
        if profile.startswith('staged'):
            outer,inner,angle,parity=STAGED_PARAMETERS[profile]
            radius,altitude=(outer,70.) if t<2 else (inner,70.)
            phase,count=t%2,2
        if profile in FIXED_SECTOR_PARAMETERS:
            radius,altitude,angle=FIXED_SECTOR_PARAMETERS[profile]
            radius+=20*(t%2)
            phase,count=0,1
        result.append([[float(center[0] + radius * math.cos(angle+2*math.pi*(i/team+phase/(team*count)))),
                        float(center[1] + radius * math.sin(angle+2*math.pi*(i/team+phase/(team*count)))),
                        30. if profile.startswith('staged') and t>=2 and (i+t+parity)%2==0 else altitude] for i in range(team)])
    return result


def clearance_waypoints(targets, screen, ceiling=70., increment=10.):
    """Training acquisition only: raise blocked columns, preserve horizontal views.

    The screen uses sampled static geometry, not target identities or detector
    scores. A failed screen at the ceiling remains a failure, never a bypass.
    Every attempt is retained; passing is not a swept-volume safety proof.
    """
    points = np.asarray(targets, float).copy()
    if (points.ndim != 3 or points.shape[-1] != 3 or not np.isfinite(points).all()
            or not math.isfinite(ceiling) or not math.isfinite(increment)
            or increment <= 0 or points[:,:,2].max() > ceiling):
        raise ValueError('Invalid clearance-acquisition dimensions')
    record = dict(kind='height_only_static_clearance_acquisition', requested=points.tolist(),
                  ceiling_m=ceiling, increment_m=increment, attempts=[], uses_object_truth=False,
                  policy_input=False, swept_volume_safety_proven=False)
    limit = int(math.ceil((ceiling-float(points[:,:,2].min()))/increment))+1
    for _ in range(limit):
        report = screen(points.tolist())
        record['attempts'].append(report)
        if report['passed']:
            record.update(passed=True, actual=points.tolist())
            return points.tolist(), record
        blocked = {(r['round'],r['agent_index']) for r in report['rays'] if r['hits']}
        if not blocked:
            raise ValueError('Failed screen omitted its blocked locations')
        if any(points[t,i,2] >= ceiling for t,i in blocked):
            record.update(passed=False, actual=points.tolist())
            return points.tolist(), record
        for t,i in blocked:
            points[t,i,2] = min(ceiling, points[t,i,2]+increment)
    raise RuntimeError('Clearance loop exceeded deterministic bound')


def lateral_clearance_waypoints(targets, screen):
    """Opt-in static-geometry acquisition repair; never a learned/oracle policy.

    Try eight fixed 20 m horizontal offsets only for blocked columns. Keep
    the 70 m capture ceiling and all ray tests. All attempts are recorded.
    This sampled screen does not prove collision-free swept trajectories.
    """
    original = np.asarray(targets, float)
    points, initial = clearance_waypoints(targets, screen)
    record = dict(kind='bounded_lateral_static_clearance_acquisition', requested=original.tolist(),
                  offsets_m=[[20,0],[-20,0],[0,20],[0,-20],[20,20],[-20,20],[20,-20],[-20,-20]],
                  attempts=[initial], uses_object_truth=False, policy_input=False,
                  swept_volume_safety_proven=False, ceiling_m=70.)
    if initial['passed']:
        record.update(passed=True, actual=points)
        return points, record
    blocked = sorted({(r['round'],r['agent_index']) for r in initial['attempts'][-1]['rays'] if r['hits']})
    for dx,dy in record['offsets_m']:
        candidate = np.asarray(points,float).copy()
        for t,i in blocked:
            candidate[t,i] = original[t,i] + [dx,dy,0]
        # Preserve the 12 m screening footprint separation at capture poses.
        if any(math.dist(batch[i][:2],batch[j][:2]) < 12
               for batch in candidate for i in range(len(batch)) for j in range(i)):
            record['attempts'].append(dict(offset_m=[dx,dy],passed=False,reason='capture_pose_separation'))
            continue
        revised, attempt = clearance_waypoints(candidate.tolist(),screen)
        attempt['offset_m'] = [dx,dy]
        record['attempts'].append(attempt)
        if attempt['passed']:
            record.update(passed=True,actual=revised)
            return revised,record
    record.update(passed=False,actual=points)
    return points,record


def transit_stages(current, targets, cruise_z=70., vertical_separation_m=0.):
    """Climb in place, translate at cruise height, then descend in place."""
    current, targets = np.asarray(current, float), np.asarray(targets, float)
    if (current.ndim != 2 or current.shape[1] != 3 or current.shape != targets.shape
            or not np.isfinite(current).all() or not np.isfinite(targets).all()
            or not math.isfinite(cruise_z) or np.max(targets[:, 2]) > cruise_z
            or not math.isfinite(vertical_separation_m) or vertical_separation_m < 0):
        raise ValueError('Invalid cruise transit')
    z = max(cruise_z, float(np.max(current[:, 2])))
    climb, translate = current.copy(), targets.copy()
    levels = z + np.arange(len(current))*vertical_separation_m
    climb[:, 2] = levels
    translate[:, 2] = levels
    return [('climb', climb.tolist()), ('cruise', translate.tolist()),
            ('approach', targets.tolist())]


def screen_descent_columns(world, targets, cruise_top=70.):
    """Evaluator-only sampled ray screen, NOT a swept-volume safety proof.

    CARLA 0.9.13 World.cast_ray returns intersecting labelled geometry. Sample
    a 12 m-wide footprint, covering the unchanged 5 m arrival gate plus 1 m
    body margin. Actual collision monitoring remains mandatory. No target IDs
    or object truth are used and the chosen route is never changed by this test.
    """
    import carla
    tests = []
    for round_index, batch in enumerate(targets):
        for agent, point in enumerate(batch):
            if point[2] >= cruise_top:
                continue
            for dx in (-6., -3., 0., 3., 6.):
                for dy in (-6., -3., 0., 3., 6.):
                    hits = world.cast_ray(carla.Location(x=point[0]+dx, y=point[1]+dy, z=cruise_top+2.),
                                          carla.Location(x=point[0]+dx, y=point[1]+dy, z=point[2]-2.))
                    tests.append({'round': round_index, 'agent_index': agent,
                                  'offset_xy_m': [dx, dy],
                                  'hits': [{'xyz': [h.location.x, h.location.y, h.location.z],
                                            'label': str(h.label)} for h in hits]})
    return {'kind': 'sampled_static_geometry_rejection_screen', 'rays': tests,
            'passed': all(not t['hits'] for t in tests),
            'swept_volume_safety_proven': False, 'policy_input': False}
