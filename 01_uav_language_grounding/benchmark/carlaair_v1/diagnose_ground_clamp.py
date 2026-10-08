"""Privileged runtime diagnosis only; never benchmark or policy evidence."""
import json
import time
from pathlib import Path
import numpy as np
from flight_feedback import reach_views
from runtime_frame_probe import save, xyz
from survey_geometry import assign_views, transit_stages


def pose_delta(estimated, rendered):
    a, b = np.asarray(estimated, float), np.asarray(rendered, float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
        raise ValueError('Expected finite XYZ positions')
    delta = b-a
    return dict(delta_local_ned_m=delta.tolist(), horizontal_gap_m=float(np.linalg.norm(delta[:2])),
                vertical_gap_m=float(abs(delta[2])), total_gap_m=float(np.linalg.norm(delta)))


def run_diagnostic(args, ac, world, calibration, local_origins):
    import carla
    config = json.loads((Path(__file__).parent/'ground_clamp_case.json').read_text())
    names = sorted(ac.listVehicles())
    origin = np.asarray(calibration['world_origin_m'])
    axis = np.array([1., 1., -1.])
    path = args.output/'ground_clamp_diagnostic.json'
    record = dict(stage='ray_audit', started_unix=time.time(), benchmark=False,
                  source=config['source'], cruise_z_m=config['cruise_z_m'],
                  ray_channel_equivalence_verified=False,
                  new_collisions={n:False for n in names}, flight_attempts=[], rays=[])
    def ray(point):
        return [{'xyz':[h.location.x,h.location.y,h.location.z], 'label':str(h.label)}
                for h in world.cast_ray(carla.Location(x=point[0],y=point[1],z=point[2]+100.),
                                        carla.Location(x=point[0],y=point[1],z=point[2]-5.))]
    def persist():
        record['updated_unix']=time.time()
        save(path, record)
    try:
        # CARLA labelled rays and UE ECC_WorldStatic need runtime cross-checking;
        # correspondence is a diagnostic, never asserted from API names alone.
        for row in config['archived_samples']:
            estimated=(np.asarray(row['position_local_ned'])+local_origins[row['agent']])*axis+origin
            rendered=(np.asarray(row['rendered_position_local_ned'])+local_origins[row['agent']])*axis+origin
            hits=ray(estimated)
            record['rays'].append(dict(agent=row['agent'],timestamp=row['timestamp'],
                estimated_world=estimated.tolist(),rendered_world=rendered.tolist(),hits=hits,
                **pose_delta(row['position_local_ned'],row['rendered_position_local_ned'])))
        persist()
        current=[(np.asarray(xyz(ac.simGetVehiclePose(vehicle_name=n).position))+local_origins[n])*axis+origin for n in names]
        targets, assignment=assign_views(current, config['targets_world'])
        record['assignment']=assignment
        record['transit_vertical_separation_m']=10.
        record['stage']='flight'
        # Stop on a 3m mismatch, before waiting at a physically inconsistent pose.
        # This is a diagnostic stop, not a relaxed arrival or collision criterion.
        def observe():
            samples=record.get('trajectory_samples',[])
            if samples:
                recent=samples[-len(names):]
                for s in recent:
                    if s['estimated_rendered_gap_m']>3.:
                        point=(np.asarray(s['position_local_ned'])+local_origins[s['agent']])*axis+origin
                        s['overhead_ray_hits']=ray(point)
                        record['stage']='pose_divergence_detected'
                        persist()
                        raise RuntimeError('Measured renderer/physics divergence exceeds diagnostic 3m stop')
            persist()
        for phase, points in transit_stages(current,targets,cruise_z=config['cruise_z_m'],vertical_separation_m=10.):
            if phase=='approach':
                break  # The controlled variable is transit altitude, not capture.
            record['current_transit_phase']=phase
            remaining= min(180., args.probe_timeout-60-(time.time()-record['started_unix']))
            if remaining<10:raise TimeoutError('Ground clamp diagnostic budget')
            reach_views(ac,names,points,local_origins,origin,0,record,observe,remaining)
        record['stage']='transit_passed'
    except BaseException as error:
        record['error']=repr(error)
        if record['stage']!='pose_divergence_detected':record['stage']='failed'
        raise
    finally:
        record['ended_unix']=time.time();persist()
