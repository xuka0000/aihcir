"""Evaluator-only projected support audit; projection is NOT pixel visibility."""
import argparse
import json
from pathlib import Path
import numpy as np
from geometry import project_carla


def world_to_camera(observation):
    q = np.asarray(observation['camera_orientation_ned_wxyz'], dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .01:
        raise ValueError('Invalid camera quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    axis = np.diag([1., 1., -1.])
    carla_rotation = axis @ r @ axis
    matrix = np.eye(4)
    matrix[:3, :3] = carla_rotation.T
    matrix[:3, 3] = -carla_rotation.T @ np.asarray(observation['camera_world_position'])
    return matrix


def projected_support(observation, vertices):
    p, depth, _ = project_carla(vertices, world_to_camera(observation), 640, 480, 90)
    if not np.all(depth > 0):
        return {'fully_in_front': False, 'box_intersects_image': False, 'bbox_xyxy': None,
                'pixel_visibility_verified': False}
    low, high = p.min(axis=0), p.max(axis=0)
    intersects = bool(high[0] > 0 and high[1] > 0 and low[0] < 640 and low[1] < 480)
    return {'fully_in_front': True, 'box_intersects_image': intersects,
            'bbox_xyxy': [*low.tolist(), *high.tolist()],
            'projected_area_px2': float(np.prod(high-low)),
            'pixel_visibility_verified': False}


def audit(root, output):
    results = []
    for team in (2, 4):
        base = root / f'sessions/runtime_views_{team}uav_v2/probe/training_views'
        for path in sorted(base.glob('train_*/observations.json')):
            row = json.loads(path.read_text())
            if row['stage'] != 'actual_views_collected_not_validated':
                continue
            frames = []
            for obs in row['observations']:
                frames.append({'image': obs['image'], 'agent': obs['agent'],
                    'objects': {key: projected_support(obs, obj['bbox_world_vertices'])
                                for key, obj in row['private_native_instances'].items()}})
            target = [f['objects']['target'] for f in frames]
            results.append({'episode_id': row['episode_id'], 'team_size': team,
                'requested_class': row['visual_condition_requested'],
                'target_box_intersecting_frames': sum(t['box_intersects_image'] for t in target),
                'maximum_target_projected_area': max((t.get('projected_area_px2', 0)
                    for t in target if t['box_intersects_image']), default=0), 'frames': frames})
    summary = {'episodes': len(results), 'no_projected_target_in_any_frame': sum(
        r['target_box_intersecting_frames'] == 0 for r in results),
        'requested_clear_below_1024px2_in_all_frames': sum(r['requested_class'] == 'clear'
        and r['maximum_target_projected_area'] < 1024 for r in results),
        'pixel_visibility_verified': False, 'runtime_validated_episodes': 0,
        'boundary': 'Evaluator-only geometric rejection audit. In-frame projections do not prove visible pixels.',
        'results': results}
    with output.open('x') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
