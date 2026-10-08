"""Measured camera geometry and class gates. No inferred visibility from labels."""
import math
import numpy as np


def project_carla(points_world, world_to_camera, width, height, horizontal_fov):
    """CARLA x-forward/y-right/z-up to pixel x-right/y-down, metres throughout.

    Follows CARLA's bounding-box camera projection, not geographic conventions.
    Behind-camera points remain invalid instead of being clipped onto the image.
    """
    points = np.asarray(points_world, dtype=float)
    matrix = np.asarray(world_to_camera, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError('Unexpected geometry dimensions')
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite geometry')
    if width <= 0 or height <= 0 or not 0 < horizontal_fov < 180:
        raise ValueError('Invalid intrinsics')
    homogeneous = np.column_stack((points, np.ones(len(points))))
    local = (matrix @ homogeneous.T).T[:, :3]
    depth = local[:, 0]
    f = width / (2 * math.tan(math.radians(horizontal_fov) / 2))
    pixels = np.full((len(points), 2), np.nan)
    valid = depth > 1e-6
    pixels[valid, 0] = f * local[valid, 1] / depth[valid] + width / 2
    pixels[valid, 1] = -f * local[valid, 2] / depth[valid] + height / 2
    inside = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels, depth, inside


def instance_metrics(mask, target_id, projected_box):
    """Target pixel support from a true instance mask, never from a bbox alone.

    Visible-box fill is NOT an occlusion ratio. Occlusion requires an amodal
    object mask or a separately validated isolated-object reference rendering.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError('Instance-ID image must be two-dimensional')
    box = np.asarray(projected_box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Invalid projected bounding box')
    hits = mask == target_id
    pixels = int(hits.sum())
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    return {'visible_pixels': pixels, 'projected_box_area_px2': area,
            'visible_box_fill': pixels / area, 'occlusion_ratio': None}


def visual_condition(measurement, requested):
    """Thresholds are this suite's declared choices, not a conference standard."""
    pixels = measurement['visible_pixels']
    area = measurement['projected_box_area_px2']
    if pixels < 16:
        return False
    if requested == 'small_target':
        return 16 <= area < 32 ** 2
    ratio = measurement.get('occlusion_ratio')
    if ratio is None or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        return False
    if requested == 'clear':
        return area >= 32 ** 2 and ratio <= 0.1
    if requested == 'occluded':
        return 0.3 <= ratio <= 0.8
    raise ValueError('Unknown requested visual condition')


def progressive_witness(initial_visible, later_visible, intermediary_ids, physical_motion_m):
    """A later view must reveal a previously unseen task-relevant intermediary."""
    new = (set(later_visible) - set(initial_visible)) & set(intermediary_ids)
    return physical_motion_m >= 0.5 and bool(new)
