"""Controlled renderer unit test. Suspended assets are NOT benchmark episodes."""
import hashlib
import math
from pathlib import Path
import numpy as np
from PIL import Image
from flight_feedback import reach_views
from pixel_visibility import PixelVisibilityAudit
from runtime_frame_probe import save, xyz


def run_control(args, ac, cc, calibration, local_origins):
    import airsim
    import carla
    out = args.output / 'controlled_occluder'
    out.mkdir(exist_ok=False)
    record = {'kind': 'renderer_unit_test_not_benchmark', 'stage': 'preparing',
              'flight_attempts': [], 'new_collisions': {f'UAV_{i}': False for i in range(args.agents)},
              'publication_scores': None, 'benchmark_episodes_validated': 0}
    def write():
        save(out / 'result.json', record)
    owned = []
    world = cc.get_world()
    origin = np.asarray(calibration['world_origin_m'])
    axis = np.array([1., 1., -1.])
    names = [f'UAV_{i}' for i in range(args.agents)]
    paused = False
    try:
        for name in names:
            ac.simSetCameraPose('0', airsim.Pose(airsim.Vector3r(.5, 0, .1),
                airsim.to_quaternion(-math.pi/2, 0, 0)), vehicle_name=name)
        targets = [[200+30*i, 200, 150] for i in range(args.agents)]
        reach_views(ac, names, targets, local_origins, origin, 0, record, write, 180)
        pixel = PixelVisibilityAudit(ac)
        instances = {}
        for key, blueprint, location in [('target', 'vehicle.tesla.model3', [200, 200, 100]),
                ('occluder', 'vehicle.carlamotors.firetruck', [260, 200, 125])]:
            before = set(ac.simListSceneObjects())
            actor = world.spawn_actor(world.get_blueprint_library().find(blueprint),
                carla.Transform(carla.Location(x=location[0], y=location[1], z=location[2])))
            owned.append(actor)
            actor.set_simulate_physics(False)
            pixel.register(key, set(ac.simListSceneObjects())-before)
            instances[key] = {'actor_id': actor.id,
                'bbox_world_vertices': [[v.x, v.y, v.z] for v in actor.bounding_box.get_world_vertices(actor.get_transform())]}
        # Spawn acknowledgement precedes first physics/render initialization.
        # Allow the world to complete initialization before the paused control.
        for _ in range(3):
            world.wait_for_tick(20)
        ac.simPause(True)
        paused = True
        observations = []
        for condition in ['clear', 'blocked', 'restored']:
            location = carla.Location(x=200 if condition == 'blocked' else 260, y=200, z=125)
            owned[1].set_transform(carla.Transform(location))
            request = [airsim.ImageRequest('0', airsim.ImageType.Scene, False, True)]
            ac.simGetImages(request, vehicle_name='UAV_0')
            response = ac.simGetImages(request, vehicle_name='UAV_0')[0]
            image_path = out / (condition+'.png')
            image_path.write_bytes(bytes(response.image_data_uint8))
            q = response.camera_orientation
            camera_world = (np.asarray(xyz(response.camera_position))+local_origins['UAV_0'])*axis+origin
            obs = {'agent': 'UAV_0', 'image': image_path.name, 'timestamp': int(response.time_stamp),
                   'camera_world_position': camera_world.tolist(),
                   'camera_orientation_ned_wxyz': [q.w_val, q.x_val, q.y_val, q.z_val]}
            metrics = pixel.capture('UAV_0', obs, {'target': instances['target']}, out, debug=True)['target']
            observations.append(dict(condition=condition, observation=obs, measurement=metrics))
            record.update(stage='measuring', observations=observations)
            write()
        masks = []
        visible = []
        for row in observations:
            with Image.open(out / row['measurement']['isolated_candidate_mask']) as img:
                masks.append(np.array(img))
            visible.append(row['measurement']['visible_pixels'])
        passes = {'clear_visible_at_least_16_pixels': visible[0] >= 16,
                  'blocked_fully_occluded': visible[1] == 0,
                  'isolated_mask_bit_identical_under_occluder': np.array_equal(masks[0], masks[1]),
                  'restored_isolated_mask_identical': np.array_equal(masks[0], masks[2]),
                  'restored_visible_count_identical': visible[0] == visible[2],
                  'same_paused_timestamp': len({r['observation']['timestamp'] for r in observations}) == 1}
        record.update(stage='passed' if all(passes.values()) else 'failed', checks=passes,
                      boundary='Single controlled shader test, not validation of every benchmark image')
        write()
        if not all(passes.values()):
            raise RuntimeError('Controlled amodal/visibility test failed')
    finally:
        if paused:
            ac.simPause(False)
        if owned:
            responses = cc.apply_batch_sync([carla.command.DestroyActor(a.id) for a in owned], False)
            if any(r.has_error() for r in responses):
                raise RuntimeError('Controlled actor cleanup failed')
