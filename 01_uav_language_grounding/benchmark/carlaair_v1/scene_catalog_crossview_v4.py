"""Two-sector cross-view candidates for a separately versioned benchmark.

The scene is constructed around a public search region. One camera route
surveys its centre and other routes survey the two road clusters. Construction
geometry is private; rendered visibility and multi-agent necessity remain
separate admission checks. This never changes the frozen v1 scene assets.
"""

import hashlib
import math
import random

from benchmark import canonical
from scene_catalog_v2 import CAR, TRUCK, VAN, counterfactual_audit, distance, enu, holds


VERSION = "crossview_two_sector_candidates_v4"
PROFILES_BY_CONDITION = {
    "clear": "crossview_center_sectors_clear_v1_independent_clearance",
    "small_target": "crossview_center_sectors_small_v1_independent_clearance",
    "occluded": "crossview_center_sectors_occluded_v1_independent_clearance",
}
CAR_COLOR = "240,240,240"
VAN_COLOR = "200,160,20"


def _neighbors(points):
    return [tuple(j for j, other in enumerate(points)
                  if j != i and 9 <= distance(point, other) <= 19)
            for i, point in enumerate(points)]


def recipe(spec, points, blueprint_ids, attempt=0, max_layout_attempts=256,candidate_accept=None):
    if spec.get("task_family") != "cross_view_disambiguation":
        raise ValueError("Only cross-view specifications are supported")
    if spec.get("split") not in ("train", "validation", "test"):
        raise ValueError("Invalid split")
    if spec.get("team_size") not in (2, 4):
        raise ValueError("Invalid team size")
    if spec.get("visual_condition_requested") not in PROFILES_BY_CONDITION:
        raise ValueError("Invalid visual condition")
    if not {CAR, VAN, TRUCK}.issubset(set(blueprint_ids)):
        raise ValueError("Required native vehicle assets are absent")
    if not 1 <= max_layout_attempts <= 1024 or attempt < 0:
        raise ValueError("Invalid bounded search")
    points = tuple(tuple(float(v) for v in point) for point in points)
    if len(points) < 8 or any(len(point) < 3 for point in points):
        raise ValueError("Insufficient native road points")

    near = _neighbors(points)
    road_clusters = [i for i, neighbors in enumerate(near) if neighbors]
    rng = random.Random(int(spec["scene_seed"]) ^ (attempt * 0x9E3779B97F4A7C15))
    engines = list(range(len(points)))
    rng.shuffle(engines)
    for layout_attempt, engine_index in enumerate(engines[:max_layout_attempts]):
        engine = points[engine_index]
        left = [i for i in road_clusters
                if -125 <= points[i][0] - engine[0] <= -85
                and abs(points[i][1] - engine[1]) <= 48]
        right = [i for i in road_clusters
                 if 85 <= points[i][0] - engine[0] <= 125
                 and abs(points[i][1] - engine[1]) <= 48]
        rng.shuffle(left)
        rng.shuffle(right)
        pairs = [(i, j) for i in left for j in right]
        rng.shuffle(pairs)
        for li, ri in pairs[:128]:
            reference_index, other_van_index = (li, ri) if rng.randrange(2) else (ri, li)
            reference, other_van = points[reference_index], points[other_van_index]
            if distance(reference, other_van) < 170:
                continue
            relations = [direction for direction in ("north", "south")
                         if holds(reference, engine, direction)
                         and not holds(other_van, engine, direction)
                         and abs(reference[1] - engine[1]) >= 12
                         and abs(other_van[1] - engine[1]) >= 12]
            if not relations:
                continue
            target_options = [points[i] for i in near[reference_index]
                              if distance(points[i], other_van) > 25
                              and distance(points[i], engine) > 25]
            distractor_options = [points[i] for i in near[other_van_index]
                                  if distance(points[i], reference) > 25
                                  and distance(points[i], engine) > 25]
            if not target_options or not distractor_options:
                continue
            target = rng.choice(target_options)
            distractor = rng.choice(distractor_options)
            extra_options = [point for point in points
                             if all(distance(point, occupied) >= 25 for occupied in
                                    (target, reference, distractor, other_van, engine))]
            if not extra_options:
                continue
            extra = rng.choice(extra_options)
            # The centre defines a public search region, not an exact target
            # coordinate. The route is fixed relative to it for all scenes.
            center = [(reference[k] + other_van[k] + engine[k]) / 3 for k in (0, 1)]
            if (abs(engine[0] - center[0]) > 25 or
                    any(abs(point[0] - center[0]) < 78 for point in (reference, other_van)) or
                    any(abs(point[k] - center[k]) > 145 for point in
                        (target, reference, distractor, other_van, engine, extra) for k in (0, 1))):
                continue
            direction = rng.choice(relations)
            objects = [
                {"key": "target", "blueprint": CAR, "color": CAR_COLOR, "transform": target},
                {"key": "reference", "blueprint": VAN, "color": VAN_COLOR, "transform": reference},
                {"key": "second_reference", "blueprint": TRUCK, "transform": engine},
                {"key": "alternative_reference", "blueprint": VAN, "color": VAN_COLOR, "transform": other_van},
                {"key": "distractor_0", "blueprint": CAR, "color": CAR_COLOR, "transform": distractor},
                {"key": "distractor_1", "blueprint": CAR, "color": CAR_COLOR, "transform": extra},
            ]
            positions = {obj["key"]: enu(obj["transform"]) for obj in objects}
            required = [
                {"relation_id": "r0", "subject": "target", "object": "reference", "relation": "near"},
                {"relation_id": "r1", "subject": "reference", "object": "second_reference", "relation": direction},
            ]
            scene = {**spec, "construction_version": VERSION, "status": "geometry_compiled",
                     "instruction": f"Find the car near the van {direction} of the fire engine. Use world directions.",
                     "objects": objects, "geometry_sha256": hashlib.sha256(canonical(objects).encode()).hexdigest(),
                     "generation_attempt": attempt, "layout_search_attempt": layout_attempt,
                     "search_region_center_xy": center,
                     "survey_profile_required": PROFILES_BY_CONDITION[spec["visual_condition_requested"]],
                     "private_truth": {
                         "target_id": "target", "target_position": positions["target"],
                         "candidate_positions": {key: value for key, value in positions.items()
                                                 if key == "target" or key.startswith("distractor")},
                         "positions": positions, "required_relations": required,
                         "crossview_construction": {
                             "endpoint_span_m": distance(reference, other_van),
                             "engine_to_target_van_m": distance(engine, reference),
                             "engine_to_other_van_m": distance(engine, other_van),
                             "native_visibility_proven": False,
                             "multi_agent_necessity_proven": False}},
                     "runtime_validation": {"spawn": False, "visual_condition": False,
                                            "crossview_necessity": False, "flight": False}}
            audit = counterfactual_audit(scene)
            if audit["unique_target"] and audit["every_relation_target_informative"]:
                scene["private_truth"]["counterfactual_audit"] = audit
                if candidate_accept is not None and not candidate_accept(scene):continue
                return scene
    raise ValueError("No informative two-sector layout in the bounded native-map search")
