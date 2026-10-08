"""Fail-closed semantic audit for held-out CARLA-AirSim captures.

This audit separates sensor integrity, requested visual-condition evidence, and
task-family observability.  It never converts a construction label into a
validated benchmark episode.  In particular, progressive-discovery scenes stay
blocked until the generator creates an implicit relation that is revealed by
physical observation rather than stated in the instruction.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from visual_condition_geometry import full_frame_projection


VIEW = re.compile(r"round(\d+)_UAV_(\d+)\.png")
MIN_VISIBLE_PIXELS = 16
LARGE_TARGET_AREA_PX2 = 32 ** 2
MIN_PROGRESSIVE_MOTION_M = 0.5


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_renderer_control(control):
    required = {
        "clear_visible_at_least_16_pixels",
        "blocked_fully_occluded",
        "isolated_mask_bit_identical_under_occluder",
        "restored_isolated_mask_identical",
        "restored_visible_count_identical",
        "same_paused_timestamp",
    }
    checks = control.get("checks", {})
    passed = (
        control.get("kind") == "renderer_unit_test_not_benchmark"
        and control.get("stage") == "passed"
        and required <= set(checks)
        and all(checks[key] is True for key in required)
    )
    return {
        "passed": passed,
        "required_checks": sorted(required),
        "failed_checks": sorted(key for key in required if checks.get(key) is not True),
        "boundary": "Renderer measurement protocol only. This is not benchmark or model validation.",
    }


def _measurement(measurement, renderer_control_passed):
    visible = measurement.get("visible_pixels")
    isolated = measurement.get("isolated_candidate_pixels")
    projected = measurement.get("projected", {}).get("projected_area_px2")
    local_controls = all(
        measurement.get(key) is True
        for key in ("visible_subset_of_isolated", "visible_aba_exact", "isolated_aba_exact")
    )
    numbers_valid = (
        isinstance(visible, int)
        and isinstance(isolated, int)
        and visible >= 0
        and isolated >= visible
        and isolated > 0
        and isinstance(projected, (int, float))
        and math.isfinite(projected)
        and projected > 0
    )
    ratio = None
    if renderer_control_passed and local_controls and numbers_valid:
        ratio = 1.0 - visible / isolated
    verified_visible = bool(
        visible is not None
        and visible >= MIN_VISIBLE_PIXELS
        and measurement.get("instance_mask_verified") is True
        and local_controls
    )
    conditions = []
    in_frame = full_frame_projection(measurement.get('projected'))
    if verified_visible and in_frame and projected < LARGE_TARGET_AREA_PX2:
        conditions.append("small_target")
    if verified_visible and in_frame and ratio is not None:
        if projected >= LARGE_TARGET_AREA_PX2 and ratio <= 0.1:
            conditions.append("clear")
        if 0.3 <= ratio <= 0.8:
            conditions.append("occluded")
    return {
        "visible_pixels": visible,
        "isolated_candidate_pixels": isolated,
        "projected_area_px2": projected,
        "occlusion_ratio": ratio,
        "verified_visible": verified_visible,
        "local_stencil_controls_passed": local_controls,
        "actual_conditions": conditions,
    }


def _task_endpoints(scene):
    relations = scene.get("private_truth", {}).get("required_relations", [])
    endpoints = {
        relation[key]
        for relation in relations
        for key in ("subject", "object")
        if key in relation
    }
    endpoints.add("target")
    return endpoints


def _minimum_cover(visible_by_agent, endpoints):
    agents = sorted(visible_by_agent)
    for size in range(1, len(agents) + 1):
        for group in itertools.combinations(agents, size):
            union = set().union(*(visible_by_agent[agent] for agent in group))
            if endpoints <= union:
                return list(group)
    return None


def audit_episode(record, scene, renderer_control_passed):
    if record.get("episode_id") != scene.get("episode_id"):
        raise ValueError("Scene/observation ID mismatch")
    if record.get("stage") != "actual_views_collected_not_validated":
        raise ValueError("Episode capture is incomplete")
    team_size = record.get("team_size")
    if team_size != scene.get("team_size") or team_size not in (2, 4):
        raise ValueError("Unexpected team size")
    if any(record.get("new_collisions", {}).values()):
        raise ValueError("Collision-tainted episode")

    grouped = {}
    for observation in record.get("observations", []):
        match = VIEW.fullmatch(observation.get("image", ""))
        if not match or observation.get("agent") != f"UAV_{int(match[2])}":
            raise ValueError("Unexpected observation identity")
        grouped.setdefault(int(match[1]), []).append(observation)
    if sorted(grouped) != list(range(len(grouped))):
        raise ValueError("Incomplete round sequence")

    endpoints = _task_endpoints(scene)
    target_condition_views = []
    round_rows = []
    initial_visible = set()
    initial_positions = {}
    progressive_witnesses = []
    for round_id, observations in sorted(grouped.items()):
        expected_agents = {f"UAV_{index}" for index in range(team_size)}
        if {row["agent"] for row in observations} != expected_agents or len(observations) != team_size:
            raise ValueError("Incomplete or duplicate team round")
        visible_by_agent = {}
        for observation in observations:
            image = observation["image"]
            measures = record.get("private_pixel_measurements", {}).get(image, {})
            if set(measures) != set(record.get("private_native_instances", {})):
                raise ValueError("Missing private instance measurement")
            normalized = {
                key: _measurement(value, renderer_control_passed)
                for key, value in measures.items()
            }
            visible = {key for key, value in normalized.items() if value["verified_visible"]}
            visible_by_agent[observation["agent"]] = visible
            target = normalized.get("target", {})
            for condition in target.get("actual_conditions", []):
                target_condition_views.append(
                    {
                        "condition": condition,
                        "round": round_id,
                        "agent": observation["agent"],
                        "image": image,
                        "visible_pixels": target["visible_pixels"],
                        "projected_area_px2": target["projected_area_px2"],
                        "occlusion_ratio": target["occlusion_ratio"],
                    }
                )
            position = observation.get("camera_world_position")
            if not isinstance(position, list) or len(position) != 3 or not all(math.isfinite(x) for x in position):
                raise ValueError("Invalid measured camera position")
            if round_id == 0:
                initial_positions[observation["agent"]] = position
            else:
                distance = math.dist(initial_positions[observation["agent"]], position)
                new_intermediaries = (visible - initial_visible) & (endpoints - {"target"})
                if distance >= MIN_PROGRESSIVE_MOTION_M:
                    for intermediary in sorted(new_intermediaries):
                        progressive_witnesses.append(
                            {
                                "round": round_id,
                                "agent": observation["agent"],
                                "intermediary": intermediary,
                                "measured_displacement_m": distance,
                                "image": image,
                            }
                        )
        team_visible = set().union(*visible_by_agent.values())
        if round_id == 0:
            initial_visible = set(team_visible)
        cover = _minimum_cover(visible_by_agent, endpoints)
        round_rows.append(
            {
                "round": round_id,
                "visible_by_agent": {key: sorted(value) for key, value in visible_by_agent.items()},
                "team_visible": sorted(team_visible),
                "missing_task_endpoints": sorted(endpoints - team_visible),
                "minimum_endpoint_cover_agents": cover,
            }
        )

    requested = scene.get("visual_condition_requested")
    matching_views = [row for row in target_condition_views if row["condition"] == requested]
    team_complete = [row for row in round_rows if not row["missing_task_endpoints"]]
    single_view_complete = [
        row for row in team_complete if len(row["minimum_endpoint_cover_agents"] or []) == 1
    ]
    cooperative_complete = [
        row for row in team_complete if len(row["minimum_endpoint_cover_agents"] or []) >= 2
    ]
    history_visible = {}
    for row in round_rows:
        for agent, visible in row["visible_by_agent"].items():
            history_visible.setdefault(agent, set()).update(visible)
    history_cover = _minimum_cover(history_visible, endpoints)
    family = scene.get("task_family")
    blockers = []
    if not renderer_control_passed:
        blockers.append("renderer_control_not_passed")
    if not matching_views:
        blockers.append("requested_visual_condition_not_observed")

    family_screen = False
    if family == "direct_spatial":
        family_screen = bool(single_view_complete)
        if not family_screen:
            blockers.append("direct_task_endpoints_not_jointly_visible_in_one_view")
    elif family == "compositional_reference":
        family_screen = bool(team_complete)
        if not family_screen:
            blockers.append("compositional_endpoints_not_visible_in_one_team_round")
    elif family == "progressive_discovery":
        observational_screen = bool(
            team_complete
            and team_complete[0]["round"] > 0
            and progressive_witnesses
        )
        if not observational_screen:
            blockers.append("progressive_physical_witness_screen_failed")
        blockers.append("implicit_relation_generation_not_implemented")
        family_screen = False
    elif family == "cross_view_disambiguation":
        if not cooperative_complete:
            blockers.append("cross_view_union_not_required_or_not_complete")
        history_requires_team = history_cover is not None and len(history_cover) >= 2
        if not history_requires_team:
            blockers.append("cross_view_not_necessary_across_complete_history")
        family_screen = bool(cooperative_complete) and history_requires_team
    else:
        raise ValueError("Unknown task family")

    return {
        "audit_schema": "semantic_full_history_v3_exact_frame_containment",
        "episode_id": record["episode_id"],
        "task_family": family,
        "requested_visual_condition": requested,
        "requested_visual_condition_observed": bool(matching_views),
        "matching_target_views": matching_views,
        "task_endpoints": sorted(endpoints),
        "rounds": round_rows,
        "single_view_complete_rounds": [row["round"] for row in single_view_complete],
        "cooperative_complete_rounds": [row["round"] for row in cooperative_complete],
        "full_history_minimum_agent_cover": history_cover,
        "history_visible_by_agent": {agent: sorted(visible) for agent, visible in history_visible.items()},
        "progressive_physical_witnesses": progressive_witnesses,
        "task_family_observability_screen_passed": family_screen,
        "semantic_gate_passed": not blockers,
        "blockers": blockers,
        "runtime_validated_episodes": 0,
        "boundary": "Truth-assisted semantic dataset audit only. No model prediction or communication result is validated.",
    }


def audit_files(observation_path, scene_path, renderer_control_path):
    observation_path = Path(observation_path)
    scene_path = Path(scene_path)
    record = _load(observation_path)
    scene_bytes = scene_path.read_bytes()
    if hashlib.sha256(scene_bytes).hexdigest() != record.get("source_sha256"):
        raise ValueError("Construction changed since collection")
    renderer = validate_renderer_control(_load(renderer_control_path))
    result = audit_episode(record, json.loads(scene_bytes.decode("utf-8-sig")), renderer["passed"])
    result["renderer_control"] = renderer
    result["observation_file_sha256"] = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    result["scene_sha256"] = record["source_sha256"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--renderer-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit = audit_files(arguments.observations, arguments.scene, arguments.renderer_control)
    with arguments.output.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps({key: value for key, value in audit.items() if key not in {"rounds", "matching_target_views"}}))
