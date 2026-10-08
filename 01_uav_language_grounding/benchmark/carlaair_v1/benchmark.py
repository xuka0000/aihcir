"""Deterministic full-scale task specification and evaluator.

Specifications are NOT validated simulator episodes. Actor truth is private.
No TRACE implementation, result claim or oracle input to agent policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_profile(path=None):
    return json.loads(Path(path or Path(__file__).with_name("profile.json")).read_text(encoding="utf-8"))


def specs(profile):
    seen_maps = set()
    cells = list(itertools.product(profile["task_families"], profile["visual_conditions"], profile["team_sizes"]))
    for split, config in profile["splits"].items():
        if config["count"] % len(cells):
            raise ValueError("Split size must be divisible by stratification cells")
        if seen_maps.intersection(config["maps"]):
            raise ValueError("Map leakage across splits")
        seen_maps.update(config["maps"])
        for index in range(config["count"]):
            family, visual, team = cells[index % len(cells)]
            replicate = index // len(cells)
            map_name = config["maps"][replicate % len(config["maps"])]
            key = f"{profile['seed']}/{split}/{index}"
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
            yield {
                "schema_version": profile["schema_version"],
                "episode_id": f"{split}_{index:06d}",
                "split": split, "map": map_name, "scene_seed": seed,
                "task_family": family, "visual_condition_requested": visual,
                "team_size": team, "status": "pending_scene_instantiation",
                "agent_names": [f"UAV_{i}" for i in range(team)],
                "budgets": dict(profile["budgets"]),
                "recipe_key": hashlib.sha256(key.encode()).hexdigest(),
                "profile_sha256": hashlib.sha256(canonical(profile).encode()).hexdigest(),
            }


def generate(root, profile):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in specs(profile):
        grouped.setdefault(row["split"], []).append(row)
    summaries = {}
    for split, rows in grouped.items():
        destination = root / f"{split}.specs.jsonl"
        serialized = "".join(canonical(row) + "\n" for row in rows)
        encoded = serialized.encode("utf-8")
        if destination.exists() and destination.read_bytes() != encoded:
            raise FileExistsError("Refusing to replace a different frozen specification")
        destination.write_bytes(encoded)
        summaries[split] = {
            "specifications": len(rows),
            "runtime_validated_episodes": 0,
            "strata": dict(Counter(
                f"{r['task_family']}/{r['visual_condition_requested']}/{r['team_size']}"
                for r in rows)),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    report = {"stage": "specifications_generated", "dataset_ready": False,
              "total_specifications": sum(x["specifications"] for x in summaries.values()),
              "runtime_validated_episodes": 0, "splits": summaries}
    (root / "specification_summary.json").write_bytes(
        (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return report


def public_episode(episode):
    """Whitelisting prevents accidental truth leakage from a future added key."""
    public_keys = ("episode_id", "instruction", "agent_names", "search_region", "budgets")
    return {key: episode[key] for key in public_keys if key in episode}


def finite_vector(values, length):
    if not isinstance(values, (tuple, list)) or len(values) != length:
        raise ValueError("Invalid coordinate vector")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite coordinate")
    return result


def relation_holds(predicate, positions, near=20.0, margin=5.0):
    """Input positions use a declared ENU evaluation frame in metres."""
    a = finite_vector(positions[predicate["subject"]], 3)
    b = finite_vector(positions[predicate["object"]], 3)
    dx, dy = a[0] - b[0], a[1] - b[1]
    relation = predicate["relation"]
    if relation == "east":
        return dx >= margin
    if relation == "west":
        return dx <= -margin
    if relation == "north":
        return dy >= margin
    if relation == "south":
        return dy <= -margin
    if relation == "near":
        return math.hypot(dx, dy) <= near
    raise ValueError(f"Unknown relation {relation}")


class EpisodeLedger:
    """Trusted runner records actions/messages; policies cannot self-report cost."""
    def __init__(self, agent_names, budgets):
        self.agent_names = tuple(agent_names)
        self.budgets = dict(budgets)
        self.messages = []
        self.observations = {}
        self.path = {a: 0.0 for a in self.agent_names}
        self.positions = {}
        self.rounds = 0
        self.elapsed = 0.0
        self.bytes = 0
        self.collisions = 0

    def observe(self, agent, source_id, metadata):
        if agent not in self.agent_names:
            raise ValueError("Unknown agent")
        self.observations.setdefault((agent, source_id), dict(metadata))

    def move(self, agent, position):
        if agent not in self.agent_names:
            raise ValueError("Unknown agent")
        position = finite_vector(position, 3)
        if agent in self.positions:
            self.path[agent] += math.dist(self.positions[agent], position)
        self.positions[agent] = position

    def communicate(self, sender, recipient, payload):
        if sender == recipient or sender not in self.agent_names or recipient not in self.agent_names:
            raise ValueError("Invalid message route")
        envelope = {"sender": sender, "recipient": recipient, "payload": payload}
        n = len(canonical(envelope).encode("utf-8"))
        if self.bytes + n > self.budgets["communication_bytes"]:
            return False
        self.bytes += n
        self.messages.append({"sender": sender, "recipient": recipient, "bytes": n})
        return True

    def within_budget(self):
        return (
            self.rounds <= self.budgets["decision_rounds"]
            and self.elapsed <= self.budgets["simulated_seconds"]
            and self.bytes <= self.budgets["communication_bytes"]
            and all(p <= self.budgets["per_agent_path_m"] for p in self.path.values())
        )


def evaluate(episode, prediction, ledger, profile):
    if episode.get("status") != "runtime_validated":
        raise ValueError("Unvalidated specifications cannot yield benchmark scores")
    truth = episode["private_truth"]
    radius = profile["evaluation"]["target_localization_radius_m"]
    target = finite_vector(truth["target_position"], 3)
    try:
        estimate = finite_vector(prediction.get("target_position"), 3)
        localization_error = math.dist(target, estimate)
    except (ValueError, TypeError):
        localization_error = None
    localized = localization_error is not None and localization_error <= radius
    # A returned coordinate must identify the correct candidate, not merely fall
    # in the radius of two nearby objects.
    if localized:
        distances = {key: math.dist(estimate, finite_vector(pos, 3))
                     for key, pos in truth["candidate_positions"].items()}
        ranked = sorted(distances.items(), key=lambda p: (p[1], p[0]))
        localized = (
            bool(ranked) and ranked[0][0] == truth["target_id"]
            and (len(ranked) < 2 or ranked[1][1] - ranked[0][1] > 1e-6)
        )
    required = truth["required_relations"]
    claims = prediction.get("relations", [])
    if not isinstance(claims, list):
        claims = []
    correct = 0
    seen = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        try:
            subject = finite_vector(claim["subject_position"], 3)
            reference = finite_vector(claim["object_position"], 3)
        except (KeyError, ValueError, TypeError):
            continue
        # Match semantic endpoints, never require evaluator-only relation IDs.
        # Each claim and each truth edge can contribute at most once.
        for index, gt in enumerate(required):
            if (index not in seen and claim.get("relation") == gt["relation"]
                    and math.dist(subject, truth["positions"][gt["subject"]]) <= radius
                    and math.dist(reference, truth["positions"][gt["object"]]) <= radius):
                correct += 1
                seen.add(index)
                break
    budget_ok = ledger.within_budget()
    witnessed = any(truth["target_id"] in obs.get("visible_entities", [])
                    for obs in ledger.observations.values())
    # Oracle-coordinate guessing without any visual observation is not success.
    # `relations` is the final set of mission claims, not the hypothesis graph.
    # Extra, invalid or repeated claims must not make enumerating every possible
    # answer a successful prediction. Auxiliary hypotheses belong in a separate
    # trace, which is evaluated independently from this final answer.
    claim_count = len(claims)
    precision = correct / claim_count if claim_count else (1.0 if not required else 0.0)
    recall = correct / len(required) if required else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_relations = correct == len(required) and claim_count == len(required)
    success = bool(localized and exact_relations and budget_ok and witnessed)
    return {
        "episode_id": episode["episode_id"], "success": success,
        "target_observed": witnessed, "collision_free_success": success and ledger.collisions == 0,
        "target_success": bool(localized), "relation_recall": recall,
        "relation_precision": precision, "relation_f1": f1,
        "relation_exact_match": exact_relations, "relation_claim_count": claim_count,
        "unmatched_relation_claims": claim_count - correct,
        "localization_error_m": localization_error, "budget_valid": budget_ok,
        "communication_bytes": ledger.bytes, "team_path_m": sum(ledger.path.values()),
        "decision_rounds": ledger.rounds, "simulated_seconds": ledger.elapsed,
        "collisions": ledger.collisions, "observations": len(ledger.observations),
    }


def summarize(scores, expected_ids):
    scores = list(scores)
    expected = set(expected_ids)
    observed = [s["episode_id"] for s in scores]
    if len(set(observed)) != len(observed) or not set(observed).issubset(expected):
        raise ValueError("Duplicate or unexpected episode scores")
    return {
        "expected": len(expected), "completed": len(scores),
        "missing": sorted(expected - set(observed)),
        "success_rate_completed_only": sum(s["success"] for s in scores) / len(scores) if scores else None,
        "success_rate_full_suite": sum(s["success"] for s in scores) / len(expected) if len(scores) == len(expected) and expected else None,
        "publication_complete": len(scores) == len(expected) and bool(expected),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, load_profile(args.profile)), indent=2))
