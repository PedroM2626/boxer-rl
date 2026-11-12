"""Pure promotion rules shared by the learner, evaluator and supervisor."""
from __future__ import annotations

import math

STAGES = ("balance", "walk_easy", "walk", "approach")
# Sparring footwork while a punch generator drives both arms, and getting up
# from the floor. Both follow the locomotion job rather than joining its
# automatic promotion sequence.
TRAINABLE_STAGES = STAGES + ("spar", "getup")
COMMANDS = [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (-0.25, 0.0, 0.0),
            (0.0, 0.2, 0.0), (0.0, -0.2, 0.0),
            (0.0, 0.0, 0.4), (0.0, 0.0, -0.4)]


def commands_for(stage):
    if stage in ("walk_easy", "walk", "spar"):
        scale = {"walk_easy": 0.25, "walk": 1.0, "spar": 0.5}[stage]
        return [tuple(value * scale for value in command) for command in COMMANDS]
    return [COMMANDS[0]]


def passes_gate(stage, trials):
    if stage not in TRAINABLE_STAGES or not trials:
        return False
    for trial in trials:
        if not all(math.isfinite(trial[key]) for key in (
                "survival_rate", "linear_tracking_error", "yaw_tracking_error",
                "directed_distance", "required_distance", "arrival_rate")):
            return False
        if trial["survival_rate"] < (0.95 if stage == "balance" else 0.90):
            return False
        if trial["linear_tracking_error"] > (0.15 if stage == "balance" else 0.20):
            return False
        if trial["yaw_tracking_error"] > 0.25:
            return False
        if stage in ("walk_easy", "walk") and trial["required_distance"] > trial["directed_distance"]:
            return False
        if stage == "approach" and trial["arrival_rate"] < 0.90:
            return False
    return True


def verified_gate(result, stage):
    """A preview, incomplete direction set or stale stage cannot certify a skill."""
    if (result.get("kind") != "promotion_gate" or result.get("stage") != stage
            or result.get("passed") is not True or result.get("assistance") is not False
            or result.get("worlds_per_trial", 0) < 32
            or result.get("episode_seconds") != 10):
        return False
    disturbances = (False, True) if stage in ("balance", "spar") else (False,)
    expected = {(seed, command, disturbed) for seed in (7, 23)
                for command in commands_for(stage) for disturbed in disturbances}
    trials = result.get("trials", [])
    actual = {(t.get("seed"), tuple(t.get("command", [])), t.get("disturbed")) for t in trials}
    if any(not math.isclose(t.get("required_distance", -1),
                            math.hypot(*t.get("command", [0, 0, 0])[:2]) * 4.5,
                            abs_tol=1e-6) for t in trials):
        return False
    return (len(trials) == len(expected) and actual == expected
            and passes_gate(stage, trials)
            and isinstance(result.get("stage_steps"), int)
            and len(result.get("checkpoint_sha256", "")) == 64)


def consecutive_passes(history, stage):
    count, next_step, next_hash = 0, float("inf"), None
    for result in reversed(history):
        if (not verified_gate(result, stage) or result["stage_steps"] >= next_step
                or result["checkpoint_sha256"] == next_hash):
            break
        count += 1
        next_step, next_hash = result["stage_steps"], result["checkpoint_sha256"]
    return count


def promotion_ready(history, stage, required=2):
    return consecutive_passes(history, stage) >= max(2, required)
