"""Promotion requires repeated, complete physical evaluations of new weights."""
import copy
import math

from boxer.curriculum_control import commands_for, promotion_ready, verified_gate


def gate(stage="balance", step=0, digest="a", passed=True):
    trials = []
    for seed in (7, 23):
        for command in commands_for(stage):
            for disturbed in ((False, True) if stage == "balance" else (False,)):
                trials.append(dict(seed=seed, command=list(command), disturbed=disturbed,
                    survival_rate=1.0 if passed else 0.5, linear_tracking_error=.02,
                    yaw_tracking_error=.02, directed_distance=3.0,
                    required_distance=math.hypot(*command[:2]) * 4.5,
                    arrival_rate=1.0))
    return dict(stage=stage, stage_steps=step, checkpoint_sha256=digest * 64,
                kind="promotion_gate", passed=passed, assistance=False,
                worlds_per_trial=32, episode_seconds=10, trials=trials)


def test_two_distinct_complete_passes_required_and_failure_resets_streak():
    first, second = gate(step=1), gate(step=2, digest="b")
    assert not promotion_ready([first], "balance")
    assert promotion_ready([first, second], "balance")
    assert not promotion_ready([first, first], "balance")
    assert not promotion_ready([first, gate(step=2)], "balance")  # Same weights.
    assert not promotion_ready([first, gate(step=2, passed=False), second], "balance")


def test_preview_small_sample_missing_direction_and_stale_stage_are_rejected():
    valid = gate("walk_easy")
    assert verified_gate(valid, "walk_easy")
    for key, value in (("kind", "development_preview"), ("worlds_per_trial", 8),
                       ("assistance", True), ("stage", "walk")):
        bad = dict(valid, **{key: value})
        assert not verified_gate(bad, "walk_easy")
    missing = copy.deepcopy(valid)
    missing["trials"].pop()
    assert not verified_gate(missing, "walk_easy")


def test_easy_commands_expand_to_full_commands():
    for easy, full in zip(commands_for("walk_easy"), commands_for("walk")):
        assert easy == tuple(v * .25 for v in full)
