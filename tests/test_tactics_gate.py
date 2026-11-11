"""The native gate picks checkpoints by native record, not by recency."""
from pathlib import Path

import numpy as np

from boxer.curriculum_train import load_policy_params, save_policy
from boxer.tactics_gate import candidates, combine, score, stage


def record(wins, losses, draws, mine, theirs, bouts=16, knockdowns=0):
    return dict(bouts=bouts, knockdowns=knockdowns,
                tactics=dict(first_wins=wins, first_losses=losses, draws=draws,
                             first_clean_per_bout=mine, second_clean_per_bout=theirs))


def test_candidates_are_the_latest_steps_in_numeric_order_and_stage_loads(tmp_path):
    for step in (0, 5_000_000, 10_000_000, 15_000_000, 20_000_000):
        save_policy(tmp_path, (np.full(2, step, float),), dict(stage_steps=step, stage="tactics"))
    picked = candidates(tmp_path, 3)
    assert [p.name for p in picked] == ["step-10000000.pkl.json", "step-15000000.pkl.json",
                                        "step-20000000.pkl.json"]
    params, metadata = load_policy_params(stage(picked[0], tmp_path / "staged"))
    assert metadata["stage_steps"] == 10_000_000 and params[0][0] == 10_000_000


def test_score_prefers_record_then_punch_margin_and_combine_weights_bouts():
    assert score(record(8, 5, 3, 10, 12)) > score(record(7, 5, 4, 20, 5))
    assert score(record(7, 7, 2, 15, 12)) > score(record(7, 7, 2, 12, 12))
    total = combine(record(10, 5, 1, 16, 16, knockdowns=1), record(11, 18, 3, 13, 16, bouts=32, knockdowns=2))
    assert (total["first_wins"], total["first_losses"], total["draws"], total["bouts"]) == (21, 23, 4, 48)
    assert abs(total["first_clean_per_bout"] - 14.0) < 1e-9 and total["falls"] == 3


def test_native_match_plays_on_the_given_motor_with_shared_arms(tmp_path, monkeypatch):
    import json
    import subprocess
    from boxer import tactics_gate
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        (tmp_path / "m.json").write_text(json.dumps(dict(bouts=2, results=[])))
    monkeypatch.setattr(subprocess, "run", run)
    tactics_gate.native_match("a", "b", [0], tmp_path / "m.json", True, "arm", "motor")
    tactics_gate.native_match("a", "b", [0], tmp_path / "m.json")
    with_motor, default = commands
    assert with_motor[with_motor.index("--checkpoint") + 1] == "motor" and "--ring" in with_motor
    assert with_motor[with_motor.index("--blue-arm") + 1] == "arm"
    assert "--checkpoint" not in default and "--red-arm" not in default


def long_record(mine, theirs):
    return dict(bouts=16, knockdowns=0, tactics=dict(first_wins=8, first_losses=8, draws=0, first_clean_per_bout=1.,
                                                     second_clean_per_bout=1., first_knockdowns=mine,
                                                     second_knockdowns=theirs))


def test_stability_allows_two_or_double_the_opponents_knockdowns():
    from boxer.tactics_gate import stable
    assert stable(long_record(2, 0)) and not stable(long_record(3, 0))
    assert stable(long_record(10, 5)) and not stable(long_record(11, 5))


def gate_with(tmp_path, monkeypatch, short, long, tries=2):
    """Runs ``gate_generation`` on three saved steps with scripted match records."""
    from boxer import tactics_gate
    for step in (1, 2, 3):
        save_policy(tmp_path / "checkpoint", (np.full(2, step, float),), dict(stage_steps=step, stage="tactics"))
    calls = []

    def match(first, second, seeds, output, ring=False, arm=None, motor=None, seconds=None, knockdowns=None):
        step = int(Path(first).name.split("-")[1])
        calls.append((step, seconds))
        if seconds:
            return long[step]
        if tuple(seeds) == tactics_gate.CONFIRMATION_SEEDS:
            return record(10, 6, 0, 2., 1., bouts=32)
        return short[step]
    monkeypatch.setattr(tactics_gate, "native_match", match)
    return tactics_gate.gate_generation(tmp_path, "champion", 3, long_tries=tries), calls



def test_gate_skips_a_best_candidate_that_falls_over_long_bouts(tmp_path, monkeypatch):
    import json
    short = {1: record(4, 4, 0, 1., 1.), 2: record(9, 1, 0, 3., 1.), 3: record(7, 3, 0, 2., 1.)}
    long = {2: long_record(9, 0), 3: long_record(1, 1)}
    selected, calls = gate_with(tmp_path, monkeypatch, short, long)
    assert selected.name == "step-3"
    assert [c for c in calls if c[1]] == [(2, 120.), (3, 120.)]
    result = json.loads((tmp_path / "gate" / "gate.json").read_text())
    assert result["stable_candidate"] and [c["stable"] for c in result["long_checks"]] == [False, True]


def test_gate_keeps_the_champion_when_no_candidate_is_stable(tmp_path, monkeypatch):
    import json
    short = {1: record(4, 4, 0, 1., 1.), 2: record(9, 1, 0, 3., 1.), 3: record(7, 3, 0, 2., 1.)}
    long = {2: long_record(9, 0), 3: long_record(5, 0)}
    selected, calls = gate_with(tmp_path, monkeypatch, short, long)
    assert str(selected) == "champion" and 1 not in [c[0] for c in calls if c[1]]
    result = json.loads((tmp_path / "gate" / "gate.json").read_text())
    assert not result["stable_candidate"] and not result["beats_champion"]


def test_a_candidate_must_be_stable_against_every_long_opponent(tmp_path, monkeypatch):
    from boxer import tactics_gate
    for step in (1, 2):
        save_policy(tmp_path / "checkpoint", (np.full(2, step, float),), dict(stage_steps=step, stage="tactics"))
    seen = []

    def match(first, second, seeds, output, ring=False, arm=None, motor=None, seconds=None, knockdowns=None):
        step = int(Path(first).name.split("-")[1])
        if seconds:
            seen.append((step, second))
            # Step 2 is stable against the champion but falls against the tactics.
            return long_record(9 if (step, second) == (2, "tactics") else 0, 0)
        if tuple(seeds) == tactics_gate.CONFIRMATION_SEEDS:
            return record(10, 6, 0, 2., 1., bouts=32)
        return record(9, 1, 0, 3., 1.) if step == 2 else record(5, 5, 0, 1., 1.)
    monkeypatch.setattr(tactics_gate, "native_match", match)
    selected = tactics_gate.gate_generation(tmp_path, "champion", 2, long_opponents=["tactics"])
    assert selected.name == "step-1"
    assert seen == [(2, "champion"), (2, "tactics"), (1, "champion"), (1, "tactics")]


def test_long_check_can_be_skipped(tmp_path, monkeypatch):
    short = {1: record(4, 4, 0, 1., 1.), 2: record(9, 1, 0, 3., 1.), 3: record(7, 3, 0, 2., 1.)}
    selected, calls = gate_with(tmp_path, monkeypatch, short, {}, tries=0)
    assert selected.name == "step-2" and not [c for c in calls if c[1]]
