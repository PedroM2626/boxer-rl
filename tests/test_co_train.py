"""Legs and striker in alternating rounds (boxer.co_train)."""
import json
from pathlib import Path

import pytest

from boxer import co_train


def fake_phases(monkeypatch, selected, fail_at=None):
    """Replaces the phase processes; the striker gate selects from ``selected`` by round."""
    calls = []

    def run(command, log):
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError("phase crashed")
        calls.append(command)
        output = command[command.index("--output") + 1]
        if command[0] == "boxer.native_finetune":
            init = command[command.index("--init-from") + 1]
            pick = selected[len([c for c in calls if c[0] == "boxer.native_finetune"]) - 1] or init
            (Path(output) / "gate").mkdir(parents=True, exist_ok=True)
            (Path(output) / "gate" / "gate.json").write_text(json.dumps(dict(selected=pick, long_checks=[])))
    monkeypatch.setattr(co_train, "run", run)
    return calls



def test_rounds_alternate_and_carry_legs_always_and_strikers_when_selected(tmp_path, monkeypatch):
    calls = fake_phases(monkeypatch, ["s1", None])
    co_train.main(["--output", str(tmp_path), "--rounds", "2", "--legs", "L0", "--striker", "S0"])
    assert [c[0] for c in calls] == ["boxer.motor_native", "boxer.native_finetune"] * 2
    first_legs, first_striker, second_legs, second_striker = calls
    assert first_legs[first_legs.index("--striker") + 1] == "S0"
    assert first_striker[first_striker.index("--motor") + 1] == str(tmp_path / "round-0" / "legs" / "checkpoint")
    assert "S0" in first_striker[first_striker.index("--pool") + 1:first_striker.index("--scripted-fraction")]
    assert first_striker[first_striker.index("--long-gate-opponents") + 1] == "checkpoints/tactics"
    assert second_legs[second_legs.index("--init-from") + 1] == str(tmp_path / "round-0" / "legs" / "checkpoint")
    assert second_legs[second_legs.index("--striker") + 1] == "s1"           # the gate replaced S0
    assert second_striker[second_striker.index("--init-from") + 1] == "s1"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["striker"] == "s1" and [r["striker_replaced"] for r in state["rounds"]] == [True, False]


def test_a_crashed_round_resumes_without_rerunning_finished_phases(tmp_path, monkeypatch):
    fake_phases(monkeypatch, ["s1"], fail_at=1)
    with pytest.raises(RuntimeError):
        co_train.main(["--output", str(tmp_path), "--rounds", "1", "--legs", "L0", "--striker", "S0"])
    calls = fake_phases(monkeypatch, ["s1"])
    co_train.main(["--output", str(tmp_path), "--rounds", "1", "--legs", "L0", "--striker", "S0"])
    assert [c[0] for c in calls] == ["boxer.native_finetune"]
