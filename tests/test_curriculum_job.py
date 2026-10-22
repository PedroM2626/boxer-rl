"""A failed physical gate must never promote a curriculum stage."""
import json
import sys
from pathlib import Path

import pytest

from boxer import curriculum_job as job


@pytest.mark.skipif(sys.platform == "win32", reason="Linux supervisor uses flock")
@pytest.mark.parametrize("pass_first", [False, True])
def test_only_successful_gate_promotes(tmp_path, monkeypatch, pass_first):
    commands = []
    def child(command, log):
        commands.append(command)
        def flag(name):
            return command[command.index(name) + 1]
        if "boxer.curriculum_train" in command:
            destination = Path(flag("--output")) / "checkpoint"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "latest.json").write_text(json.dumps(dict(stage_steps=100)))
        else:
            destination = Path(flag("--output"))
            passed = pass_first and destination.parent.name == "balance"
            destination.write_text(json.dumps(dict(passed=passed)))
    monkeypatch.setattr(job, "run_child", child)
    output = tmp_path / "run"
    assert job.main(["--output", str(output), "--steps-per-stage", "100"]) == 0
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "gate_not_passed"
    assert state["completed"] == (["balance"] if pass_first else [])
    assert len(commands) == (4 if pass_first else 2)
    if pass_first:
        assert "--init-from" in commands[2]
    assert not (output / "approach").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Linux supervisor uses flock")
def test_second_output_cannot_publish_over_a_running_job(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(job, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    with (tmp_path / "runs/curriculum_publish.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            job.main(["--output", str(tmp_path / "other"), "--publish"])
    assert not (tmp_path / "other/state.json").exists()
