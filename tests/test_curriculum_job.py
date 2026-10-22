"""A failed physical gate must never promote a curriculum stage."""
import json
import sys
from pathlib import Path

import pytest

from boxer import curriculum_job as job
from test_curriculum_control import gate


@pytest.mark.skipif(sys.platform == "win32", reason="Linux supervisor uses flock")
@pytest.mark.parametrize("pass_first", [False, True])
def test_only_successful_gate_promotes(tmp_path, monkeypatch, pass_first):
    commands = []
    def child(command, log):
        commands.append(command)
        def flag(name):
            return command[command.index(name) + 1]
        destination = Path(flag("--output"))
        checkpoint = destination / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "latest.json").write_text(json.dumps(dict(stage_steps=20480, sha256="b" * 64)))
        stage = flag("--stage")
        passed = pass_first and stage == "balance"
        history = [gate(stage, 0), gate(stage, 20480, "b", passed)]
        (destination / "gate_history.json").write_text(json.dumps(history))
    monkeypatch.setattr(job, "run_child", child)
    output = tmp_path / "run"
    assert job.main(["--output", str(output), "--max-total-steps", "100000", "--worlds", "1"]) == 0
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "budget_reached"
    assert state["completed"] == (["balance"] if pass_first else [])
    assert len(commands) == (2 if pass_first else 1)
    if pass_first:
        assert "--init-from" in commands[1]
        assert commands[1][commands[1].index("--steps") + 1] == "79520"
    assert state["total_steps"] == (40960 if pass_first else 20480)
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
