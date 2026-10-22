"""A failed physical gate must never promote a curriculum stage."""
import json
import sys
from pathlib import Path

import pytest

from boxer import curriculum_job as job
from test_curriculum_control import gate


@pytest.mark.skipif(sys.platform == "win32", reason="Linux supervisor uses flock")
@pytest.mark.parametrize("pass_first", [False, True])
@pytest.mark.parametrize("sigma", [None, ".01"])
def test_only_successful_gate_promotes(tmp_path, monkeypatch, pass_first, sigma):
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
    flags = ["--output", str(output), "--max-total-steps", "100000", "--worlds", "1",
             "--walk-tracking-sigma", "0.04"]
    if sigma is not None:
        flags += ["--walk-easy-tracking-sigma", sigma]
    assert job.main(flags) == 0
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "budget_reached"
    assert state["completed"] == (["balance"] if pass_first else [])
    assert len(commands) == (2 if pass_first else 1)
    assert "--tracking-sigma" not in commands[0]  # Balance remains unchanged.
    if pass_first:
        assert "--init-from" in commands[1]
        assert commands[1][commands[1].index("--steps") + 1] == "79520"
        assert ("--tracking-sigma" in commands[1]) == (sigma is not None)
    assert state["total_steps"] == (40960 if pass_first else 20480)
    assert not (output / "approach").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Linux supervisor uses flock")
@pytest.mark.parametrize("active_stage", ["walk_easy", "walk"])
@pytest.mark.parametrize("override", [False, True])
def test_walking_resume_inherits_own_reward_profile(tmp_path, monkeypatch, active_stage, override):
    output = tmp_path / "run"
    checkpoint = output / active_stage / "checkpoint"
    checkpoint.mkdir(parents=True)
    (output / "state.json").write_text("{}")
    (checkpoint / "latest.json").write_text(json.dumps(dict(stage_steps=1000,
                                                          training_tracking_sigma=.01 if active_stage == "walk_easy" else .04,
                                                          lateral_practice_probability=.5)))
    previous = job.STAGES[:job.STAGES.index(active_stage)]
    monkeypatch.setattr(job, "certified", lambda path, stage: stage in previous)
    class Captured(Exception):
        pass
    def child(command, log):
        if active_stage == "walk_easy":
            assert command[command.index("--tracking-sigma") + 1] == "0.01"
            assert command[command.index("--lateral-practice-probability") + 1] == "0.5"
        else:
            assert command[command.index("--tracking-sigma") + 1] == ("0.06" if override else "0.04")
            assert "--lateral-practice-probability" not in command
        assert "--warm-start" in command and "--restart-from" not in command
        raise Captured
    monkeypatch.setattr(job, "run_child", child)
    with pytest.raises(Captured):
        job.main(["--output", str(output), "--warm-start", "--worlds", "1",
                  "--walk-easy-tracking-sigma", "0.01"]
                 + (["--walk-tracking-sigma", "0.06"] if override else []))


@pytest.mark.skipif(sys.platform == "win32", reason="Linux supervisor uses flock")
@pytest.mark.parametrize("sigma", ["0", "-0.1", "nan", "inf"])
def test_full_walking_profile_rejects_invalid_width(sigma):
    with pytest.raises(ValueError, match="full-walking tracking sigma"):
        job.main(["--walk-tracking-sigma", sigma])


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
