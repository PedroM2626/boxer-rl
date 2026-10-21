"""Exercise the durable supervisor without pretending to run a long GPU experiment."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxer import long_train as job

linux_job = pytest.mark.skipif(sys.platform == "win32", reason="supervisor runs in Linux/WSL2")


def seed(root):
    path = root / "checkpoints/seed"
    (path / "checkpoint").mkdir(parents=True)
    (path / "checkpoint/latest.txt").write_text("100")
    (path / "checkpoint/step-100.pkl").write_bytes(b"synthetic test fixture")
    config = dict(environment="skills", mode="selfplay", worlds=2, n_frames=5,
                  spawn_distance=.62, action_scale=1, round_steps=4,
                  damage_cap=6, stun_steps=10, policy_layers=[8], value_layers=[8],
                  seed=11, ppo=dict(batch_size=2, num_minibatches=2, unroll_length=2))
    (path / "warp_config.json").write_text(json.dumps(config))
    return config


def arguments():
    return ["--seed-run", "checkpoints/seed", "--output", "runs/long",
            "--milestones", "200", "400", "--eval-worlds", "2", "--eval-seeds", "7"]


def fake_children(command, log_path, env):
    def flag(name):
        return command[command.index(name) + 1]
    if "boxer.warp_train" in command:
        output, total = Path(flag("--output")), int(flag("--total-steps"))
        (output / "checkpoint" / f"step-{total}.pkl").write_bytes(f"test-{total}".encode())
        (output / "checkpoint/latest.txt").write_text(str(total))
    elif "boxer.warp_arena" in command:
        path = Path(flag("--output"))
        path.parent.mkdir(parents=True, exist_ok=True)
        result = dict(dropped_rounds=0, a_head=.1, b_head=.2, a_torso=2.,
                      b_torso=3., a_win_with_points=.5)
        path.write_text(json.dumps(dict(results=result)))
    else:
        path = Path(flag("--output"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic renderer output")


@linux_job
def test_supervisor_pins_milestones_and_resumes_without_retraining(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "ROOT", tmp_path)
    seed(tmp_path)
    commands = []
    def child(*args):
        commands.append(args[0])
        fake_children(*args)
    monkeypatch.setattr(job, "run_command", child)
    assert job.main(arguments()) == 0
    state = json.loads((tmp_path / "runs/long/job_state.json").read_text())
    assert state["status"] == "complete"
    assert [stage["actual_steps"] for stage in state["completed"]] == [200, 400]
    assert len(commands) == 8
    assert all("step-" in stage["checkpoint"] for stage in state["completed"])
    assert (tmp_path / "checkpoints/skills_200m/step-200/artifact.json").exists()
    assert job.main(arguments()) == 0
    assert len(commands) == 8


@linux_job
def test_child_failure_keeps_last_valid_weights_and_records_the_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "ROOT", tmp_path)
    seed(tmp_path)
    def fail(*_args):
        raise RuntimeError("simulated GPU failure")
    monkeypatch.setattr(job, "run_command", fail)
    with pytest.raises(RuntimeError, match="GPU failure"):
        job.main(arguments())
    state = json.loads((tmp_path / "runs/long/job_state.json").read_text())
    assert state["status"] == "failed" and "GPU failure" in state["error"]
    assert job.checkpoint_step(tmp_path / "runs/long") == 100


def test_plan_only_has_no_training_or_output_side_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "ROOT", tmp_path)
    seed(tmp_path)
    assert job.main([*arguments(), "--plan-only"]) == 0
    assert not (tmp_path / "runs").exists()


def test_commands_restore_recorded_seed_network_and_ppo_settings(tmp_path):
    config = seed(tmp_path)
    command = job.training_command(config, "runs/long", 400, 50)
    assert command[command.index("--seed") + 1] == "11"
    assert command[command.index("--batch-size") + 1] == "2"
    assert command[command.index("--policy-layers") + 1] == "8"
    assert "--resume" in command


def test_publisher_refuses_unrelated_staged_files(monkeypatch):
    calls = []
    def git(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="master" if "branch" in command else "README.md\n")
    monkeypatch.setattr(job.subprocess, "run", git)
    args = SimpleNamespace(git_executable="git", git_worktree="project", branch="master")
    with pytest.raises(RuntimeError, match="unrelated staged changes"):
        job.publish(args, "test")
    assert not any("add" in command for command in calls)


@linux_job
def test_failed_publication_does_not_cancel_the_training_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "ROOT", tmp_path)
    seed(tmp_path)
    monkeypatch.setattr(job, "run_command", fake_children)
    def unavailable(*_args):
        raise RuntimeError("remote unavailable")
    monkeypatch.setattr(job, "publish", unavailable)
    assert job.main([*arguments(), "--publish"]) == 1
    state = json.loads((tmp_path / "runs/long/job_state.json").read_text())
    assert state["status"] == "complete" and len(state["completed"]) == 2
    assert "remote unavailable" in state["publish_error"]
