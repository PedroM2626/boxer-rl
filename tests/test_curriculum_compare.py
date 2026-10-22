"""Comparison arms share an immutable policy and cannot replace the live learner."""
import hashlib
import json

import pytest

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer import curriculum_compare as comparison
from test_curriculum_control import gate


def prepare(tmp_path):
    run = tmp_path / "main"
    atomic_json(run / "state.json", dict(stage="walk_easy", completed=["balance"]))
    checkpoint = run / "walk_easy/checkpoint"
    payload = b"preserved-main-policy"
    atomic_bytes(checkpoint / "policy.pkl", payload)
    atomic_json(checkpoint / "latest.json", dict(file="policy.pkl", stage_steps=112000000,
                total_steps=315000000, sha256=hashlib.sha256(payload).hexdigest()))
    return run


def test_paired_arms_keep_original_policy_and_separate_compute(tmp_path):
    run, output = prepare(tmp_path), tmp_path / "comparison"
    commands = []
    def execute(command, log):
        commands.append(command)
        def flag(name):
            return command[command.index(name) + 1]
        if "boxer.curriculum_train" in command:
            arm = output / ("default" if flag("--tracking-sigma") == "0.25" else "sharp")
            payload = flag("--tracking-sigma").encode()
            atomic_bytes(arm / "checkpoint/policy.pkl", payload)
            atomic_json(arm / "checkpoint/latest.json", dict(file="policy.pkl", stage_steps=9994240,
                        sha256=hashlib.sha256(payload).hexdigest()))
        else:
            checkpoint = command[command.index("boxer.curriculum_eval") + 1]
            manifest = comparison.read(comparison.Path(checkpoint) / "latest.json")
            result = gate("walk_easy", step=manifest["stage_steps"], passed=False)
            result["checkpoint_sha256"] = manifest["sha256"]
            atomic_json(comparison.Path(flag("--output")), result)
    result = comparison.compare(run, output, execute=execute)
    assert result["status"] == "completed"
    assert result["diagnostic_steps"] == 19988480
    assert result["main_total_steps"] == 315000000
    assert not result["automatic_adoption"]
    assert len(commands) == 5  # One shared baseline, two trains, two full gates.
    trains = [c for c in commands if "boxer.curriculum_train" in c]
    assert all(c[c.index("--init-from") + 1] == str(output / "seed") for c in trains)
    assert all(c[c.index("--seed") + 1] == "42" for c in trains)
    assert (run / "walk_easy/checkpoint/policy.pkl").read_bytes() == b"preserved-main-policy"
    assert (output / "seed/policy.pkl").read_bytes() == b"preserved-main-policy"
    with pytest.raises(FileExistsError):
        comparison.compare(run, output, execute=execute)


def test_comparison_rejects_preview_and_preserves_failure_report(tmp_path):
    run, output = prepare(tmp_path), tmp_path / "comparison"
    def execute(command, log):
        manifest = comparison.read(output / "seed/latest.json")
        result = gate("walk_easy", step=manifest["stage_steps"], passed=False)
        result.update(kind="development_preview", checkpoint_sha256=manifest["sha256"])
        atomic_json(comparison.Path(command[command.index("--output") + 1]), result)
    with pytest.raises(ValueError, match="complete physical gate"):
        comparison.compare(run, output, execute=execute)
    result = comparison.read(output / "state.json")
    assert result["status"] == "failed" and result["diagnostic_steps"] == 0
    assert (run / "walk_easy/checkpoint/policy.pkl").read_bytes() == b"preserved-main-policy"


def test_failed_training_retains_partial_diagnostic_accounting(tmp_path):
    run, output = prepare(tmp_path), tmp_path / "comparison"
    def execute(command, log):
        if "boxer.curriculum_train" in command:
            atomic_json(output / "default/checkpoint/latest.json", dict(stage_steps=123456))
            raise RuntimeError("simulated child failure")
        manifest = comparison.read(output / "seed/latest.json")
        result = gate("walk_easy", step=manifest["stage_steps"], passed=False)
        result["checkpoint_sha256"] = manifest["sha256"]
        atomic_json(output / "baseline.json", result)
    with pytest.raises(RuntimeError, match="child failure"):
        comparison.compare(run, output, execute=execute)
    assert comparison.read(output / "state.json")["diagnostic_steps"] == 123456


def test_active_main_lock_prevents_comparison_or_resume(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    run = prepare(tmp_path)
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    def forbidden(*args):
        pytest.fail("an active main job must never be overlapped or resumed")
    monkeypatch.setattr(comparison.os, "execv", forbidden)
    before = (run / "state.json").read_bytes()
    with (run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            comparison.main(["--main-run", str(run), "--output", str(tmp_path / "comparison"),
                             "--resume-main"])
    assert (run / "state.json").read_bytes() == before
