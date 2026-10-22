"""Persistent curriculum supervisor and scoped milestone publication.

One policy progresses after repeated physical gates, under a shared total
budget ceiling. No training reward threshold or timer advances a stage.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_control import STAGES, promotion_ready

ROOT = Path(__file__).resolve().parents[1]
OWNED = ("checkpoints/curriculum", "docs/results/curriculum", "fights/curriculum")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def publish(run, stage, git_executable="git", git_worktree=None):
    source = Path(run) / stage
    checkpoint = source / "checkpoint"
    metadata = json.loads((checkpoint / "latest.json").read_text())
    payload = (checkpoint / metadata["file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("refusing to publish an inconsistent checkpoint")
    destination = ROOT / "checkpoints/curriculum" / stage
    # Fixed snapshot filenames keep the working tree small. Git retains each
    # historical policy; local step files remain available for recovery.
    metadata["file"] = "policy.pkl"
    atomic_bytes(destination / "policy.pkl", payload)
    atomic_json(destination / "latest.json", metadata)
    atomic_bytes(destination / "config.json", (source / "config.json").read_bytes())
    reports = ROOT / "docs/results/curriculum"
    for name in ("progress.json", "evaluation.json", "preview.json", "gate_history.json", "training_result.json"):
        if (source / name).exists():
            atomic_bytes(reports / f"{stage}-{name}", (source / name).read_bytes())
    if (source / "evaluation.mp4").exists():
        atomic_bytes(ROOT / "fights/curriculum" / f"{stage}.mp4",
                     (source / "evaluation.mp4").read_bytes())
    if (source / "preview.mp4").exists():
        atomic_bytes(ROOT / "fights/curriculum" / f"{stage}-preview.mp4",
                     (source / "preview.mp4").read_bytes())
    state_path = Path(run) / "state.json"
    if state_path.exists():
        atomic_bytes(reports / "state.json", state_path.read_bytes())
    git = [git_executable, "-C", str(git_worktree or ROOT)]
    staged = subprocess.check_output(git + ["diff", "--cached", "--name-only"], text=True)
    for path in staged.splitlines():
        if not any(path == p or path.startswith(p + "/") for p in OWNED):
            raise RuntimeError(f"unrelated staged file blocks publishing: {path}")
    paths = [p for p in OWNED if (ROOT / p).exists()]
    subprocess.run(git + ["add", "--", *paths], check=True)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode:
        subprocess.run(git + ["commit", "-m",
            f"Record unassisted {stage} at {metadata['stage_steps']} transitions"], check=True)
    subprocess.run(git + ["push", "origin", "HEAD"], check=True)


def run_child(command, log):
    with log.open("a") as stream:
        offset = log.stat().st_size
        overlap = b""
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while True:
                try:
                    code = child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    code = None
                with log.open("rb") as reader:
                    reader.seek(offset)
                    chunk = reader.read()
                    offset = reader.tell()
                scan = overlap + chunk
                overlap = scan[-64:]
                if b"overflow -" in scan.lower():
                    raise RuntimeError(f"Physics allocation overflow; child stopped. Inspect {log}")
                if code is not None:
                    break
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    if code:
        raise RuntimeError(f"child exited {code}; inspect {log}")


def saved_steps(run, stage):
    path = Path(run) / stage / "checkpoint/latest.json"
    return int(json.loads(path.read_text())["stage_steps"]) if path.exists() else 0


def total_steps(run):
    return sum(saved_steps(run, stage) for stage in STAGES)


def certified(stage_dir, stage):
    history_path, checkpoint = stage_dir / "gate_history.json", stage_dir / "checkpoint/latest.json"
    if not history_path.exists() or not checkpoint.exists():
        return False
    history = json.loads(history_path.read_text())
    metadata = json.loads(checkpoint.read_text())
    return (promotion_ready(history, stage)
            and history[-1]["stage_steps"] == metadata["stage_steps"]
            and history[-1]["checkpoint_sha256"] == metadata["sha256"])


def main(argv=None):
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/curriculum"))
    parser.add_argument("--max-total-steps", type=int, default=200_000_000)
    parser.add_argument("--gate-every", type=int, default=5_000_000)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--git-executable", default="git")
    parser.add_argument("--git-worktree")
    args = parser.parse_args(argv)
    if args.max_total_steps <= 0 or args.gate_every <= 0:
        raise ValueError("total ceiling and gate interval must be positive")
    if args.force_restart and not (args.warm_start and args.restart_from):
        raise ValueError("forced rollback requires a warm start and a recovery source")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((args.output / ".lock").open("w"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.publish:
            namespace = ROOT / "runs/curriculum_publish.lock"
            namespace.parent.mkdir(parents=True, exist_ok=True)
            publication_lock = stack.enter_context(namespace.open("w"))
            fcntl.flock(publication_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()))
        lock.flush()
        state_file = args.output / "state.json"
        previous = json.loads(state_file.read_text()) if state_file.exists() else {}
        if previous and not args.warm_start:
            raise ValueError("existing job requires --warm-start (fresh Adam and RNG)")
        state = dict(status="running", phase="starting", pid=os.getpid(),
                     started_utc=utc(), protocol="recovery-v3",
                     max_total_steps=args.max_total_steps, gate_every=args.gate_every,
                     completed=[stage for stage in STAGES
                                if certified(args.output / stage, stage)], assistance=False,
                     publishing=args.publish)

        def record(**changes):
            state.update(changes, total_steps=total_steps(args.output), updated_utc=utc())
            atomic_json(state_file, state)
            print(json.dumps(state), flush=True)

        def interrupt(signum, frame):
            del signum, frame
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            for index, stage in enumerate(STAGES):
                if stage in state["completed"]:
                    continue
                stage_dir = args.output / stage
                stage_dir.mkdir(exist_ok=True)
                saved = saved_steps(args.output, stage)
                used = total_steps(args.output)
                remaining = args.max_total_steps - used
                block = max(128, args.worlds // 4) * 20 * 4
                if remaining < block:
                    record(status="budget_reached", phase="stopped", stage=stage,
                           reason="Shared total ceiling reached; no automatic promotion or reset.")
                    if args.publish and (saved or index):
                        publish(args.output, stage if saved else STAGES[index - 1],
                                args.git_executable, args.git_worktree)
                    return 0
                record(phase="training", stage=stage)
                command = [sys.executable, "-u", "-m", "boxer.curriculum_train",
                    "--stage", stage, "--output", str(stage_dir),
                    "--steps", str(saved + remaining), "--worlds", str(args.worlds),
                    "--seed", str(41 + index), "--adaptive-gates",
                    "--gate-every", str(args.gate_every), "--total-offset", str(used - saved)]
                if saved:
                    command += ["--warm-start"]
                    if args.restart_from:
                        source = json.loads((args.restart_from / "latest.json").read_text())
                        if source["stage"] == stage:
                            command += ["--restart-from", str(args.restart_from.resolve())]
                            if args.force_restart:
                                command += ["--force-restart"]
                elif index:
                    command += ["--init-from", str(args.output / STAGES[index - 1] / "checkpoint")]
                if args.publish:
                    command += ["--publish-run", str(args.output),
                        "--git-executable", args.git_executable]
                    if args.git_worktree:
                        command += ["--git-worktree", args.git_worktree]
                run_child(command, stage_dir / "train.log")
                if not certified(stage_dir, stage):
                    record(status="budget_reached", phase="stopped",
                           reason="Available training budget ended without two consecutive physical passes. No promotion.")
                else:
                    state["completed"].append(stage)
                    record(phase="stage_completed")
                if args.publish:
                    publish(args.output, stage, args.git_executable, args.git_worktree)
                if not certified(stage_dir, stage):
                    return 0
            record(status="foundation_completed", phase="stopped",
                   reason="Two consecutive gates passed at every level. Paired boxing transfer is a separate experiment.")
            if args.publish:
                publish(args.output, "approach", args.git_executable, args.git_worktree)
        except KeyboardInterrupt:
            record(status="stopped_by_user", phase="stopped")
            if args.publish and state.get("stage") and (args.output / state["stage"] / "checkpoint/latest.json").exists():
                publish(args.output, state["stage"], args.git_executable, args.git_worktree)
            return 130
        except BaseException as error:
            record(status="failed", phase="stopped", error=str(error))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
