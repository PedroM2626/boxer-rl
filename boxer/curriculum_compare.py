"""Bounded, isolated reward comparison followed by the unchanged main warm start."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_job import ROOT, run_child, utc
from boxer.curriculum_control import commands_for, passes_gate


def read(path):
    return json.loads(Path(path).read_text())


def compare(main_run, output, steps=10_000_000, worlds=1024, execute=run_child):
    """Caller holds main run/publication locks; never replace the main policy."""
    main_run, output = Path(main_run), Path(output)
    state = read(main_run / "state.json")
    if state.get("stage") != "walk_easy" or "balance" not in state.get("completed", []):
        raise ValueError("comparison requires a transferred easy-walking policy")
    if steps <= 0 or worlds <= 0 or worlds & (worlds - 1):
        raise ValueError("positive budget and power-of-two worlds required")
    source = main_run / "walk_easy"
    metadata = read(source / "checkpoint/latest.json")
    payload = (source / "checkpoint" / metadata["file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("comparison source checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    seed = output / "seed"
    atomic_bytes(seed / metadata["file"], payload)
    atomic_json(seed / "latest.json", metadata)
    for name in ("config.json", "gate_history.json", "progress.json", "training_result.json"):
        if (source / name).exists():
            atomic_bytes(output / ("original-" + name), (source / name).read_bytes())
    record = dict(status="running", pid=os.getpid(), started_utc=utc(),
                  main_total_steps=metadata["total_steps"],
                  main_stage_steps=metadata["stage_steps"], seed_sha256=metadata["sha256"],
                  requested_steps_per_arm=steps, worlds=worlds, training_seed=42,
                  diagnostic_steps=0, arms={}, automatic_adoption=False)

    def update(**changes):
        record.update(changes, updated_utc=utc())
        atomic_json(output / "state.json", record)
        print(json.dumps(record), flush=True)

    def evaluate(checkpoint, destination):
        execute([sys.executable, "-u", "-m", "boxer.curriculum_eval", str(checkpoint),
                 "--output", str(destination)], output / "evaluation.log")
        result = read(destination)
        manifest = read(checkpoint / "latest.json")
        if result["checkpoint_sha256"] != manifest["sha256"]:
            raise ValueError("comparison gate checksum differs from evaluated policy")
        expected = {(seed, command, False) for seed in (7, 23)
                    for command in commands_for("walk_easy")}
        trials = result.get("trials", [])
        actual = {(t["seed"], tuple(t["command"]), t["disturbed"]) for t in trials}
        if (result.get("kind") != "promotion_gate" or result.get("stage") != "walk_easy"
                or result.get("worlds_per_trial") != 32 or result.get("episode_seconds") != 10
                or result.get("assistance") is not False or len(trials) != 14
                or actual != expected or result.get("stage_steps") != manifest["stage_steps"]
                or result.get("passed") != passes_gate("walk_easy", trials)):
            raise ValueError("comparison requires the unchanged complete physical gate")
        return result

    try:
        update(phase="baseline_evaluation")
        evaluate(seed, output / "baseline.json")
        for name, sigma in (("default", .25), ("sharp", .01)):
            arm = output / name
            update(phase="training", active_arm=name)
            execute([sys.executable, "-u", "-m", "boxer.curriculum_train",
                     "--stage", "walk_easy", "--output", str(arm),
                     "--init-from", str(seed), "--steps", str(steps),
                     "--worlds", str(worlds), "--seed", "42", "--checkpoints", "10",
                     "--tracking-sigma", str(sigma)], output / (name + ".log"))
            trained = read(arm / "checkpoint/latest.json")
            record["arms"][name] = dict(tracking_sigma=sigma,
                transitions=trained["stage_steps"], checkpoint_sha256=trained["sha256"])
            record["diagnostic_steps"] = sum(a["transitions"] for a in record["arms"].values())
            update(phase="evaluation")
            result = evaluate(arm / "checkpoint", arm / "evaluation.json")
            record["arms"][name]["gate_passed"] = result["passed"]
            update(phase="arm_completed")
        update(status="completed", phase="awaiting_analysis", active_arm=None,
               decision="No candidate adopted; resume the original main policy pending analysis.")
    except BaseException as error:
        # Preserve completed and partial diagnostic compute even when a child fails.
        for name in ("default", "sharp"):
            manifest = output / name / "checkpoint/latest.json"
            if manifest.exists():
                record["arms"].setdefault(name, {})["transitions"] = read(manifest)["stage_steps"]
        update(status="failed", phase="stopped", error=str(error),
               diagnostic_steps=sum(a.get("transitions", 0) for a in record["arms"].values()))
        raise
    return record


def main(argv=None):
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, default=ROOT / "runs/curriculum")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--resume-main", action="store_true")
    args = parser.parse_args(argv)
    previous = read(args.main_run / "state.json")
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    failure = None
    with ExitStack() as stack:
        for path in (args.main_run / ".lock", ROOT / "runs/curriculum_publish.lock"):
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # An active main supervisor holds these locks, so comparison cannot overlap it.
        paused = dict(previous, status="paused_for_diagnostic", phase="reward_comparison",
                      diagnostic_output=str(args.output), diagnostic_pid=os.getpid(), updated_utc=utc())
        atomic_json(args.main_run / "state.json", paused)
        try:
            compare(args.main_run, args.output, args.steps, args.worlds)
        except KeyboardInterrupt:
            raise  # An explicit stop must not start another learner.
        except Exception as error:
            failure = error
            print(f"Comparison failed; preserving original policy: {error}", flush=True)
    if args.resume_main:
        # Locks have been released. No experiment candidate or forced rollback is used.
        os.execv("/bin/bash", ["bash", str(ROOT / "tools/run_curriculum.sh"),
                 "--output", str(args.main_run), "--warm-start", "--publish",
                 "--max-total-steps", str(previous["max_total_steps"]),
                 "--gate-every", str(previous["gate_every"]), "--worlds", str(args.worlds)])
    if failure:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
