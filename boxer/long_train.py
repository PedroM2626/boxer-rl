"""Durable assisted-boxing continuation with milestone evaluation and publishing.

Run in the Warp venv. PPO, evaluation and rendering run sequentially in separate
processes, so their CUDA allocations do not compete. Published milestones contain
weights, checksums, pinned evaluations, video and an English experiment report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from boxer.checkpoint_io import atomic_bytes, atomic_json

ROOT = Path(__file__).resolve().parents[1]
MILESTONES = [10_000_000, 25_000_000, 50_000_000, 100_000_000,
              150_000_000, 200_000_000]
OWNED = ("checkpoints/skills_200m", "docs/results/skills_200m",
         "fights/skills_200m", "docs/skills-200m.md")


def checkpoint_step(run):
    pointer = Path(run) / "checkpoint/latest.txt"
    return int(pointer.read_text().strip()) if pointer.exists() else 0


def training_command(config, output, total_steps, save_every):
    command = [sys.executable, "-u", "-m", "boxer.warp_train", "--resume",
               "--output", str(output), "--total-steps", str(total_steps),
               "--save-every", str(save_every), "--seed", str(config.get("seed", 0))]
    for name in ("environment", "worlds", "n_frames", "spawn_distance", "action_scale",
                 "round_steps", "damage_cap", "stun_steps", "mode"):
        command.extend(["--" + name.replace("_", "-"), str(config[name])])
    for name in ("policy_layers", "value_layers"):
        command.extend(["--" + name.replace("_", "-"), *map(str, config[name])])
    for name, value in config["ppo"].items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    if not config.get("skill_prior", True):
        command.append("--no-skill-prior")
    return command


def export_checkpoint(source, destination):
    """Copy only a complete, pinned checkpoint, never an entire runs/ tree."""
    source, destination = Path(source), Path(destination)
    step = checkpoint_step(source)
    if not step:
        raise ValueError(f"no checkpoint under {source}")
    name = f"step-{step}.pkl"
    data = (source / "checkpoint" / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    manifest_path = source / "checkpoint" / f"step-{step}.state.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != digest or manifest["step"] != step:
            raise ValueError(f"checkpoint verification failed under {source}")
        atomic_bytes(destination / "checkpoint" / manifest_path.name,
                     manifest_path.read_bytes())
    atomic_bytes(destination / "checkpoint" / name, data)
    atomic_bytes(destination / "warp_config.json", (source / "warp_config.json").read_bytes())
    atomic_bytes(destination / "checkpoint/latest.txt", str(step).encode())
    atomic_json(destination / "artifact.json", dict(step=step, sha256=digest))
    return destination / "checkpoint" / name


def scoped_path(path):
    path = path.replace("\\", "/")
    return any(path == prefix or path.startswith(prefix + "/") for prefix in OWNED)


def run_command(command, log_path, env):
    print(f"running {command[3] if len(command) > 3 else command[0]}; log: {log_path}",
          flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT)
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    if code:
        raise RuntimeError(f"child exited with code {code}; inspect {log_path}")


def write_report(state):
    lines = ["# Assisted boxing: 200M continuation", "",
             f"Status: **{state['status']}**. Updated: {state['updated_utc']}.", "",
             f"Requested budget: {state['target_steps']:,} learner transitions total, "
             "including the 2,109,440-transition seed. A world step counts twice.", "",
             "The environment, motor controller and PPO hyperparameters stay fixed. "
             "This is the assisted curriculum; balance and locomotion are not learned.", "",
             "Training is allowed to use the full budget. Short-term plateaus, lower "
             "reward or a checkpoint losing an evaluation do not cancel it. Non-finite "
             "data, an external interruption or a failed subprocess stop the job with "
             "the last valid checkpoint preserved.", "",
             "Milestones are evaluated with deterministic self-play and against the "
             "pinned 2.11M seed, at seeds 7 and 23, exchanging corners. All reports "
             "and videos identify the exact checkpoint. The latest checkpoint is not "
             "automatically called the best. Points readings settle standing timeouts "
             "using legal punch value; the original KO/fall win readings are retained.", "",
             "| Requested milestone | Actual checkpoint | Self-play hits A / B | "
             "Self-play head hits A / B | Win with points vs seed |",
             "| --- | --- | --- | --- | --- |"]
    for stage in state.get("completed", []):
        self_results = [r["results"] for r in stage["evaluations"] if r["kind"] == "self"]
        baseline = [r["results"] for r in stage["evaluations"] if r["kind"] == "seed"]
        avg = lambda values, key: sum(r[key] for r in values) / len(values)
        hits = lambda side: avg(self_results, f"{side}_head") + avg(self_results, f"{side}_torso")
        lines.append(f"| {stage['requested_steps']:,} | {stage['actual_steps']:,} | "
                     f"{hits('a'):.3f} / {hits('b'):.3f} | "
                     f"{avg(self_results, 'a_head'):.3f} / {avg(self_results, 'b_head'):.3f} | "
                     f"{avg(baseline, 'a_win_with_points'):.3f} |")
    lines += ["", "Machine-readable records: [milestones](results/skills_200m/state.json).",
              "Weights: `checkpoints/skills_200m/`. Videos: `fights/skills_200m/`.", "",
              "Raw progress, subprocess logs and the live job state remain under "
              "`runs/skills_200m/` and are ignored by Git. Published milestones are "
              "small reproducible bundles; intermediate saves remain local.", "",
              "A resume preserves actor, critic, optimizer, normalization and the saved "
              "PPO sampling key. Each milestone starts fresh physical rounds; simulator "
              "state is not restored. Legacy seeds without a saved key fold the seed "
              "step into the new key. This is not a bit-identical uninterrupted rollout.", "",
              "The computer and WSL must remain running. The job uses an exclusive "
              "run lock, atomic checkpoint writes and explicit output-path ownership "
              "for automatic Git commits. Unrelated staged changes block publishing."]
    if state.get("error"):
        lines += ["", f"Failure: `{state['error']}`."]
    if state.get("publish_error"):
        lines += ["", f"Publishing requires attention: `{state['publish_error']}`."]
    atomic_bytes(ROOT / "docs/skills-200m.md", ("\n".join(lines) + "\n").encode())
    atomic_json(ROOT / "docs/results/skills_200m/state.json", state)


def publish(args, message):
    prefix = [args.git_executable, "-C", args.git_worktree]
    def git(*arguments):
        return subprocess.run([*prefix, *arguments], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    if git("branch", "--show-current") != args.branch:
        raise RuntimeError("branch changed while training; refusing automatic publication")
    staged = git("diff", "--cached", "--name-only").splitlines()
    if any(not scoped_path(path) for path in staged):
        raise RuntimeError("unrelated staged changes; refusing to include them in a training commit")
    paths = [path for path in OWNED if (ROOT / path).exists()]
    git("add", "--", *paths)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", message)
    git("push", "origin", args.branch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/skills_200m"))
    parser.add_argument("--seed-run", type=Path, default=Path("checkpoints/skills_seed_2m"))
    parser.add_argument("--milestones", type=int, nargs="+", default=MILESTONES)
    parser.add_argument("--eval-worlds", type=int, default=64)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[7, 23])
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--git-executable", default="git")
    parser.add_argument("--git-worktree", default=str(ROOT))
    parser.add_argument("--branch", default="master")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    args.output = (ROOT / args.output).resolve()
    args.seed_run = (ROOT / args.seed_run).resolve()
    if (not args.output.is_relative_to(ROOT / "runs") or args.output == ROOT / "runs"
            or args.output == args.seed_run or not args.seed_run.is_relative_to(ROOT)):
        parser.error("output must be a dedicated runs/ subdirectory and the seed must be inside the project")
    if args.milestones != sorted(set(args.milestones)) or min(args.milestones) <= checkpoint_step(args.seed_run):
        parser.error("milestones must be unique, increasing and beyond the seed checkpoint")
    if min(args.eval_worlds, args.save_every) < 1:
        parser.error("eval-worlds and save-every must be positive")
    config = json.loads((args.seed_run / "warp_config.json").read_text())
    if config.get("environment") != "skills" or config.get("mode") != "selfplay":
        parser.error("the long-run seed must be an assisted self-play checkpoint")
    protocol = dict(seed_run=str(args.seed_run.relative_to(ROOT)),
                    milestones=args.milestones, eval_worlds=args.eval_worlds,
                    eval_seeds=args.eval_seeds, target_steps=max(args.milestones))
    if args.plan_only:
        print(json.dumps(protocol, indent=2))
        return 0
    import fcntl  # The durable job runs on Linux/WSL2 with MuJoCo Warp.
    args.output.mkdir(parents=True, exist_ok=True)
    # All outputs share one published namespace, so a second output directory
    # must not bypass the lock and overwrite another job's milestone reports.
    with (ROOT / "runs/.skills_200m_namespace.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("this run already has an active supervisor")
        return run_job(args, config, protocol)


def run_job(args, config, protocol):
    state_path = args.output / "job_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        protocol=protocol, completed=[], target_steps=protocol["target_steps"])
    if state["protocol"] != protocol:
        raise ValueError("resume protocol differs from the saved long-run protocol")
    state.update(status="running", pid=os.getpid(), error=None, publish_error=None)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu", MUJOCO_GL="egl")
    def record():
        state["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        state["current_checkpoint_steps"] = checkpoint_step(args.output)
        atomic_json(state_path, state)
        write_report(state)
    def maybe_publish(message):
        if args.publish:
            state["publish_error"] = None
            record()
            try:
                publish(args, message)
            except Exception as error:
                state["publish_error"] = str(error)
                print(f"publishing failed; training artifacts retained: {error}", flush=True)
                record()
    def stop(_signum, _frame):
        raise KeyboardInterrupt("training interrupted")
    signal.signal(signal.SIGTERM, stop)
    try:
        if not checkpoint_step(args.output):
            export_checkpoint(args.seed_run, args.output)
        record()
        for target in args.milestones:
            if any(stage["requested_steps"] == target for stage in state["completed"]):
                continue
            state.update(phase="training", milestone=target)
            record()
            if checkpoint_step(args.output) < target:
                run_command(training_command(config, args.output, target, args.save_every),
                            args.output / f"train_{target}.log", env)
            actual = checkpoint_step(args.output)
            if actual < target:
                raise RuntimeError("trainer exited before reaching the milestone")
            bundle = ROOT / "checkpoints/skills_200m" / f"step-{actual}"
            candidate = export_checkpoint(args.output, bundle).relative_to(ROOT)
            # Export the seed too so both sides remain pinned if local runs change.
            baseline = export_checkpoint(args.seed_run,
                ROOT / "checkpoints/skills_200m/seed").relative_to(ROOT)
            state["phase"] = "evaluating"
            record()
            evaluations = []
            for seed in args.eval_seeds:
                for kind, opponent in (("self", candidate), ("seed", baseline)):
                    result_path = ROOT / "docs/results/skills_200m" / f"{actual}_{kind}_{seed}.json"
                    run_command([sys.executable, "-u", "-m", "boxer.warp_arena",
                        str(candidate), str(opponent), "--worlds", str(args.eval_worlds),
                        "--steps", str(config["round_steps"]), "--seed", str(seed),
                        "--output", str(result_path)],
                        args.output / f"eval_{actual}_{kind}_{seed}.log", env)
                    result = json.loads(result_path.read_text())
                    if result["results"]["dropped_rounds"] != 0:
                        raise RuntimeError("evaluation dropped non-finite rounds")
                    evaluations.append(dict(kind=kind, seed=seed, results=result["results"]))
            state["phase"] = "rendering"
            record()
            video = ROOT / "fights/skills_200m" / f"step-{actual}.mp4"
            run_command([sys.executable, "-u", "tools/warp_video.py", str(candidate),
                         "--length", str(config["round_steps"]), "--output", str(video)],
                        args.output / f"render_{actual}.log", env)
            state["completed"].append(dict(requested_steps=target, actual_steps=actual,
                checkpoint=str(candidate), evaluations=evaluations,
                video=str(video.relative_to(ROOT))))
            state["phase"] = "milestone_complete"
            record()
            maybe_publish(f"Record assisted boxing milestone at {actual:,} transitions")
        state.update(status="complete", phase="complete")
        record()
        maybe_publish("Record completed 200M assisted boxing continuation")
        return 1 if state.get("publish_error") else 0
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="failed", error=str(error))
        record()
        maybe_publish("Record interrupted assisted boxing continuation")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
