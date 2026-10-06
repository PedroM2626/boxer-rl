"""Third training mode: each corner owns a separate network.

Brax's PPO trainer holds exactly one parameter set, so two networks learning
*simultaneously* would mean forking its update loop. Alternating best response
gets the same end product more cheaply: train red's net against a frozen copy of
blue's, then train blue against red's updated net, and repeat. Each corner keeps
its own weights, its own running observation normaliser and its own value
function, which is what "each agent has its own policy" means here; what it does
not have is a single gradient step that sees both learners at once. The
distinction is worth stating because a best response to a stale opponent can
lose to a policy that kept adapting, and symmetric self-play cannot.

Each corner therefore points at a persistent ``--output`` directory and is
resumed rather than restarted, so iteration *k* adds ``--steps-per-iter`` learner
steps to a net that already exists. Opponents are loaded as frozen policies
straight out of the other corner's checkpoint directory, which is also how the
arena picks them up, so nothing here has a special-cased reader.

    python -m boxer.alternating --iterations 3 --steps-per-iter 2000000

Run it on the GPU with ``JAX_PLATFORMS=cuda,cpu``, as with ``boxer.train``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SIDES = ("red", "blue")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("runs/alt"),
                        help="root; one subdirectory per corner holds that net's run")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--steps-per-iter", type=int, default=2_000_000,
                       help="new learner steps per corner per iteration")
    parser.add_argument("--opponent", default="passive",
                        help="who the corners fight in iteration 0, before the other "
                             "net exists; 'passive', 'jabber', or a checkpoint dir")
    parser.add_argument("--seed-checkpoint", type=Path, default=None,
                        help="weights each corner starts from, instead of a random "
                             "net. Without it a league seeded against a trained "
                             "opponent spends iteration 0 being beaten up")
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--num-evals", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the trainer commands without running them")
    return parser.parse_args(argv)


def command(args: argparse.Namespace, side: str, iteration: int) -> list[str]:
    """The ``boxer.train`` invocation for one corner at one iteration.

    ``--total-steps`` is the horizon of a single invocation, not a cumulative
    one: Brax's restore path reloads the normaliser and both networks but
    reinitialises the optimizer and resets ``env_steps`` to zero, so every
    iteration trains ``--steps-per-iter`` *new* steps. The same reset means the
    learning-rate schedule restarts each time, which is the price of not forking
    Brax's trainer.
    """
    directory = args.output / side
    foe = args.output / SIDES[1 - SIDES.index(side)]
    argv = [
        sys.executable, "-u", "-m", "boxer.train",
        "--mode", "frozen",
        "--learner-side", side,
        "--opponent", str(foe / "checkpoint") if latest_steps(foe) else args.opponent,
        "--total-steps", str(args.steps_per_iter),
        "--worlds", str(args.worlds),
        "--batch-size", str(args.batch_size),
        "--num-minibatches", str(args.num_minibatches),
        "--num-evals", str(args.num_evals),
        "--episode-length", str(args.episode_length),
        "--output", str(directory),
    ]
    if args.seed_checkpoint is not None:
        # boxer.train keeps the corner's own weights if it already has them, so
        # the seed only ever lands on a corner that has never trained.
        argv += ["--init-from", str(args.seed_checkpoint)]
    # Passing --resume with nothing to resume would abort inside orbax, and
    # iteration 0 of a fresh league is exactly that case.
    return argv + ["--resume"] if latest_steps(directory) else argv


def latest_steps(side_dir: Path) -> int | None:
    """The step count of the newest checkpoint a corner has written."""
    root = side_dir / "checkpoint"
    if not root.is_dir():
        return None
    steps = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    return int(steps[-1]) if steps else None


def roster(root: Path) -> list[dict]:
    """Summarises what each corner of the league currently holds."""
    entries = []
    for side in SIDES:
        directory = root / side
        step = latest_steps(directory)
        row = {"side": side, "checkpoint": str(directory / "checkpoint"), "steps": step}
        log = directory / "league.json"
        row["opponent"] = json.loads(log.read_text()).get("last_opponent") if log.is_file() else None
        entries.append(row)
    return entries


def main(argv=None) -> int:
    args = _parse_args(argv)
    if (args.batch_size * args.num_minibatches) % args.worlds:
        raise SystemExit(
            "--batch-size * --num-minibatches must be a multiple of --worlds; "
            "frozen mode has one learner per physics world."
        )
    for side in SIDES:
        (args.output / side).mkdir(parents=True, exist_ok=True)

    for iteration in range(args.iterations):
        for side in SIDES:
            argv_ = command(args, side, iteration)
            print(f"=== iteration {iteration} / corner {side} ===", flush=True)
            print(" ".join(argv_[2:]), flush=True)
            if args.dry_run:
                continue
            opponent = argv_[argv_.index("--opponent") + 1]
            done = subprocess.run(argv_, check=False)
            directory = args.output / side
            (directory / "league.json").write_text(json.dumps(
                {"last_opponent": opponent, "last_iteration": iteration,
                 "steps": latest_steps(directory)}, indent=2
            ) + "\n")
            if done.returncode:
                print(f"corner {side} exited with {done.returncode}", flush=True)
                return done.returncode

    if args.dry_run:
        return 0
    print(json.dumps({"league": roster(args.output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
