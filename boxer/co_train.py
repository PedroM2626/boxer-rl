"""Legs and striker trained in alternating rounds.

Every part of a boxer so far trained with the others frozen, and changing one
broke another: a striker that learned a new arm style fell on legs trained
under the old one (docs/striker.md). Here the two adapt to each other. Each
round:

1. legs: ``motor_native`` fine-tunes the current legs in native two-robot
   bouts whose pool holds the current striker (weighted up), the
   ``--reference-striker``, the champion tactics and the scripted brain;
2. striker: ``native_finetune --striker`` fine-tunes the current striker on
   those new legs, against the champion pool and the current striker itself,
   and its native gate (with the 120 s stability check against the current
   striker and the first pool policy, the champion tactics; ``tactics_gate``)
   decides whether the new striker replaces the current one.

The new legs always carry on; the striker only when its gate selects it.
Rounds resume from ``state.json``: a finished phase is never rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from boxer.checkpoint_io import atomic_json


def run(command, log):
    """Runs one phase as its own process, its output appended to ``log``."""
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
    with open(log, "a") as out:
        subprocess.run([sys.executable, "-u", "-m", *command], env=env, check=True, stdout=out,
                       stderr=subprocess.STDOUT)


def legs_command(args, k, legs, striker, output):
    return ["boxer.motor_native", "--init-from", str(legs), "--output", str(output),
            "--striker", str(striker), "--second-striker", str(args.reference_striker),
            "--weights", *map(str, args.leg_weights), "--steps", str(args.leg_steps),
            "--checkpoint-every", str(args.leg_steps), "--seed", str(100 + k)]


def striker_command(args, k, legs, striker, output):
    return ["boxer.native_finetune", "--striker", "--init-from", str(striker), "--output", str(output),
            "--motor", str(legs), "--ring", "--arm", str(args.arm),
            "--pool", *map(str, args.pool), str(striker),
            "--scripted-fraction", str(args.scripted_fraction), "--high-guard-fraction", ".6",
            "--steps", str(args.striker_steps), "--checkpoint-every", str(args.striker_steps // 2),
            "--min-std", ".1", "--arm-motion-cost", str(args.arm_motion_cost),
            "--native-gate", "2", "--long-gate", "2", "--long-gate-opponents", str(args.pool[0]),
            "--seed", str(200 + k)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legs", type=Path, default=Path("checkpoints/spar-native"))
    parser.add_argument("--striker", type=Path, default=Path("checkpoints/striker"))
    parser.add_argument("--reference-striker", type=Path, default=Path("checkpoints/striker-v1"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--leg-steps", type=int, default=1_000_000, help="robot transitions per legs phase")
    parser.add_argument("--leg-weights", type=float, nargs=4, default=[2., 1., 1., .5],
                        help="current striker, reference striker, champion, scripted (motor_native BRAINS)")
    parser.add_argument("--striker-steps", type=int, default=750_000)
    parser.add_argument("--pool", type=Path, nargs="*",
                        default=[Path("checkpoints/tactics"), Path("checkpoints/tactics-highguard-v1")])
    parser.add_argument("--arm", type=Path, default=Path("checkpoints/arm-skill"))
    parser.add_argument("--scripted-fraction", type=float, default=.3)
    parser.add_argument("--arm-motion-cost", type=float, default=.05)
    args = parser.parse_args(argv)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    path = root / "state.json"
    state = json.loads(path.read_text()) if path.exists() else dict(
        legs=str(args.legs), striker=str(args.striker), rounds=[])
    log = root / "co_train.log"
    for k in range(len(state["rounds"]), args.rounds):
        directory = root / f"round-{k}"
        entry = state.get("current") or dict(round=k)
        if "legs" not in entry:
            run(legs_command(args, k, state["legs"], state["striker"], directory / "legs"), log)
            entry["legs"] = str(directory / "legs" / "checkpoint")
            state["current"] = entry
            atomic_json(path, state)
            print(json.dumps(dict(event="legs", round=k, legs=entry["legs"])), flush=True)
        if "striker" not in entry:
            run(striker_command(args, k, entry["legs"], state["striker"], directory / "striker"), log)
            gate = json.loads((directory / "striker" / "gate" / "gate.json").read_text())
            entry.update(striker=gate["selected"], striker_replaced=gate["selected"] != state["striker"],
                         gate=dict(long_checks=gate["long_checks"], total=gate.get("total")))
            print(json.dumps(dict(event="striker", round=k, striker=entry["striker"],
                                  replaced=entry["striker_replaced"])), flush=True)
        state.update(legs=entry["legs"], striker=entry["striker"])
        state["rounds"].append(entry)
        state.pop("current", None)
        atomic_json(path, state)
    print(json.dumps(dict(event="done", legs=state["legs"], striker=state["striker"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
