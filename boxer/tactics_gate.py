"""Native-physics gate for tactics checkpoints.

Batched self-play can improve while native play gets worse (docs/tactics.md),
so a generation's final checkpoint is not trusted. After each generation, the
last few checkpoints play the champion natively (``boxer.spar_match``, seeds
0-7, corners swapped). The best one continues training and joins the
opponent pool. It is then confirmed on fresh seeds 8-23, for a 48-bout record
against the champion.

Short bouts end at the first fall and last 20 s, but some candidates only
fall over minutes: a striker trained with an arm-motion cost had no falls in
the gate and 29 knockdowns in 32 bouts of 120 s. So before confirmation the
best candidates, in order of their short-bout score, also box 16 bouts of
120 s under knockdown rules (fresh seeds 24-31) against the champion and any
further long-bout opponents: that striker fell twice against the striker it
started from, but 29 times against the champion tactics. A candidate is
stable unless, against some opponent, it suffers more than
``opponent knockdowns + max(2, opponent knockdowns)``; the first stable one
is selected. If none is, the champion stays selected.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from boxer.checkpoint_io import atomic_json

SELECTION_SEEDS = tuple(range(8))
CONFIRMATION_SEEDS = tuple(range(8, 24))
LONG_SEEDS = tuple(range(24, 32))
LONG_SECONDS = 120.
LONG_KNOCKDOWNS = 3


def candidates(checkpoint_dir, last):
    """The ``last`` saved steps of a generation, oldest first."""
    steps = sorted(Path(checkpoint_dir).glob("step-*.pkl.json"),
                   key=lambda path: int(path.name.split("-")[1].split(".")[0]))
    return steps[-last:]


def stage(manifest, destination):
    """A loadable checkpoint directory (latest.json + weights) for one step."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(manifest).read_text())
    shutil.copy2(Path(manifest).parent / metadata["file"], destination / metadata["file"])
    shutil.copy2(manifest, destination / Path(manifest).name)
    atomic_json(destination / "latest.json", metadata)
    return destination


def native_match(first, second, seeds, output, ring=False, arm=None, motor=None, seconds=None,
                 knockdowns=None):
    """Runs ``spar_match`` on CPU and returns its summary with the record of ``first``."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(Path.cwd()))
    command = [sys.executable, "-u", "-m", "boxer.spar_match", "--red-tactics", str(first),
               "--blue-tactics", str(second), "--swap", "--output", str(output),
               "--seeds", *map(str, seeds)] + (["--ring"] if ring else [])
    if arm:
        # Both corners punch with the same arm skill: only the tactics differ.
        command += ["--red-arm", str(arm), "--blue-arm", str(arm)]
    if motor:
        command += ["--checkpoint", str(motor)]
    if seconds:
        command += ["--seconds", str(seconds)]
    if knockdowns:
        command += ["--knockdowns", str(knockdowns)]
    subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summary = json.loads(Path(output).read_text())
    return {k: v for k, v in summary.items() if k != "results"}


def score(record):
    """Wins minus losses, then clean-punch margin."""
    tactics = record["tactics"]
    return (tactics["first_wins"] - tactics["first_losses"],
            tactics["first_clean_per_bout"] - tactics["second_clean_per_bout"])


def stable(record):
    """Did ``first`` suffer no more than ``second + max(2, second)`` knockdowns?"""
    tactics = record["tactics"]
    mine, theirs = tactics["first_knockdowns"], tactics["second_knockdowns"]
    return mine <= theirs + max(2, theirs)


def combine(*records):
    keys = ("first_wins", "first_losses", "draws")
    total = {k: sum(r["tactics"][k] for r in records) for k in keys}
    bouts = sum(r["bouts"] for r in records)
    for side in ("first_clean_per_bout", "second_clean_per_bout"):
        total[side] = sum(r["tactics"][side] * r["bouts"] for r in records) / bouts
    total.update(bouts=bouts, falls=sum(r["knockdowns"] for r in records))
    return total


def mirror_stable(record, limit):
    """In bouts of a candidate against itself: no more than ``limit`` knockdowns in all."""
    tactics = record["tactics"]
    return tactics["first_knockdowns"] + tactics["second_knockdowns"] <= limit


def gate_generation(root, champion, last=4, ring=False, arm=None, motor=None, long_tries=2,
                    long_opponents=(), mirror_knockdowns=None):
    """Selects and confirms a generation's best checkpoint; returns its directory.

    ``ring``: all gate bouts are fought inside the roped ring. ``arm``: an
    arm-skill checkpoint steering both corners' punches. ``motor``: the leg
    motor both corners balance with (default ``checkpoints/spar``).
    ``long_tries``: how many candidates, best first, may take the long-bout
    stability check (0 skips it); ``long_opponents`` are checked besides
    ``champion``. ``mirror_knockdowns``: also box each long-checked candidate
    against itself and allow at most this many knockdowns in all (None skips
    it). Returns ``champion`` when none is stable.
    """
    root = Path(root)
    gate = root / "gate"
    path = gate / "gate.json"
    if path.exists():
        return Path(json.loads(path.read_text())["selected"])
    trials = []
    for manifest in candidates(root / "checkpoint", last):
        step = json.loads(manifest.read_text())["stage_steps"]
        directory = stage(manifest, gate / f"step-{step}")
        record = native_match(directory, champion, SELECTION_SEEDS, gate / f"step-{step}.json", ring, arm, motor)
        trials.append(dict(step=step, directory=str(directory), record=record))
        print(json.dumps(dict(event="native_gate", step=step, **record["tactics"])), flush=True)
    ranked = sorted(trials, key=lambda trial: score(trial["record"]), reverse=True)
    best, checks = ranked[0], []
    if long_tries:
        best = None
        for trial in ranked[:long_tries]:
            fine = True
            for n, opponent in enumerate([champion, *long_opponents]):
                record = native_match(trial["directory"], opponent, LONG_SEEDS,
                                      gate / f"long-{trial['step']}-{n}.json", ring, arm, motor, LONG_SECONDS,
                                      LONG_KNOCKDOWNS)
                checks.append(dict(step=trial["step"], opponent=str(opponent), stable=stable(record),
                                   **record["tactics"]))
                print(json.dumps(dict(event="native_long_gate", **checks[-1])), flush=True)
                fine = fine and checks[-1]["stable"]
                if not fine:
                    break
            if fine and mirror_knockdowns is not None:
                record = native_match(trial["directory"], trial["directory"], LONG_SEEDS,
                                      gate / f"long-{trial['step']}-mirror.json", ring, arm, motor, LONG_SECONDS,
                                      LONG_KNOCKDOWNS)
                checks.append(dict(step=trial["step"], opponent="itself",
                                   stable=mirror_stable(record, mirror_knockdowns), **record["tactics"]))
                print(json.dumps(dict(event="native_long_gate", **checks[-1])), flush=True)
                fine = checks[-1]["stable"]
            if fine:
                best = trial
                break
    common = dict(champion=str(champion), ring=ring, arm=str(arm) if arm else None,
                  motor=str(motor) if motor else None,
                  candidates=[dict(step=t["step"], **t["record"]["tactics"]) for t in trials], long_checks=checks)
    if best is None:
        atomic_json(path, dict(common, selected=str(champion), selected_step=None, beats_champion=False,
                               stable_candidate=False))
        print(json.dumps(dict(event="native_gate_rejected", reason="no candidate stable over long bouts")),
              flush=True)
        return Path(champion)
    fresh = native_match(best["directory"], champion, CONFIRMATION_SEEDS, gate / "confirmation.json", ring, arm, motor)
    total = combine(best["record"], fresh)
    result = dict(common, selected=best["directory"], selected_step=best["step"], stable_candidate=True,
                  confirmation=fresh["tactics"], total=total,
                  beats_champion=total["first_wins"] > total["first_losses"])
    atomic_json(path, result)
    print(json.dumps(dict(event="native_confirmation", selected_step=best["step"], **total)), flush=True)
    return Path(best["directory"])
