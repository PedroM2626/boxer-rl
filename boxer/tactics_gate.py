"""Native-physics gate for tactics checkpoints.

Batched self-play can improve while native play gets worse (docs/tactics.md),
so a generation's final checkpoint is not trusted. After each generation, the
last few checkpoints play the champion natively (``boxer.spar_match``, seeds
0-7, corners swapped). The best one continues training and joins the
opponent pool. It is then confirmed on fresh seeds 8-23, for a 48-bout record
against the champion.
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


def native_match(first, second, seeds, output, ring=False):
    """Runs ``spar_match`` on CPU and returns its summary with the record of ``first``."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(Path.cwd()))
    command = [sys.executable, "-u", "-m", "boxer.spar_match", "--red-tactics", str(first),
               "--blue-tactics", str(second), "--swap", "--output", str(output),
               "--seeds", *map(str, seeds)] + (["--ring"] if ring else [])
    subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summary = json.loads(Path(output).read_text())
    return {k: v for k, v in summary.items() if k != "results"}


def score(record):
    """Wins minus losses, then clean-punch margin."""
    tactics = record["tactics"]
    return (tactics["first_wins"] - tactics["first_losses"],
            tactics["first_clean_per_bout"] - tactics["second_clean_per_bout"])


def combine(*records):
    keys = ("first_wins", "first_losses", "draws")
    total = {k: sum(r["tactics"][k] for r in records) for k in keys}
    bouts = sum(r["bouts"] for r in records)
    for side in ("first_clean_per_bout", "second_clean_per_bout"):
        total[side] = sum(r["tactics"][side] * r["bouts"] for r in records) / bouts
    total.update(bouts=bouts, falls=sum(r["knockdowns"] for r in records))
    return total


def gate_generation(root, champion, last=4, ring=False):
    """Selects and confirms a generation's best checkpoint; returns its directory.

    ``ring``: all gate bouts are fought inside the roped ring.
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
        record = native_match(directory, champion, SELECTION_SEEDS, gate / f"step-{step}.json", ring)
        trials.append(dict(step=step, directory=str(directory), record=record))
        print(json.dumps(dict(event="native_gate", step=step, **record["tactics"])), flush=True)
    best = max(trials, key=lambda trial: score(trial["record"]))
    fresh = native_match(best["directory"], champion, CONFIRMATION_SEEDS, gate / "confirmation.json", ring)
    total = combine(best["record"], fresh)
    result = dict(champion=str(champion), ring=ring, selected=best["directory"], selected_step=best["step"],
                  candidates=[dict(step=t["step"], **t["record"]["tactics"]) for t in trials],
                  confirmation=fresh["tactics"], total=total,
                  beats_champion=total["first_wins"] > total["first_losses"])
    atomic_json(path, result)
    print(json.dumps(dict(event="native_confirmation", selected_step=best["step"], **total)), flush=True)
    return Path(best["directory"])
