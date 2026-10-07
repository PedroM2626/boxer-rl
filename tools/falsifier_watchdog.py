"""Stop the long run when its own falsifier fires, instead of after 26 hours.

tools/long_run.sh commits to a claim: if `down_red` has not gone below ~0.3 by 50M
learner steps, training scale is not what keeps the boxers on the floor, and the
remaining 150M steps of the same run cannot fix it. Reading that and acting on it
by hand means either watching a log all night or spending a day of a laptop GPU on
a hypothesis that is already dead.

Checkpoints land every 5,242,880 steps and `--resume` works, so stopping here loses
at most the block in flight.

    python -m tools.falsifier_watchdog            # watch the 200M run
    python -m tools.falsifier_watchdog --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LIMIT = 50_000_000
THRESHOLD = 0.30
POLL_SECONDS = 300


def latest(run: Path) -> dict | None:
    log = run / "progress.jsonl"
    if not log.is_file():
        return None
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "episode/down_red" in r]
    return rows[-1] if rows else None


def trainer_pid() -> int | None:
    """The boxer.train process, if one is running."""
    out = subprocess.run(["pgrep", "-f", "boxer.train"], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    return pids[0] if pids else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/frozen_200m"))
    parser.add_argument("--dry-run", action="store_true", help="report without killing")
    args = parser.parse_args(argv)

    print(f"watching {args.run}: limit {LIMIT:,} steps, threshold down_red {THRESHOLD}",
          flush=True)
    while True:
        row = latest(args.run)
        pid = trainer_pid()
        if pid is None and row and row["step"] >= LIMIT:
            print(f"trainer is gone; final reading down_red={row['episode/down_red']:.3f} "
                  f"at {row['step']:,} steps", flush=True)
            return 0
        if row and row["step"] >= LIMIT:
            down = row["episode/down_red"]
            if down > THRESHOLD:
                print(f"FALSIFIED at {row['step']:,} steps: down_red={down:.3f} exceeds "
                      f"{THRESHOLD}. Scale is not the blocker; stopping the run.", flush=True)
                if not args.dry_run:
                    os.kill(pid, signal.SIGTERM)
                    print(f"sent SIGTERM to pid {pid}", flush=True)
                return 1
            print(f"holding at {row['step']:,} steps: down_red={down:.3f} <= {THRESHOLD}. "
                  "Letting the run continue.", flush=True)
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
