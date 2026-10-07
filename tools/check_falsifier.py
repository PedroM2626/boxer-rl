"""Read the 200M run against the falsifier that was written before it started.

The claim being tested is in tools/long_run.sh: if `down_red` has not gone below
~0.3 by 50M learner steps, then training scale is not the blocker and what is left
is the control interface. This prints the curve and the verdict, so checking
whether a day of GPU time is still worth spending is one command.

    python -m tools.check_falsifier [run_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LIMIT = 50_000_000
THRESHOLD = 0.30
KEYS = ("episode/length", "episode/down_red", "episode/hits_red",
        "episode/head_shots", "episode/contacts", "episode/upright")


def main(argv=None) -> int:
    run = Path(argv[0] if argv else "runs/frozen_200m")
    log = run / "progress.jsonl"
    if not log.is_file():
        raise SystemExit(f"no progress log at {log}")

    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "episode/down_red" in r]
    if not rows:
        raise SystemExit(f"{log} has no episode metrics yet")

    print(f"{'step':>12} " + " ".join(f"{k.split('/')[-1]:>10}" for k in KEYS))
    for r in rows:
        print(f"{r['step']:>12,} " + " ".join(f"{r.get(k, float('nan')):>10.2f}" for k in KEYS))

    last = rows[-1]
    print(f"\nlast checkpoint {last['step']:,} learner steps, "
          f"down_red={last['episode/down_red']:.3f}")
    if last["step"] < LIMIT:
        print(f"not decided yet: the falsifier reads at {LIMIT:,} steps, "
              f"{LIMIT - last['step']:,} to go")
    elif last["episode/down_red"] > THRESHOLD:
        print(f"FALSIFIED: down_red is above {THRESHOLD} at {LIMIT:,} steps, so scale "
              "is not the blocker and the control interface is the remaining suspect")
    else:
        print(f"holding: down_red is below {THRESHOLD} at {last['step']:,} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
