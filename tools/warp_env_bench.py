"""Throughput of the full WarpBoxing control step, at training scale.

``tools/warp_train_check.sh`` measured the bare physics loop (2048 worlds,
4,275 control steps/s). The trainer pays for more than that every step: the
packed contact buffer sync, the contact_force kernel, the scorecard's numpy
reductions, the 128-wide observation build, and the reward. This is the number
the 200M-step estimate for the first warp arm has to use.

    PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python tools/warp_env_bench.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from boxer.warp_env import WarpBoxing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args(argv)

    env = WarpBoxing(nworld=args.worlds)
    action = np.zeros((args.worlds, 2, env.action_size), dtype=np.float32)
    env.reset(0)
    for _ in range(args.warmup):
        env.step(action)

    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(action)
    elapsed = time.perf_counter() - start
    rate = args.steps * args.worlds / elapsed

    mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    print(f"{args.worlds} worlds, {args.steps} control steps in {elapsed:.2f}s: "
          f"{rate:,.0f} control steps/s (device memory {mem})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
