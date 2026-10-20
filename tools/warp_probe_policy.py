"""What is a warp checkpoint actually doing, measured in its own engine?

The training log reports reward, distance and hit counts, but not the fact that
decides whether any of them mean anything: how much of the round each boxer
spends on the floor. warp2's reward sat at -0.01 with the corners 1.2 m apart,
which is either "they stood up and walked away" or "they fell, and every dense
term is gated on BOTH corners being upright so the fall zeroes the gradient".
Those two need opposite fixes, so this measures which one it is.

    PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python \
        tools/warp_probe_policy.py runs/warp2 --steps 480
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxer import scene  # noqa: E402
from boxer.warp_env import WarpBoxing, quat_rotate, WORLD_UP  # noqa: E402
from tools.warp_video import make_forward, resolve  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("checkpoint", help="runs/<warpN>, its checkpoint dir, or a step .pkl")
    ap.add_argument("--steps", type=int, default=480,
                    help="control steps to roll out; since the round truncates on "
                         "a knockdown, 480 is several rounds not one")
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--bins", type=int, default=8, help="timeline bins")
    args = ap.parse_args(argv)

    root, step = resolve(str(args.checkpoint))
    cfg = json.loads((root / "warp_config.json").read_text())
    with step.open("rb") as handle:
        params_p, _v, _o, norm, trained = pickle.load(handle)
    forward = make_forward(cfg, params_p, norm, args.stochastic)

    env = WarpBoxing(nworld=args.worlds, n_frames=cfg["n_frames"],
                     spawn_distance=cfg["spawn_distance"],
                     action_scale=cfg["action_scale"],
                     round_steps=cfg["round_steps"],
                     imitation_path=cfg["imitation"]["path"])
    obs = env.reset(seed=args.seed)
    # The observation vector grows when a fight-state entry is added, and the new
    # entries go on the end, so an older checkpoint reads the leading slice.
    ckpt_obs = cfg["obs_size"]

    down = np.zeros((args.bins, 2))          # (world, step) pairs on the floor
    bin_len = np.zeros(args.bins)
    engaged = np.zeros(args.bins)            # both corners up, the reward gate
    posture = np.zeros((2, 2))               # [torso up-z, head z] x [red, blue]
    band, dist, hits, rew, carry = np.zeros(args.bins), [], np.zeros(2), np.zeros(2), 0.0
    # closures and their reasons: a truncated round can end because someone fell,
    # because someone left the ring, or because the two drifted apart, and only the
    # first of those is the pressure this env was changed to create.
    ends = np.zeros(4)
    sep_max = 0.0
    for t in range(args.steps):
        b = min(t * args.bins // args.steps, args.bins - 1)
        bin_len[b] += 1
        action = forward(obs[..., :ckpt_obs].reshape(-1, ckpt_obs)).reshape(
            args.worlds, 2, env.action_size)
        obs, reward, done_, parts = env.step(action)
        ends[0] += int(np.sum(done_))
        ends[1] += float(np.sum(parts["fell"]))
        ends[2] += float(np.sum(parts["left_ring"]))
        ends[3] += float(np.sum(parts["too_far"]))
        sep_max = max(sep_max, float(parts["separation"].max()))
        # The env's own down flags, taken from the step's metrics. Recomputing them
        # from the sensors reads the state AFTER the autoreset, which erases the fall
        # that just closed the round: measured 0.001 where a round every 87 steps
        # implies ~0.011. The instrument has to look before the respawn, not after.
        on_floor = parts["on_floor"] > 0.0
        for i in (0, 1):
            down[b, i] += np.count_nonzero(on_floor[:, i])
        engaged[b] += float(np.mean(~on_floor[:, 0] & ~on_floor[:, 1]))
        sensors = env._d.sensordata.numpy()
        for i, side in enumerate(scene.SIDES):
            f = env._frame(side, sensors)
            posture[0, i] += float(np.mean(
                quat_rotate(f["quat"], WORLD_UP[None, :])[:, 2]))
            posture[1, i] += float(np.mean(f["head"][:, 2]))
        band[b] += float(np.mean(parts["in_range"] > 0.5))
        carry += float(np.mean(parts["carry"] > 0.0))
        dist.append(float(np.mean(parts["distance"])))
        hits += [float(np.sum(parts["scored"])), float(np.sum(parts["taken"]))]
        rew += np.mean(reward, axis=0)

    unit = args.worlds * args.steps
    frac = down / (args.worlds * np.maximum(bin_len[:, None], 1.0))
    print(f"{step.name} trained to {trained:,} learner steps, "
          f"{'sampled' if args.stochastic else 'deterministic'} actions; "
          f"{args.worlds} worlds x {args.steps} control steps")
    print(f"  on the floor: red {frac[:, 0].mean():.3f} of the round, "
          f"blue {frac[:, 1].mean():.3f}; both up (the reward gate) "
          f"{(engaged / np.maximum(bin_len, 1.0)).mean():.3f}")
    print(f"  rounds closed: {ends[0]:.0f} in {args.steps} steps x {args.worlds} worlds"
          f" = a round every {args.worlds * args.steps / max(ends[0], 1.0):.0f} steps;"
          f" reasons per closure: fell {ends[1] / max(ends[0], 1):.2f}, ring "
          f"{ends[2] / max(ends[0], 1):.2f}, too far {ends[3] / max(ends[0], 1):.2f}"
          f" (widest separation seen {sep_max:.2f} m)")
    print("  timeline, fraction of the bin:")
    for b in range(args.bins):
        lo, hi = b * args.steps // args.bins, (b + 1) * args.steps // args.bins
        print(f"    steps {lo:4d}-{hi:4d}  down red {frac[b, 0]:.3f} blue {frac[b, 1]:.3f}"
              f"  both up {engaged[b] / bin_len[b]:.3f}  in band {band[b] / bin_len[b]:.3f}")
    print(f"  torso up-z (post-respawn, so standing worlds dominate) red "
          f"{posture[0, 0] / args.steps:.3f} "
          f"blue {posture[0, 1] / args.steps:.3f} (down under {env._down_up_z}); "
          f"head z red {posture[1, 0] / args.steps:.3f} "
          f"blue {posture[1, 1] / args.steps:.3f} (down under {env._down_head_z})")
    print(f"  distance: mean {np.mean(dist):.3f} m (start {dist[0]:.3f}, end {dist[-1]:.3f}); "
          f"band {env._clinch_range:.2f}-{env._punch_reach:.2f} m, "
          f"target {env._target_range:.2f} m")
    print(f"  steps in the band: {band.sum() / args.steps:.3f}; "
          f"red in band of a DOWN blue: {carry / args.steps:.3f}")
    print(f"  punch value per world-step (head 1.0, torso 0.5): red threw "
          f"{hits[0] / unit:.5f}, red took {hits[1] / unit:.5f}; per round of "
          f"{args.worlds * args.steps / max(ends[0], 1.0):.0f} steps that is "
          f"{hits[0] / max(ends[0], 1.0):.2f} red, "
          f"{hits[1] / max(ends[0], 1.0):.2f} against")
    print(f"  mean reward: red {rew[0] / args.steps:+.4f} blue {rew[1] / args.steps:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
