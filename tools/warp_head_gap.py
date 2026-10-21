"""How close does the fist ever get to the head, and does it try?

warp6 lands 6.4 torso contacts a round and one head shot per ~400 rounds. Two
opposite explanations survive that number: the head is physically out of reach at
the distance the reward pays (then no weight will fix it), or it is reachable and
the policy simply does not aim there (then the objective is the lever). This
measures which, by rolling a checkpoint out and tracking the fist's distance to
each target surface -- and the elevation it would have to lift the arm to reach
the head -- against the collision geometry that decides contact.

    PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python \
        tools/warp_head_gap.py runs/warp6 --worlds 128 --steps 480
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxer import scene  # noqa: E402
from boxer.warp_env import WarpBoxing  # noqa: E402
from boxer.warp_train import numpy_forward, warp_run  # noqa: E402

# A fist (capsule radius 0.035) touching the head sphere (radius 0.06) has its
# centre 0.095 m from the head centre; that is what "in contact range" means here.
CONTACT_RANGE = 0.095


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("checkpoint", help="runs/<warpN>, its checkpoint dir, or a step .pkl")
    ap.add_argument("--worlds", type=int, default=128)
    ap.add_argument("--steps", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args(argv)

    _root, step, cfg = warp_run(str(args.checkpoint))
    with step.open("rb") as handle:
        params_p, _v, _o, norm, trained = pickle.load(handle)
    forward = numpy_forward(cfg, params_p, norm, args.stochastic)
    env = WarpBoxing(nworld=args.worlds, n_frames=cfg["n_frames"],
                     spawn_distance=cfg["spawn_distance"],
                     action_scale=cfg["action_scale"],
                     round_steps=cfg["round_steps"],
                     imitation_path=cfg["imitation"]["path"])
    obs = env.reset(seed=args.seed)
    # The vector grew by three fight-state entries after this checkpoint was
    # trained, and they were appended, so the old one is the leading slice.
    ckpt_obs = cfg["obs_size"]
    if obs.shape[-1] != ckpt_obs:
        print(f"note: env observes {obs.shape[-1]} dims, this checkpoint was trained "
              f"on the first {ckpt_obs} -- slicing")

    gap = {"head": [], "torso": []}          # fist-to-surface distance, per step
    lift = {"head": [], "torso": []}         # surface_z - fist_z for that fist
    reach = {"head": 0, "torso": 0}          # steps inside contact range
    both = 0
    hits = {"head": 0.0, "torso": 0.0, "body": 0.0}
    near_head_scored = 0                     # scoring events that happened at the head
    dist = []
    for _ in range(args.steps):
        action = forward(obs[..., :ckpt_obs].reshape(-1, ckpt_obs)).reshape(
            args.worlds, 2, env.action_size)
        obs, reward, done, parts = env.step(action)
        sensors = env._d.sensordata.numpy()
        frames = {side: env._frame(side, sensors) for side in scene.SIDES}
        for i, side in enumerate(scene.SIDES):
            me, foe = frames[side], frames[scene.SIDES[1 - i]]
            palms = np.stack([me["left_palm"], me["right_palm"]], axis=1)  # (w,2,3)
            for key, target in (("head", foe["head"]), ("torso", foe["torso"])):
                d = np.linalg.norm(palms - target[:, None, :], axis=2)
                best = np.argmin(d, axis=1)
                rows = np.arange(args.worlds)
                gap[key].append(d[rows, best])
                lift[key].append(target[rows, 2] - palms[rows, best, 2])
                reach[key] += int(np.count_nonzero(d.min(axis=1) <= CONTACT_RANGE))
            both += args.worlds
        for key, lo, hi in (("head", 0.999, 9.0), ("torso", 0.499, 0.999),
                            ("body", 0.001, 0.499)):
            s = parts["scored_by"]
            hits[key] += float(np.count_nonzero((s >= lo) & (s < hi)))
        near_head_scored += int(np.count_nonzero(s >= 0.999))
        dist.append(float(np.mean(parts["distance"])))

    gh = np.concatenate(gap["head"])
    gt = np.concatenate(gap["torso"])
    lh = np.concatenate(lift["head"])
    print(f"{step.name} trained to {trained:,} learner steps, "
          f"{'sampled' if args.stochastic else 'deterministic'}; "
          f"{args.worlds} worlds x {args.steps} steps, {both} corner-steps")
    print(f"  torso-torso distance: mean {np.mean(dist):.3f} m (punch_reach "
          f"{env._punch_reach:.2f}, target {env._target_range:.2f})")
    for key, arr, lifts in (("head", gh, lh), ("torso", gt, None)):
        pct = np.percentile(arr, [50, 95, 99])
        print(f"  fist-to-{key}: median {pct[0]:.3f} m, p95 {pct[1]:.3f}, "
              f"p99 {pct[2]:.3f}, CLOSEST EVER {arr.min():.3f} m; inside "
              f"{CONTACT_RANGE:.3f} m (touching range) on {reach[key] / both:.4f} "
              f"of corner-steps")
    print(f"  elevation the arm would need: head above the fist by median "
          f"{np.median(lh):+.3f} m (p95 {np.percentile(lh, 95):+.3f}); the torso "
          f"sits at {np.median(np.concatenate(lift['torso'])):+.3f} m")
    tot = sum(hits.values())
    print(f"  scoring events: head {hits['head']:.0f}, torso {hits['torso']:.0f}, "
          f"body {hits['body']:.0f} -> head is "
          f"{100 * hits['head'] / max(tot, 1):.2f}% of landed punches")
    inside, landed = reach["head"], sum(hits.values())
    print(f"  reading: the fist enters touching range of the head on {inside} of "
          f"{both} corner-steps and converts {hits['head']:.0f} of {landed:.0f} "
          f"scoring events there")
    if inside == 0:
        print("    -> REACH: at the distance the reward pays, the head is never in "
              "range; no incentive fixes that, the band or the pose has to change")
    elif hits["head"] == 0:
        print("    -> AIM: the elevation is available and unexploited -- the policy "
              "keeps its fists at torso height and never buys the +0.32 m")
    else:
        print("    -> neither: the head is reached and scored; the rate is the "
              "question, not the possibility")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
