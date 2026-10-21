"""Can this arm score a head hit at all, under the gate the env actually uses?

The head has been measured three ways and every one of them said the same thing:
warp6 landed 0.0025 head hits a round, warp7 landed 0.0000 after the head's damage
weight was doubled, and `tools/warp_head_gap.py` found the fist inside touching
range of the head once in 122,880 corner-steps, sitting a median 0.319 m BELOW it.
Two readings survive that: the elevation is available and the policy never buys it
(aim), or the elevation is not available to this robot at this range (a wall). The
README's open experiment is the second one, and no rollout can answer it -- someone
has to try every pose on purpose.

`tools/reach_sweep.py` swept the same question statically on the MJX env and is the
floor for it: a pose that fails there may still land, because a real punch carries
torso rotation and translation on top of the arm. This is the dynamic version, on
the physics the arms actually train in: red's four left-arm joints swept over their
model ranges, one pose per world, driven from the stance to the target fast enough
that the palm closes at the speed the scorecard asks for, with both of blue's arms
at their zero pose so no forearm blocks the line. A hit counts only as the env
counts it -- force over the threshold on a head geom, the striker's palm closing at
`punch_speed_min`, that fist armed.

    PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python tools/warp_head_sweep.py \
        --distances 0.45 0.52 0.60

If some pose scores on the head, the head is a target the policy is not choosing and
the objective may keep paying double for it. If none does, the 2x weight is
decoration and what has to change is the geometry: the guard envelope, the distance
band, or what counts as the head. A grid is not a proof -- it says what this sweep
found, at this resolution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxer import scene  # noqa: E402
from boxer.warp_env import WarpBoxing  # noqa: E402

ARM = ("left_shoulder_pitch_joint", "left_shoulder_roll_joint",
       "left_shoulder_yaw_joint", "left_elbow_joint")
# The scorecard's bands on one hit's value: head 1.0, torso 0.5, body 0.35. A fist
# capsule (radius 0.035) touching the head sphere (radius 0.06) has its centre
# 0.095 m from the head centre, which is what "in touching range" means here.
HEAD, TORSO, TOUCH = 0.999, 0.499, 0.095


def arm_columns(env, side, names):
    """(action column, actuator id, joint id) per named joint of one side."""
    ids = np.asarray(env._sides[side]["actuators"])
    out = []
    for name in names:
        aid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, side + name)
        jid = int(env._model.actuator_trnid[aid, 0])
        out.append((int(np.flatnonzero(ids == aid)[0]), aid, jid))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--distances", type=float, nargs="+", default=[0.45, 0.52, 0.60],
                    help="torso-to-torso separation to try; ~0.52 m is where the arms "
                         "actually fight and 0.45 m is the env self-check's jab distance")
    ap.add_argument("--samples", type=int, nargs=4, default=[7, 6, 6, 5],
                    help="grid points per arm joint (pitch, roll, yaw, elbow)")
    ap.add_argument("--ramp", type=int, default=8,
                    help="control steps from stance to target; this sets the closing "
                         "speed, and the gate wants >= punch_speed_min")
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--settle", type=int, default=6,
                    help="steps at the stance before the swing, so the sensors and the "
                         "scorecard's arm window are both settled")
    ap.add_argument("--guard-pose", default="natural")
    args = ap.parse_args(argv)

    grid = np.stack(np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in args.samples],
                                indexing="ij"), axis=-1).reshape(-1, 4)
    nworld = grid.shape[0]
    print(f"{nworld} poses per distance, ramp {args.ramp} steps "
          f"({args.ramp * 0.02:.2f} s), hold {args.hold}")

    for distance in args.distances:
        env = WarpBoxing(nworld=nworld, spawn_distance=distance, spawn_jitter=0.0,
                         yaw_jitter=0.0, pose_noise=0.0, guard_pose=args.guard_pose,
                         round_steps=10_000)
        env.reset(0)
        model, ctrl0 = env._model, np.asarray(env._ctrl0)
        red = arm_columns(env, scene.RED, ARM)
        blue = arm_columns(env, scene.BLUE,
                           [j.replace("left", "right") for j in ARM]
                           + [j for j in ARM])
        act = np.array([aid for _c, aid, _j in red])
        start = ctrl0[act]                                   # stance ctrl per joint
        lo = np.array([model.jnt_range[jid][0] if model.jnt_limited[jid]
                       else start[k] - 1.0 for k, (_c, _a, jid) in enumerate(red)])
        hi = np.array([model.jnt_range[jid][1] if model.jnt_limited[jid]
                       else start[k] + 1.0 for k, (_c, _a, jid) in enumerate(red)])
        targets = lo[None, :] + (hi - lo)[None, :] * grid

        action = np.zeros((nworld, 2, env.action_size), np.float32)
        for col, aid, _jid in blue:
            action[:, 1, col] = (0.0 - ctrl0[aid]) / env._action_scale

        best = np.zeros(nworld)
        raw_best = np.zeros(nworld)
        head_best = np.zeros(nworld)
        head_raw = np.zeros(nworld)
        closing_max = np.zeros(nworld)
        fist_head = np.full(nworld, np.inf)
        for t in range(args.settle + args.ramp + args.hold):
            if t < args.settle:
                frac = 0.0
            elif t < args.settle + args.ramp:
                frac = (t - args.settle + 1) / args.ramp
            else:
                frac = 1.0
            pose = start[None, :] + frac * (targets - start[None, :])
            for k, (col, aid, _jid) in enumerate(red):
                action[:, 0, col] = (pose[:, k] - ctrl0[aid]) / env._action_scale
            _obs, _reward, _done, parts = env.step(action)

            sensors = env._d.sensordata.numpy()
            me, foe = env._frame(scene.RED, sensors), env._frame(scene.BLUE, sensors)
            line = foe["torso"] - me["torso"]
            line /= np.linalg.norm(line, axis=1, keepdims=True) + 1e-9
            closing_max = np.maximum(
                closing_max,
                np.maximum(np.sum(me["left_palm_vel"] * line, axis=1),
                           np.sum(me["right_palm_vel"] * line, axis=1)))
            fist_head = np.minimum(fist_head,
                                   np.linalg.norm(me["left_palm"] - foe["head"], axis=1))
            scored = parts["scored_by"][:, 0]
            raw = parts["hit"]          # red's value before the arming gate
            best = np.maximum(best, scored)
            raw_best = np.maximum(raw_best, raw)
            head_best = np.maximum(head_best, np.where(scored >= HEAD, scored, 0.0))
            head_raw = np.maximum(head_raw, np.where(raw >= HEAD, raw, 0.0))

        gate = env._punch_speed_min
        credited = int(np.count_nonzero(head_best > 0.0))
        raw_head = int(np.count_nonzero(head_raw > 0.0))
        print(f"\nspawn_distance {distance:.2f} m, {nworld} poses:")
        print(f"  credited (armed) hits: any {int(np.count_nonzero(best > 0.0))}, "
              f"head {credited}, torso "
              f"{int(np.count_nonzero((best >= TORSO) & (best < HEAD)))}, body "
              f"{int(np.count_nonzero((best > 0.0) & (best < TORSO)))}")
        print(f"  raw scorecard hits (before the arming gate): any "
              f"{int(np.count_nonzero(raw_best > 0.0))}, head {raw_head}, best value "
              f"on any surface {raw_best.max():.2f}")
        print(f"  fist inside head-touching range ({TOUCH:.3f} m, site to site) during "
              f"a swing: {int(np.count_nonzero(fist_head <= TOUCH))} poses; closest "
              f"approach anywhere {fist_head.min():.3f} m")
        print(f"  closing speed: best {closing_max.max():.2f} m/s against the "
              f"{gate:.2f} gate; {int(np.count_nonzero(closing_max >= gate))} poses "
              f"swung fast enough to be a punch")
        if credited:
            k = int(np.argmax(head_best))
            print(f"  -> THE HEAD IS SCORABLE: value {head_best.max():.2f} at "
                  f"pitch {targets[k, 0]:+.2f} roll {targets[k, 1]:+.2f} "
                  f"yaw {targets[k, 2]:+.2f} elbow {targets[k, 3]:+.2f} rad, "
                  f"closing {closing_max[k]:.2f} m/s")
        elif raw_head:
            print(f"  -> the head takes a threshold contact in {raw_head} poses and "
                  "every one of them is denied credit, so the wall is the gate, not "
                  "the arm")
        else:
            fast = int(np.count_nonzero(closing_max >= gate))
            print("  -> NO POSE PUT 50 N ON THE HEAD. The palm gets to its touching "
                  f"distance ({fist_head.min():.3f} m) and closes at up to "
                  f"{closing_max.max():.1f} m/s, and the head geom still never takes a "
                  "scoring contact -- so the elevation is reachable and the head is "
                  "not, which is a geometry answer, not an incentive one"
                  + ("" if fast else "; and no pose closed fast enough, so lower "
                     "--ramp before trusting this"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
