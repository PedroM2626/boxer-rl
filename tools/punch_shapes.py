"""What punches does a corner throw? Shapes of fist paths, whatever drives the arms.

Plays native ring bouts with ``spar_match`` and, for the red corner, follows
each fist in red's torso frame (x forward, y toward the fist's own side,
z up; the right fist mirrored). A punch is a stretch where the fist is more
than ``AWAY`` from both its normal-guard and high-guard positions. Each punch
is named by where its fist gets farthest from the guard (its peak),
compared with where the scripted moves (``boxer.moves``) put it, found by
forward kinematics at every aim from body to head:

* straight / hook: the peak is within ``NEAR`` of a scripted straight's
  path (guard to strike point; a punch stopped by contact ends on it) or a
  hook's swing (wind-up to strike point); a hook also sweeps sideways;
* new: a peak no scripted move reaches; subdivided by the peak's largest
  displacement from the guard (forward, side, up or down).

Each punch also keeps its forward reach, sideways sweep and rise relative to
the normal guard.

The same measure applies to scripted moves, arm-skill corrections and a
striker's free arm targets, so their repertoires can be compared.
"""
import argparse
import collections
import json
from pathlib import Path

import jax
import mujoco
import numpy as np

from boxer import moves
from boxer.arm_skill import ArmSkill
from boxer.curriculum_env import LEFT_ARM, UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.paired_foundation import PairedFoundation
from boxer.spar_match import bout, load_brain

AWAY = .12          # m from both guards
NEAR = .08          # m from a scripted strike point
MIN_FRAMES = 2      # frames are recorded every second control step (25 Hz)


def fist_paths(pair, frames, corner=0):
    """(frames, hand, 3) fist positions in ``corner``'s torso frame (right fist mirrored)."""
    data = mujoco.MjData(pair.model)
    site = pair._torso_sites[corner]
    out = np.zeros((len(frames), 2, 3))
    for f, qpos in enumerate(frames):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(pair.model, data)
        rot, origin = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]
        for hand in range(2):
            local = rot.T @ (data.geom_xpos[int(pair.fists[corner][hand])] - origin)
            out[f, hand] = local * ([1, -1, 1] if hand else 1)
    return out


def guard_fists(pair, qpos, corner=0):
    """(guard, hand, 3) fist positions at the normal and high guard, same frame as ``fist_paths``."""
    q = pair.maps[corner]['q']
    guards = []
    for high in (False, True):
        frame = qpos.copy()
        for hand in range(2):
            start = LEFT_ARM + 7 * hand
            rest = moves.rest_pose(np, moves.mirror(np, pair.pose[start:start + 4], hand), high)
            frame[q[start:start + 4]] = moves.mirror(np, rest, hand)
        guards.append(fist_paths(pair, [frame], corner)[0])
    return np.stack(guards)


def strike_points(pair, qpos, corner=0):
    """{move: (start, end)} left-fist segments of the scripted moves at 11 aims, frame as above.

    Straights run from the guard to the strike point, hooks from the wind-up.
    """
    q = pair.maps[corner]['q'][LEFT_ARM:LEFT_ARM + 4]

    def fist(pose):
        frame = qpos.copy()
        frame[q] = pose
        return fist_paths(pair, [frame], corner)[0, 0]
    aims = np.linspace(0, 1, 11)
    guard = fist(pair.pose[LEFT_ARM:LEFT_ARM + 4])
    return dict(
        straight=(np.repeat(guard[None], len(aims), 0),
                  np.array([fist(moves.strike_pose(np, moves.STRAIGHT, a)) for a in aims])),
        hook=(np.array([fist(moves.WINDUP_BODY + a * (moves.WINDUP_HEAD - moves.WINDUP_BODY)) for a in aims]),
              np.array([fist(moves.strike_pose(np, moves.HOOK, a)) for a in aims])))


def segment_distance(point, start, end):
    """Distance from ``point`` to the nearest of the segments ``start[k]``-``end[k]``."""
    span = end - start
    t = np.clip(((point - start) * span).sum(1) / np.maximum((span ** 2).sum(1), 1e-12), 0, 1)
    return float(np.linalg.norm(start + t[:, None] * span - point, axis=1).min())


def punches(path, guards, points):
    """Descriptors of every punch in one hand's (frames, 3) path."""
    away = np.linalg.norm(path[:, None] - guards[None], axis=-1).min(axis=1) > AWAY
    found, start = [], None
    for f, flag in enumerate(np.append(away, False)):
        if flag and start is None:
            start = f
        elif not flag and start is not None:
            if f - start >= MIN_FRAMES:
                stretch = path[start:f]
                guard = guards[0]
                peak = stretch[np.argmax(np.linalg.norm(stretch - guard, axis=1))]
                reach = float(stretch[:, 0].max() - guard[0])
                sweep = float(stretch[:, 1].max() - stretch[:, 1].min())
                rise = float(stretch[:, 2].max() - guard[2])
                found.append(dict(reach=reach, sweep=sweep, rise=rise, frames=f - start,
                                  peak=(peak - guard).round(3).tolist(),
                                  kind=classify(peak - guard, peak, sweep, points)))
            start = None
    return found


def classify(displacement, peak, sweep, points):
    near = {name: segment_distance(peak, *ends) for name, ends in points.items()}
    if near["hook"] < NEAR and (sweep > .15 or near["straight"] >= NEAR):
        return "hook"
    if near["straight"] < NEAR:
        return "straight"
    axis = int(np.argmax(np.abs(displacement)))
    return "new " + (("forward", "back"), ("side", "inside"), ("up", "down"))[axis][int(displacement[axis] < 0)]


def summarize(found):
    kinds = collections.Counter(p["kind"] for p in found)
    total = max(len(found), 1)
    stats = {k: dict(mean=round(float(np.mean([p[k] for p in found])), 3),
                     p90=round(float(np.quantile([p[k] for p in found], .9)), 3))
             for k in ("reach", "sweep", "rise")} if found else {}
    return dict(punches=len(found), share={k: round(v / total, 3) for k, v in sorted(kinds.items())},
                **stats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/spar"))
    parser.add_argument("--red-tactics", type=Path, required=True)
    parser.add_argument("--red-arm", type=Path)
    parser.add_argument("--blue-tactics", type=Path, default=Path("checkpoints/tactics"))
    parser.add_argument("--blue-arm", type=Path, default=Path("checkpoints/arm-skill"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(4)))
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    foundation = UnassistedG1("spar")
    motor = jax.jit(inference(foundation, load_policy_params(args.checkpoint)[0]))
    pair = PairedFoundation(foundation, ring=True)
    skills = {i: ArmSkill(load_policy_params(p)[0]) for i, p in enumerate((args.red_arm, args.blue_arm)) if p}
    found = []
    for seed in args.seeds:
        brains = {0: load_brain(args.red_tactics), 1: load_brain(args.blue_tactics)}
        result, frames, _ = bout(pair, motor, seed, args.seconds, True, brains, 3, skills)
        if not frames:
            continue
        guards = guard_fists(pair, frames[0])
        points = strike_points(pair, frames[0])
        paths = fist_paths(pair, frames)
        for hand in range(2):
            found += punches(paths[:, hand], guards[:, hand], points)
        print(json.dumps(dict(seed=seed, clean=result["clean_punches"], knockdowns=result["knockdowns_suffered"])),
              flush=True)
    summary = dict(red=str(args.red_tactics), red_arm=str(args.red_arm) if args.red_arm else None,
                   **summarize(found), detail=found)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "detail"}), flush=True)


if __name__ == "__main__":
    main()
