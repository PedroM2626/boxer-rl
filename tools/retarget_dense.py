"""Dense frame-by-frame retarget of the recorded jab onto the G1's arm joints.

``tools/retarget_probe.py`` settled VIABILITY with a coarse 8^4 grid: 249/250
clip frames fit inside 15 deg on all four rig-invariant quantities. This turns
that existence proof into a trajectory: per clip frame, a bounded least-squares
over (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow) minimises the same
four-quantity residual, started from the grid's best pose and re-started from
the previous refined frame so the result is temporally continuous. The torso
and the wrists stay at the natural stance -- a fit means the arm alone carries
the motion, the same rule that made the probe honest.

Writes ``runs/reference/jab_targets.json`` (out of git with the clip: the
Mixamo terms are not cleared for redistribution) and prints the residual
before/after plus the joint excursions an imitation channel would have to
track.

    PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python tools/retarget_dense.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.optimize import least_squares

from boxer import scene

QUANTS = ("elbow_deg", "fore_elev_deg", "upper_elev_deg", "fore_align_deg")


def joint_info(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if bool(model.jnt_limited[jid]):
        rng = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
    else:
        rng = (-2.5, 2.5)
    return int(model.jnt_qposadr[jid]), rng, jid


def arm_quantities(upper, fore, punch):
    unit = lambda v: v / np.linalg.norm(v)
    elbow = np.degrees(np.arccos(np.clip(np.dot(unit(upper), unit(fore)), -1.0, 1.0)))
    elev = lambda v: np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
    align = np.degrees(np.arccos(np.clip(np.dot(unit(fore), punch), -1.0, 1.0)))
    return np.array([elbow, elev(fore), elev(upper), align])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--geometry", default="runs/reference/arm_dirs.json")
    ap.add_argument("--out", default="runs/reference/jab_targets.json")
    ap.add_argument("--grid", type=int, default=12,
                    help="coarse seeds per joint before the least-squares")
    args = ap.parse_args(argv)

    with open(args.geometry) as handle:
        clip = json.load(handle)
    side = "right"
    rows = clip["sides"][side]["rows"]
    window = clip["sides"][side]["window_frames"]
    clip_q = np.array([[r[q] for q in QUANTS] for r in rows])

    model = scene.build_model(spawn_distance=1.3, visual=False)
    base, _ctrl = scene.stance(model, "natural")
    names = [f"red/{side}_{j}_joint" for j in
             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
    infos = [joint_info(model, n) for n in names]
    adr = np.array([i[0] for i in infos])
    lo = np.array([i[1][0] for i in infos])
    hi = np.array([i[1][1] for i in infos])
    _, _, wrist_jid = joint_info(model, f"red/{side}_wrist_roll_joint")
    punch = np.array([1.0, 0.0, 0.0])  # red faces blue along +x
    data = mujoco.MjData(model)

    def quantities(q):
        data.qpos[:] = base
        data.qpos[adr] = q
        mujoco.mj_forward(model, data)
        shoulder = data.xanchor[infos[0][2]]
        elbow_p = data.xanchor[infos[3][2]]
        wrist = data.xanchor[wrist_jid]
        return arm_quantities(elbow_p - shoulder, wrist - elbow_p, punch)

    # Coarse seeds: the grid the probe already validated, this time denser.
    # Quantities are frame-independent, so evaluate the grid once.
    axes = [np.linspace(lo[k], hi[k], args.grid) for k in range(4)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(4, -1).T
    grid_quant = np.array([quantities(q) for q in grid])
    best_q, worst0 = np.zeros((len(rows), 4)), np.zeros(len(rows))
    for i in range(len(rows)):
        res = np.abs(grid_quant - clip_q[i]).max(axis=1)
        j = int(res.argmin())
        best_q[i], worst0[i] = grid[j], float(res[j])

    # Refine with temporal coherence: 4 joints against 4 quantities is an
    # exact-determined inverse problem, and the solver hops between IK branches
    # (elbow-up/elbow-down) every frame -- measured jumps of 50-93 deg/frame.
    # Seeding from the previous refined frame and paying a small continuity
    # penalty keeps one branch; the fit loosens from exact to whatever the
    # branch can still express.
    eps = np.radians(0.5)
    w_continuity = 0.15

    def residual_cont(q, target, prev):
        return np.concatenate([quantities(q) - target,
                               w_continuity * np.degrees(q - prev)])

    refined, worst1 = best_q.copy(), worst0.copy()
    for i in range(len(rows)):
        seeds = [refined[i - 1]] if i else [best_q[i]]
        seeds.append(best_q[i])
        best = None
        for s in seeds:
            try:
                sol = least_squares(
                    residual_cont, np.clip(s, lo + eps, hi - eps),
                    bounds=(lo + eps, hi - eps),
                    args=(clip_q[i], seeds[0]), xtol=1e-10, ftol=1e-10)
            except Exception:
                continue
            w = float(np.abs(quantities(sol.x) - clip_q[i]).max())
            if best is None or w < best[0]:
                best = (w, sol.x)
        if best is not None:
            refined[i], worst1[i] = best[1], best[0]

    speed = np.abs(np.diff(refined, axis=0)).max(axis=0)
    print(f"{len(rows)} frames refined; worst-quantity residual: "
          f"grid {np.mean(worst0):.1f} deg mean / {np.median(worst0):.1f} median "
          f"-> refined {np.mean(worst1):.1f} / {np.median(worst1):.1f}")
    print(f"  frames within 2 deg after refine: {int((worst1 <= 2).sum())}/{len(rows)}")
    print(f"  max per-frame joint jump: {np.degrees(speed).round(1)} deg/frame "
          f"(clip at {clip['fps']} fps)")
    print("  excursion (deg): " + ", ".join(
        f"{n.split('/')[-1].replace('_joint', '')}"
        f"[{np.degrees(refined[:, k].min()):.0f},"
        f"{np.degrees(refined[:, k].max()):.0f}]"
        for k, n in enumerate(names)))
    w = slice(window[0], window[1] + 1)
    print(f"  strike window {window}: refined residual max "
          f"{worst1[w].max():.1f} deg")

    out = {
        "clip": clip["clip"], "side": side, "fps": clip["fps"],
        "joints": names, "strike_window": window,
        "targets_radians": [t.round(4).tolist() for t in refined],
        "worst_residual_deg": worst1.round(2).tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"wrote {args.out}")

    print("\n  frame  clip(elbow,fore_elev,upper_elev,align)  "
          "fit(elbow,fore_elev,upper_elev,align) | pitch roll yaw elbow (deg)")
    for i in list(range(window[0] - 1, window[1] + 2)):
        got = quantities(refined[i]).round(1)
        j = np.degrees(refined[i]).round(0)
        print(f"  {i:5d}  {clip_q[i].round(1)}  {got} | {j}  worst {worst1[i]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
