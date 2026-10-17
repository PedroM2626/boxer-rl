"""What part of a recorded jab's arm geometry does the G1 joint box cover?

Reads the geometry JSON that ``tools/fbx_arm_dirs.py`` dumped from the Mixamo
clip (``runs/reference/arm_dirs.json``, out of git) and asks the MuJoCo model
directly: a grid of shoulder_pitch/roll/yaw + elbow values inside the model
ranges, body held in the natural stance, is forwarded on the host, and every
pose's elbow angle, forearm elevation and forearm angle to the facing axis are
computed exactly the way the clip's were -- world frame, angle-invariant. A
clip frame "retargets" when some grid pose brings all four quantities within
tolerance; the winning joint values are what an imitation channel would drive.

Torso rotation is deliberately not fitted: the grid holds the torso at the
stance, so a match means the arm alone can express the geometry, and a miss
points at the waist joints instead of guessing.

    PYTHONPATH=. python tools/retarget_probe.py [--grid 8] [--tolerance 15]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

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
    return [float(elbow), float(elev(fore)), float(elev(upper)), float(align)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--geometry", default="runs/reference/arm_dirs.json")
    ap.add_argument("--grid", type=int, default=8,
                    help="samples per joint; poses = grid**4")
    ap.add_argument("--tolerance", type=float, default=15.0,
                    help="degrees per quantity for a frame to count as retargeted")
    args = ap.parse_args(argv)

    with open(args.geometry) as handle:
        clip = json.load(handle)

    model = scene.build_model(spawn_distance=1.3, visual=False)
    base, _ctrl = scene.stance(model, "natural")

    names = [f"red/right_{j}_joint" for j in
             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
    infos = [joint_info(model, n) for n in names]
    adr = np.array([i[0] for i in infos])
    lo = np.array([i[1][0] for i in infos])
    hi = np.array([i[1][1] for i in infos])
    _, _, wrist_jid = joint_info(model, "red/right_wrist_roll_joint")
    # The robot's punch line: red faces blue along +x in the composed scene.
    punch = np.array([1.0, 0.0, 0.0])

    data = mujoco.MjData(model)
    axes = [np.linspace(lo[k], hi[k], args.grid) for k in range(4)]
    poses, quants = [], []
    for pitch in axes[0]:
        for roll in axes[1]:
            for yaw in axes[2]:
                for elbow in axes[3]:
                    q = base.copy()
                    q[adr[0]], q[adr[1]], q[adr[2]], q[adr[3]] = pitch, roll, yaw, elbow
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    shoulder = data.xanchor[infos[0][2]]
                    elbow_p = data.xanchor[infos[3][2]]
                    wrist = data.xanchor[wrist_jid]
                    poses.append((pitch, roll, yaw, elbow))
                    quants.append(arm_quantities(elbow_p - shoulder, wrist - elbow_p,
                                                 punch))
    poses = np.array(poses)
    quants = np.array(quants)
    print(f"G1 grid: {len(poses)} poses over the right arm's 4 joints; "
          f"torso held in the natural stance")
    for k, n in enumerate(names):
        print(f"  {n.split('/')[-1]:28s} range "
              f"[{np.degrees(lo[k]):7.1f}, {np.degrees(hi[k]):7.1f}] deg")

    elbow_col, fore_col, upper_col, align_col = quants.T
    forwardish = align_col <= 15.0
    print(f"  realized elbow angle      {elbow_col.min():6.1f}..{elbow_col.max():6.1f} deg")
    print(f"  realized forearm elevation {fore_col.min():6.1f}..{fore_col.max():6.1f} deg")
    print(f"  poses with forearm within 15 deg of the punch line: "
          f"{int(forwardish.sum())}")
    if forwardish.any():
        flat = forwardish & (np.abs(fore_col) <= 10.0)
        print(f"  ... and elevation within 10 deg of horizontal: {int(flat.sum())}")

    rows = clip["sides"]["right"]["rows"]
    window = clip["sides"]["right"]["window_frames"]
    clip_q = np.array([[r[q] for q in QUANTS] for r in rows])
    # (frames, poses, quants): per frame the smallest worst-quantity residual
    residual = np.abs(clip_q[:, None, :] - quants[None, :, :]).max(axis=2)
    best = residual.argmin(axis=1)
    misfit = residual[np.arange(len(rows)), best]

    print(f"\nclip: {clip['clip']}, right arm, frames 0..{len(rows) - 1}, "
          f"strike window {window}")
    fits = misfit <= args.tolerance
    print(f"  frames within {args.tolerance:.0f} deg on all four quantities: "
          f"{int(fits.sum())}/{len(rows)} overall")
    w = slice(window[0], window[1] + 1)
    print(f"  inside the strike window: {int(fits[w].sum())}/{window[1] - window[0] + 1}")

    print("\n  frame  elbow  fore_el  upper_el  align | worst-fit  | fitted pitch/roll/yaw/elbow (deg)")
    for i in list(range(w.start - 2, w.stop + 3)):
        joints = np.degrees(poses[best[i]])
        row = rows[i]
        print(f"  {i:5d} {row['elbow_deg']:6.1f} {row['fore_elev_deg']:8.2f} "
              f"{row['upper_elev_deg']:9.2f} {row['fore_align_deg']:6.1f} | "
              f"{misfit[i]:9.1f} | {joints.round(0)}")

    anchor = int(np.argmin([rows[i]["fore_align_deg"] for i in range(*window)])) + window[0]
    joints = np.degrees(poses[best[anchor]])
    print(f"\n  closest-to-contact frame {anchor}: clip "
          f"{ {q: rows[anchor][q] for q in QUANTS} }")
    print(f"  G1 fit joints (deg): pitch {joints[0]:.0f}, roll {joints[1]:.0f}, "
          f"yaw {joints[2]:.0f}, elbow {joints[3]:.0f}; worst residual "
          f"{misfit[anchor]:.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
