"""Dump a Mixamo boxing clip's fist trajectory, headless in Blender.

Why this exists: the fifth arm's video shows a fist that rises to strike, and the
question is whether that arc is the robot's fault or the reward's. A human jab clip
answers it in the same currency: if a recorded jab's fist also travels mostly
upward, the arc is what a punch looks like and the gate is fighting physics; if the
clip's fist travels flat while our G1's must climb 0.38 m, the ceiling is the
guard pose and the arm's envelope, and no reward writes its way past that.

Run headless (Blender 4.2, Windows side):

    blender -b --python tools/fbx_trace.py -- <clip.fbx> <out.json>

The clip stays out of git: runs/reference/ is ignored. The Mixamo terms were
checked on 2026-10-17 -- Adobe's General Terms section 3.6 allow using and
modifying Content Files inside a product you author but forbid distributing them
"on a stand-alone basis, outside of the End Use" -- so the published motion
reference for the imitation channel is CMU's instead (see LICENSES.md and
tools/cmu_fetch.sh). The flat-travel measurement below still stands.
"""
import json
import sys

import bpy
import numpy as np

WANTED = ("Hips", "Head", "RightHand", "LeftHand", "RightArm", "RightForeArm")
# Speed above which a frame counts as part of a strike, as a fraction of the
# hand's own peak speed in the clip: robust to scale and fps without thresholds
# written in metres by hand.
STRIKE_SPEED_FRACTION = 0.4


def _bone_world(arm, suffix):
    """World translation of the pose bone whose name ends in ``suffix``."""
    for bone in arm.pose.bones:
        if bone.name.split(":")[-1] == suffix:
            return (arm.matrix_world @ bone.matrix).translation
    return None


def _window_stats(positions, fps, mask):
    """Travel, height change and elevation of the resultant velocity in a window."""
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return None
    vel = np.gradient(positions, axis=0) * fps
    resultant = vel[idx].sum(axis=0)
    horizontal = float(np.linalg.norm(resultant[:2]))
    rise = float(resultant[2])
    travel = [float(np.linalg.norm(positions[i + 1] - positions[i])) for i in
              range(idx[0], idx[-1])]
    return {
        "frames": [int(idx[0]), int(idx[-1])],
        "seconds": [round(idx[0] / fps, 3), round(idx[-1] / fps, 3)],
        "path_m": round(sum(travel), 4),
        "net_displacement_m": round(
            float(np.linalg.norm(positions[idx[-1]] - positions[idx[0]])), 4),
        "rise_m": round(float(positions[idx[-1]][2] - positions[idx[0]][2]), 4),
        "resultant_speed_mps": round(
            float(np.linalg.norm(resultant)) / len(idx), 3),
        "elevation_deg": round(
            float(np.degrees(np.arctan2(rise, horizontal + 1e-9))), 2),
    }


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:]
    fbx_path, out_path = argv[0], argv[1]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    scene = bpy.context.scene
    fps = scene.render.fps

    bones = sorted(b.name for b in arm.pose.bones)
    fingerish = [b for b in bones if any(
        f in b for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"))]

    frames = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        row = {"frame": frame}
        for suffix in WANTED:
            pos = _bone_world(arm, suffix)
            if pos is not None:
                row[suffix] = [round(c, 5) for c in pos]
        frames.append(row)

    arrays = {s: np.array([f[s] for f in frames if s in f]) for s in WANTED}
    hips, head = arrays["Hips"], arrays["Head"]

    out = {
        "clip": fbx_path.rsplit("/", 1)[-1],
        "fps": int(fps),
        "frames": len(frames),
        "bones": len(bones),
        "finger_bones": len(fingerish),
        # Human scale in the clip, so the robot comparison is in proportions,
        # not in metres: the G1 numbers are fist +0.05 m and head +0.43 m above
        # torso centre.
        "head_above_hips_m": round(float(np.mean(head[:, 2] - hips[:, 2])), 4),
        "heights": {},
        "strikes": {},
    }
    for side in ("RightHand", "LeftHand"):
        pos = arrays[side]
        above = pos[:, 2] - hips[:, 2]
        out["heights"][side] = {
            "min_above_hips_m": round(float(above.min()), 4),
            "mean_above_hips_m": round(float(above.mean()), 4),
            "max_above_hips_m": round(float(above.max()), 4),
        }
        speed = np.linalg.norm(np.gradient(pos, axis=0) * fps, axis=1)
        peak = float(speed.max())
        mask = speed >= STRIKE_SPEED_FRACTION * peak
        out["strikes"][side] = {
            "peak_speed_mps": round(peak, 3),
            "strike_frames": int(mask.sum()),
        }
        out["strikes"][side]["fastest_window"] = _window_stats(pos, fps, mask)

    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=1)

    print(json.dumps({k: v for k, v in out.items() if k != "heights"}, indent=1))
    print("heights:", json.dumps(out["heights"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
