"""Extract a Mixamo clip's arm geometry per frame, headless in Blender.

The imitation fork -- train the G1 with a public clip -- is only real if the
recorded jab's arm geometry is expressible by the G1's shoulder/elbow joint
box. Rig differences kill most comparisons, so this dumps only quantities that
survive them: the elbow's interior angle (frame-invariant), and the upper arm
and forearm directions measured against the world horizontal and the punch
line, never against a rig's local axes. ``tools/retarget_probe.py`` reads the
JSON and asks the G1 model what part of that geometry its joint ranges cover.

Run headless (Blender 4.2, Windows side):

    blender -b --python tools/fbx_arm_dirs.py -- <clip.fbx> <trace.json> <out.json>

``trace.json`` is the output of ``tools/fbx_trace.py`` and supplies the strike
window frames. The clip stays out of git: runs/reference/ is ignored, and the
Mixamo terms have not been checked for redistribution.
"""
import json
import sys

import bpy
import numpy as np

SIDES = {
    "right": ("RightArm", "RightForeArm", "RightHand"),
    "left": ("LeftArm", "LeftForeArm", "LeftHand"),
}


def _bone_world(arm, suffix):
    """World position of the pose bone's origin (its joint) whose name ends in
    ``suffix`` -- Mixamo bones run joint to joint, so the origin is the joint."""
    for bone in arm.pose.bones:
        if bone.name.split(":")[-1] == suffix:
            return np.array((arm.matrix_world @ bone.matrix).translation)
    return None


def _row(shoulder, elbow, wrist, punch):
    upper = elbow - shoulder
    fore = wrist - elbow
    unit = lambda v: v / np.linalg.norm(v)
    elbow_deg = np.degrees(np.arccos(np.clip(
        np.dot(unit(upper), unit(fore)), -1.0, 1.0)))
    elevation = lambda v: np.degrees(np.arctan2(
        v[2], np.hypot(v[0], v[1])))
    return {
        "elbow_deg": round(float(elbow_deg), 2),
        "fore_elev_deg": round(float(elevation(fore)), 2),
        "upper_elev_deg": round(float(elevation(upper)), 2),
        "fore_align_deg": round(float(np.degrees(np.arccos(np.clip(
            np.dot(unit(fore), punch), -1.0, 1.0)))), 2),
    }


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:]
    fbx_path, trace_path, out_path = argv[0], argv[1], argv[2]
    with open(trace_path) as handle:
        trace = json.load(handle)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    scene = bpy.context.scene
    fps = scene.render.fps

    frames = list(range(scene.frame_start, scene.frame_end + 1))
    joints = {side: {role: [] for role in ("shoulder", "elbow", "wrist")}
              for side in SIDES}
    for frame in frames:
        scene.frame_set(frame)
        for side, (shoulder_bone, elbow_bone, hand_bone) in SIDES.items():
            joints[side]["shoulder"].append(_bone_world(arm, shoulder_bone))
            joints[side]["elbow"].append(_bone_world(arm, elbow_bone))
            joints[side]["wrist"].append(_bone_world(arm, hand_bone))

    out = {"clip": fbx_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
           "fps": int(fps), "frames": len(frames), "sides": {}}
    for side in SIDES:
        wrist = np.array(joints[side]["wrist"])
        first, last = trace["strikes"][f"{side.capitalize()}Hand"][
            "fastest_window"]["frames"]
        punch = wrist[last] - wrist[first]
        punch = punch / np.linalg.norm(punch)
        rows = [_row(joints[side]["shoulder"][i], joints[side]["elbow"][i],
                     wrist[i], punch) for i in range(len(frames))]
        window = [r for r in rows[first:last + 1]]
        out["sides"][side] = {
            "punch_dir": [round(c, 4) for c in punch],
            "window_frames": [first, last],
            "rows": rows,
            "window": {
                "elbow_deg": [window[0]["elbow_deg"], window[-1]["elbow_deg"],
                              round(min(r["elbow_deg"] for r in window), 2)],
                "mean_fore_elev_deg": round(
                    float(np.mean([r["fore_elev_deg"] for r in window])), 2),
                "mean_fore_align_deg": round(
                    float(np.mean([r["fore_align_deg"] for r in window])), 2),
            },
        }

    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=1)

    for side, data in out["sides"].items():
        print(side, "window", data["window_frames"], data["window"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
