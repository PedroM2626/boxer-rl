"""Turn a BVH motion file into the same arm-geometry currency as the clip tools.

The imitation channel needs a human punch expressed in quantities that survive a
change of rig: elbow deflection from straight, upper-arm and forearm elevation
above the world horizontal, and the forearm's angle to the direction the fist
travels. ``tools/fbx_arm_dirs.py`` extracted those from a Mixamo FBX in Blender;
this reads BVH instead, because the motion reference now comes from the CMU
Graphics Lab database -- "The motion capture data may be copied, modified, or
redistributed without permission" (http://mocap.cs.cmu.edu/faqs.php) -- rather
than from Mixamo, whose Adobe General Terms section 3.6 forbid distributing
Content Files "on a stand-alone basis, outside of the End Use".

BVH is unambiguous where CMU's own Vicon-dialect ASF/AMC pair is not: a JOINT's
OFFSET is the vector to that joint in its parent's rotated frame, and CHANNELS
gives the rotation order literally. That mattered: the AMC reading left the
elbow deflection pinned at 90 deg through the whole trial -- the elbow channel
was spinning the forearm about its own axis -- which no recorded punch can do.

Joint naming follows the MotionBuilder convention the CMU conversions use: the
RightArm node sits at the shoulder, RightForeArm at the elbow, RightHand at the
wrist. Vertical axis and unit scale are measured off the reconstructed skeleton
rather than assumed, so the person comes out ~1.7 m tall whichever way the
exporter oriented it.

    python tools/bvh_to_dirs.py runs/reference/cmu/13_17.bvh out.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

DEG = np.pi / 180.0
# (segment-start joint, in the order upper arm / forearm / hand)
CHAINS = {
    "right": ("RightArm", "RightForeArm", "RightHand"),
    "left": ("LeftArm", "LeftForeArm", "LeftHand"),
}


class Joint:
    __slots__ = ("name", "offset", "channels", "children")

    def __init__(self, name):
        self.name, self.offset, self.channels, self.children = \
            name, np.zeros(3), [], []


def rot(axis, angle_deg):
    a, c, s = angle_deg * DEG, 0, 0
    c, s = np.cos(a), np.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def parse_block(lines, i):
    header = lines[i].strip().split()
    joint = Joint(header[-1])
    i += 1
    while lines[i].strip() != "{":
        i += 1
    i += 1
    while lines[i].strip() != "}":
        parts = lines[i].strip().split()
        key = parts[0].upper() if parts else ""
        if key == "OFFSET":
            joint.offset = np.array([float(v) for v in parts[1:4]])
        elif key == "CHANNELS":
            joint.channels = list(parts[2:2 + int(parts[1])])
        elif key in ("JOINT", "ROOT"):
            child, i = parse_block(lines, i)
            joint.children.append(child)
            continue
        elif parts and parts[0].capitalize() == "End":
            child = Joint(parts[1] if len(parts) > 1 else "site")
            i += 1
            while lines[i].strip() != "{":
                i += 1
            while lines[i].strip() != "}":
                sp = lines[i].strip().split()
                if sp and sp[0].upper() == "OFFSET":
                    child.offset = np.array([float(v) for v in sp[1:4]])
                i += 1
            joint.children.append(child)
        i += 1
    return joint, i + 1


def parse_bvh(path):
    lines = open(path).read().splitlines()
    # Start at the ROOT line, not at the "HIERARCHY" header: parsing from the
    # header swallows "ROOT Hips" and the real root name disappears.
    start = next(i for i, line in enumerate(lines)
                 if line.strip().upper().startswith("ROOT"))
    root, i = parse_block(lines, start)
    while i < len(lines) and lines[i].strip().upper() != "MOTION":
        i += 1
    frame_time = 1.0 / 120.0
    i += 1
    while ":" in lines[i] and not lines[i].strip().split(":")[0].strip().isdigit():
        parts = lines[i].strip().split()
        if parts[0].lower() == "frametime" or parts[0].lower() == "frame":
            frame_time = float(parts[-1])
        i += 1
    rows = [[float(v) for v in raw.split()] for raw in lines[i:] if raw.strip()]
    return root, frame_time, np.array(rows)


def order_of(joint, out=None):
    out = [] if out is None else out
    out.append(joint)
    for child in joint.children:
        order_of(child, out)
    return out


def positions(root, order, row):
    """World position of every joint for one motion row.

    The row's numbers are laid out in the file's pre-order DFS, so the values
    are sliced by walking that same sequence -- a stack walk would hand the
    elbow's rotations to the shoulder and the fist would appear to move at
    hundreds of metres per second.
    """
    values, cursor = {}, 0
    for joint in order:
        n = len(joint.channels)
        values[joint.name] = dict(zip(joint.channels, row[cursor:cursor + n]))
        cursor += n

    out = {}
    def walk(joint, parent_pos, parent_rot):
        v = values[joint.name]
        here = parent_pos + parent_rot @ joint.offset + np.array([
            v.get("Xposition", 0.0), v.get("Yposition", 0.0),
            v.get("Zposition", 0.0)])
        rotation = parent_rot
        for channel in joint.channels:
            if channel.endswith("rotation"):
                rotation = rotation @ rot(channel[0], v[channel])
        out[joint.name] = here
        for child in joint.children:
            walk(child, here, rotation)

    walk(root, np.zeros(3), np.eye(3))
    return out


def arm_quantities(upper, fore, punch):
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    elbow = np.degrees(np.arccos(np.clip(np.dot(unit(upper), unit(fore)), -1, 1)))
    elev = lambda v: np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
    align = np.degrees(np.arccos(np.clip(np.dot(unit(fore), punch), -1, 1)))
    return {"elbow_deg": round(float(elbow), 2),
            "fore_elev_deg": round(float(elev(fore)), 2),
            "upper_elev_deg": round(float(elev(upper)), 2),
            "fore_align_deg": round(float(align), 2)}


def pick(names, present):
    for name in names:
        if name in present:
            return name
    return None


def median_filter(values, width=3):
    pad = width // 2
    padded = np.concatenate([values[:1].repeat(pad), values, values[-1:].repeat(pad)])
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def fast_runs(smooth, fraction=0.7, min_len=6):
    """Contiguous runs at or above ``fraction`` of the smoothed peak speed."""
    threshold = fraction * float(smooth.max())
    runs, i = [], 0
    while i < len(smooth):
        if smooth[i] >= threshold:
            j = i
            while j + 1 < len(smooth) and smooth[j + 1] >= threshold:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def choose_strike(wrist, speed, elbow, shoulder, fps):
    """The straightest outward strike: a jab, not the fastest uppercut.

    Three measurements have to be taken on the OUTWARD path, from the start of
    the run to the frame where the fist is furthest from the shoulder. Net
    displacement between the run's edges is meaningless, because a run that ends
    with the fist back on the guard barely displaces at all -- and an unfiltered
    elevation ranking then rewards exactly those non-strokes. Ranking by speed
    alone picks the hook or uppercut, which is faster than a jab; the recorded
    reference this channel exists to encode travels flat (+1.25 deg measured over
    the strike in the Mixamo clip), so among runs that extend the arm the
    flattest wins.
    """
    smooth = median_filter(speed)
    scored = []
    for lo, hi in fast_runs(smooth, fraction=0.5, min_len=5):
        window = slice(lo, hi + 1)
        reach = np.linalg.norm(wrist[window] - shoulder[window], axis=1)
        far = lo + int(reach.argmax())
        if far - lo < 4:
            continue
        raw = wrist[far] - wrist[lo]
        horiz = float(np.linalg.norm(raw[:2]))
        rise = float(raw[2])
        elevation = float(np.degrees(np.arctan2(rise, horiz + 1e-9)))
        bend_at = lambda i: arm_quantities(elbow[i] - shoulder[i],
                                           wrist[i] - elbow[i], raw)["elbow_deg"]
        straightening = bend_at(lo) - bend_at(far)
        if straightening < 15.0:
            continue
        scored.append((abs(elevation), (lo, far), rise, elevation,
                       float(smooth[lo:far + 1].max()),
                       bend_at(lo), bend_at(far)))
    if not scored:
        return None
    scored.sort()
    _, (lo, far), rise, elevation, run_speed, bend_from, bend_to = scored[0]
    return ([int(lo), int(far)], rise, elevation,
            [round(s[3], 1) for s in scored[:8]], float(run_speed),
            bend_from, bend_to)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    bvh_path, out_path = argv[0], argv[1]
    root, frame_time, rows = parse_bvh(bvh_path)
    fps = 1.0 / frame_time
    sequence = order_of(root)
    print(f"{bvh_path.rsplit('/', 1)[-1]}: {len(rows)} frames at {fps:.1f} fps, "
          f"{len(sequence)} joints")

    every = [positions(root, sequence, row) for row in rows]
    names = set(every[0])
    hips_name = pick(["Hips", "hips", "Pelvis", "hip"], names)
    head_name = pick(["Head", "head"], names)
    hips = np.array([f[hips_name] for f in every])
    heads = np.array([f[head_name] for f in every])
    up = int(np.argmax(np.abs(np.median(heads - hips, axis=0))))
    perm = (np.eye(3) if up == 2 else
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) if up == 1 else
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    every = [{k: (perm @ v) for k, v in f.items()} for f in every]
    # Scale from the whole skeleton's vertical extent, not one bone pair: the
    # "Head" joint sits at the base of the skull, so a hip-to-head estimate
    # gave 0.117 m/unit and an upper arm of 0.72 m.
    cloud = np.stack([np.stack(list(f.values())) for f in every])
    heights = cloud[:, :, 2].max(axis=1) - cloud[:, :, 2].min(axis=1)
    scale = 1.70 / float(np.percentile(heights, 90))
    print(f"vertical axis index {up}; standing height {np.percentile(heights, 90):.1f} "
          f"units -> {scale:.5f} m/unit")

    # These conversions lead with a rest pose that the motion cuts out of: the
    # wrist moves 8 units in one frame while the median frame moves a tenth of
    # that, and no speed threshold survives that. Trim the discontinuous frames
    # before anything is measured on the series.
    jumps = np.abs(np.diff(cloud, axis=0)).max(axis=(1, 2))
    quiet = float(np.median(jumps))
    lead = 0
    while lead < len(jumps) and jumps[lead] > 5.0 * quiet:
        lead += 1
    if lead:
        print(f"trimmed {lead} rest-pose frame(s) before the motion starts")
        every, cloud, rows = every[lead:], cloud[lead:], rows[lead:]

    out = {"clip": bvh_path.rsplit("/", 1)[-1], "fps": int(round(fps)),
           "frames": len(rows), "source": "CMU mocap (free for all uses)",
           "sides": {}}
    for side, (shoulder_j, elbow_j, wrist_j) in CHAINS.items():
        if not {shoulder_j, elbow_j, wrist_j} <= names:
            continue
        shoulder = np.array([f[shoulder_j] for f in every]) * scale
        elbow = np.array([f[elbow_j] for f in every]) * scale
        wrist = np.array([f[wrist_j] for f in every]) * scale
        speed = np.linalg.norm(np.gradient(wrist, axis=0) * fps, axis=1)
        found = choose_strike(wrist, speed, elbow, shoulder, fps)
        if found is None:
            print(f"  {side}: no candidate strike -- no fast run long and bending "
                  f"enough to be a punch")
            continue
        window, rise, elevation, choices, run_speed, bend_from, bend_to = found
        peak = float(median_filter(speed).max())
        a, b = window
        raw = wrist[b] - wrist[a]
        punch = raw / (np.linalg.norm(raw) + 1e-12)
        rows_out = [arm_quantities(elbow[i] - shoulder[i], wrist[i] - elbow[i], punch)
                    for i in range(len(rows))]
        out["sides"][side] = {
            "punch_dir": [round(float(v), 4) for v in punch],
            "window_frames": window,
            "peak_speed_mps": round(run_speed, 3),
            "session_peak_mps": round(peak, 3),
            "rise_m": round(rise, 4),
            "elevation_deg": round(elevation, 2),
            "elbow_through_strike": [round(bend_from, 1), round(bend_to, 1)],
            "candidate_elevations_deg": choices,
            "rows": rows_out,
        }
        print(f"  {side}: strike {window} "
              f"({window[1] - window[0] + 1} frames, "
              f"{(window[1] - window[0]) / fps:.2f} s) at {run_speed:.1f} m/s "
              f"(session peak {peak:.1f}), rise {rise:+.3f} m, elevation "
              f"{elevation:+.1f} deg, elbow {bend_from:.0f}->{bend_to:.0f} deg, "
              f"candidate elevations {choices}")

    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
