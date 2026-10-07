"""Is the stance's centre of mass even inside the feet?

A humanoid that topples in 1.08 s with zero actions and no opponent in reach is
either badly damped or statically impossible, and only one of those can be fixed
by a controller. This asks the static question with plain MuJoCo -- no JAX, no
training -- by putting the model in each candidate pose and projecting the whole
body's centre of mass onto the ground against the support box the feet define.

    python -m tools.com_check
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxer import scene

MARGIN_CM = 0.02  # how far inside the support box the CoM must sit to count


def support_box(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Ground xy extents of every geom that touches the floor."""
    low, high = np.full(2, np.inf), np.full(2, -np.inf)
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "foot" not in name:
            continue
        pos, rot = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        extent = np.abs(rot[:2, :]) @ size  # conservative xy half-width
        low = np.minimum(low, pos[:2] - extent)
        high = np.maximum(high, pos[:2] + extent)
    return low, high


def report(model: mujoco.MjModel, qpos: np.ndarray, label: str) -> dict:
    data = mujoco.MjData(model)
    n = min(len(qpos), model.nq)
    data.qpos[:n] = qpos[:n]
    mujoco.mj_forward(model, data)
    com = data.subtree_com[0][:2]
    low, high = support_box(model, data)
    inside = (com >= low + MARGIN_CM) & (com <= high - MARGIN_CM)
    margin = float(np.min(np.where(com <= (low + high) / 2.0, com - low, high - com)))
    print(f"{label:22} com=({com[0]:+.3f},{com[1]:+.3f}) "
          f"box=[({low[0]:+.3f},{low[1]:+.3f}) ({high[0]:+.3f},{high[1]:+.3f})] "
          f"margin={margin:+.3f} {'INSIDE' if inside.all() else 'OUTSIDE'}", flush=True)
    return {"label": label, "com": com, "margin": margin, "inside": bool(inside.all())}


def guard_arms(robot: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Apply the fitted guard pose to the arm joints of a single-robot qpos.

    This is what ``scene.stance`` does for each boxer: legs and waist from the
    upstream home keyframe, arms from :data:`scene.GUARD_ARMS`.
    """
    out = qpos.copy()
    for half, values in scene.GUARD_ARMS.items():
        for joint, value in zip(scene.ARM_JOINTS, values):
            name = f"{half}_{joint}_joint"
            index = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
            out[robot.jnt_qposadr[index]] = value
    return out


def main() -> int:
    robot = mujoco.MjModel.from_xml_path(str(scene.robot_xml_path()))
    print(f"{robot.nq} qpos, {robot.nu} actuators\n", flush=True)

    home_qpos, _ = scene.home_pose()
    report(robot, robot.qpos0, "robot qpos0")
    report(robot, home_qpos, "robot home keyframe")
    report(robot, guard_arms(robot, home_qpos), "home + guard arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
