"""Re-fit the guard pose so the arms look like a guard and not a broken wrist.

The current GUARD_ARMS came out of a random search whose only objective was fist
*position* in the torso frame. The arm is a redundant chain, so many joint sets
reach the same point, and the cheapest one in position error keeps the elbow
nearly straight and cranks the wrist by ~90 degrees -- because the palm site is
offset from the wrist joint, so bending the wrist moves the site without moving
the arm. That is geometrically valid and visually wrong, and the `guard` reward
inherits it since it too only measures palm position.

This fit adds the two things the old one lacked: a prior pulling the arm toward
the upstream home keyframe, and a hard cap on the wrist joints. It prints the
resulting GUARD_ARMS dict to paste into boxer/scene.py.

    python -m tools.fit_guard
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxer import scene

#: Where each fist should sit in its own torso frame: forward, to the side, up.
#: The height is the whole argument. The first fit was handed +0.22 m (chin height)
#: with no wrist cap and met it by cranking both wrists to their +/-1.614 rad stops,
#: because the palm site is 8 cm down the forearm from the wrist yaw joint -- a broken
#: wrist moves the site without moving the arm. With the wrists capped at WRIST_LIMIT
#: the same search cannot close that height: it lands near +0.02 m and leaves ~8 cm of
#: z-residual behind, and that residual is the measurement of how high a
#: straight-wristed G1 fist can actually be held. This target is set from it.
TARGET_X, TARGET_Y, TARGET_Z = 0.28, 0.20, 0.10
TORSO_BODY = "torso_link"          # the frame the offset is expressed in
PALM_SITE = "{half}_palm"
WRIST_LIMIT = 0.35   # rad; a guard does not bend the wrist more than ~20 degrees
NATURALNESS = 0.03   # weight on staying near the home keyframe's arm angles
ITERATIONS = 60_000


def arm_indices(robot: mujoco.MjModel, half: str) -> np.ndarray:
    return np.array([
        robot.jnt_qposadr[mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT,
                                            f"{half}_{joint}_joint")]
        for joint in scene.ARM_JOINTS
    ])


def fist_offset(robot: mujoco.MjModel, home: np.ndarray, adr: np.ndarray,
                values: np.ndarray, half: str) -> np.ndarray:
    """Position of the palm site relative to the torso site, in the torso frame."""
    data = mujoco.MjData(robot)
    data.qpos[:] = home
    data.qpos[adr] = values
    mujoco.mj_forward(robot, data)
    torso = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
    palm = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE.format(half=half))
    if torso < 0 or palm < 0:
        raise SystemExit(f"name not found: torso={torso} palm={palm}")
    to_torso_frame = data.xmat[torso].reshape(3, 3).T
    return to_torso_frame @ (data.site_xpos[palm] - data.xpos[torso])


def target_for(half: str) -> np.ndarray:
    return np.array([TARGET_X, TARGET_Y if half == "left" else -TARGET_Y, TARGET_Z])


def fit_half(robot: mujoco.MjModel, home: np.ndarray, half: str, seed: int):
    adr = arm_indices(robot, half)
    home_arm = home[adr].copy()
    rng = np.random.default_rng(seed)
    wrist_mask = np.array([j.startswith("wrist") for j in scene.ARM_JOINTS])

    def cost(values: np.ndarray) -> float:
        offset = fist_offset(robot, home, adr, values, half)
        return float(np.sum((offset - target) ** 2)
                     + NATURALNESS * np.sum((values - home_arm) ** 2))

    target = target_for(half)
    best = home_arm.copy()
    best_cost = cost(best)
    scale = 0.35
    for step in range(ITERATIONS):
        if step and step % 8000 == 0:
            scale *= 0.5
        trial = best + rng.normal(0.0, scale, best.shape)
        trial[wrist_mask] = np.clip(trial[wrist_mask], -WRIST_LIMIT, WRIST_LIMIT)
        value = cost(trial)
        if value < best_cost:
            best, best_cost = trial, value
    return best, best_cost


def main() -> int:
    robot = mujoco.MjModel.from_xml_path(str(scene.robot_xml_path()))
    home, _ = scene.home_pose()
    out = {}
    print(f"target fist offset: ({TARGET_X:+.2f}, +-{TARGET_Y:.2f}, {TARGET_Z:+.2f}) m "
          "in the torso frame")
    print(f"wrist cap {WRIST_LIMIT} rad, naturalness weight {NATURALNESS}\n")
    for i, half in enumerate(("left", "right")):
        values, cost = fit_half(robot, home, half, seed=7 + i)
        offset = fist_offset(robot, home, arm_indices(robot, half), values, half)
        residual = float(np.linalg.norm(offset - target_for(half)))
        print(f"{half}: residual {residual * 100:.2f} cm  cost {cost:.4f}")
        print(f"   {', '.join(f'{j}={v:+.3f}' for j, v in zip(scene.ARM_JOINTS, values))}")
        out[half] = tuple(round(float(v), 3) for v in values)
    print("\nGUARD_ARMS = {")
    for half, values in out.items():
        print(f'    "{half}": {values},')
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
