"""Why does a boxer with zero actions fall over, and how long does it take?

Measured: the guard stance topples in 54 control steps -- 1.08 s -- with no
opponent anywhere near it. Every number in the arena is downstream of that, so
this script isolates the candidates instead of guessing: the fitted arm pose
shifts the centre of mass, and the scene disables Euler damping, which is the
implicit joint damping that keeps a position-controlled humanoid from oscillating
into a fall.

    python -m tools.stand_test
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boxer import env as env_module, scene
from boxer.env import NUM_AGENTS, BoxingEnv


def home_arm_targets():
    """The upstream home keyframe's arm joints, in GUARD_ARMS' own layout."""
    home_qpos, _ = scene.home_pose()
    robot = mujoco.MjModel.from_xml_path(str(scene.robot_xml_path()))
    targets = {}
    for half in ("left", "right"):
        values = []
        for joint in scene.ARM_JOINTS:
            index = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, f"{half}_{joint}_joint")
            values.append(float(home_qpos[robot.jnt_qposadr[index]]))
        targets[half] = tuple(values)
    return targets


def steps_until_down(env: BoxingEnv, limit: int = 400) -> int | None:
    state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    zero = jp.zeros((NUM_AGENTS, env.action_size))
    for t in range(limit):
        state = step(state, zero)
        if float(jp.asarray(state.metrics["down_red"])) > 0.0:
            return t + 1
    return None


def build(arms: str, euler_damping: bool) -> BoxingEnv:
    """One world, both boxers 3 m apart, so nothing but posture can make it fall."""
    original_arms, original_build = scene.GUARD_ARMS, scene.build_model
    scene.GUARD_ARMS = home_arm_targets() if arms == "home" else original_arms
    if euler_damping:
        def patched(**kwargs):
            model = original_build(**kwargs)
            model.opt.disableflags &= ~mujoco.mjtDisableBit.mjDSBL_EULERDAMP
            return model
        scene.build_model = patched
    try:
        return BoxingEnv(spawn_distance=3.0, spawn_jitter=0.0, yaw_jitter=0.0,
                         pose_noise=0.0, action_scale=0.25)
    finally:
        scene.GUARD_ARMS, scene.build_model = original_arms, original_build


def main() -> int:
    print(f"{'arms':6} {'euler damping':14} steps until down", flush=True)
    for arms in ("guard", "home"):
        for damping in (False, True):
            try:
                stood = steps_until_down(build(arms, damping))
            except Exception as error:  # a variant that cannot even load is a result
                print(f"{arms:6} {str(damping):14} ERROR {type(error).__name__} {error}",
                      flush=True)
                continue
            print(f"{arms:6} {str(damping):14} {stood if stood else '>400 (stands)'}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
