"""What the articulated-hand robot would cost, measured before it is adopted.

The boxers render with open hands, and the obvious fix is to switch the robot
description from ``g1_mjx.xml`` to ``g1_with_hands.xml`` out of the same Menagerie
folder and lock the fingers into a fist. Before spending a day on that rewrite,
this measures what it charges:

* the hand variant ships at 2 ms with 100 solver iterations and has no MJX-tuned
  sibling, so it is run here under the *current* scene's settings -- 4 ms, 5
  iterations -- which is the only way the two costs are comparable, and also the
  optimistic case for the hand variant;
* its collision geoms are unnamed meshes with contype/conaffinity set, so two of
  them in one world generate cross-robot candidates by broadphase instead of by
  the 208 explicit pairs this scene declares;
* MJX cost scales with the contact buffer, so the buffer size is the number that
  decides whether a 200M-step arm still fits inside a day.

Measured on 2026-10-14, the second probe never got that far: ``mjx.put_model``
raises ``NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_MESH) collisions not
implemented`` on the hand variant, because its colliders are exactly that pair.
The objection to swapping is therefore not cost but feasibility -- the variant
cannot be simulated by this MJX at any speed without authoring new colliders,
which is the one thing this project's brief rules out. The tuned robot, for
reference, steps two robots in one world at ~20 steps/s.

  JAX_PLATFORMS=cuda,cpu python tools/hand_cost.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from boxer import scene

STEPS = 500


def two_handed_robots(spawn_distance: float = 1.3) -> mujoco.MjModel:
    """Two ``g1_with_hands`` robots under this scene's own solver settings."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.004
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 5
    spec.option.ls_iterations = 8
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    spec.compiler.degree = False
    spec.compiler.discardvisual = True

    floor = spec.worldbody.add_geom()
    floor.name = scene.FLOOR
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.contype = 0
    floor.conaffinity = 0

    hands = scene.robot_xml_path().parent / "g1_with_hands.xml"
    half = spawn_distance / 2.0
    for side, x, quat in (
        (scene.RED, -half, [1.0, 0.0, 0.0, 0.0]),
        (scene.BLUE, +half, [0.0, 0.0, 0.0, 1.0]),
    ):
        robot = mujoco.MjSpec.from_file(str(hands))
        for sensor in list(robot.sensors):
            robot.delete(sensor)
        frame = spec.worldbody.add_frame(pos=[x, 0.0, 0.0], quat=quat)
        spec.attach(robot, prefix=side, frame=frame)
    return spec.compile()


def probe(label: str, model: mujoco.MjModel) -> None:
    print(label)
    print(f"  ngeom={model.ngeom} npair={model.npair} nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"  timestep={model.opt.timestep} iterations={model.opt.iterations} "
          f"ls_iterations={model.opt.ls_iterations}")
    try:
        mx = mjx.put_model(model)
    except NotImplementedError as exc:
        # The answer, not a crash: MJX's collision driver has no cylinder-vs-mesh
        # function, and the articulated hand's colliders are exactly that pair. The
        # variant cannot be simulated here at any speed, so its cost per step is not
        # the objection to swapping to it -- its collider set is.
        print(f"  MJX refuses this model: {exc}")
        print("  no steps/s to measure; the swap would need new colliders first")
        return

    data = mjx.make_data(mx)
    data = data.replace(ctrl=jp.zeros(model.nu))
    data = mjx.forward(mx, data)

    step = jax.jit(lambda d: mjx.step(mx, d))
    jax.block_until_ready(step(data))

    started = time.perf_counter()
    state = data
    for _ in range(STEPS):
        state = step(state)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - started

    print(f"  contact buffer slots: {np.asarray(data.contact.geom).shape[0]}")
    print(f"  {STEPS / elapsed:8.1f} steps/s   ({elapsed:.2f} s for {STEPS} steps)")
    print(f"  qpos finite afterwards: {bool(np.all(np.isfinite(np.asarray(state.qpos))))}")


def main() -> None:
    probe("== the robot this project trains on: g1_mjx, 208 declared pairs ==",
          scene.build_model(visual=False))
    print()
    probe("== the articulated-hand robot: g1_with_hands, broadphase candidates ==",
          two_handed_robots())
    print("\nA 200M-step arm is 200e6 control steps; at ~19 steps/s of two-robot "
          "physics that is the budget every change to the robot has to fit in.")


if __name__ == "__main__":
    main()
