"""Can a real fist still land a scored punch now that the arms are worth nothing?

`test_a_moving_fist_that_connects_scores` used to drive a scripted jabber at a
teleported opponent and ask for one counted punch, and it went to zero when the
arms went to zero. ``tools/jab_probe.py`` showed why: against a guard-holding
opponent every cross-corner contact the jabber makes is wrist-on-elbow or
hand-on-hand, so under the new scorecard it is contact without a score, and closer
than 0.75 m the two are in contact on 31 of 40 steps so the rearm window never
reopens either.

That leaves a question the suite should not answer by fabrication: is there still
a real physical punch in this environment that scores? This tries the obvious one
-- the same jabber, against an opponent commanded to hold its arms down instead of
its guard, so the jab arrives on the torso rather than on a glove -- and prints,
per control step, what crossed the margin by geom name and whether it counted.

    JAX_PLATFORMS=cuda,cpu python tools/fist_probe.py [--gap 0.5] [--steps 80]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp
import mujoco
import numpy as np

from boxer import policy as policy_module, scene
from boxer.env import BoxingEnv, NUM_AGENTS


def arms_down(level: BoxingEnv, side: str) -> jax.Array:
    """The action that holds one corner's arms at zero instead of in the guard.

    Zero action holds the *stance*, and the stance is the guard, so a passive
    opponent covers its own torso with its forearms. The action is the inverse of
    the env's own mapping: ``ctrl = stance + action * span``.
    """
    base = 0 if side == scene.RED else level.action_size
    span = (level._ctrl_hi - level._ctrl_lo) / 2.0 * level._action_scale
    action = jp.zeros(level.action_size)
    for half in ("left", "right"):
        for joint in scene.ARM_JOINTS:
            actuator = level.actuator_index(f"{side}{half}_{joint}_joint")
            wanted = (0.0 - level._stance_ctrl[actuator]) / span[actuator]
            action = action.at[actuator - base].set(float(jp.clip(wanted, -1.0, 1.0)))
    return action


POSES = {
    # What the red corner's lead arm is doing when the corners are moved together,
    # and how fast the shoulder is driving it. The guard is the stance; "straight"
    # extends the elbow so the fist is what arrives first rather than the forearm.
    "guard": {},
    "still": {"left_shoulder_pitch_joint": -1.5, "left_elbow_joint": 0.0},
    "thrown": {"left_shoulder_pitch_joint": -1.5, "left_elbow_joint": 0.0,
               "velocity": ("left_shoulder_pitch_joint", -8.0)},
}


def static_sweep(level: BoxingEnv, gaps: list[float]) -> None:
    """One step per (arm pose, teleport): what touches, and does any of it score?

    The tests that move a corner onto the other need to know which combination puts
    a fist on something that scores rather than on a glove, and guessing at it is how
    a test comes to assert something the scene does not do.

    The state is rebuilt through MuJoCo on the host, sensors included, because
    writing qpos and qvel into a pipeline state leaves ``sensordata`` describing the
    pose the state was reset in -- and the gate reads fist positions and velocities
    from the sensors, so a stale block measures a punch that is not there. Measured:
    a shoulder rate of -8 rad/s set in qvel reads 0.00 m/s of fist at step 0 and
    2.71 at step 1, by which time the contact is no longer a rising edge.
    """
    model = level._model
    data = mujoco.MjData(model)
    red = level._sides[scene.RED]

    def address(joint: str) -> int:
        return int(red["qpos_adr"][level.actuator_index(f"red/{joint}")])

    def dof(joint: str) -> int:
        return int(red["dof_adr"][level.actuator_index(f"red/{joint}")])

    configs = [(pose, POSES[pose], gap) for pose in POSES for gap in gaps]
    base = jax.vmap(level.reset)(jax.random.split(jax.random.PRNGKey(5), len(configs)))
    free = level._sides[scene.BLUE]["free_adr"]

    qpos = np.asarray(base.pipeline_state.qpos).copy()
    qvel = np.asarray(base.pipeline_state.qvel).copy()
    # `model.nsensor` counts sensors (18); the block they write is the sum of their
    # dimensions (56), which is also the length the pipeline state carries.
    sensors = np.zeros((len(configs), np.asarray(base.pipeline_state.sensordata).shape[-1]))
    for world, (_, angles, gap) in enumerate(configs):
        qpos[world, free] -= gap
        for joint, angle in angles.items():
            if joint != "velocity":
                qpos[world, address(joint)] = angle
        joint, rate = angles.get("velocity", (None, 0.0))
        if joint is not None:
            qvel[world, dof(joint)] = rate
        data.qpos[:] = qpos[world]
        data.qvel[:] = qvel[world]
        mujoco.mj_forward(model, data)
        sensors[world] = data.sensordata

    pipeline = base.pipeline_state.replace(
        qpos=jp.array(qpos.astype(np.float32)),
        qvel=jp.array(qvel.astype(np.float32)),
        sensordata=jp.array(sensors.astype(np.float32)),
    )
    state = base.replace(pipeline_state=pipeline)

    frames = {side: level._frame(pipeline, side) for side in scene.SIDES}
    # Trimmed to what the gates read, and vmap'd rather than called on the batch:
    # the frame also carries a qpos/qvel gather whose axis order only works per
    # world, and the gates index their per-hand axis with literals.
    me = {k: frames[scene.RED][k] for k in
          ("left_palm", "right_palm", "torso_vel", "left_palm_vel", "right_palm_vel", "quat")}
    foe = {k: frames[scene.BLUE][k] for k in ("head", "torso")}
    committed = np.asarray(jax.vmap(level._committed)(me, foe))
    speed, closing = jax.vmap(level._punch_signals)(me, foe)
    speed, closing = np.asarray(speed.max(-1)), np.asarray(closing.max(-1))

    stepped = jax.jit(jax.vmap(level.step))(
        state, jp.zeros((len(configs), NUM_AGENTS, level.action_size)))
    contacts = np.asarray(stepped.metrics["contacts"])
    hits = np.asarray(stepped.metrics["hits_red"])
    damage = np.asarray(stepped.metrics["damage"])

    name = lambda gid: (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<unnamed {gid}>")
    inside = np.asarray(stepped.pipeline_state.contact.dist
                        < stepped.pipeline_state.contact.includemargin)
    geoms = np.asarray(stepped.pipeline_state.contact.geom)

    print("one step per cell, sensors rebuilt on the host; `committed` is read before "
          "the step, the rest after it")
    for world, (pose, _, gap) in enumerate(configs):
        pairs = set()
        for slot in np.flatnonzero(inside[world]):
            a, b = name(geoms[world, slot, 0]), name(geoms[world, slot, 1])
            if "/" in a and "/" in b and a.split("/")[0] != b.split("/")[0]:
                pairs.add(f"{a}x{b}")
        print(f"  {pose:8s} gap {gap:4.2f} m ({1.3 - gap:4.2f} m apart): "
              f"fist {speed[world]:4.1f} m/s closing {closing[world]:4.1f} "
              f"committed {committed[world]:.2f} | contacts {contacts[world]:.2f} "
              f"hits {hits[world]:.2f} damage {damage[world]:.3f}")
        for pair in sorted(pairs):
            print(f"      {pair}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gap", type=float, default=0.5, help="metres to teleport the jabber closer")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--rearm", type=int, default=20)
    ap.add_argument("--static-gaps", type=float, nargs="*", default=None,
                    help="skip the jabber and report one zero-action step at each teleport")
    args = ap.parse_args(argv)

    level = BoxingEnv(spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0,
                      rearm_steps=args.rearm)
    if args.static_gaps is not None:
        static_sweep(level, args.static_gaps or [0.5, 0.6, 0.75, 0.85, 0.9])
        return 0

    model = level._model

    def name(gid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<unnamed {gid}>"

    # The action vector is per agent, so a corner's own actuators are indexed from
    # zero within it: red's scene indices are already local, and blue's are the same
    # joints minus one robot's width.
    jab = policy_module.jabber(
        level.action_size,
        pitch_adr=level.actuator_index("red/left_shoulder_pitch_joint"),
        elbow_adr=level.actuator_index("red/left_elbow_joint"),
    )
    down = arms_down(level, scene.RED)

    state = level.reset(jax.random.PRNGKey(6))
    pipeline = state.pipeline_state
    blue_free = level._sides[scene.BLUE]["free_adr"]
    state = state.replace(
        pipeline_state=pipeline.replace(qpos=pipeline.qpos.at[blue_free].add(-args.gap)))

    def body(state, _):
        clock = jp.atleast_1d(state.pipeline_state.time)[:1]
        action = jp.stack([down, jab(state.obs[1][None], clock, jax.random.PRNGKey(0))[0]])
        inside = (state.pipeline_state.contact.dist
                  < state.pipeline_state.contact.includemargin)
        state = level.step(state, action)
        return state, (inside, state.pipeline_state.contact.geom,
                       state.metrics["contacts"], state.metrics["hits_blue"],
                       state.metrics["damage_taken"], state.info["steps_off"][1],
                       state.info["last_contact"][1])

    _, logged = jax.lax.scan(body, state, None, length=args.steps)
    inside, geoms, contacts, hits, damage, steps_off, last_contact = (
        np.asarray(x) for x in logged)

    print(f"jabber is blue, target holds its arms down, teleport {args.gap} m, "
          f"rearm_steps {args.rearm}")
    print(f"  contact on {int((contacts > 0).sum())} of {args.steps} steps, "
          f"scored on {int((hits > 0).sum())}")
    print("  step armed edge  contacts   hits  damage | cross-corner slots inside the margin")
    for step in range(args.steps):
        pairs = []
        for slot in np.flatnonzero(inside[step]):
            a, b = name(geoms[step, slot, 0]), name(geoms[step, slot, 1])
            if "/" in a and "/" in b and a.split("/")[0] != b.split("/")[0]:
                pairs.append(f"{a.split('/')[-1]}x{b.split('/')[-1]}")
        armed = int(steps_off[step - 1] >= args.rearm) if step else 1
        edge = int(contacts[step] > 0 and (last_contact[step - 1] <= 0 if step else True))
        if not pairs and hits[step] == 0:
            continue
        print(f"  {step:4d} {armed:5d} {edge:4d}  {contacts[step]:8.2f} {hits[step]:6.2f} "
              f"{damage[step]:7.3f} | {'; '.join(pairs) if pairs else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
