"""What the MJX contact buffer actually reports for far-away declared pairs.

`tools/reach_sweep.py` printed "the arm meets blue's hand on 100% of poses" at
0.90 m, which is not physically plausible and matters for how its blocker lines
are read. Two candidate explanations, with different consequences:

* the hand-vs-hand pairs carry a large contact margin, so a true separation of
  a few tenths of a metre still reads "inside" -- then the blocker lines are
  margin artefacts and only the margin-zero torso/head slots are meaningful;
* ``dist`` is not the true separation for pairs the broadphase never engaged --
  then "closest approach" lines are suspect too, and the sweep's zeros would
  need a different instrument.

This prints, for one stance state and one 0.9 m state, every declared slot with
a nonzero margin or a small separation, plus whether the per-state
``includemargin`` matches the one the sweep reused from the stance -- it is
supposed to be a static pair parameter.

    JAX_PLATFORMS=cpu python tools/margin_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp
import mujoco
import numpy as np

from boxer import scene
from boxer.env import BoxingEnv


def main() -> None:
    level = BoxingEnv(spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
    model = level._model
    qvel0 = jp.zeros(level.sys.nv)
    ctrl0 = level._stance_ctrl
    blue_free = level._sides[scene.BLUE]["free_adr"]

    far = np.asarray(level._stance_qpos).astype(np.float32).copy()
    near = far.copy()
    near[blue_free] -= 0.4  # 0.9 m apart, both holding the guard

    state_far = level.pipeline_init(jp.array(far), qvel0, None, ctrl0)
    state_near = level.pipeline_init(jp.array(near), qvel0, None, ctrl0)

    def name(gid):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<{gid}>"

    geoms = np.asarray(state_far.contact.geom)
    for label, state in (("1.30 m", state_far), ("0.90 m", state_near)):
        dist = np.asarray(state.contact.dist)
        margin = np.asarray(state.contact.includemargin)
        print(f"=== {label} ===")
        nonzero = np.flatnonzero((margin != 0.0) | (np.abs(dist) < 0.5))
        print(f"  slots with nonzero margin or |dist| < 0.5: {len(nonzero)} of {len(dist)}")
        for slot in nonzero:
            print(f"    slot {slot:4d} {name(geoms[slot, 0])} x {name(geoms[slot, 1])}"
                  f"  dist {dist[slot]:+.4f}  margin {margin[slot]:+.4f}")

    m_far = np.asarray(state_far.contact.includemargin)
    m_near = np.asarray(state_near.contact.includemargin)
    print(f"\nincludemargin identical across states: {np.array_equal(m_far, m_near)}")
    pair_margin = np.asarray(model.pair_margin)
    pair_gap = np.asarray(model.pair_gap)
    declared = np.flatnonzero(pair_margin != 0.0)
    print(f"declared pairs with nonzero model.pair_margin: {len(declared)} of {model.npair}")
    for i in declared[:20]:
        a = name(model.pair_geom1[i])
        b = name(model.pair_geom2[i])
        print(f"    {a} x {b}  margin {pair_margin[i]:+.4f} gap {pair_gap[i]:+.4f}"
              f"  -> includemargin {pair_margin[i] - pair_gap[i]:+.4f}")

    # The sweep reads dist through a jitted vmap; this tool reads it eagerly. If
    # the two execution paths disagree, the sweep's lines measure a path, not
    # geometry -- so compare them on the same random arm poses at 0.9 m.
    from boxer import scene as scene_module
    red_adr = {}
    for joint in scene_module.ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                scene_module.RED + f"left_{joint}_joint")
        red_adr[joint] = int(model.jnt_qposadr[jid])
    adr = np.array([red_adr[j] for j in scene_module.ARM_JOINTS])
    rng = np.random.default_rng(3)
    poses = rng.uniform(-2.0, 2.0, (16, len(adr))).astype(np.float32)
    qpos = np.tile(near, (len(poses), 1))
    qpos[:, adr] = poses

    batch = jax.jit(jax.vmap(lambda q: level.pipeline_init(q, qvel0, None, ctrl0)))
    batched_dist = np.asarray(batch(jp.array(qpos)).contact.dist)
    eager_dist = np.stack([
        np.asarray(level.pipeline_init(jp.array(q), qvel0, None, ctrl0).contact.dist)
        for q in qpos
    ])
    diff = np.abs(batched_dist - eager_dist)
    print(f"\nvmapped vs eager dist over 16 posed states: max |diff| {diff.max():.6f}")
    worst = int(np.unravel_index(np.argmax(diff), diff.shape)[1])
    print(f"  worst slot: {worst} {name(geoms[worst, 0])} x {name(geoms[worst, 1])}")
    hand_hand = [s for s in range(len(geoms))
                 if "hand_collision" in name(geoms[s, 0])
                 and "hand_collision" in name(geoms[s, 1])
                 and name(geoms[s, 0]).split("/")[0] != name(geoms[s, 1]).split("/")[0]]
    for s in hand_hand:
        print(f"  slot {s} {name(geoms[s, 0])} x {name(geoms[s, 1])}: "
              f"batched {batched_dist[:, s].min():+.4f}..{batched_dist[:, s].max():+.4f}  "
              f"eager {eager_dist[:, s].min():+.4f}..{eager_dist[:, s].max():+.4f}")


if __name__ == "__main__":
    main()
