"""Compares MJX's contact buffer against MuJoCo's own collision detection for
the same forced-overlap configuration, and dumps what the env's role mask says
about every touching slot."""

import warnings

warnings.filterwarnings("ignore")

import jax
import mujoco
import numpy as np
from jax import numpy as jp

from boxer import scene
from boxer.env import BLUE_STRIKE, BLUE_TARGET, RED_STRIKE, RED_TARGET, BoxingEnv

env = BoxingEnv(spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
model = env.sys.mj_model
name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or "floor"

q0, c0 = scene.stance(model)
blue_free = int(env._sides[scene.BLUE]["free_adr"])
role = np.asarray(env._contact_role)
weight = np.asarray(env._contact_weight)

for dx in (0.55, 0.75, 0.95, 1.15):
    # --- what plain MuJoCo sees at that configuration
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    data.ctrl[:] = c0
    data.qpos[blue_free] -= dx
    mujoco.mj_forward(model, data)
    mj_pairs = []
    for i in range(data.ncon):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        if name(g1).startswith(scene.RED) != name(g2).startswith(scene.RED):
            mj_pairs.append((name(g1), name(g2), float(data.contact[i].dist)))

    # --- what MJX sees after pipeline init, before any step
    from brax.mjx import pipeline as mp

    ps = mp.init(env.sys, jp.array(q0).at[blue_free].add(-dx), jp.zeros(env.sys.nv), None, jp.array(c0))
    c = ps.contact
    dist = np.asarray(c.dist)
    include = np.asarray(c.includemargin)
    geoms = np.asarray(c.geom)
    touching = dist < include
    cross = []
    for k in np.where(touching)[0]:
        g1, g2 = int(geoms[k][0]), int(geoms[k][1])
        if name(g1).startswith(scene.RED) != name(g2).startswith(scene.RED):
            r1, r2 = role[g1], role[g2]
            strike_target = (r1 & RED_STRIKE and r2 & BLUE_TARGET) or (
                r2 & RED_STRIKE and r1 & BLUE_TARGET
            )
            cross.append((name(g1), name(g2), round(float(dist[k]), 6), bool(strike_target)))

    print(f"dx={dx:.2f}  mujoco cross-contacts={len(mj_pairs)}  mjx touching={int(touching.sum())} mjx cross={len(cross)}")
    for pair in mj_pairs[:4]:
        print("   mujoco:", pair[0], "|", pair[1], "| dist", round(pair[2], 6))
    for pair in cross[:6]:
        print("   mjx    :", pair)
