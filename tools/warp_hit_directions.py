"""Which corner does the scorecard credit, and does geometry order decide it?

The warp run logged hits_red ~730/iteration that never moved with the policy
while hits_blue stayed 0.0 for 82 iterations -- impossible in symmetric
self-play with one shared policy, so the bug is in the measurement, not the
boxers. This probe steps two scripted worlds and prints the truth table:

* red punches (blue's arms held down) -> which agent gets the credit,
* blue punches (red's arms held down) -> which agent gets the credit,
* chest-to-chest pressing with nobody punching -> who gets "hit",
* and, for every strong cross-corner contact, which colour sits in column 0.

Run before and after the attribution fix:

    PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python tools/warp_hit_directions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from boxer import scene
from boxer.warp_env import WarpBoxing

NW = 8
PROBE_PITCH, PROBE_ELBOW = -1.5, 0.0  # the reach_sweep pose that lands a fist
SETTLE, PUNCH = 25, 60


def column_colours(env) -> list:
    """Colour of geom column 0 over the current strong cross contacts."""
    nacon = int(env._d.nacon.numpy()[0])
    if not nacon:
        return []
    rows = slice(0, nacon)
    geoms = env._d.contact.geom.numpy()[rows]
    efc = env._d.contact.efc_address.numpy()[rows, 0]
    red0 = np.isin(geoms[:, 0], env._red_ids) & np.isin(geoms[:, 1], env._blue_ids)
    blue0 = np.isin(geoms[:, 0], env._blue_ids) & np.isin(geoms[:, 1], env._red_ids)
    out = []
    for mask, colour in ((red0, "red"), (blue0, "blue")):
        for r in np.flatnonzero(mask & (efc >= 0)):
            out.append((colour, mujoco.mj_id2name(
                env._model, mujoco.mjtObj.mjOBJ_GEOM, int(geoms[r, 0])),
                mujoco.mj_id2name(
                    env._model, mujoco.mjtObj.mjOBJ_GEOM, int(geoms[r, 1]))))
    return out


def run(striker: str, label: str) -> dict:
    env = WarpBoxing(nworld=NW, spawn_distance=0.45,
                     spawn_jitter=0.0, yaw_jitter=0.0, pose_noise=0.0)
    env.reset(0)
    side = {0: scene.RED, 1: scene.BLUE}
    punching = {"red": 0, "blue": 1, "none": -1}[striker]
    cols = {}
    for agent, prefix in ((0, scene.RED), (1, scene.BLUE)):
        ids = env._sides[prefix]["actuators"]
        for joint, probe in (("left_shoulder_pitch_joint", PROBE_PITCH),
                             ("left_elbow_joint", PROBE_ELBOW)):
            aid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    prefix + joint)
            col = int(np.flatnonzero(ids == aid)[0])
            cols[(agent, col)] = (probe - np.asarray(env._ctrl0)[aid]) / env._action_scale

    def build(step: int, punching_agent: int):
        a = np.zeros((NW, 2, env.action_size), np.float32)
        if punching_agent >= 0:
            for (agent, col), amp in cols.items():
                if agent == punching_agent:
                    a[:, agent, col] = amp * step
        return a

    peak = np.zeros(2)
    orders = []
    action = np.zeros((NW, 2, env.action_size), np.float32)
    for t in range(SETTLE):  # both hold the guard, then we press in chest-to-chest
        _, _, _, parts = env.step(action)
        peak[0] = max(peak[0], float(parts["hit"].max()))
        peak[1] = max(peak[1], float(parts["taken"].max()))
    for t in range(PUNCH):
        punch = max(0.0, np.sin(2.0 * np.pi * t / 30.0)) ** 2
        action = build(punch, punching)
        _, _, _, parts = env.step(action)
        peak[0] = max(peak[0], float(parts["hit"].max()))      # red's raw hit value
        peak[1] = max(peak[1], float(parts["taken"].max()))    # blue's raw hit value
        orders += column_colours(env)
    tally = {}
    for colour, g0, g1 in orders:
        tally[colour] = tally.get(colour, 0) + 1
    print(f"{label:28s} peak red-hit {peak[0]:.2f}  peak blue-hit {peak[1]:.2f}"
          f"  | cross contacts by column-0 colour: {tally or 'none'}")
    return {"red": float(peak[0]), "blue": float(peak[1]), "orders": tally}


def main() -> int:
    print(f"{NW} worlds, 0.45 m apart, 50 N threshold; peak per-agent hit value")
    settle = run("none", "nobody punching (chest press)")
    red = run("red", "RED jabs (blue holds guard)")
    blue = run("blue", "BLUE jabs (red holds guard)")
    ok = (red["red"] > 0 and blue["blue"] > 0 and settle["red"] == 0.0
          and settle["blue"] == 0.0)
    print("\nverdict:", "attributions follow the STRIKER, presses score nothing"
          if ok else "BROKEN: credit does not follow the striker, or a press scores")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
