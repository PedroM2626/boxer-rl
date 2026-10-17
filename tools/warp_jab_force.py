"""Does a force-threshold scorecard fire on a scripted jab in the warp loop?

The last unproven piece of the migration: not that ``contact_force`` works on
one posed world (`tools/warp_scene_check.sh` settled that), but that a batched
control loop -- step N worlds, sync the packed contact buffer, threshold the
per-contact forces, count events under a rearm window -- sees a scripted jab
land. Blue stands with arms down at 0.45 m; red throws shoulder-pitch + elbow
extensions on a clock, the way the jabber fixture does; worlds are offset in
phase so the batch is never synchronised.

The weights table is duplicated from boxer/env.py's TARGET_WEIGHTS rather than
imported, because that module is jax-bound and this tool runs in the warp venv.

    PYTHONPATH=. /home/pedro/venvs/warp_test/bin/python tools/warp_jab_force.py
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from boxer import scene

WEIGHTS = {"head_collision": 1.0, "torso_collision": 0.5}
N_FRAMES = 5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--rearm", type=int, default=20)
    ap.add_argument("--period", type=int, default=30,
                    help="control steps between jab launches")
    ap.add_argument("--force-threshold", type=float, default=50.0)
    args = ap.parse_args(argv)

    model = scene.build_model(spawn_distance=1.3, visual=False)
    base, ctrl0 = scene.stance(model, "natural")

    def qadr(name):
        return int(model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)])

    def cadr(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    pitch, elbow = (cadr("red/left_shoulder_pitch_joint"),
                    cadr("red/left_elbow_joint"))
    # The reach_sweep probe pose: where these two joints put a hand on the torso.
    probe_pitch, probe_elbow = -1.5, 0.0

    qpos = base.copy()
    qpos[qadr("blue/floating_base_joint")] -= 0.85
    for half in ("left", "right"):
        for joint in scene.ARM_JOINTS:
            qpos[qadr(f"blue/{half}_{joint}_joint")] = 0.0

    host = mujoco.MjData(model)
    host.qpos[:] = qpos
    host.ctrl[:] = ctrl0
    m = mjwarp.put_model(model)
    d = mjwarp.put_data(model, host, nworld=args.worlds, nconmax=384, njmax=2048)
    try:
        m.opt.warn_overflow = 0
    except AttributeError:
        pass
    replicated = bool(np.all(d.qpos.numpy() == d.qpos.numpy()[0]))
    print(f"{args.worlds} worlds, {N_FRAMES} substeps/control step; "
          f"put_data replicated the pose across worlds: {replicated}")

    red = {g for g in range(model.ngeom) if
           (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("red")}
    blue = {g for g in range(model.ngeom) if
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("blue")}
    weight_of = {}
    for g in blue:
        bare = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g).split("/")[-1]
        weight_of[g] = WEIGHTS.get(bare, 0.35 if "pelvis" in bare or "hip" in bare
                                   or "thigh" in bare else 0.0)

    rng = np.random.default_rng(0)
    phase = 2.0 * np.pi * np.arange(args.worlds) / args.worlds
    steps_off = np.full(args.worlds, args.rearm, dtype=np.int64)
    scored_total = np.zeros(args.worlds, dtype=np.int64)
    fired = collections.Counter()
    ctrl = np.tile(np.asarray(ctrl0, dtype=np.float32), (args.worlds, 1))

    for step in range(args.steps):
        clock = step / args.period
        punch = np.clip(np.sin(2.0 * np.pi * clock + phase), 0.0, 1.0) ** 2
        ctrl[:, pitch] = ctrl0[pitch] + punch * (probe_pitch - ctrl0[pitch])
        ctrl[:, elbow] = ctrl0[elbow] + punch * (probe_elbow - ctrl0[elbow])
        d.ctrl.assign(ctrl)
        for _ in range(N_FRAMES):
            mjwarp.step(m, d)

        nacon = int(d.nacon.numpy()[0])
        event = np.zeros(args.worlds, dtype=bool)
        if nacon:
            rows = slice(0, nacon)
            geoms = d.contact.geom.numpy()[rows]
            worldid = d.contact.worldid.numpy()[rows]
            efc = d.contact.efc_address.numpy()[rows, 0]
            cross = (((np.isin(geoms[:, 0], list(red)) &
                       np.isin(geoms[:, 1], list(blue))) |
                      (np.isin(geoms[:, 0], list(blue)) &
                       np.isin(geoms[:, 1], list(red)))) & (efc >= 0))
            ids = np.flatnonzero(cross)
            if len(ids):
                force = wp.zeros(len(ids), dtype=wp.spatial_vector)
                mjwarp.contact_force(m, d, wp.array(ids.astype(np.int32)),
                                     True, force)
                norms = np.linalg.norm(force.numpy()[:, :3], axis=1)
                for i, n in zip(ids, norms):
                    if n >= args.force_threshold:
                        blue_geom = (geoms[i, 1] if geoms[i, 0] in red
                                     else geoms[i, 0])
                        if weight_of[int(blue_geom)] > 0.0:
                            event[int(worldid[i])] = True
                            fired[mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_GEOM,
                                int(blue_geom)).split("/")[-1]] += 1

        armed = steps_off >= args.rearm
        scored = event & armed
        steps_off = np.where(scored, 0, steps_off + 1)
        scored_total += scored
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}: nacon {nacon:4d}, worlds with force-events "
                  f"{int(event.sum()):3d}, scored this step {int(scored.sum()):3d}, "
                  f"total scored per world "
                  f"min {scored_total.min()} mean {scored_total.mean():.1f} "
                  f"max {scored_total.max()}")

    print(f"\nforce-threshold scorecard over {args.steps} steps "
          f"(threshold {args.force_threshold:.0f} N, rearm {args.rearm}):")
    print(f"  scored jabs per world: min {scored_total.min()}, "
          f"mean {scored_total.mean():.1f}, max {scored_total.max()}")
    print(f"  landed on: {dict(fired) or 'nothing'}")
    if scored_total.max() == 0:
        print("  no event ever crossed the threshold -- the scorecard did not fire")
        return 1
    print("  the scorecard fires on the warp loop -- events land and the rearm "
          "window paces them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
