#!/usr/bin/env bash
# Does the composed two-boxer scene run on mujoco_warp, with force on its contacts?
#
# tools/warp_check.sh settled that mujoco_warp ships a per-contact force kernel
# -- but it read the installed package, not a model. The open question is whether
# OUR scene (two Unitree G1s attached under one spec, every contact declared as
# an explicit <pair> row, the model MJX trains on) loads through put_model/
# put_data, steps on the GPU engine, and has the solver leave a nonzero force in
# the contact rows. A force-threshold scorecard -- the published recipe on this
# same robot -- is only real if all three hold at once on this model.
#
# The probe is the reach_sweep preflight pose: blue teleported to 0.45 m with
# arms at zero, red's left arm extended -- the pose whose penetrating pair set
# host MuJoCo and MJX already agree on. Host mj_forward is again the ground
# truth: the device's force norms must line up with mujoco.mj_contactForce on
# the same pose, or the kernel is reading a buffer nothing populated.
#
# Runs on a free GPU (a second CUDA context next to a trainer has restarted
# this VM before) in its own venv, so boxer313's pinned stack is untouched:
#   wsl bash -lc 'cd /mnt/e/boxer-agent && bash tools/warp_scene_check.sh'
set -uo pipefail

UV=/home/pedro/venvs/boxer/bin/uv
VENV=/home/pedro/venvs/warp_test
PY="$VENV/bin/python"

# boxer/scene.py imports mujoco_menagerie; the warp venv was built bare.
# Pinned to boxer313's version so the G1 XML is the one the scene composes.
"$UV" pip install --python "$PY" mujoco-menagerie==2026.9.4 2>&1 | tail -2

export PYTHONPATH=/mnt/e/boxer-agent
cd /mnt/e/boxer-agent

"$PY" - <<'PY'
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from boxer import scene

model = scene.build_model(spawn_distance=1.3, visual=False)
base, ctrl0 = scene.stance(model, "natural")
print(f"model: {model.ngeom} geoms, {model.npair} declared pairs, "
      f"solver {int(model.opt.solver)}, integrator {int(model.opt.integrator)}, "
      f"cone {int(model.opt.cone)}")


def qadr(joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])


probe = base.copy()
probe[qadr("blue/floating_base_joint")] -= 0.85
for half in ("left", "right"):
    for joint in scene.ARM_JOINTS:
        probe[qadr(f"blue/{half}_{joint}_joint")] = 0.0
probe[qadr("red/left_shoulder_pitch_joint")] = -1.5
probe[qadr("red/left_elbow_joint")] = 0.0


def gname(gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<{gid}>"


def cross(pair):
    return {n.split("/")[0] for n in pair} == {"red", "blue"}


# Ground truth: host mj_forward at the probe pose, forces read the classic way.
host = mujoco.MjData(model)
host.qpos[:] = probe
host.ctrl[:] = ctrl0
mujoco.mj_forward(model, host)
host_rows = {}
for i in range(host.ncon):
    pair = tuple(sorted((gname(host.contact[i].geom1), gname(host.contact[i].geom2))))
    if cross(pair):
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, host, i, wrench)
        host_rows[pair] = (float(host.contact[i].dist), float(np.linalg.norm(wrench[:3])))
print(f"\nhost mj_forward at the probe: {host.ncon} contacts, "
      f"{len(host_rows)} cross-corner")
for pair, (dist, force) in sorted(host_rows.items()):
    print(f"  {' x '.join(pair):64s} dist {dist:+.4f} m  force {force:8.2f} N")

# Device: the same posed state through the warp bridge, one full step.
print(f"\nwarp device: {wp.get_device()}")
device_in = mujoco.MjData(model)
device_in.qpos[:] = probe
device_in.ctrl[:] = ctrl0
m = mjwarp.put_model(model)
# njmax sized by hand: put_data derives the default from the host data's ncon,
# which is zero before any forward -- the first run overflowed at 64 rows, the
# solver silently dropped the excess constraints, and the dropped rows were
# exactly the ones that read 0 N on device against nonzero on host (including
# the torso contact). A fight round carries more contacts than this probe.
d = mjwarp.put_data(model, device_in, nconmax=128, njmax=512)
mjwarp.step(m, d)
nacon = int(d.nacon.numpy()[0])
nefc = int(d.nefc.numpy()[0])
print(f"warp step: {nacon} contacts, {nefc} constraint rows")

ids = wp.array(np.arange(nacon, dtype=np.int32))
force = wp.zeros(nacon, dtype=wp.spatial_vector)
mjwarp.contact_force(m, d, ids, True, force)
forces = force.numpy()
geoms = d.contact.geom.numpy()
dists = d.contact.dist.numpy()

device_rows = {}
for i in range(nacon):
    pair = tuple(sorted((gname(geoms[i, 0]), gname(geoms[i, 1]))))
    if cross(pair):
        device_rows[pair] = (float(dists[i]), float(np.linalg.norm(forces[i][:3])))
print(f"warp contact_force, world frame: {len(device_rows)} cross-corner rows")
for pair, (dist, force) in sorted(device_rows.items()):
    print(f"  {' x '.join(pair):64s} dist {dist:+.4f} m  force {force:8.2f} N")

only_host = set(host_rows) - set(device_rows)
only_device = set(device_rows) - set(host_rows)
ratios = []
for pair, (hdist, hforce) in host_rows.items():
    if pair in device_rows:
        ddist, dforce = device_rows[pair]
        if hforce > 1e-6 and dforce > 1e-6:
            ratios.append(dforce / hforce)
        print(f"  {' x '.join(pair):64s} host {hforce:8.2f} N  warp {dforce:8.2f} N  "
              f"dist {hdist - ddist:+.5f} m")

print()
if only_host or only_device:
    print(f"pair sets disagree -- host-only {sorted(map(list, only_host))}, "
          f"warp-only {sorted(map(list, only_device))}")
elif not host_rows:
    print("no cross-corner contacts at the probe on either engine -- "
          "the probe pose no longer penetrates, the check measured nothing")
elif not ratios:
    print("same pairs on both engines but no nonzero force pair to compare")
else:
    spread = f"{min(ratios):.3f}..{max(ratios):.3f}" if len(ratios) > 1 else f"{ratios[0]:.3f}"
    best = max(device_rows.items(), key=lambda kv: kv[1][1])
    print(f"force ratio warp/host {spread} over {len(ratios)} pairs; "
          f"largest device force {best[1][1]:.2f} N on {' x '.join(best[0])}")
    print("the scene steps on mujoco_warp and its contacts carry force -- "
          "a force-threshold scorecard is implementable on this model")
PY
