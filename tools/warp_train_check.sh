#!/usr/bin/env bash
# The two migration checks: does warp compute OUR sensors, and at what throughput?
#
# The port to mujoco_warp only makes sense if the env's observation path survives
# it. boxer/env.py's _frame reads site sensors -- FRAMEPOS on torso/head/palms,
# FRAMELINVEL on torso/palms, FRAMEQUAT on the torso, GYRO in the imu -- so the
# first check compares warp's sensordata against host mj_forward on the same
# posed state, sensor by sensor. The module compiling in the smoke test proved
# the code path exists; only the values prove it is right.
#
# The second check times mjwarp.step at training-scale world counts on this
# 6 GB card, from the contact-heavy probe pose (all worlds stepping through
# real contacts, not a free-fall), and reads the card's memory. The bar is the
# MJX trainer's measured aggregate: ~2,270 env steps/s at n_frames=5.
#
#   wsl bash -lc 'cd /mnt/e/boxer-agent && bash tools/warp_train_check.sh'
set -uo pipefail

VENV=/home/pedro/venvs/warp_test
PY="$VENV/bin/python"

export PYTHONPATH=/mnt/e/boxer-agent
cd /mnt/e/boxer-agent

"$PY" - <<'PY'
import subprocess
import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from boxer import scene

model = scene.build_model(spawn_distance=1.3, visual=False)
base, ctrl0 = scene.stance(model, "natural")


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

sensor_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                for i in range(model.nsensor)]
read_by_env = [n for n in sensor_names if n.split("_")[-1] in
               ("pos", "linvel", "quat", "gyro")]
print(f"sensors: {model.nsensor} declared, {len(read_by_env)} of the kinds "
      f"_frame reads; nsensordata {model.nsensordata}")

host = mujoco.MjData(model)
host.qpos[:] = probe
host.ctrl[:] = ctrl0
mujoco.mj_forward(model, host)

device_in = mujoco.MjData(model)
device_in.qpos[:] = probe
device_in.ctrl[:] = ctrl0
m = mjwarp.put_model(model)
d = mjwarp.put_data(model, device_in, nconmax=384, njmax=2048)
mjwarp.step(m, d)
sensors = np.asarray(d.sensordata.numpy()).reshape(-1)[:model.nsensordata]
diff = np.abs(sensors - host.sensordata)
print(f"\nsensordata vs host mj_forward on the same pose: max |diff| "
      f"{diff.max():.2e} over {model.nsensordata} values")
worst = []
for i, name in enumerate(sensor_names):
    adr, dim = int(model.sensor_adr[i]), int(model.sensor_dim[i])
    worst.append((float(diff[adr:adr + dim].max()), name))
worst.sort(reverse=True)
for value, name in worst[:6]:
    print(f"  {name:34s} max |diff| {value:.2e}")
pos_ok = float(diff.max()) < 1e-4
print("  " + ("sensor values agree -- the observation path ports"
              if pos_ok else "SENSOR VALUES DISAGREE -- the port is blocked"))

print(f"\nthroughput on {wp.get_device()} (probe state, contacts live in "
      f"every world), warmup 3 + timed 50 steps:")
for nworld in (256, 512, 1024, 2048):
    dd = mjwarp.put_data(model, device_in, nworld=nworld, nconmax=384,
                         njmax=2048)
    for _ in range(3):
        mjwarp.step(m, dd)
    steps = 50
    t0 = time.perf_counter()
    for _ in range(steps):
        mjwarp.step(m, dd)
    dt = time.perf_counter() - t0
    physics = steps * nworld / dt
    mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    print(f"  nworld {nworld:5d}: {dt / steps * 1e3:7.1f} ms/step, "
          f"{physics:10,.0f} physics steps/s, "
          f"{physics / 5:10,.0f} control steps/s (n_frames 5), "
          f"card {mem}")
    del dd

print("\nMJX trainer baseline: ~2,270 control env steps/s aggregate "
      "(arm 5: 204,472,320 steps in ~25 h)")
PY
