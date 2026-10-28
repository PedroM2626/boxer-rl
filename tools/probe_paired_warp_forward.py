"""Compare one current unassisted paired model forward on native MuJoCo and Warp.

This is a hardware/kinematics preflight, not training, contact-force parity,
trajectory parity, throughput benchmarking, or boxing certification.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp
from boxer.curriculum_env import UnassistedG1
from boxer.paired_foundation import PairedFoundation


def main():
    out = Path('docs/results/striking-pilot/paired-warp-preflight.json')
    if out.exists():
        raise ValueError('Existing preflight must be preserved and reviewed before retry')
    pair = PairedFoundation(UnassistedG1('approach', impl='jax'))
    pair.reset(seed=34002003, swap=False)
    model, host = pair.model, pair.data
    assert model.neq == 0 and not np.any(model.body_gravcomp)
    wp.init()
    device = wp.get_device('cuda:0')
    with wp.ScopedDevice(device):
        gpu_model = mjwarp.put_model(model)
        gpu_data = mjwarp.put_data(model, host, nworld=1, nconmax=256, njmax=2048)
        mjwarp.forward(gpu_model, gpu_data)
        wp.synchronize_device(device)
        qpos = gpu_data.qpos.numpy()[0]
        xpos = gpu_data.xpos.numpy()[0]
        qvel = gpu_data.qvel.numpy()[0]
    assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(xpos).all()
    errors = dict(qpos_max_abs=float(np.max(np.abs(qpos-host.qpos))),
                  qvel_max_abs=float(np.max(np.abs(qvel-host.qvel))),
                  body_position_max_abs=float(np.max(np.abs(xpos-host.xpos))))
    result = dict(status='completed', device=str(device), seed=34002003, swap=False,
                  model_nq=model.nq, model_nv=model.nv, actuator_count=model.nu,
                  equality_constraints=model.neq, gravity_compensation=False,
                  native_contact_count=int(host.ncon), gpu_contact_capacity=256, gpu_constraint_capacity=2048,
                  paired_native_resets=2, gpu_forward_calls=1, physical_steps=0,
                  new_learner_steps=0, main_consumed_changed=False, numerical_errors=errors,
                  limitation='One GPU forward only; inspect Warp output for solver warnings. No contact-force, constraints, trajectory, policy, reward, buffer-overflow or throughput equivalence established.')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
