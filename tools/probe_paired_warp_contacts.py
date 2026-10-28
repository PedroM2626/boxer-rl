"""Compare contact forces at two saved v15 states; no stepping or policy execution.

The saved qpos/qvel/ctrl do not contain solver warm-start state. This is a
cold-start diagnostic, not a replay of the original dynamic trajectory.
"""
import hashlib
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp
from boxer.curriculum_env import UnassistedG1
from boxer.paired_foundation import PairedFoundation

ROOT = Path('docs/results/striking-pilot/contact-progress-v15')
OUT = Path('docs/results/striking-pilot/paired-warp-contact-forwarded.json')


def main():
    if OUT.exists():
        raise ValueError('Existing probe must be reviewed before retry')
    chosen = json.loads((ROOT / 'first-captured-event-review.json').read_text())['selected_for_render']
    assert len(chosen) == 2
    pair = PairedFoundation(UnassistedG1('approach', impl='jax'))
    model = pair.model
    assert model.neq == 0 and not np.any(model.body_gravcomp)
    wp.init()
    device = wp.get_device('cuda:0')
    cases = []
    with wp.ScopedDevice(device):
        gpu_model = mjwarp.put_model(model)
        for selected in chosen:
            path = ROOT / 'reviewed-training-events' / selected['native_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == selected['native_sha256']
            with np.load(path) as data:
                index = int(np.argmin(abs(data['time_before'] - selected['event']['event_time'])))
                host = mujoco.MjData(model)
                host.qpos[:] = data['evidence_qpos_before'][index]
                host.qvel[:] = data['evidence_qvel_before'][index]
                host.ctrl[:] = data['evidence_ctrl'][index]
                host.time = float(data['time_before'][index])
            mujoco.mj_forward(model, host)
            # Avoid put_data copying native scratch fields: some MuJoCo bindings
            # produce uninitialized efm_fluid even when fluid is disabled.
            # Transfer only explicit physical state into fresh zeroed Warp data.
            with warnings.catch_warnings():
                warnings.filterwarnings('error', message='overflow encountered in cast')
                gpu = mjwarp.make_data(model, nworld=1, nconmax=256, njmax=2048)
                gpu.qpos.assign(np.asarray(host.qpos, dtype=np.float32)[None, :])
                gpu.qvel.assign(np.asarray(host.qvel, dtype=np.float32)[None, :])
                gpu.ctrl.assign(np.asarray(host.ctrl, dtype=np.float32)[None, :])
                gpu.time.assign(np.asarray([host.time], dtype=np.float32))
            mjwarp.forward(gpu_model, gpu)
            wp.synchronize_device(device)
            count = int(gpu.nacon.numpy()[0]); constraints = int(gpu.nefc.numpy()[0])
            if count >= 256 or constraints >= 2048:
                raise RuntimeError('Contact or constraint capacity reached: diagnostic invalid')
            force = wp.zeros(count, dtype=wp.spatial_vector)
            mjwarp.contact_force(gpu_model, gpu, wp.array(np.arange(count, dtype=np.int32)), False, force)
            gpuforces = force.numpy(); gpupairs = gpu.contact.geom.numpy()[:count]
            cpu_rows, gpu_rows = {}, {}
            for i in range(host.ncon):
                key = tuple(sorted((int(host.contact[i].geom1), int(host.contact[i].geom2))))
                wrench = np.zeros(6); mujoco.mj_contactForce(model, host, i, wrench)
                cpu_rows[key] = cpu_rows.get(key, 0.) + max(0., float(wrench[0]))
            for i, geoms in enumerate(gpupairs):
                key = tuple(sorted(map(int, geoms)))
                gpu_rows[key] = gpu_rows.get(key, 0.) + max(0., float(gpuforces[i][0]))
            matched = []
            for a, b in sorted(set(cpu_rows) & set(gpu_rows)):
                names = [model.geom(i).name for i in (a, b)]
                if not (any(x.startswith('red/') for x in names) and any(x.startswith('blue/') for x in names)):
                    continue
                matched.append(dict(geoms=names, native_normal_force=cpu_rows[(a, b)],
                                    gpu_normal_force=gpu_rows[(a, b)]))
            cases.append(dict(source=path.name, source_sha256=selected['native_sha256'],
                              active_opponent=selected['active_opponent'], event_time=selected['event']['event_time'],
                              saved_index=index, native_contacts=host.ncon, gpu_contacts=count, gpu_constraints=constraints,
                              native_only_pairs=[list(x) for x in sorted(set(cpu_rows)-set(gpu_rows))],
                              gpu_only_pairs=[list(x) for x in sorted(set(gpu_rows)-set(cpu_rows))],
                              matched_interbody_forces=matched,
                              body_position_max_abs=float(np.max(np.abs(gpu.xpos.numpy()[0]-host.xpos)))))
    result = dict(status='completed', device=str(device), native_cold_forwards=2, gpu_cold_forwards=2,
                  physical_steps=0, new_learner_steps=0, new_formal_controls=0, cases=cases,
                  state_transfer='Fresh zeroed GPU data with explicit qpos/qvel/ctrl/time; no native scratch copy',
                  fluid_density=model.opt.density, fluid_viscosity=model.opt.viscosity,
                  limitation='Cold-start saved poses are not original warm-started solver states; no rollout, strike classification, throughput or general physics equivalence established. Inspect solver output for warnings.')
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
