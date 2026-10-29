"""Bounded CPU/CUDA solver and open-loop dynamics study; no policy adoption.

Three solver settings are compared on BOTH engines. Higher iteration settings
are diagnostic counterfactuals, never silent edits to the training environment.
"""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import mujoco
import numpy as np
import warp as wp

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.paired_foundation import PairedFoundation
from boxer.paired_warp_physics import PairedWarpPhysics

EVENTS = Path('docs/results/striking-pilot/contact-progress-v15')


def sources(pair):
    pair.reset(seed=34002003)
    result = [dict(name='separated-reset', qpos=pair.data.qpos.copy(), qvel=pair.data.qvel.copy(),
                   controls=np.tile(pair.data.ctrl, (250, 1)), time=0., source_sha256=None)]
    selected = json.loads((EVENTS/'first-captured-event-review.json').read_text())['selected_for_render']
    for row in selected:
        path = EVENTS/'reviewed-training-events'/row['native_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['native_sha256']
        with np.load(path) as data:
            event = int(np.argmin(abs(data['time_before']-row['event']['event_time'])))
            start = event-125
            result.append(dict(name=path.stem, qpos=data['evidence_qpos_before'][start],
                qvel=data['evidence_qvel_before'][start], controls=data['evidence_ctrl'][start:start+250],
                time=float(data['time_before'][start]), source_sha256=row['native_sha256']))
    return result


def native_force(model, data, fists, targets):
    result = np.zeros((2, 2))
    for index, contact in enumerate(data.contact):
        geoms = set(map(int, contact.geom))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        for corner in range(2):
            for hand, fist in enumerate(fists[corner]):
                if fist in geoms and geoms & targets[1-corner]:
                    result[corner, hand] += max(0., wrench[0])
    return result


def gpu_force(physics, fists, targets):
    result = np.zeros((2, 2))
    _, pairs, forces = physics.normal_forces()
    for geoms, force in zip(pairs, forces):
        geoms = set(map(int, geoms))
        for corner in range(2):
            for hand, fist in enumerate(fists[corner]):
                if fist in geoms and geoms & targets[1-corner]:
                    result[corner, hand] += force
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--line-search-study', action='store_true')
    args = parser.parse_args()
    OUT = Path('/home/pedro/boxing-runs/gpu-dynamics-ls-v1' if args.line_search_study else '/home/pedro/boxing-runs/gpu-dynamics-v1')
    OUT.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', cases=[], native_substeps=0, gpu_world_substeps=0,
        main_consumed_retained=476041216, diagnostic_ppo_retained=9504000,
        formal_controls_retained=2271001, new_learner_steps=0, new_formal_controls=0,
        scope='Experimental open-loop physics, not gate evaluation or policy training')
    atomic_json(OUT/'progress.json', state)
    with Path('runs/unassisted_boxing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        pair = PairedFoundation(UnassistedG1('approach', impl='jax'))
        model = pair.model
        state['original_solver'] = dict(iterations=model.opt.iterations, ls_iterations=model.opt.ls_iterations,
            tolerance=model.opt.tolerance, ls_tolerance=model.opt.ls_tolerance, solver=int(model.opt.solver),
            timestep=model.opt.timestep, integrator=int(model.opt.integrator))
        cases = sources(pair)
        settings = ((3, 20), (30, 20), (30, 50)) if args.line_search_study else ((3, 5), (10, 5), (30, 5))
        for iterations, ls_iterations in settings:
            model.opt.iterations = iterations
            model.opt.ls_iterations = ls_iterations
            gpu = PairedWarpPhysics(model)
            warm = cases[0]
            gpu.load(warm['qpos'][None], warm['qvel'][None], warm['controls'][:1])
            state['gpu_world_substeps'] += 1
            atomic_json(OUT/'progress.json', state)
            gpu.step(); gpu.health(); gpu.capture_step()
            for source in cases:
                host = mujoco.MjData(model)
                host.qpos[:] = source['qpos']; host.qvel[:] = source['qvel']
                host.ctrl[:] = source['controls'][0]; host.time = source['time']
                mujoco.mj_forward(model, host)
                gpu.load(host.qpos[None], host.qvel[None], host.ctrl[None], [host.time])
                initial = dict(qacc_max_abs=float(np.max(abs(host.qacc-gpu.data.qacc.numpy()[0]))),
                    sensors_max_abs=float(np.max(abs(host.sensordata-gpu.data.sensordata.numpy()[0]))),
                    health=gpu.health(), native_iterations=int(host.solver_niter[0]))
                arrays = {k: [] for k in ('native_qpos', 'gpu_qpos', 'native_qvel', 'gpu_qvel',
                    'native_force', 'gpu_force', 'gpu_flags', 'native_iterations', 'gpu_iterations')}
                for ctrl in source['controls']:
                    host.ctrl[:] = ctrl
                    # Debit before attempted stepping, including a failed diagnostic.
                    state['native_substeps'] += 1; state['gpu_world_substeps'] += 1
                    atomic_json(OUT/'progress.json', state)
                    mujoco.mj_step(model, host); gpu.step(ctrl[None])
                    health = gpu.health()
                    if any(w.number for w in host.warning):
                        raise FloatingPointError('native MuJoCo warning invalidates study')
                    for key, value in dict(native_qpos=host.qpos.copy(), gpu_qpos=gpu.data.qpos.numpy()[0],
                        native_qvel=host.qvel.copy(), gpu_qvel=gpu.data.qvel.numpy()[0],
                        native_force=native_force(model, host, pair.fists, pair.targets),
                        gpu_force=gpu_force(gpu, pair.fists, pair.targets),
                        gpu_flags=health['overflow_flags'][0], native_iterations=host.solver_niter[0],
                        gpu_iterations=gpu.data.solver_niter.numpy()[0]).items():
                        arrays[key].append(value)
                arrays = {key: np.asarray(value) for key, value in arrays.items()}
                path = OUT/f"{source['name']}-iterations-{iterations}-ls-{ls_iterations}.npz"
                np.savez_compressed(path, **arrays)
                pos_error = np.max(abs(arrays['native_qpos']-arrays['gpu_qpos']), axis=1)
                vel_error = np.max(abs(arrays['native_qvel']-arrays['gpu_qvel']), axis=1)
                result = dict(name=source['name'], iterations=iterations, ls_iterations=ls_iterations,
                    source_sha256=source['source_sha256'],
                    trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), steps=len(pos_error), initial=initial,
                    qpos_max_abs=float(pos_error.max()), qpos_final_max_abs=float(pos_error[-1]),
                    qvel_max_abs=float(vel_error.max()), first_step_qpos_error=float(pos_error[0]),
                    native_legal_impulse=(arrays['native_force'].sum(axis=0)*model.opt.timestep).tolist(),
                    gpu_legal_impulse=(arrays['gpu_force'].sum(axis=0)*model.opt.timestep).tolist(),
                    solver_flagged_substeps=int(np.count_nonzero(arrays['gpu_flags'])),
                    native_max_iterations=int(arrays['native_iterations'].max()),
                    gpu_max_iterations=int(arrays['gpu_iterations'].max()))
                state['cases'].append(result); atomic_json(OUT/'progress.json', state)
                print(json.dumps(result), flush=True)
        state.update(status='completed', limitation='Cold-start open-loop 0.5 s windows; solver counterfactuals are not approved environment changes. No closed-loop, termination, reward, event-cycle or throughput equivalence.')
        atomic_json(OUT/'result.json', state); atomic_json(OUT/'progress.json', state)


if __name__ == '__main__':
    main()
