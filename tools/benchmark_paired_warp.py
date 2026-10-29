"""Bounded current-model CUDA benchmark including real frozen-policy inference.

No optimizer, punch reward, autoreset, promotion or formal gate. Throughput is
reported in world controls (one shared physics transition) and not learner steps.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jp
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import inference, load_policy_params
from boxer.paired_foundation import PairedFoundation
from boxer.paired_warp_features import PairedFeatures, snapshot
from boxer.paired_warp_physics import PairedWarpPhysics
from boxer.striking_train import Actor, normalize, require_ppo_checkpoint

OUT = Path('/home/pedro/boxing-runs/gpu-benchmark-v1')
SHA = 'f311640d78941546938412f5802cff872e0a3611c392f49a3b4719eefaa1f26a'
SOURCE = Path('checkpoints/striking-validation/arm-effort-v17/step-2097152')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', cases=[], gpu_world_substeps=0, new_learner_steps=0,
                 new_formal_controls=0, checkpoint_sha256=SHA, main_consumed_retained=476041216,
                 diagnostic_ppo_retained=9504000, formal_controls_retained=2271001)
    atomic_json(OUT/'progress.json', state)
    payload = (SOURCE/'policy.pkl').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SHA
    saved = pickle.loads(payload); require_ppo_checkpoint(saved)
    foundation = UnassistedG1('approach', impl='jax')
    params, motor_manifest = load_policy_params('checkpoints/curriculum/approach')
    assert motor_manifest['sha256'] == 'b39b1990945e8a374802c3a190e5ac5e026ae39e7fd5ef61c5c2b01ad291c6fc'
    motor = jax.jit(inference(foundation, params))
    actor = jax.jit(Actor().apply)
    config = json.loads((SOURCE/'config.json').read_text())
    pair = PairedFoundation(foundation)
    features = PairedFeatures(pair, engagement_distance=config['engagement_distance'])
    arms = np.array([i for i in range(29) if any(word in foundation.mj_model.actuator(i).name
                     for word in ('shoulder', 'elbow', 'wrist'))])
    assert len(arms) == 14 and pair.substeps == 10
    with Path('runs/unassisted_boxing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for worlds in (1, 32, 128, 512):
            physics = PairedWarpPhysics(pair.model, worlds=worlds)
            # Disable ONLY repetitive solver-limit printing for timed runs. The
            # unchanged solver/capacity flags are read and retained each control.
            # All other warnings remain enabled. This does not change physics.
            physics.model.opt.warn_overflow &= ~(mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS)
            qpos, qvel, ctrl, frequency = [], [], [], []
            for world in range(worlds):
                pair.reset(seed=35000000+world, swap=bool(world % 2))
                qpos.append(pair.data.qpos.copy()); qvel.append(pair.data.qvel.copy())
                ctrl.append(pair.data.ctrl.copy()); frequency.append(pair.frequency.copy())
            qpos, qvel, ctrl, frequency = map(np.asarray, (qpos, qvel, ctrl, frequency))
            physics.load(qpos, qvel, ctrl)
            state['gpu_world_substeps'] += worlds
            atomic_json(OUT/'progress.json', state)
            physics.step(); physics.health(); physics.capture_step()
            phase = np.tile([0., np.pi], (worlds, 2, 1))
            last = np.zeros((worlds, 2, 29)); previous = np.zeros((worlds, 2, 17))
            residual = np.zeros((worlds, 2, 14)); history = np.zeros((worlds, 2, 2))
            physics.load(qpos, qvel, ctrl)
            latencies, flags = [], []
            for step in range(45):
                start = time.perf_counter()
                snap = snapshot(physics.data, gpu=True)
                obs = features.striking_observation(snap, last, phase, previous, history)
                mean, _ = actor(saved['actor'], jp.asarray(normalize(obs.reshape(-1, 148), saved['norm'])))
                actions = np.tanh(np.asarray(mean)).reshape(worlds, 2, 17)
                command = features.commands(snap, offsets=actions[..., :3])
                base = features.motor_observation(snap, command, last, phase).reshape(-1, 103)
                output, _ = motor({'state': jp.asarray(base), 'privileged_state': jp.zeros((worlds*2, 216))}, jax.random.PRNGKey(0))
                last = np.asarray(output).reshape(worlds, 2, 29).copy()
                residual = .8*residual+.2*actions[..., 3:]
                last[..., arms] += .8/pair.action_scale*residual
                targets = np.zeros((worlds, pair.model.nu), np.float32)
                for corner, mapping in enumerate(pair.maps):
                    targets[:, mapping['act']] = pair.pose+pair.action_scale*last[:, corner]
                state['gpu_world_substeps'] += worlds*10
                atomic_json(OUT/'progress.json', state)
                physics.data.ctrl.assign(targets)
                # Keep each substep's flags: final-state checks must not erase
                # an earlier overflow. Reading flags is part of measured cost.
                for _ in range(10):
                    physics.step()
                    overflow = physics.data.overflow.numpy()
                    solver_mask = int(mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS)
                    if np.any(overflow & ~solver_mask):
                        raise FloatingPointError(f'GPU non-solver overflow: {overflow.tolist()}')
                    flags.append(int(np.count_nonzero(overflow)))
                health = physics.health()
                phase = (phase+2*np.pi*.02*frequency+np.pi) % (2*np.pi)-np.pi
                previous = actions
                elapsed = time.perf_counter()-start
                if step >= 5:
                    latencies.append(elapsed)
            result = dict(worlds=worlds, measured_controls=40, warmup_controls=5,
                seconds=float(sum(latencies)), world_controls_per_second=float(worlds*40/sum(latencies)),
                mean_control_latency_seconds=float(np.mean(latencies)), final_health=health,
                solver_flag_observations=sum(flags), actor_and_motor_device='JAX CPU', physics_device='CUDA:0',
                model_iterations=pair.model.opt.iterations, model_ls_iterations=pair.model.opt.ls_iterations,
                timestep=pair.model.opt.timestep, substeps_per_control=10,
                accounting='Includes observation, both actors, frozen motor, state transfers, all substep overflow reads and durable accounting writes')
            state['cases'].append(result); atomic_json(OUT/'progress.json', state)
            print(json.dumps(result), flush=True)
            del physics
        state.update(status='completed', limitation='Throughput prototype: no PPO, exact punch scorecard, reset-on-fall, contact history updates or formal gate; not certified training throughput.')
        atomic_json(OUT/'result.json', state); atomic_json(OUT/'progress.json', state)


if __name__ == '__main__':
    main()
