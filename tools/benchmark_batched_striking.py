"""Bounded frozen-v17 collection benchmark including full rewards and resets."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jp
import numpy as np
from boxer.batched_striking_env import BatchedStrikingEnv
from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params
from boxer.striking_train import Actor, normalize, require_ppo_checkpoint
from boxer.striking_sampling_profile import sampling_offset
from boxer.result_archive import archive_tree
from tools.striking_contact_compare import load_checked
from tools.run_gpu_closed_loop import valid

OUT = Path('/home/pedro/boxing-runs/batched-gpu-benchmark-v1')
ARCHIVE = Path('docs/results/striking-pilot/batched-gpu-benchmark-v1')
SOURCE = Path('checkpoints/striking-validation/arm-effort-v17/step-2097152')
SHA = 'f311640d78941546938412f5802cff872e0a3611c392f49a3b4719eefaa1f26a'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    mechanical = json.loads(Path('/home/pedro/boxing-runs/batched-gpu-mechanics-v1/attempt-2/progress.json').read_text())
    assert mechanical['status'] == 'completed' and len(mechanical['checks']) == 5
    prior = mechanical['prior_attempt_gpu_substeps']+sum(r['gpu_world_substeps'] for r in mechanical['engine_ledgers'])
    state = dict(status='running', source_sha256=SHA, cases=[], ledgers=[], prior_development_gpu_substeps=prior,
        total_development_gpu_substep_ceiling=200000, new_ppo_steps=0, new_formal_controls=0,
        main_retained=476041216, diagnostic_ppo_retained=9504000, formal_retained=2298509)
    atomic_json(OUT/'progress.json', state)
    with Path('runs/unassisted_boxing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved, manifest = load_checked(SOURCE, SHA); require_ppo_checkpoint(saved)
        config = json.loads((SOURCE/'config.json').read_text())
        params, motor = load_policy_params('checkpoints/curriculum/approach')
        assert motor['sha256'] == saved['foundation_sha256']
        foundation = UnassistedG1('approach', impl='jax')
        actor = Actor(sampling_log_std_offset=sampling_offset(saved))
        predict = jax.jit(lambda weights, obs: jp.tanh(actor.apply(weights, obs)[0]))
        def ledger(case):
            def record(value):
                while len(state['ledgers']) <= case: state['ledgers'].append({})
                state['ledgers'][case] = value
                assert prior+sum(v['gpu_world_substeps'] for v in state['ledgers']) <= 200000
                atomic_json(OUT/'progress.json', state)
            return record
        try:
            for case, worlds in enumerate((8, 128)):
                env = BatchedStrikingEnv(foundation, params, worlds=worlds,
                    engagement_distance=config['engagement_distance'],
                    engagement_reward_distance=config['engagement_reward_distance'], account=ledger(case))
                episode = np.zeros(worlds, np.int64)
                base_seed = 35004053+1009*np.arange(worlds)
                seed = base_seed+episode; swaps = np.arange(worlds) % 2 == 1
                observation = env.reset(np.arange(worlds), seed, swaps)
                corners = seed % 2
                active = np.isin(np.arange(worlds) % 8, config['active_world_indices'])
                timings, falls, resets = [], 0, 0
                cycles, captured, first_saved = np.zeros(2), 0, set()
                for step in range(128):
                    start = time.perf_counter()
                    own = observation[np.arange(worlds), corners]
                    other = observation[np.arange(worlds), 1-corners]
                    action = np.asarray(predict(saved['actor'], jp.asarray(normalize(own, saved['norm']))))
                    opponent = np.asarray(predict(saved['frozen_opponent']['actor'],
                        jp.asarray(normalize(other, saved['frozen_opponent']['norm']))))
                    paired = np.zeros((worlds, 2, 17), np.float32)
                    paired[np.arange(worlds), corners] = action
                    paired[np.flatnonzero(active), 1-corners[active]] = opponent[active]
                    observation, reward, done, outcome = env.step(paired, optimized_steps=saved['optimized_steps'])
                    assert np.isfinite(reward).all() and np.isfinite(observation).all()
                    cycles += outcome['cycles'].sum(axis=0)
                    for world, monitor in enumerate(env.monitors):
                        for captured_event in monitor.take_completed_evidence():
                            event = captured_event['event']; assert valid(event)
                            captured += 1
                            corner = event['corner']
                            if corner not in first_saved:
                                payload = io.BytesIO(); np.savez_compressed(payload, **captured_event['arrays'])
                                path = OUT/f'worlds-{worlds}-first-corner-{corner}.npz'
                                atomic_bytes(path, payload.getvalue())
                                atomic_json(path.with_suffix('.json'), dict(event=event, world=int(world),
                                    learner_corner=int(corners[world]), active_opponent=bool(active[world]),
                                    reset_seed=int(seed[world]), swap=bool(swaps[world]),
                                    source_sha256=SHA, native_file=path.name,
                                    native_sha256=hashlib.sha256(payload.getvalue()).hexdigest(),
                                    scope='Frozen-policy development benchmark, not a formal gate or learned update'))
                                first_saved.add(corner)
                    rows = np.flatnonzero(done)
                    if len(rows):
                        falls += int(outcome['fallen'][rows].any(axis=1).sum()); resets += len(rows)
                        episode[rows] += 1; seed[rows] = base_seed[rows]+episode[rows]
                        swaps[rows] = episode[rows] % 2 == 1
                        observation = env.reset(rows, seed[rows], swaps[rows])
                        corners[rows] = seed[rows] % 2
                    elapsed = time.perf_counter()-start
                    if step >= 8: timings.append(elapsed)
                result = dict(worlds=worlds, controls_per_world=128, measured_controls=120, warmup_controls=8,
                    measured_seconds=sum(timings), full_collection_world_controls_per_second=worlds*120/sum(timings),
                    reset_episodes=resets, fall_episodes=falls, valid_cycles=cycles.tolist(), captured_valid_events=captured,
                    includes='Both actor means, motor, physics, substep metrics, histories, v17 rewards, full randomized resets, evidence handling and durable accounting',
                    excludes='Gaussian sampling, critic/PPO updates and independent native promotion gates',
                    solver_iterations=env.model.opt.iterations, line_search_iterations=env.model.opt.ls_iterations)
                state['cases'].append(result); atomic_json(OUT/'progress.json', state)
                print(json.dumps(result), flush=True)
                del env
            state.update(status='completed', decision='Review complete collection performance and correctness before the conditional 1,024-credit integration')
        except BaseException as error:
            state.update(status='failed', error=repr(error)); raise
        finally:
            atomic_json(OUT/'progress.json', state)
            archive_tree(OUT, ARCHIVE)


if __name__ == '__main__': main()
