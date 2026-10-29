"""Bounded complete frozen-collector profile; preserve all development compute."""
import argparse
import cProfile
import fcntl
import io
import json
import os
from pathlib import Path
import pstats
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jp
import numpy as np
from boxer.batched_striking_env import BatchedStrikingEnv
from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params, utc
from boxer.result_archive import archive_tree
from boxer.striking_train import Actor, normalize
from boxer.striking_sampling_profile import sampling_offset
from tools.striking_contact_compare import load_checked

BASE = Path('/home/pedro/boxing-runs/gpu-collection-scaling-v1')
ARCHIVE = Path('docs/results/striking-pilot/gpu-collection-scaling-v1')
SOURCE = Path('checkpoints/striking-validation/gpu-batched-integration-v1')
SHA = 'be1b5a8d64d88ac5ae2a266f1b975dc79ac39f90f95df1082d438fc86d5327e9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, choices=('baseline', 'packed'))
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    out = BASE/args.name
    with Path('runs/unassisted_boxing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = sum(json.loads(p.read_text()).get('engine_ledger', {}).get('gpu_world_substeps', 0)
                    for p in BASE.glob('*/progress.json'))
        out.mkdir(exist_ok=False)
        state = dict(status='running', started_utc=utc(), variant=args.name, source_sha256=SHA,
            prior_development_gpu_world_substeps=prior, development_ceiling=300000,
            engine_ledger={}, new_ppo_steps=0, new_formal_controls=0, main_retained=476041216,
            formal_retained=2298509, diagnostic_ppo_retained=9505024)
        def account(value):
            state['engine_ledger'] = value
            assert prior+value['gpu_world_substeps'] <= 300000
            atomic_json(out/'progress.json', state)
        atomic_json(out/'progress.json', state)
        try:
            saved, _ = load_checked(SOURCE, SHA)
            config = json.loads((SOURCE/'config.json').read_text())
            params, manifest = load_policy_params('checkpoints/curriculum/approach')
            assert manifest['sha256'] == saved['foundation_sha256']
            foundation = UnassistedG1('approach', impl='jax')
            actor = Actor(sampling_log_std_offset=sampling_offset(saved))
            predict = jax.jit(lambda weights, obs: jp.tanh(actor.apply(weights, obs)[0]))
            env = BatchedStrikingEnv(foundation, params, worlds=128, account=account,
                engagement_distance=.45, engagement_reward_distance=.45,
                **(dict(packed_readback=True) if args.name == 'packed' else {}))
            episode = np.zeros(128, np.int64)
            seeds = 36004053+1009*np.arange(128); swap = np.arange(128)%2 == 1
            obs = env.reset(np.arange(128), seeds, swap); corner = seeds%2
            active = np.isin(np.arange(128)%8, config['active_world_indices'])
            profiler = cProfile.Profile(); timings = []; completed = cycles = captured = 0
            for control in range(64):
                if control == 8: profiler.enable()
                start = time.perf_counter()
                paired = np.zeros((128, 2, 17), np.float32)
                paired[np.arange(128), corner] = np.asarray(predict(saved['actor'],
                    jp.asarray(normalize(obs[np.arange(128), corner], saved['norm']))))
                opponent = np.asarray(predict(saved['frozen_opponent']['actor'],
                    jp.asarray(normalize(obs[np.arange(128), 1-corner], saved['frozen_opponent']['norm']))))
                paired[np.flatnonzero(active), 1-corner[active]] = opponent[active]
                obs, reward, done, outcome = env.step(paired, optimized_steps=saved['optimized_steps'])
                assert np.isfinite(obs).all() and np.isfinite(reward).all()
                cycles += int(outcome['cycles'].sum())
                for monitor in env.monitors: captured += len(monitor.take_completed_evidence())
                rows = np.flatnonzero(done)
                if len(rows):
                    completed += len(rows); episode[rows] += 1
                    obs = env.reset(rows, seeds[rows]+episode[rows], episode[rows]%2 == 1)
                    corner[rows] = (seeds[rows]+episode[rows])%2
                if control >= 8: timings.append(time.perf_counter()-start)
            profiler.disable()
            output = io.StringIO(); pstats.Stats(profiler, stream=output).strip_dirs().sort_stats('cumulative').print_stats(45)
            atomic_bytes(out/'profile.txt', output.getvalue().encode())
            state.update(status='completed', controls_per_world=64, measured_controls=56,
                measured_seconds=sum(timings), collection_world_controls_per_second=128*56/sum(timings),
                completed_episodes=completed, valid_cycles=cycles, captured_cycles=captured,
                physics_health=env.health_summary, completed_utc=utc(),
                scope='Frozen means; full motor/physics/scorecard/history/rewards/resets/accounting, with cProfile overhead; no PPO')
        except BaseException as error:
            state.update(status='failed', error=repr(error)); raise
        finally:
            atomic_json(out/'progress.json', state); archive_tree(out, ARCHIVE/args.name)
        print(json.dumps(state, indent=2))


if __name__ == '__main__': main()
