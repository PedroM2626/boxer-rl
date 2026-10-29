"""Bounded persistent scaled-GPU PPO, with unchanged independent native gates."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
from contextlib import ExitStack

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jp
import numpy as np
from boxer.batched_ppo_update import BatchedPPOUpdate
from boxer.batched_striking_env import BatchedStrikingEnv
from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params, utc
from boxer.result_archive import archive_tree
from boxer.striking_event_evidence import save_completed_evidence
from boxer.striking_gpu_profile import BATCH_PROFILE
from boxer.striking_punch_trace import describe_events
from boxer.striking_train import normalize, log_probability, require_ppo_checkpoint
from tools.striking_arm_effort_supervisor import independent_cycle
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites
from tools.review_striking_credit_horizon import full_profiles

ROOT = Path('/home/pedro/boxing-runs/gpu-learning-v1')
ARCHIVE = Path('docs/results/striking-pilot/gpu-learning-v1')
MODELS = Path('checkpoints/striking-validation/gpu-learning-v1')
SOURCE = Path('checkpoints/striking-validation/gpu-batched-scaling-v1')
SHA = '6432029087cb00e73e39bee3a974e07f47f598771bae05c519e1bcad616c2103'
MAX_PPO, PPO_BEFORE, EVAL_BEFORE, MAX_EVAL = 1048576, 9521408, 2298509, 72000
OFFSET, WORLD_OFFSET, BLOCK = 9438208, 8815616, 16384
GIT = ['/mnt/c/Program Files/Git/cmd/git.exe', '-C', 'E:/boxer-agent']
SEEDS, FRESH = [7, 23, 31, 47], [37002003, 37002004, 37002005, 37002006]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (ROOT/'.lock', Path('runs/striking_diagnostic_supervisor.lock'), Path('runs/unassisted_boxing.lock')):
            fcntl.flock(stack.enter_context(path.open('a')), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for field in ('user.name', 'user.email'):
            assert subprocess.check_output(GIT+['config', '--get', field], text=True).strip()
        reviewed = json.loads(Path('docs/results/striking-pilot/gpu-batched-scaling-v1/independent-review.json').read_text())
        assert reviewed['status'] == 'reviewed' and reviewed['final_sha256'] == SHA
        assert reviewed['full_update_and_sampler_replayed_exactly']
        source, manifest = load_checked(SOURCE, SHA); require_ppo_checkpoint(source)
        assert source['gpu_collection_profile'] == BATCH_PROFILE
        config = json.loads((SOURCE/'config.json').read_text())
        prerequisites = verify_prerequisites(); cp = ROOT/'checkpoint'
        state_path = ROOT/'state.json'; child = [None]
        if state_path.exists():
            if not args.resume: raise ValueError('existing job requires inspected ordinary --resume')
            state = json.loads(state_path.read_text())
            assert state['source_sha256'] == SHA and state['maximum_new_ppo_steps'] == MAX_PPO
            assert state['prerequisites'] == prerequisites
            current_manifest = json.loads((cp/'latest.json').read_text())
            saved, _ = load_checked(cp, current_manifest['sha256'])
            assert state['consumed_steps'] == state['optimized_steps'] == saved['optimized_steps']
            prior_engine = state['engine_ledger'].copy()
            state.setdefault('past_process_physics_health', []).append(state.get('physics_health', {}))
            state['resumes'] += 1
            if state['resumes'] > 7: raise ValueError('bounded warm-up allocation exhausted; inspect before another resume')
        else:
            if args.resume: raise ValueError('no job to resume')
            cp.mkdir(exist_ok=False)
            saved = source
            atomic_bytes(cp/'initial.pkl', (SOURCE/manifest['file']).read_bytes())
            atomic_json(cp/'latest.json', dict(manifest, file='initial.pkl'))
            atomic_json(ROOT/'config.json', config)
            prior_engine = {}
            state = dict(status='running', phase='initializing', source_sha256=SHA,
                diagnostic_output=str(ROOT), started_utc=utc(), resumes=0, maximum_new_ppo_steps=MAX_PPO,
                maximum_new_formal_controls=MAX_EVAL, diagnostic_ppo_before=PPO_BEFORE,
                diagnostic_ppo_ceiling=PPO_BEFORE+MAX_PPO, diagnostic_ppo_total=PPO_BEFORE,
                formal_before=EVAL_BEFORE, formal_ceiling=EVAL_BEFORE+MAX_EVAL,
                formal_total=EVAL_BEFORE, formal_new=0, main_retained=476041216,
                consumed_steps=OFFSET, optimized_steps=OFFSET, world_control_steps=WORLD_OFFSET,
                new_ppo_steps=0, engine_ledger={}, gates=[], rollouts=[], prerequisites=prerequisites,
                assistance=False, optimizer_reset=False, main_adoption=False)
            state['retained_auxiliary_counters'] = dict(standalone_bc_updates=8000,
                cloned_queries=6349, surface_queries=960, render_forwards=2947)
        state.update(status='running', pid=os.getpid(), updated_utc=utc())
        def record():
            state['new_ppo_steps'] = state['consumed_steps']-OFFSET
            state['diagnostic_ppo_total'] = PPO_BEFORE+state['new_ppo_steps']
            state['formal_new'] = sum(json.loads(p.read_text())['simulated_world_transitions']
                for p in (ROOT/'gates').rglob('*-progress.json'))
            state['formal_total'] = EVAL_BEFORE+state['formal_new']
            assert 0 <= state['new_ppo_steps'] <= MAX_PPO and 0 <= state['formal_new'] <= MAX_EVAL
            state['updated_utc'] = utc(); atomic_json(state_path, state)
        record()
        def stop(signum, frame):
            if child[0] is not None and child[0].poll() is None: os.killpg(child[0].pid, signal.SIGTERM)
            raise InterruptedError('stopped; retain attempted data and inspect before ordinary resume')
        signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
        def checkpoint(value):
            require_ppo_checkpoint(value)
            used = value['optimized_steps']-OFFSET
            assert value['consumed_steps'] == value['optimized_steps'] == OFFSET+used
            assert value['world_control_steps'] == WORLD_OFFSET+used
            assert int(value['optimizer']['actor_adam'][0].count) == 129944+used//1024*16
            assert int(value['optimizer']['critic_adam'][0].count) == 153616+used//1024*16
            payload = pickle.dumps(jax.device_get(value), protocol=pickle.HIGHEST_PROTOCOL)
            name = f"step-{value['optimized_steps']}.pkl"
            atomic_bytes(cp/name, payload)
            metadata = dict(manifest, file=name, sha256=hashlib.sha256(payload).hexdigest(),
                consumed_steps=value['consumed_steps'], optimized_steps=value['optimized_steps'],
                world_control_steps=value['world_control_steps'])
            atomic_json(cp/'latest.json', metadata)
            return metadata
        def publish(label):
            record()
            try:
                archive_tree(ROOT/'gates', ARCHIVE/'gates')
                archive_tree(ROOT/'event-evidence', ARCHIVE/'event-evidence') if (ROOT/'event-evidence').exists() else None
                archive_tree(cp, MODELS); atomic_json(MODELS/'config.json', config)
                atomic_json(ARCHIVE/'state.json', state)
                atomic_json(ARCHIVE/'config.json', config)
                staged = subprocess.check_output(GIT+['diff', '--cached', '--name-only'], text=True).splitlines()
                prefixes = (str(ARCHIVE)+'/', str(MODELS)+'/')
                if any(not path.startswith(prefixes) for path in staged): raise ValueError('unrelated staged work blocks automatic publication')
                subprocess.run(GIT+['add', '--', str(ARCHIVE), str(MODELS)], check=True)
                errors = [str(path) for path in ARCHIVE.rglob('*-errors.log')]
                if errors: subprocess.run(GIT+['add', '-f', '--', *errors], check=True)
                if subprocess.run(GIT+['diff', '--cached', '--quiet']).returncode:
                    subprocess.run(GIT+['commit', '-m', f'Record bounded GPU learning {label}'], check=True)
                    subprocess.run(GIT+['push'], check=True)
                state.pop('publication_error', None)
            except Exception as error:
                state['publication_error'] = repr(error); record()
        def evaluate(label, seeds, value):
            if any(g['label'] == label for g in state['gates']): return
            assert state['formal_new']+12000 <= MAX_EVAL
            destination = ROOT/'gates'/label; destination.mkdir(parents=True, exist_ok=True)
            point = destination/'checkpoint'; point.mkdir(exist_ok=True)
            latest = json.loads((cp/'latest.json').read_text())
            atomic_bytes(point/'policy.pkl', (cp/latest['file']).read_bytes())
            atomic_json(point/'latest.json', dict(latest, file='policy.pkl'))
            atomic_json(point/'config.json', config)
            for role in (0, 1, None):
                name = 'selfplay' if role is None else f'neutral-{role}'
                if (destination/(name+'.json')).exists(): continue
                if (destination/(name+'-progress.json')).exists():
                    raise RuntimeError('partial native evaluation requires consumed-attempt review; never overwrite')
                state.update(phase='native_evaluation', active_gate=label, active_role=name); record()
                cmd = [sys.executable, '-u', 'tools/evaluate_striking.py', '--checkpoint', str(point),
                    '--seeds', *map(str, seeds), '--output', str(destination/(name+'.json')),
                    '--progress', str(destination/(name+'-progress.json')),
                    '--punch-trace-directory', str(destination/name/'native'), '--trace-directory', str(destination/name/'full')]
                if role is not None: cmd += ['--neutral-partner', str(role)]
                with (destination/(name+'.log')).open('w') as stdout, (destination/(name+'-errors.log')).open('w') as stderr:
                    child[0] = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, start_new_session=True)
                    state['active_child_pid'] = child[0].pid; record()
                    code = child[0].wait(); child[0] = None; state.pop('active_child_pid', None); record()
                if code or (destination/(name+'-errors.log')).read_text().strip():
                    raise RuntimeError(f'native evaluation failed/warned; inspect {destination/name}')
            original, controls = full_profiles(destination, seeds, latest['sha256'])
            motion = {}; events = []
            for role in (0, 1, None):
                name = 'selfplay' if role is None else f'neutral-{role}'
                report = json.loads((destination/(name+'.json')).read_text())
                own, reciprocal, counts = set(), set(), np.zeros(2, int)
                for trial in report['trials']:
                    path = Path(trial['punch_trace_file']); assert digest(path) == trial['punch_trace_sha256']
                    with np.load(path) as raw:
                        assert all(np.isfinite(raw[key]).all() for key in raw.files)
                        measured = describe_events(raw)
                        assert measured == trial['punch_measurements']
                        qualified = [e for e in measured if independent_cycle(e)]
                    full = Path(trial['trace_file']); assert digest(full) == trial['trace_sha256']
                    with np.load(full) as raw:
                        assert all(np.isfinite(raw[key]).all() for key in raw.files)
                    count = np.array([sum(e['corner'] == c for e in qualified) for c in (0, 1)])
                    counts += count; events += [dict(e, role=name, seed=trial['seed'], swap=trial['swap']) for e in qualified]
                    if trial['survived']:
                        if role is not None and count[role]: own.add(trial['seed'])
                        if role is None and count.all(): reciprocal.add(trial['seed'])
                motion[name] = dict(valid_cycles_by_corner=counts.tolist(),
                    own_surviving_valid_seeds=sorted(own), reciprocal_surviving_valid_seeds=sorted(reciprocal),
                    passes=len(reciprocal if role is None else own) >= 2)
            gate = dict(label=label, new_ppo_steps=value['optimized_steps']-OFFSET, sha256=latest['sha256'],
                seeds=seeds, controls=controls, original_contact_gate=original, motion=motion, events=events,
                passes_complete=original['passes_complete'] and all(m['passes'] for m in motion.values()),
                assistance=False, main_adoption=False, reviewed_utc=utc())
            atomic_json(destination/'motion-gate.json', gate); state['gates'].append(gate)
            assert verify_prerequisites() == prerequisites
            publish(label)
        try:
            evaluate('baseline', SEEDS, saved)
            if saved['optimized_steps']-OFFSET < MAX_PPO:
                foundation = UnassistedG1('approach', impl='jax')
                motor, motor_manifest = load_policy_params('checkpoints/curriculum/approach')
                assert motor_manifest['sha256'] == saved['foundation_sha256']
                updater = BatchedPPOUpdate(saved, config)
                @jax.jit
                def sample(weights, obs, rng):
                    mean, std = updater.actor.apply(weights, obs)
                    raw = mean+jp.exp(std)*jax.random.normal(rng, mean.shape)
                    return jp.tanh(raw), raw, log_probability(raw, mean, std)
                predict = jax.jit(lambda weights, obs: jp.tanh(updater.actor.apply(weights, obs)[0]))
                critic = jax.jit(updater.critic.apply)
                def engine_account(ledger):
                    state['engine_ledger'] = {k: v+prior_engine.get(k, 0) for k, v in ledger.items()}
                    assert state['engine_ledger']['gpu_world_substeps'] <= MAX_PPO*10+1024
                    record()
                env = BatchedStrikingEnv(foundation, motor, worlds=128, packed_readback=True,
                    engagement_distance=.45, engagement_reward_distance=.45, account=engine_account)
                episode = np.full(128, saved['consumed_steps']//128, np.int64)
                seed = 53+1009*np.arange(128)+episode; swap = np.arange(128)%2 == 1
                obs = env.reset(np.arange(128), seed, swap); corner = seed%2
                active = np.isin(np.arange(128)%8, config['active_world_indices'])
                while state['optimized_steps']-OFFSET < MAX_PPO:
                    state.update(phase='gpu_collection'); record()
                    storage = {k: [] for k in ('obs', 'raw', 'logp', 'reward', 'done', 'value')}
                    mixed = np.asarray(saved['mixed_credits_by_mode_and_corner'], np.int64).copy()
                    key = saved['key']; begun = time.perf_counter(); cycles = np.zeros(2)
                    for control in range(128):
                        own = obs[np.arange(128), corner].copy(); normalized = normalize(own, saved['norm'])
                        key, sub = jax.random.split(key)
                        action, raw, logp = sample(saved['actor'], jp.asarray(normalized), sub)
                        predicted = np.asarray(critic(saved['critic'], jp.asarray(normalized)))
                        opponent = np.asarray(predict(saved['frozen_opponent']['actor'],
                            jp.asarray(normalize(obs[np.arange(128), 1-corner], saved['frozen_opponent']['norm']))))
                        paired = np.zeros((128, 2, 17), np.float32)
                        paired[np.arange(128), corner] = np.asarray(action)
                        paired[np.flatnonzero(active), 1-corner[active]] = opponent[active]
                        state['consumed_steps'] += 128; state['world_control_steps'] += 128
                        np.add.at(mixed, (active.astype(int), corner), 1); record()
                        following, reward, done, outcome = env.step(paired, optimized_steps=saved['optimized_steps'])
                        cycles += outcome['cycles'].sum(axis=0)
                        for world, monitor in enumerate(env.monitors):
                            save_completed_evidence(monitor, ROOT/'event-evidence', cp,
                                consumed=state['consumed_steps'], world=world, learner_corner=int(corner[world]),
                                reset_seed=int(seed[world]), swap=bool(swap[world]), active_opponent=bool(active[world]),
                                foundation_sha256=saved['foundation_sha256'])
                        for name, value in dict(obs=own, raw=np.asarray(raw), logp=np.asarray(logp),
                            reward=reward[np.arange(128), corner].copy(), done=done.astype(np.float32), value=predicted).items():
                            storage[name].append(value)
                        rows = np.flatnonzero(done)
                        if len(rows):
                            episode[rows] += 1; seed[rows] = 53+1009*rows+episode[rows]; swap[rows] = episode[rows]%2 == 1
                            following = env.reset(rows, seed[rows], swap[rows]); corner[rows] = seed[rows]%2
                        obs = following
                    collection_seconds = time.perf_counter()-begun
                    rollout = {k: np.asarray(v) for k, v in storage.items()}
                    bootstrap = np.asarray(critic(saved['critic'], jp.asarray(normalize(obs[np.arange(128), corner], saved['norm']))))
                    buffer = io.BytesIO(); np.savez_compressed(buffer, **rollout, bootstrap=bootstrap)
                    atomic_bytes(ROOT/'latest-rollout.npz', buffer.getvalue())
                    state['phase'] = 'ppo_update'; record(); begun = time.perf_counter()
                    changes, losses = updater.apply(saved, rollout, bootstrap, key)
                    saved = dict(saved, **changes, consumed_steps=state['consumed_steps'],
                        optimized_steps=saved['optimized_steps']+BLOCK, world_control_steps=state['world_control_steps'],
                        mixed_credits_by_mode_and_corner=mixed.tolist())
                    metadata = checkpoint(saved); state['optimized_steps'] = saved['optimized_steps']
                    state['physics_health'] = dict(env.health_summary)
                    state['latest_sha256'] = metadata['sha256']
                    state['rollouts'].append(dict(new_ppo_steps=state['optimized_steps']-OFFSET,
                        collection_seconds=collection_seconds, ppo_seconds=time.perf_counter()-begun,
                        mean_loss=float(np.mean(losses)), completed_episodes=int(rollout['done'].sum()),
                        valid_training_cycles_by_corner=cycles.tolist(), sha256=metadata['sha256']))
                    record(); print(json.dumps(state['rollouts'][-1]), flush=True)
                    if (state['optimized_steps']-OFFSET)%262144 == 0:
                        evaluate(f"step-{state['optimized_steps']-OFFSET}", SEEDS, saved)
                        trained = [g for g in state['gates'] if g['label'].startswith('step-')]
                        if len(trained) >= 2 and all(g['passes_complete'] for g in trained[-2:]): break
            evaluate('fresh-final', FRESH, saved)
            state.update(status='completed', phase='awaiting_budget_and_physical_review', completed_utc=utc())
            publish('final')
        except BaseException as error:
            state.update(status='failed', phase='requires_investigation', error=repr(error)); record()
            publish('failed'); raise


if __name__ == '__main__': main()
