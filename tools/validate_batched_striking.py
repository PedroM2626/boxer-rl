"""Bounded independent mechanics tests for batched current-model CUDA collection."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import mujoco
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.batched_striking_env import BatchedStrikingEnv
from boxer.paired_foundation import PairedFoundation
from boxer.warp_readback import WarpReadback
from boxer.striking_contact import ContactAwareStrikingEnv
from boxer.striking_contact_progress import ContactProgressMonitor
from boxer.striking_motion_reward import replace_contact_payment
from boxer.striking_contact_rebalance import correction
from boxer.striking_arm_effort import effort_cost
from boxer.striking_punch_trace import describe_events
from tools.run_gpu_closed_loop import valid
from boxer.result_archive import archive_tree

BASE = Path('/home/pedro/boxing-runs/batched-gpu-mechanics-v1')
ARCHIVE_BASE = Path('docs/results/striking-pilot/batched-gpu-mechanics-v1')


def zero_motor(obs, key):
    return np.zeros((obs['state'].shape[0], 29), np.float32), {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--scaling-check', action='store_true', help='new bounded packed-readback regression; never rewrite prior results')
    args = parser.parse_args()
    if args.attempt < 1: raise ValueError('positive reviewed attempt number required')
    validation_root, archive_base = BASE, ARCHIVE_BASE
    if args.scaling_check:
        validation_root = Path('/home/pedro/boxing-runs/gpu-collection-scaling-v1')
        archive_base = Path('docs/results/striking-pilot/gpu-collection-scaling-v1')
    name = 'semantics' if args.attempt == 1 else f'semantics-attempt-{args.attempt}'
    OUT = validation_root/name if args.scaling_check else validation_root if args.attempt == 1 else validation_root/f'attempt-{args.attempt}'
    ARCHIVE = archive_base/name if args.scaling_check else archive_base if args.attempt == 1 else archive_base/f'attempt-{args.attempt}'
    prior = 0
    if args.scaling_check:
        for p in validation_root.glob('*/progress.json'):
            previous = json.loads(p.read_text())
            prior += previous.get('engine_ledger', {}).get('gpu_world_substeps', 0)
            prior += sum(e['gpu_world_substeps'] for e in previous.get('engine_ledgers', []))
            prior += previous.get('reference_gpu_substeps', 0)
        for n in range(1, args.attempt):
            previous_name = 'semantics' if n == 1 else f'semantics-attempt-{n}'
            assert json.loads((validation_root/previous_name/'progress.json').read_text())['status'] == 'failed'
    for n in range(1, args.attempt) if not args.scaling_check else []:
        path = BASE/'progress.json' if n == 1 else BASE/f'attempt-{n}/progress.json'
        old = json.loads(path.read_text()); assert old['status'] == 'failed'
        prior += sum(e['gpu_world_substeps'] for e in old['engine_ledgers'])+old['reference_gpu_substeps']
    OUT.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', checks=[], engine_ledgers=[], reference_gpu_substeps=0,
        prior_attempt_gpu_substeps=prior, manual_gpu_world_forwards=0,
        maximum_development_gpu_substeps=300000 if args.scaling_check else 200000, new_ppo_steps=0, new_formal_controls=0,
        main_retained=476041216, diagnostic_ppo_retained=9505024 if args.scaling_check else 9504000, formal_retained=2298509)
    atomic_json(OUT/'progress.json', state)
    with Path('runs/unassisted_boxing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        def ledger(index):
            def account(value):
                while len(state['engine_ledgers']) <= index: state['engine_ledgers'].append({})
                state['engine_ledgers'][index] = value
                used = prior+sum(r.get('gpu_world_substeps', 0) for r in state['engine_ledgers'])+state['reference_gpu_substeps']
                assert used <= state['maximum_development_gpu_substeps']
                atomic_json(OUT/'progress.json', state)
                atomic_json(validation_root/'current.json', dict(attempt=args.attempt, output=str(OUT), status=state['status']))
            return account
        def checked(name, **details):
            state['checks'].append(dict(name=name, **details)); atomic_json(OUT/'progress.json', state)
            print(json.dumps(state['checks'][-1]), flush=True)
        try:
            foundation = UnassistedG1('approach', impl='jax')
            env = BatchedStrikingEnv(foundation, worlds=1, motor=zero_motor, account=ledger(0), capture_evidence=False,
                                     packed_readback=args.scaling_check)
            reference_pair = PairedFoundation(foundation)
            reference = ContactAwareStrikingEnv(foundation, None, pair=reference_pair, motor=zero_motor,
                engagement_distance=.45, engagement_reward_distance=.45)
            readback = WarpReadback(env.physics.data)
            snapshots = []
            def observe(physics):
                data = mujoco.MjData(reference_pair.model)
                readback.export(data, reference_pair.model)
                snapshots.append(data)
            env.substep_observer = observe
            rng = np.random.default_rng(73)
            max_error = 0.; controls = 0
            for swap in (False, True):
                actual = env.reset([0], [73], [swap])[0]
                reference.reset(73, swap)
                readback.export(reference_pair.data, reference_pair.model)
                expected = reference.observation()
                monitor = ContactProgressMonitor(); reference_pair.diagnostic_punch_trace = monitor
                np.testing.assert_allclose(actual, expected, atol=1e-7)
                for _ in range(30):
                    action = rng.uniform(-.2, .2, (2, 17)).astype(np.float32)
                    snapshots.clear()
                    obs, reward, done, result = env.step(action[None])
                    replay_index = [0]
                    def replay():
                        data = snapshots[replay_index[0]]
                        np.testing.assert_array_equal(reference_pair.data.ctrl.astype(np.float32), data.ctrl.astype(np.float32))
                        mujoco.mj_copyData(reference_pair.data, reference_pair.model, data)
                        replay_index[0] += 1
                    reference_pair._physics_step = replay
                    robs, base, rdone, rout = reference.step(action)
                    assert replay_index[0] == len(snapshots)
                    cy, dense = monitor.consume(terminal=rdone)
                    expected_reward = [replace_contact_payment(base[c:c+1], rout, cy, dense, c)[0]
                        +correction(monitor.last_components, c)-effort_cost(rout, c, .02) for c in (0, 1)]
                    max_error = max(max_error, float(np.max(abs(obs[0]-robs))))
                    np.testing.assert_allclose(obs[0], robs, atol=3e-6, rtol=1e-6)
                    np.testing.assert_allclose(reward[0], expected_reward, atol=2e-7)
                    np.testing.assert_array_equal(result['scored'][0], rout['scored'])
                    np.testing.assert_array_equal(result['fallen'][0], rout['fallen'])
                    np.testing.assert_array_equal(result['cycles'][0], cy)
                    assert bool(done[0]) == rdone
                    controls += 1
                    if rdone: break
            checked('identical_cuda_trajectory_native_control_reward_and_termination', controls=controls, observation_error_max=max_error)
            batch = BatchedStrikingEnv(foundation, worlds=2, motor=zero_motor, account=ledger(1), capture_evidence=False,
                                       packed_readback=args.scaling_check)
            batch.reset([0, 1], [47, 73], [False, True])
            for _ in range(3): batch.step(rng.uniform(-.2, .2, (2, 2, 17)))
            physics_before = {k: getattr(batch.physics.data, k).numpy()[0].copy() for k in batch.reset_fields}
            controller_before = {k: getattr(batch, k)[0].copy() for k in ('phase', 'frequency', 'last_action', 'previous', 'residual', 'age')}
            history_before = batch.monitors[0].arrays()
            batch.reset([1], [123], [False])
            for k, before in physics_before.items(): np.testing.assert_array_equal(getattr(batch.physics.data, k).numpy()[0], before)
            for k, before in controller_before.items(): np.testing.assert_array_equal(getattr(batch, k)[0], before)
            for k, before in history_before.items(): np.testing.assert_array_equal(batch.monitors[0].arrays()[k], before)
            assert not batch.monitors[1].rows and not np.isfinite(batch.scorecard.last_score[1]).any()
            assert batch.age[1] == 0 and not batch.previous[1].any() and not batch.residual[1].any()
            np.testing.assert_array_equal(batch.physics.data.qacc_warmstart.numpy()[1], 0.)
            checked('selective_reset_preserves_live_physics_controller_and_history')
            # Initial low-pelvis state tests the stopping boundary, not stand-up
            # learning. The second world continues without receiving support.
            batch.reset([0, 1], [47, 73], [False, True])
            qpos = batch.physics.data.qpos.numpy(); qpos[0, batch.pair.maps[0]['root']+2] = .5
            state['manual_gpu_world_forwards'] += 2
            atomic_json(OUT/'progress.json', state)
            batch.physics.load(qpos, batch.physics.data.qvel.numpy(), batch.physics.data.ctrl.numpy())
            from boxer.paired_batch_metrics import metric_snapshot
            batch.state = metric_snapshot(batch.physics.data, gpu=True)
            _, _, done, result = batch.step(np.zeros((2, 2, 17)))
            assert done[0] and not done[1] and result['fallen'][0, 0]
            assert abs(result['elapsed'][0]-.002) < 1e-8 and abs(result['elapsed'][1]-.02) < 1e-7
            assert len(batch.monitors[0].rows) == 1 and len(batch.monitors[1].rows) == 10
            assert batch.accounting['gpu_discarded_terminal_world_substeps'] >= 9
            batch.reset([0], [101], [True])
            batch.age[1] = 499
            _, _, done, result = batch.step(np.zeros((2, 2, 17)))
            assert done[1] and result['timed_out'][1] and not result['fallen'][1].any()
            checked('first_fall_snapshot_partial_control_discard_and_timeout')
            roots = Path('docs/results/striking-pilot/contact-progress-v15')
            selected = json.loads((roots/'first-captured-event-review.json').read_text())['selected_for_render']
            for event_source in selected:
                path = roots/'reviewed-training-events'/event_source['native_file']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == event_source['native_sha256']
                with np.load(path) as f: data = {k: f[k].copy() for k in f.files}
                monitor = ContactProgressMonitor(); total = np.zeros(2)
                for start in range(0, len(data['time_before']), 10):
                    end = min(start+10, len(data['time_before']))
                    for i in range(start, end): monitor.append_row({k: a[i].copy() for k, a in data.items()})
                    after_distance = data['target_center_distance'][min(end, len(data['time_before'])-1)]
                    cycles, _ = monitor.consume(target_distances_after=after_distance)
                    total += cycles
                    before_phi = np.exp(-data['target_center_distance'][start]/.3).sum(axis=1)
                    after_phi = np.exp(-after_distance/.3).sum(axis=1)
                    np.testing.assert_allclose(monitor.last_components['target_potential'], 2*(.999*after_phi-before_phi), atol=1e-14)
                events = describe_events(data)
                expected = [sum(e['corner'] == c and valid(e) and data['time_after'][-1] >= e['event_time']+.5 for e in events) for c in (0, 1)]
                np.testing.assert_array_equal(total, expected)
                assert total.sum() > 0
                checked('positive_recorded_cycle_history_and_potential', source_sha256=event_source['native_sha256'], cycles=total.tolist())
            state.update(status='completed', decision='Mechanical checks passed; validate learned collection/benchmark before any PPO integration')
        except BaseException as error:
            state.update(status='failed', error=repr(error)); raise
        finally:
            atomic_json(OUT/'progress.json', state)
            atomic_json(validation_root/'current.json', dict(attempt=args.attempt, output=str(OUT), status=state['status']))
            archive_tree(OUT, ARCHIVE)


if __name__ == '__main__': main()
