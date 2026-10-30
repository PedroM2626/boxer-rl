"""Independent completed-job audit; no new simulation, learner data or rendering."""
import hashlib
import json
from pathlib import Path
import pickle
import time
import jax
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_punch_trace import describe_events
from tools.striking_arm_effort_supervisor import independent_cycle
from tools.striking_pool_compare import verify_prerequisites
from tools.review_striking_credit_horizon import full_profiles

ROOT = Path('/home/pedro/boxing-runs/gpu-learning-v1')
OUT = Path('docs/results/striking-pilot/gpu-learning-v1')
MODELS = Path('checkpoints/striking-validation/gpu-learning-v1')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(x, y)


def main():
    begun = time.perf_counter()
    state = json.loads((ROOT/'state.json').read_text())
    assert state['status'] == 'completed' and state['new_ppo_steps'] == 1048576
    assert state['consumed_steps'] == state['optimized_steps'] == 10486784
    assert state['world_control_steps'] == 9864192 and state['diagnostic_ppo_total'] == 10569984
    assert state['main_retained'] == 476041216 and state['formal_total'] == 2340326
    assert state['prerequisites'] == verify_prerequisites()
    source = pickle.loads((ROOT/'checkpoint/initial.pkl').read_bytes())
    assert sha(ROOT/'checkpoint/initial.pkl') == state['source_sha256']
    require_ppo_checkpoint(source)
    rollouts = state['rollouts']; assert len(rollouts) == 64
    policy_hashes = {source['optimized_steps']: state['source_sha256']}
    unchanged = source.keys()-{'actor', 'critic', 'optimizer', 'norm', 'key',
        'aux_sampler_state', 'retention_sampler_state', 'consumed_steps', 'optimized_steps',
        'world_control_steps', 'mixed_credits_by_mode_and_corner'}
    for number, row in enumerate(rollouts, 1):
        delta = 16384*number
        assert row['new_ppo_steps'] == delta and np.isfinite(row['mean_loss'])
        path = ROOT/'checkpoint'/f'step-{9438208+delta}.pkl'
        assert sha(path) == row['sha256'] == sha(MODELS/path.name)
        saved = pickle.loads(path.read_bytes()); require_ppo_checkpoint(saved)
        assert saved['consumed_steps'] == saved['optimized_steps'] == 9438208+delta
        assert saved['world_control_steps'] == 8815616+delta
        assert int(saved['optimizer']['actor_adam'][0].count) == 129944+delta//1024*16
        assert int(saved['optimizer']['critic_adam'][0].count) == 153616+delta//1024*16
        assert saved['norm']['count'] == source['norm']['count']+delta
        for key in unchanged: same(source[key], saved[key])
        credits = np.asarray(saved['mixed_credits_by_mode_and_corner'])-np.asarray(source['mixed_credits_by_mode_and_corner'])
        assert (credits >= 0).all(); np.testing.assert_array_equal(credits.sum(axis=1), [delta//2, delta//2])
        policy_hashes[saved['optimized_steps']] = row['sha256']
    assert state['latest_sha256'] == rollouts[-1]['sha256']
    ledger = state['engine_ledger']
    assert ledger['world_controls'] == 1048576 and ledger['gpu_world_substeps'] == 10485888
    assert ledger['gpu_world_substeps'] == sum(ledger[k] for k in
        ('gpu_active_world_substeps', 'gpu_discarded_terminal_world_substeps', 'gpu_warmup_world_substeps'))
    assert state['physics_health']['overflow_flags_seen'] & ~(512 | 1024) == 0
    gates = []; controls = native_windows = full_windows = 0
    for gate in state['gates']:
        folder = ROOT/'gates'/gate['label']
        original, used = full_profiles(folder, gate['seeds'], gate['sha256'])
        assert original == gate['original_contact_gate'] and used == gate['controls']
        assert sha(folder/'checkpoint/policy.pkl') == gate['sha256']
        controls += used; motions = {}
        for role in (0, 1, None):
            name = 'selfplay' if role is None else f'neutral-{role}'
            assert not (folder/(name+'-errors.log')).read_text().strip()
            report = json.loads((folder/(name+'.json')).read_text())
            count = np.zeros(2, int); own = set(); reciprocal = set()
            for row in report['trials']:
                native, full = Path(row['punch_trace_file']), Path(row['trace_file'])
                assert sha(native) == row['punch_trace_sha256'] == sha(OUT/'gates'/gate['label']/name/'native'/native.name)
                assert sha(full) == row['trace_sha256'] == sha(OUT/'gates'/gate['label']/name/'full'/full.name)
                with np.load(full) as arrays: assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
                with np.load(native) as arrays:
                    assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
                    events = describe_events(arrays); assert events == row['punch_measurements']
                accepted = [e for e in events if independent_cycle(e)]
                assert all(e['native_contact_window_impulse_ns'] >= .5 for e in accepted)
                per_corner = np.array([sum(e['corner'] == c for e in accepted) for c in (0, 1)])
                count += per_corner
                if row['survived']:
                    if role is not None and per_corner[role]: own.add(row['seed'])
                    if role is None and per_corner.all(): reciprocal.add(row['seed'])
                native_windows += 1; full_windows += 1
            motions[name] = dict(valid_cycles_by_corner=count.tolist(), own_surviving_valid_seeds=sorted(own),
                reciprocal_surviving_valid_seeds=sorted(reciprocal), passes=len(reciprocal if role is None else own) >= 2)
        assert motions == gate['motion']
        assert gate['passes_complete'] == (original['passes_complete'] and all(m['passes'] for m in motions.values()))
        gates.append(dict(label=gate['label'], new_ppo_steps=gate['new_ppo_steps'], controls=used,
            sha256=gate['sha256'], passes_complete=gate['passes_complete'],
            survival={k:v['pair_survivors'] for k,v in original['profiles'].items()}, motion=motions))
    assert len(gates) == 6 and controls == 41817 and native_windows == full_windows == 144
    captures = 0; captured_policies = set()
    for path in (ROOT/'event-evidence').glob('*.json'):
        row = json.loads(path.read_text()); native = Path(row['native_file'])
        assert sha(native) == row['native_sha256'] == sha(OUT/'event-evidence'/native.name)
        for policy in row['policy_checkpoints_in_window']:
            assert policy['sha256'] == policy_hashes[policy['optimized_steps']]
            captured_policies.add(policy['optimized_steps'])
        with np.load(native) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            assert not arrays['applied_force_norm'].any()
            assert row['event'] in describe_events(arrays) and independent_cycle(row['event'])
        captures += 1
    assert captures == 2900
    rates = [dict(first_rollout=a+1, last_rollout=b,
        collection_and_update_credits_per_second=(b-a)*16384/sum(
            r['collection_seconds']+r['ppo_seconds'] for r in rollouts[a:b])) for a,b in ((0,16),(16,32),(32,48),(48,64))]
    result = dict(status='reviewed', source_sha256=state['source_sha256'], final_sha256=state['latest_sha256'],
        checkpoint_count=65, finite_and_provenance_checked=True, native_and_full_windows_checked=[native_windows, full_windows],
        captured_training_cycles_checked=captures, captured_policy_versions_checked=len(captured_policies),
        new_ppo_steps=1048576, diagnostic_ppo_total=10569984, main_retained=476041216,
        formal_new=controls, formal_total=2340326, unused_native_controls_expired=72000-controls,
        actor_adam=146328, critic_adam=170000, consumed_optimized=10486784, world_controls=9864192,
        engine_ledger=ledger, physics_health=state['physics_health'], gates=gates,
        collection_seconds=sum(r['collection_seconds'] for r in rollouts),
        ppo_seconds=sum(r['ppo_seconds'] for r in rollouts), rates_by_quarter=rates,
        new_simulation_for_review=0, new_renderer_calls=0, promotion=False, main_adoption=False,
        review_seconds=time.perf_counter()-begun,
        limitation='Known-set neutral cycles improve but no complete gate passes; fresh mirror survives 3/8 without valid reciprocal seeds. This interval does not establish convergence or native/CUDA equivalence.',
        decision='Keep the actual final; investigate growing supervision/evidence I/O and characterize native failure timing before a separately bounded continuation.')
    atomic_json(OUT/'completed-review.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('gates','engine_ledger')}, indent=2))


if __name__ == '__main__': main()
