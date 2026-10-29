"""Read-only independent audit of the one-block CUDA PPO integration.

Recomputes its sampling stream on CPU from the stored source and rollout. This
does not run physics, train another policy, render, or consume evaluation credit.
"""
import hashlib
import argparse
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.result_archive import archive_tree
from boxer.striking_gpu_profile import (SOURCE, SOURCE_SHA, PROFILE, TRANSITION,
    BATCH_SOURCE, BATCH_SOURCE_SHA, BATCH_PROFILE, BATCH_TRANSITION)
from boxer.striking_sampling_profile import sampling_offset
from boxer.striking_train import Actor, normalize, log_probability, require_ppo_checkpoint
from tools.striking_contact_compare import load_checked

OUT = Path('/home/pedro/boxing-runs/gpu-batched-integration-v1')
ARCHIVE = Path('docs/results/striking-pilot/gpu-batched-integration-v1')


def same(before, after):
    assert jax.tree_util.tree_structure(before) == jax.tree_util.tree_structure(after)
    for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after)):
        np.testing.assert_array_equal(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scaled', action='store_true')
    args = parser.parse_args()
    OUT = Path('/home/pedro/boxing-runs')/('gpu-batched-scaling-v1' if args.scaled else 'gpu-batched-integration-v1')
    ARCHIVE = Path('docs/results/striking-pilot')/OUT.name
    source_path, source_sha = (BATCH_SOURCE, BATCH_SOURCE_SHA) if args.scaled else (SOURCE, SOURCE_SHA)
    block, worlds, updates, total = (16384, 128, 256, 9521408) if args.scaled else (1024, 8, 16, 9505024)
    progress = json.loads((OUT/'progress.json').read_text())
    assert progress['status'] == 'completed' and progress['new_ppo_steps'] == block
    source, _ = load_checked(Path(source_path), source_sha)
    cp = OUT/'checkpoint'
    payload = (cp/'initial.pkl').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == progress['initial_sha256']
    initial = pickle.loads(payload)
    manifest = json.loads((cp/'latest.json').read_text())
    payload = (cp/manifest['file']).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest['sha256'] == progress['final_sha256']
    final = pickle.loads(payload)
    for key in source.keys()-({'gpu_collection_profile'} if args.scaled else set()): same(source[key], initial[key])
    for candidate in (initial, final):
        require_ppo_checkpoint(candidate)
        assert candidate['gpu_collection_profile'] == (BATCH_PROFILE if args.scaled else PROFILE)
        assert candidate['gpu_collection_transition'] == TRANSITION
        if args.scaled: assert candidate['gpu_batch_transition'] == BATCH_TRANSITION
    changed = {'actor', 'critic', 'optimizer', 'norm', 'key', 'consumed_steps',
        'optimized_steps', 'world_control_steps', 'mixed_credits_by_mode_and_corner',
        'aux_sampler_state', 'retention_sampler_state'}
    for key in initial.keys()-changed: same(initial[key], final[key])
    for key in ('consumed_steps', 'optimized_steps', 'world_control_steps'):
        assert final[key] == initial[key]+block == progress[key]
    counts = {}
    for role in ('actor', 'critic'):
        before = int(initial['optimizer'][role+'_adam'][0].count)
        after = int(final['optimizer'][role+'_adam'][0].count)
        assert after-before == updates and progress[role+'_adam_count'] == after
        assert any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(initial[role]), jax.tree_util.tree_leaves(final[role])))
        counts[role] = dict(before=before, after=after, updates=after-before)
    assert final['norm']['count'] == initial['norm']['count']+block
    credits = np.asarray(final['mixed_credits_by_mode_and_corner'])-np.asarray(initial['mixed_credits_by_mode_and_corner'])
    assert credits.shape == (2, 2) and (credits >= 0).all()
    np.testing.assert_array_equal(credits.sum(axis=1), [block//2, block//2])
    with np.load(OUT/'rollout.npz') as stored:
        rollout = {k: stored[k] for k in stored.files}
    assert rollout['obs'].shape == (128, worlds, 148) and rollout['raw'].shape == (128, worlds, 17)
    assert all(np.isfinite(value).all() for value in rollout.values())
    assert np.isin(rollout['done'], [0., 1.]).all()
    actor = Actor(sampling_log_std_offset=sampling_offset(initial))
    @jax.jit
    def sample(weights, obs, rng):
        mean, std = actor.apply(weights, obs)
        raw = mean+jp.exp(std)*jax.random.normal(rng, mean.shape)
        return raw, log_probability(raw, mean, std)
    key = initial['key']
    for control in range(128):
        key, sub = jax.random.split(key)
        raw, logp = sample(initial['actor'], jp.asarray(normalize(rollout['obs'][control], initial['norm'])), sub)
        same(raw, rollout['raw'][control]); same(logp, rollout['logp'][control])
    update_replayed = False
    if args.scaled:
        from boxer.batched_ppo_update import BatchedPPOUpdate
        config = json.loads((OUT/'config.json').read_text())
        changes, losses = BatchedPPOUpdate(initial, config).apply(initial, rollout, rollout['bootstrap'], key)
        for name, values in changes.items(): same(values, final[name])
        assert len(losses) == 256
        update_replayed = True
    for _ in range(4): key, _ = jax.random.split(key)
    same(key, final['key'])
    for name in ('aux_sampler_state', 'retention_sampler_state'):
        assert initial[name] != final[name]
    assert progress['diagnostic_ppo_total'] == progress['diagnostic_ppo_ceiling'] == total
    assert progress['main_retained'] == 476041216 and progress['formal_retained'] == 2298509
    assert progress['new_formal_controls'] == 0 and progress['optimizer_reset'] is False
    ledger = progress['engine_ledger']
    assert ledger['world_controls'] == block and ledger['gpu_world_substeps'] <= block*10+worlds
    assert ledger['gpu_world_substeps'] == (ledger['gpu_active_world_substeps']
        +ledger['gpu_discarded_terminal_world_substeps']+ledger['gpu_warmup_world_substeps'])
    # Original 3/5 solver-limit bits remain visible; all capacity/float flags
    # were fatal in the live collector, including discarded terminal compute.
    assert progress['physics_health']['overflow_flags_seen'] & ~(512 | 1024) == 0
    result = dict(status='reviewed', source_sha256=source_sha, initial_sha256=progress['initial_sha256'],
        final_sha256=manifest['sha256'], all_source_leaves_preserved_at_transition=True,
        unchanged_metadata_and_frozen_opponent_preserved=True, stochastic_raw_and_logp_replayed_exactly=True,
        rng_splits_replayed_exactly=True, optimizer_counts=counts,
        full_update_and_sampler_replayed_exactly=update_replayed,
        retained_consumed=final['consumed_steps'], retained_optimized=final['optimized_steps'],
        retained_world_controls=final['world_control_steps'], diagnostic_ppo_total=total,
        mixed_mode_corner_credits=credits.tolist(), completed_episodes=int(rollout['done'].sum()),
        finite_rollout_and_parameters=True, engine_ledger=ledger, physics_health=progress['physics_health'],
        collection_seconds=progress['collection_seconds_including_initial_jit'],
        ppo_seconds=progress['ppo_update_seconds_including_initial_jit'],
        new_simulation_controls_for_review=0, new_formal_controls=0, main_adoption=False,
        limitation='One integration block is not a convergence test or a native boxing gate. The original 3/5 solver limits still occur; backend dynamics are not certified equivalent.',
        decision=('Preserve the finite scaled checkpoint. A longer GPU interval requires a prospective bounded learning/native-gate protocol.' if args.scaled else
                  'Preserve the finite integration checkpoint. Optimize collection and explicitly validate batch/update scaling before any larger GPU allocation.'))
    atomic_json(OUT/'independent-review.json', result)
    archive_tree(OUT, ARCHIVE)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
