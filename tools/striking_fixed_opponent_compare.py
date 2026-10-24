"""Bounded stationary-opponent versus current-opponent matched training comparison."""
import argparse
import json
import os
from pathlib import Path
import pickle
import hashlib
import shutil
import subprocess
import sys

import numpy as np

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_train import utc
from boxer.striking_fixed_opponent import prepare_fixed_state, fixed_config, SOURCE_SHA, FROZEN_FINGERPRINT
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_mixed_partner_continue import no_measured_improvement,progress_vector

SOURCE = Path('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192')
SEEDS = list(range(18000003, 18000007))


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'comparison.json'
        if path.exists(): raise ValueError('Fixed-opponent comparison exists; inspect before recovery')
        review = json.loads(Path('docs/results/striking-pilot/base-yaw-1859584/analysis.json').read_text())
        assert review['status'] == 'reviewed' and review['formal_world_controls_total'] == 1139761 and review['diagnostic_ppo_total'] == 1844480
        source, manifest = load_checked(SOURCE, SOURCE_SHA)
        require_ppo_checkpoint(source)
        config = json.loads((SOURCE / 'config.json').read_text())
        baseline=json.loads(Path('docs/results/striking-pilot/mixed-confirm-1851392/analysis.json').read_text())['gates'][-1]['screen']
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), phase='source_fresh_evaluation',
            diagnostic_output=str(args.output), source_sha256=SOURCE_SHA, source_consumed_offset=1859584,
            source_world_offset=1236992, maximum_new_ppo_steps=32768, new_ppo_steps_per_arm=16384,
            maximum_evaluation_world_controls=84000, diagnostic_consumed_steps=0,
            diagnostic_collection_world_controls=0, diagnostic_evaluation_world_controls=0,
            diagnostic_ppo_before_job=1844480, formal_evaluation_world_controls_before_job=1139761,
            main_consumed_retained=1048576, main_project_consumed_retained=476041216,
            cumulative_supervised_updates_retained=8000, new_separate_supervised_updates=0,
            assistance=False, unchanged_rewards=True, policy_adoption='none', promotion=False,
            fresh_seeds=SEEDS, arms=[])
        atomic_json(path, state)

        def account():
            rows = [json.loads(p.read_text()) for p in (args.output / 'control/progress.json', args.output / 'fixed_opponent/progress.json') if p.exists()]
            state['diagnostic_consumed_steps'] = sum(p['consumed_steps'] - 1859584 for p in rows)
            state['diagnostic_collection_world_controls'] = sum(p['world_control_steps'] - 1236992 for p in rows)
            state['diagnostic_evaluation_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            state['source_fresh'] = evaluate(SOURCE, args.output / 'source-fresh', SEEDS, state, path)
            # Validate the new native path first; each integration block is retained.
            for name, fixed in (('fixed_opponent', True), ('control', False)):
                folder = args.output / name
                checkpoint = folder / 'checkpoint'
                checkpoint.mkdir(parents=True, exist_ok=False)
                if fixed:
                    snapshot = prepare_fixed_state(source, SOURCE_SHA)
                    data = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
                    initial_sha = hashlib.sha256(data).hexdigest()
                    atomic_bytes(checkpoint / 'initial.pkl', data)
                    atomic_json(checkpoint / 'latest.json', dict(manifest, file='initial.pkl', sha256=initial_sha,
                        fixed_opponent_ppo=True, fixed_opponent_fingerprint=FROZEN_FINGERPRINT, opponent_mode='mixed_frozen_mean'))
                    atomic_json(folder / 'config.json', fixed_config(config))
                else:
                    initial_sha = SOURCE_SHA
                    for filename in (manifest['file'], 'latest.json'):
                        shutil.copy2(SOURCE / filename, checkpoint / filename)
                    atomic_json(folder / 'config.json', config)
                progress = dict(consumed_steps=1859584, optimized_steps=1859584, world_control_steps=1236992,
                    neutral_credits_by_corner=source['neutral_credits_by_corner'])
                progress['mixed_credits_by_mode_and_corner'] = source['mixed_credits_by_mode_and_corner']
                atomic_json(folder / 'progress.json', progress)
                arm = dict(arm=name, initial_sha256=initial_sha, gates=[])
                state['arms'].append(arm)
                for target in (1024, 8192, 16384):
                    state.update(phase='training', active_arm=name, active_training_output=str(folder), updated_utc=utc())
                    atomic_json(path, state)
                    command = [sys.executable, '-u', '-m', 'boxer.striking_train', '--output', str(folder), '--resume',
                        '--contact-ppo', '--demonstration-ppo', '--neutral-contact-ppo', '--credit-horizon-ppo', '--mixed-partner-ppo',
                        '--training-gamma', '.999', '--retention-loss-weight', '1', '--ceiling', str(1859584+target),
                        '--worlds', '8', '--unroll', '128', '--seed', '53']
                    if fixed: command += ['--fixed-opponent-ppo']
                    if target == 1024: command += ['--update-audit-directory', str(folder / 'update-audit')]
                    with (folder / f'train-{target}.log').open('w') as out, (folder / f'train-{target}-errors.log').open('w') as err:
                        subprocess.run(command, stdout=out, stderr=err, check=True)
                    assert not (folder / f'train-{target}-errors.log').read_text().strip()
                    account()
                    m = json.loads((checkpoint / 'latest.json').read_text())
                    current, _ = load_checked(checkpoint, m['sha256'])
                    require_ppo_checkpoint(current)
                    assert current['consumed_steps'] == current['optimized_steps'] == 1859584+target
                    assert current['world_control_steps'] == 1236992+target
                    assert int(current['optimizer']['actor_adam'][0].count) == 11528+target//1024*16
                    assert int(current['optimizer']['critic_adam'][0].count) == 35200+target//1024*16
                    np.testing.assert_array_equal(np.asarray(current['mixed_credits_by_mode_and_corner']).sum(axis=1), [61440+target//2]*2)
                    assert current['neutral_credits_by_corner'] == [50229,48075]
                    if target == 1024:
                        audit=json.loads((folder/'update-audit/minibatch-measurements.json').read_text())
                        assert len(audit['records'])==18
                        arm['initial_update_geometry']=audit['records'][-1]
                        if fixed:
                            assert current['fixed_opponent_fingerprint']==FROZEN_FINGERPRINT
                        else:
                            other=pickle.loads((args.output/'fixed_opponent/checkpoint/step-1860608.pkl').read_bytes())
                            import jax
                            for key in ('actor','critic','optimizer','norm','key','aux_sampler_state','retention_sampler_state'):
                                for a,b in zip(jax.tree_util.tree_leaves(other[key]),jax.tree_util.tree_leaves(current[key])):
                                    np.testing.assert_equal(a,b)
                            arm['initial_block_learned_state_matches_fixed_arm']=True
                        atomic_json(folder / 'initial-block-verified.json', dict(json.loads((folder/'progress.json').read_text()),
                            checkpoint_sha256=m['sha256'], included_in_arm_allocation=True))
                        atomic_json(path, state)
                        continue
                    gate = folder / f'gate-{target}' / 'checkpoint'
                    gate.mkdir(parents=True, exist_ok=False)
                    for filename in (m['file'], 'latest.json'): shutil.copy2(checkpoint / filename, gate / filename)
                    shutil.copy2(folder / 'config.json', gate / 'config.json')
                    state.update(phase='screen_evaluation', updated_utc=utc())
                    atomic_json(path, state)
                    screen = evaluate(gate, folder / f'gate-{target}' / 'screen', [7,23,31,47], state, path)
                    complete = (screen['neutral-0']['pair_survivors'] >= 7 and screen['neutral-1']['pair_survivors'] == 8
                        and screen['selfplay']['pair_survivors'] >= 7
                        and all(len(screen[f'neutral-{r}']['own_surviving_scoring_seeds']) >= 2 for r in (0,1)))
                    arm['gates'].append(dict(new_ppo_steps=target, sha256=m['sha256'], screen=screen, passes_complete=complete))
                    account(); atomic_json(path, state)
                    full,_=full_profiles(folder/f'gate-{target}'/'screen',[7,23,31,47],m['sha256'])
                    if target==8192 and no_measured_improvement(baseline,full):
                        arm['early_stop_reason']='No first complete-screen measurement improves over source; stop and review'
                        break
                    if min(p['pair_survivors'] for p in screen.values()) <= 3:
                        arm['early_stop_reason'] = 'Severe survival regression: at most three pairs survive a complete screen profile'
                        break
                arm['final_checkpoint_sha256'] = m['sha256']
                arm['fresh'] = evaluate(checkpoint, folder / 'fresh', SEEDS, state, path)
                account(); atomic_json(path, state)
            assert state['diagnostic_consumed_steps'] <= 32768 and state['diagnostic_evaluation_world_controls'] <= 84000
            state.update(status='completed', phase='awaiting_review', updated_utc=utc())
            atomic_json(path, state)
        except BaseException as error:
            account()
            state.update(status='failed', error=str(error), updated_utc=utc())
            atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
