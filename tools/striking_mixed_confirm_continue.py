"""One bounded confirmation continuation of the latest unchanged-rate mixed final."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import require_ppo_checkpoint
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate

SOURCE = Path('checkpoints/striking-validation/actor-rate-1835008/control/gate-16384')
SHA = 'c6f7fbf4c5ef941c43b8a71be7f6f6680dd657d38e96a7ed112a0f2f8f79c04b'
SEEDS = list(range(16000003, 16000007))


def progress_vector(check):
    """An early-stop measurement, never a replacement for complete physical criteria."""
    profiles = check['profiles']
    return tuple(v for name in ('neutral-0', 'neutral-1', 'selfplay') for v in
        (profiles[name]['pair_survivors'], len(profiles[name]['own_surviving_scoring_seeds']))) + (
        profiles['selfplay']['surviving_two_way_episodes'],)


def no_measured_improvement(source, current):
    return not current['passes_complete'] and all(b <= a for a, b in zip(progress_vector(source), progress_vector(current)))


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.continuation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'continuation.json'
        if path.exists(): raise ValueError('Mixed continuation exists; inspect before recovery')
        review = json.loads(Path('docs/results/striking-pilot/actor-rate-1835008/analysis.json').read_text())
        assert review['diagnostic_ppo_total'] == 1836288 and review['formal_world_controls_total'] == 1052177
        arm = next(a for a in review['arms'] if a['arm'] == 'control')
        assert arm['final_sha256'] == SHA
        baseline = arm['gates'][-1]['screen']
        saved, manifest = load_checked(SOURCE, SHA); require_ppo_checkpoint(saved)
        assert not saved.get('conservative_actor_ppo') and saved['mixed_partner_ppo'] and saved['consumed_steps'] == saved['optimized_steps'] == 1851392
        assert saved['world_control_steps'] == 1228800
        config = json.loads((SOURCE / 'config.json').read_text()); assert config['version'] == 6
        cp = args.output / 'checkpoint'; cp.mkdir(exist_ok=False)
        for filename in (manifest['file'], 'latest.json'): shutil.copy2(SOURCE / filename, cp / filename)
        shutil.copy2(SOURCE / 'config.json', args.output / 'config.json')
        atomic_json(args.output / 'progress.json', dict(consumed_steps=1851392, optimized_steps=1851392,
            world_control_steps=1228800, neutral_credits_by_corner=saved['neutral_credits_by_corner'],
            mixed_credits_by_mode_and_corner=saved['mixed_credits_by_mode_and_corner']))
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), phase='source_fresh_evaluation',
            diagnostic_output=str(args.output), source_sha256=SHA, source_consumed_offset=1851392,
            source_world_offset=1228800, maximum_new_ppo_steps=16384, maximum_evaluation_world_controls=48000,
            diagnostic_consumed_steps=0, diagnostic_collection_world_controls=0,
            diagnostic_evaluation_world_controls=0, diagnostic_ppo_before_job=1836288,
            formal_evaluation_world_controls_before_job=1052177, main_consumed_retained=1048576,
            main_project_consumed_retained=476041216, cumulative_supervised_updates_retained=8000,
            new_separate_supervised_updates=0, assistance=False, unchanged_rewards=True,
            policy_adoption='none', promotion=False, fresh_seeds=SEEDS, source_screen_vector=progress_vector(baseline), gates=[])
        atomic_json(path, state)

        def account():
            progress = json.loads((args.output / 'progress.json').read_text())
            state['diagnostic_consumed_steps'] = progress['consumed_steps'] - 1851392
            state['diagnostic_collection_world_controls'] = progress['world_control_steps'] - 1228800
            state['diagnostic_evaluation_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            state['source_fresh'] = evaluate(SOURCE, args.output / 'source-fresh', SEEDS, state, path)
            for target in (8192, 16384):
                state.update(phase='training', active_training_output=str(args.output), updated_utc=utc()); atomic_json(path, state)
                command = [sys.executable, '-u', '-m', 'boxer.striking_train', '--output', str(args.output), '--resume',
                    '--contact-ppo', '--demonstration-ppo', '--neutral-contact-ppo', '--credit-horizon-ppo',
                    '--mixed-partner-ppo', '--training-gamma', '.999', '--retention-loss-weight', '1',
                    '--ceiling', str(1851392+target), '--worlds', '8', '--unroll', '128', '--seed', '53']
                with (args.output / f'train-{target}.log').open('w') as out, (args.output / f'train-{target}-errors.log').open('w') as err:
                    subprocess.run(command, stdout=out, stderr=err, check=True)
                assert not (args.output / f'train-{target}-errors.log').read_text().strip()
                account(); assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == target
                m = json.loads((cp / 'latest.json').read_text()); current, _ = load_checked(cp, m['sha256'])
                require_ppo_checkpoint(current)
                assert current['consumed_steps'] == current['optimized_steps'] == 1851392+target
                assert current['world_control_steps'] == 1228800+target
                assert int(current['optimizer']['actor_adam'][0].count) == 11400+target//1024*16
                assert int(current['optimizer']['critic_adam'][0].count) == 35072+target//1024*16
                np.testing.assert_array_equal(np.asarray(current['mixed_credits_by_mode_and_corner']).sum(axis=1), [57344+target//2]*2)
                gate = args.output / f'gate-{target}' / 'checkpoint'; gate.mkdir(parents=True, exist_ok=False)
                for filename in (m['file'], 'latest.json'): shutil.copy2(cp / filename, gate / filename)
                shutil.copy2(args.output / 'config.json', gate / 'config.json')
                state.update(phase='screen_evaluation', updated_utc=utc()); atomic_json(path, state)
                evaluate(gate, args.output / f'gate-{target}' / 'screen', [7,23,31,47], state, path)
                full, _ = full_profiles(args.output / f'gate-{target}' / 'screen', [7,23,31,47], m['sha256'])
                state['gates'].append(dict(new_ppo_steps=target, sha256=m['sha256'], screen=full,
                    passes_complete=full['passes_complete'], progress_vector=progress_vector(full)))
                account(); atomic_json(path, state)
                if min(p['pair_survivors'] for p in full['profiles'].values()) <= 3:
                    state['early_stop_reason'] = 'Severe survival regression: at most three pairs survive a complete profile'
                    break
                if target == 8192 and no_measured_improvement(baseline, full):
                    state['early_stop_reason'] = 'No full-screen measurement improves over source at 8192; stop and review'
                    break
            state['final_fresh'] = evaluate(cp, args.output / 'final-fresh', SEEDS, state, path)
            account(); assert state['diagnostic_consumed_steps'] <= 16384 and state['diagnostic_evaluation_world_controls'] <= 48000
            state.update(status='completed', phase='awaiting_review', updated_utc=utc()); atomic_json(path, state)
        except BaseException as error:
            account(); state.update(status='failed', error=str(error), updated_utc=utc()); atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
