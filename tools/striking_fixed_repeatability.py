"""One short complete-gate repeatability check with an exact source video replay."""
import argparse
import hashlib
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
from boxer.striking_fixed_opponent import FROZEN_FINGERPRINT,fingerprint

SOURCE = Path('checkpoints/striking-validation/fixed-opponent-continue-1875968/gate-32768')
SHA = '8003a7bfdf829141634ce63019dea82a700db79613d78afaae0974346130053b'
SEEDS = list(range(20000003, 20000007))


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
    parser.add_argument('--resume',action='store_true',help='reuse a completed verified source replay after a stopped export-warning review')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.continuation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'continuation.json'
        previous=None
        if path.exists():
            if not args.resume:raise ValueError('Repeatability job exists; inspect before explicit recovery')
            previous=json.loads(path.read_text())
            assert previous['status']=='failed' and previous['phase']=='source_video_replay'
            assert previous['diagnostic_consumed_steps']==0 and previous['diagnostic_evaluation_world_controls']==1000 and not previous['gates']
            progress=json.loads((args.output/'replay-progress.json').read_text())
            assert progress['status']=='completed' and progress['simulated_world_transitions']==1000
            assert hashlib.sha256((args.output/'replay-errors.log').read_bytes()).hexdigest()=='5f745dc5b2c316c0cfcc27eaa60e9f2de3001e10ce05a73e26c4207b6073eebe'
            assert hashlib.sha256((args.output/'first-full-screen-pass-1908736.mp4').read_bytes()).hexdigest()=='69270d39a2b33e0da2df490ec2deb57e2a1e1b2cc09b010b8eb14aa1ab0944fa'
            assert not list(args.output.glob('train-*.log'))
            atomic_json(args.output/'failed-before-verified-replay-resume.json',previous)
        elif args.resume:raise ValueError('No stopped source replay exists to recover')
        review = json.loads(Path('docs/results/striking-pilot/fixed-opponent-continue-1875968/analysis.json').read_text())
        assert review['diagnostic_ppo_total'] == 1910016 and review['formal_world_controls_total'] == 1260949
        assert review['gates'][-1]['sha256']==SHA and review['original_complete_pass_streak']==1
        baseline=review['gates'][-1]['screen']
        saved, manifest = load_checked(SOURCE, SHA); require_ppo_checkpoint(saved)
        assert saved.get('fixed_opponent_ppo') and not saved.get('conservative_actor_ppo') and saved['mixed_partner_ppo'] and saved['consumed_steps'] == saved['optimized_steps'] == 1908736
        assert saved['world_control_steps'] == 1286144
        config = json.loads((SOURCE / 'config.json').read_text()); assert config['version'] == 8
        cp = args.output / 'checkpoint'
        if previous is None:
            cp.mkdir(exist_ok=False)
            for filename in (manifest['file'], 'latest.json'): shutil.copy2(SOURCE / filename, cp / filename)
            shutil.copy2(SOURCE / 'config.json', args.output / 'config.json')
            atomic_json(args.output / 'progress.json', dict(consumed_steps=1908736, optimized_steps=1908736,
                world_control_steps=1286144, neutral_credits_by_corner=saved['neutral_credits_by_corner'],
                mixed_credits_by_mode_and_corner=saved['mixed_credits_by_mode_and_corner']))
        else:
            retained,_=load_checked(cp,SHA);require_ppo_checkpoint(retained)
            assert (args.output/'config.json').read_bytes()==(SOURCE/'config.json').read_bytes()
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), phase='source_fresh_evaluation',
            diagnostic_output=str(args.output), source_sha256=SHA, source_consumed_offset=1908736,
            source_world_offset=1286144, maximum_new_ppo_steps=8192, maximum_evaluation_world_controls=37000,
            diagnostic_consumed_steps=0, diagnostic_collection_world_controls=0,
            diagnostic_evaluation_world_controls=0, diagnostic_ppo_before_job=1910016,
            formal_evaluation_world_controls_before_job=1260949, main_consumed_retained=1048576,
            main_project_consumed_retained=476041216, cumulative_supervised_updates_retained=8000,
            new_separate_supervised_updates=0, assistance=False, unchanged_rewards=True,
            policy_adoption='none', promotion=False, fresh_seeds=SEEDS, source_screen_vector=progress_vector(baseline), gates=[])
        if previous is not None:
            state.update(resumed_after_verified_renderer_warning=True,diagnostic_evaluation_world_controls=1000,render_warning_retained=True)
        atomic_json(path, state)

        def account():
            progress = json.loads((args.output / 'progress.json').read_text())
            state['diagnostic_consumed_steps'] = progress['consumed_steps'] - 1908736
            state['diagnostic_collection_world_controls'] = progress['world_control_steps'] - 1286144
            state['diagnostic_evaluation_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            # Replay a preselected complete surviving two-way source episode. All
            # physical trial fields must match; this clip cannot certify a phase.
            state.update(phase='source_video_replay',updated_utc=utc());atomic_json(path,state)
            command=[sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(SOURCE),
                '--seeds','7','--output',str(args.output/'replay.json'),'--progress',str(args.output/'replay-progress.json'),
                '--video',str(args.output/'first-full-screen-pass-1908736.mp4')]
            if previous is None:
                with (args.output/'replay.log').open('w') as out,(args.output/'replay-errors.log').open('w') as err:
                    subprocess.run(command,stdout=out,stderr=err,check=True)
                assert not (args.output/'replay-errors.log').read_text().strip()
            else:
                # Only the exact reviewed export warning is accepted; no physics
                # or training-warning exemption is introduced. Replay is not rerun.
                assert hashlib.sha256((args.output/'replay-errors.log').read_bytes()).hexdigest()=='5f745dc5b2c316c0cfcc27eaa60e9f2de3001e10ce05a73e26c4207b6073eebe'
            replay=json.loads((args.output/'replay.json').read_text())
            original=json.loads(Path('docs/results/striking-pilot/fixed-opponent-continue-1875968/gate-32768/screen/selfplay.json').read_text())
            old={(t['seed'],t['swap']):t for t in original['trials']}
            assert len(replay['trials'])==2 and replay['striking_sha256']==SHA
            for trial in replay['trials']:
                assert all(trial.get(k)==v for k,v in old[trial['seed'],trial['swap']].items()), 'Video recording changed physical outcome'
            assert replay['trials'][0]['survived'] and all(v>0 for v in replay['trials'][0]['scored_by_corner'])
            account();state.update(source_replay_physical_fields_exact=True,phase='source_fresh_evaluation',updated_utc=utc());atomic_json(path,state)
            state['source_fresh'] = evaluate(SOURCE, args.output / 'source-fresh', SEEDS, state, path)
            for target in (8192,):
                state.update(phase='training', active_training_output=str(args.output), updated_utc=utc()); atomic_json(path, state)
                command = [sys.executable, '-u', '-m', 'boxer.striking_train', '--output', str(args.output), '--resume',
                    '--contact-ppo', '--demonstration-ppo', '--neutral-contact-ppo', '--credit-horizon-ppo',
                    '--mixed-partner-ppo', '--fixed-opponent-ppo', '--training-gamma', '.999', '--retention-loss-weight', '1',
                    '--ceiling', str(1908736+target), '--worlds', '8', '--unroll', '128', '--seed', '53']
                with (args.output / f'train-{target}.log').open('w') as out, (args.output / f'train-{target}-errors.log').open('w') as err:
                    subprocess.run(command, stdout=out, stderr=err, check=True)
                assert not (args.output / f'train-{target}-errors.log').read_text().strip()
                account(); assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == target
                m = json.loads((cp / 'latest.json').read_text()); current, _ = load_checked(cp, m['sha256'])
                require_ppo_checkpoint(current)
                assert fingerprint(current['frozen_opponent']) == FROZEN_FINGERPRINT
                assert current['consumed_steps'] == current['optimized_steps'] == 1908736+target
                assert current['world_control_steps'] == 1286144+target
                assert int(current['optimizer']['actor_adam'][0].count) == 12296+target//1024*16
                assert int(current['optimizer']['critic_adam'][0].count) == 35968+target//1024*16
                np.testing.assert_array_equal(np.asarray(current['mixed_credits_by_mode_and_corner']).sum(axis=1), [86016+target//2]*2)
                gate = args.output / f'gate-{target}' / 'checkpoint'; gate.mkdir(parents=True, exist_ok=False)
                for filename in (m['file'], 'latest.json'): shutil.copy2(cp / filename, gate / filename)
                shutil.copy2(args.output / 'config.json', gate / 'config.json')
                state.update(phase='screen_evaluation', updated_utc=utc()); atomic_json(path, state)
                evaluate(gate, args.output / f'gate-{target}' / 'screen', [7,23,31,47], state, path)
                full, _ = full_profiles(args.output / f'gate-{target}' / 'screen', [7,23,31,47], m['sha256'])
                state['original_screen_pass_streak_pending_review']=2 if full['passes_complete'] else 0
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
            account(); assert state['diagnostic_consumed_steps'] <= 8192 and state['diagnostic_evaluation_world_controls'] <= 37000
            state.update(status='completed', phase='awaiting_review', updated_utc=utc()); atomic_json(path, state)
        except BaseException as error:
            account(); state.update(status='failed', error=str(error), updated_utc=utc()); atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
