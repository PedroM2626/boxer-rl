"""One bounded ordinary continuation of the latest longer-GAE final with unchanged patient consolidation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_train import require_ppo_checkpoint
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate
from boxer.striking_fixed_opponent import FROZEN_FINGERPRINT,fingerprint

SOURCE = Path('checkpoints/striking-validation/gae-continue-1982464/gate-16384')
SHA = '3052e0f69fcadc823be14069ed462ee3057ae75a8028cb2d8aa5bf3615aa48b5'
SEEDS = list(range(29000003, 29000007))


from tools.striking_mixed_partner_continue import progress_vector



def advance_frontier(frontier, failed_streak, screen):
    current=progress_vector(screen)
    if len(frontier)!=len(current):raise ValueError('unchanged progress-vector schema required')
    improved=any(b>a for a,b in zip(frontier,current))
    return tuple(max(a,b) for a,b in zip(frontier,current)), (failed_streak+1 if not screen['passes_complete'] and not improved else 0)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.continuation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'continuation.json'
        if path.exists(): raise ValueError('GAE continuation exists; inspect before recovery')
        review = json.loads(Path('docs/results/striking-pilot/gae-gradient-1998848/analysis.json').read_text())
        assert review['status']=='reviewed' and review['diagnostic_ppo_total']==2065664 and review['formal_world_controls_total']==1804660
        assert next(v for v in review['variants'] if v['variant']=='final')['sha256']==SHA
        prior=json.loads(Path('docs/results/striking-pilot/gae-continue-1982464/analysis.json').read_text())
        assert prior['gates'][-1]['sha256']==SHA
        baseline=prior['gates'][-1]['screen']
        family=json.loads(Path('docs/results/striking-pilot/gae-1966080/version-12-original-gate-history.json').read_text())
        assert family['records'][-1]['sha256']==SHA and not family['records'][-1]['screen']['passes_complete'] and family['pass_streak']==0
        saved, manifest = load_checked(SOURCE, SHA); require_ppo_checkpoint(saved)
        assert saved.get('long_gae_ppo') and saved['training_gae_lambda']==.99 and saved.get('fixed_opponent_ppo') and not saved.get('conservative_actor_ppo') and saved['mixed_partner_ppo'] and saved['consumed_steps'] == saved['optimized_steps'] == 1998848
        assert saved['world_control_steps'] == 1376256
        config = json.loads((SOURCE / 'config.json').read_text()); assert config['version'] == 12 and config['gae_lambda'] == .99
        cp = args.output / 'checkpoint'; cp.mkdir(exist_ok=False)
        for filename in (manifest['file'], 'latest.json'): shutil.copy2(SOURCE / filename, cp / filename)
        shutil.copy2(SOURCE / 'config.json', args.output / 'config.json')
        atomic_json(args.output / 'progress.json', dict(consumed_steps=1998848, optimized_steps=1998848,
            world_control_steps=1376256, neutral_credits_by_corner=saved['neutral_credits_by_corner'],
            mixed_credits_by_mode_and_corner=saved['mixed_credits_by_mode_and_corner']))
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), phase='source_fresh_evaluation',
            diagnostic_output=str(args.output), source_sha256=SHA, source_consumed_offset=1998848,
            source_world_offset=1376256, maximum_new_ppo_steps=65536, maximum_evaluation_world_controls=72000,
            diagnostic_consumed_steps=0, diagnostic_collection_world_controls=0,
            diagnostic_evaluation_world_controls=0, diagnostic_ppo_before_job=2065664,
            formal_evaluation_world_controls_before_job=1804660, main_consumed_retained=1048576,
            main_project_consumed_retained=476041216, cumulative_supervised_updates_retained=8000,
            new_separate_supervised_updates=0, assistance=False, unchanged_rewards=True,
            policy_adoption='none', promotion=False, fresh_seeds=SEEDS, source_screen_vector=progress_vector(baseline), gates=[])
        frontier=progress_vector(baseline);failed_streak=pass_streak=0
        state.update(progress_frontier=frontier,failed_plateau_streak=0,original_complete_pass_streak=0,maximum_failed_plateau_screens=3,required_distinct_consecutive_passes=2)
        atomic_json(path, state)

        def account():
            progress = json.loads((args.output / 'progress.json').read_text())
            state['diagnostic_consumed_steps'] = progress['consumed_steps'] - 1998848
            state['diagnostic_collection_world_controls'] = progress['world_control_steps'] - 1376256
            state['diagnostic_evaluation_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))
            assert state['diagnostic_consumed_steps'] <= 65536 and state['diagnostic_evaluation_world_controls'] <= 72000

        try:
            evaluate(SOURCE, args.output / 'source-fresh', SEEDS, state, path)
            state['source_fresh'], _ = full_profiles(args.output / 'source-fresh', SEEDS, SHA)
            account(); atomic_json(path,state)
            for target in (16384, 32768, 49152, 65536):
                state.update(phase='training', active_training_output=str(args.output), updated_utc=utc()); atomic_json(path, state)
                command = [sys.executable, '-u', '-m', 'boxer.striking_train', '--output', str(args.output), '--resume',
                    '--contact-ppo', '--demonstration-ppo', '--neutral-contact-ppo', '--credit-horizon-ppo',
                    '--mixed-partner-ppo', '--fixed-opponent-ppo', '--long-gae-ppo', '--training-gamma', '.999', '--retention-loss-weight', '1',
                    '--ceiling', str(1998848+target), '--worlds', '8', '--unroll', '128', '--seed', '53']
                with (args.output / f'train-{target}.log').open('w') as out, (args.output / f'train-{target}-errors.log').open('w') as err:
                    subprocess.run(command, stdout=out, stderr=err, check=True)
                assert not (args.output / f'train-{target}-errors.log').read_text().strip()
                account(); assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == target
                m = json.loads((cp / 'latest.json').read_text()); current, _ = load_checked(cp, m['sha256'])
                require_ppo_checkpoint(current)
                assert fingerprint(current['frozen_opponent']) == FROZEN_FINGERPRINT
                assert current.get('long_gae_ppo') and current['training_gae_lambda']==.99
                assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(current[k] for k in ('actor','critic','optimizer','norm','key'))))
                assert current['consumed_steps'] == current['optimized_steps'] == 1998848+target
                assert current['world_control_steps'] == 1376256+target
                assert int(current['optimizer']['actor_adam'][0].count) == 13704+target//1024*16
                assert int(current['optimizer']['critic_adam'][0].count) == 37376+target//1024*16
                np.testing.assert_array_equal(np.asarray(current['mixed_credits_by_mode_and_corner']).sum(axis=1), [131072+target//2]*2)
                gate = args.output / f'gate-{target}' / 'checkpoint'; gate.mkdir(parents=True, exist_ok=False)
                for filename in (m['file'], 'latest.json'): shutil.copy2(cp / filename, gate / filename)
                shutil.copy2(args.output / 'config.json', gate / 'config.json')
                state.update(phase='screen_evaluation', updated_utc=utc()); atomic_json(path, state)
                evaluate(gate, args.output / f'gate-{target}' / 'screen', [7,23,31,47], state, path)
                full, _ = full_profiles(args.output / f'gate-{target}' / 'screen', [7,23,31,47], m['sha256'])
                assert m['sha256'] not in {g['sha256'] for g in state['gates']}
                frontier,failed_streak=advance_frontier(frontier,failed_streak,full)
                pass_streak=pass_streak+1 if full['passes_complete'] else 0
                state.update(progress_frontier=frontier,failed_plateau_streak=failed_streak,original_complete_pass_streak=pass_streak)
                state['gates'].append(dict(new_ppo_steps=target, sha256=m['sha256'], screen=full,
                    passes_complete=full['passes_complete'], progress_vector=progress_vector(full)))
                account(); atomic_json(path, state)
                if min(p['pair_survivors'] for p in full['profiles'].values()) <= 3:
                    state['early_stop_reason'] = 'Severe survival regression: at most three pairs survive a complete profile'
                    break
                if pass_streak>=2:
                    state['early_stop_reason']='Two distinct consecutive complete passes; freeze latest for independent fresh review'
                    break
                if failed_streak>=3:
                    state['early_stop_reason']='Three consecutive failed complete screens without a new progress-frontier component'
                    break
            evaluate(cp, args.output / 'final-fresh', SEEDS, state, path)
            state['final_fresh'], _ = full_profiles(args.output / 'final-fresh', SEEDS, m['sha256'])
            account(); assert state['diagnostic_consumed_steps'] <= 65536 and state['diagnostic_evaluation_world_controls'] <= 72000
            state.update(status='completed', phase='awaiting_review', updated_utc=utc()); atomic_json(path, state)
        except BaseException as error:
            account(); state.update(status='failed', error=str(error), updated_utc=utc()); atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
