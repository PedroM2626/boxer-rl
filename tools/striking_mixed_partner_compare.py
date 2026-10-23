"""Bounded matched neutral-only versus 50/50 active-opponent coverage diagnostic."""
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
from boxer.striking_mixed_partner import prepare_mixed_state, mixed_config, SOURCE_SHA
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate

SOURCE = Path('checkpoints/striking-validation/credit-horizon-continue-1703936/gate-32768')
SEEDS = list(range(13000003, 13000007))


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'comparison.json'
        if path.exists(): raise ValueError('Mixed comparison exists; inspect before recovery')
        review = json.loads(Path('docs/results/striking-pilot/zero-command-holdout-1736704/analysis.json').read_text())
        assert review['status'] == 'reviewed' and review['formal_world_controls_total'] == 870055
        source, manifest = load_checked(SOURCE, SOURCE_SHA)
        require_ppo_checkpoint(source)
        config = json.loads((SOURCE / 'config.json').read_text())
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), phase='source_fresh_evaluation',
            diagnostic_output=str(args.output), source_sha256=SOURCE_SHA, source_consumed_offset=1736704,
            source_world_offset=1114112, maximum_new_ppo_steps=65536, new_ppo_steps_per_arm=32768,
            maximum_evaluation_world_controls=84000, diagnostic_consumed_steps=0,
            diagnostic_collection_world_controls=0, diagnostic_evaluation_world_controls=0,
            diagnostic_ppo_before_job=1679616, formal_evaluation_world_controls_before_job=870055,
            main_consumed_retained=1048576, main_project_consumed_retained=476041216,
            cumulative_supervised_updates_retained=8000, new_separate_supervised_updates=0,
            assistance=False, unchanged_rewards=True, policy_adoption='none', promotion=False,
            fresh_seeds=SEEDS, arms=[])
        atomic_json(path, state)

        def account():
            rows = [json.loads(p.read_text()) for p in (args.output / 'neutral/progress.json', args.output / 'mixed/progress.json') if p.exists()]
            state['diagnostic_consumed_steps'] = sum(p['consumed_steps'] - 1736704 for p in rows)
            state['diagnostic_collection_world_controls'] = sum(p['world_control_steps'] - 1114112 for p in rows)
            state['diagnostic_evaluation_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            state['source_fresh'] = evaluate(SOURCE, args.output / 'source-fresh', SEEDS, state, path)
            # Validate the new native path first; each integration block is retained.
            for name, mixed in (('mixed', True), ('neutral', False)):
                folder = args.output / name
                checkpoint = folder / 'checkpoint'
                checkpoint.mkdir(parents=True, exist_ok=False)
                if mixed:
                    snapshot = prepare_mixed_state(source, SOURCE_SHA)
                    data = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
                    initial_sha = hashlib.sha256(data).hexdigest()
                    atomic_bytes(checkpoint / 'initial.pkl', data)
                    atomic_json(checkpoint / 'latest.json', dict(manifest, file='initial.pkl', sha256=initial_sha,
                        mixed_partner_ppo=True, opponent_mode='mixed_current_mean'))
                    atomic_json(folder / 'config.json', mixed_config(config))
                else:
                    initial_sha = SOURCE_SHA
                    for filename in (manifest['file'], 'latest.json'):
                        shutil.copy2(SOURCE / filename, checkpoint / filename)
                    atomic_json(folder / 'config.json', config)
                progress = dict(consumed_steps=1736704, optimized_steps=1736704, world_control_steps=1114112,
                    neutral_credits_by_corner=source['neutral_credits_by_corner'])
                if mixed: progress['mixed_credits_by_mode_and_corner'] = [[0, 0], [0, 0]]
                atomic_json(folder / 'progress.json', progress)
                arm = dict(arm=name, initial_sha256=initial_sha, gates=[])
                state['arms'].append(arm)
                for target in (1024, 16384, 32768):
                    state.update(phase='training', active_arm=name, active_training_output=str(folder), updated_utc=utc())
                    atomic_json(path, state)
                    command = [sys.executable, '-u', '-m', 'boxer.striking_train', '--output', str(folder), '--resume',
                        '--contact-ppo', '--demonstration-ppo', '--neutral-contact-ppo', '--credit-horizon-ppo',
                        '--training-gamma', '.999', '--retention-loss-weight', '1', '--ceiling', str(1736704+target),
                        '--worlds', '8', '--unroll', '128', '--seed', '53']
                    if mixed: command += ['--mixed-partner-ppo']
                    with (folder / f'train-{target}.log').open('w') as out, (folder / f'train-{target}-errors.log').open('w') as err:
                        subprocess.run(command, stdout=out, stderr=err, check=True)
                    assert not (folder / f'train-{target}-errors.log').read_text().strip()
                    account()
                    m = json.loads((checkpoint / 'latest.json').read_text())
                    current, _ = load_checked(checkpoint, m['sha256'])
                    require_ppo_checkpoint(current)
                    assert current['consumed_steps'] == current['optimized_steps'] == 1736704+target
                    assert current['world_control_steps'] == 1114112+target
                    assert int(current['optimizer']['actor_adam'][0].count) == 9608+target//1024*16
                    assert int(current['optimizer']['critic_adam'][0].count) == 33280+target//1024*16
                    if mixed:
                        np.testing.assert_array_equal(np.asarray(current['mixed_credits_by_mode_and_corner']).sum(axis=1), [target//2]*2)
                    else: assert sum(current['neutral_credits_by_corner']) == 98304+target
                    if target == 1024:
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
                    if min(p['pair_survivors'] for p in screen.values()) <= 3:
                        arm['early_stop_reason'] = 'Severe survival regression: at most three pairs survive a complete screen profile'
                        break
                arm['final_checkpoint_sha256'] = m['sha256']
                arm['fresh'] = evaluate(checkpoint, folder / 'fresh', SEEDS, state, path)
                account(); atomic_json(path, state)
            assert state['diagnostic_consumed_steps'] <= 65536 and state['diagnostic_evaluation_world_controls'] <= 84000
            state.update(status='completed', phase='awaiting_review', updated_utc=utc())
            atomic_json(path, state)
        except BaseException as error:
            account()
            state.update(status='failed', error=str(error), updated_utc=utc())
            atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
