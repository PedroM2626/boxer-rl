"""Single bounded ordinary continuation of the reviewed gamma-0.999 actor."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_credit_horizon import validate_horizon_state
from tools.striking_contact_compare import load_checked
from tools.striking_ppo_anchor_compare import evaluate

SOURCE = Path('checkpoints/striking-validation/credit-horizon-1671168/gamma_0999')
SHA = '22f4908af40b7c5269952d40a98f35630c2300d50d2bf8ba54c125f36cb6dd55'
SEEDS = list(range(11000003, 11000007))


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.continuation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'continuation.json'
        if path.exists():
            raise ValueError('Continuation exists; inspect before recovery')
        review = json.loads(Path('docs/results/striking-pilot/credit-horizon-1671168/analysis.json').read_text())
        assert review['status'] == 'reviewed' and review['diagnostic_ppo_total'] == 1646848
        assert review['formal_world_controls_total'] == 770086
        assert review['arms'][1]['sha256'] == SHA
        saved, manifest = load_checked(SOURCE, SHA)
        validate_horizon_state(saved)
        assert saved['training_gamma'] == .999 and saved['consumed_steps'] == saved['optimized_steps'] == 1703936
        assert saved['world_control_steps'] == 1081344
        config = json.loads((SOURCE / 'config.json').read_text())
        assert config['version'] == 5 and config['gamma'] == .999 and config['worlds'] == 8
        checkpoint = args.output / 'checkpoint'
        checkpoint.mkdir(exist_ok=False)
        for filename in (manifest['file'], 'latest.json'):
            shutil.copy2(SOURCE / filename, checkpoint / filename)
        shutil.copy2(SOURCE / 'config.json', args.output / 'config.json')
        atomic_json(args.output / 'progress.json', dict(consumed_steps=1703936,
            optimized_steps=1703936, world_control_steps=1081344,
            neutral_credits_by_corner=saved['neutral_credits_by_corner']))
        state = dict(status='running', pid=os.getpid(), started_utc=utc(),
            phase='source_fresh_evaluation', diagnostic_output=str(args.output), source_sha256=SHA,
            source_consumed_offset=1703936, source_world_offset=1081344,
            maximum_new_ppo_steps=32768, maximum_evaluation_world_controls=48000,
            diagnostic_consumed_steps=0, diagnostic_collection_world_controls=0,
            diagnostic_evaluation_world_controls=0, diagnostic_ppo_before_job=1646848,
            formal_evaluation_world_controls_before_job=770086,
            main_consumed_retained=1048576, main_project_consumed_retained=476041216,
            cumulative_supervised_updates_retained=8000, new_separate_supervised_updates=0,
            assistance=False, unchanged_rewards=True, policy_adoption='none', promotion=False,
            fresh_seeds=SEEDS, gates=[])
        atomic_json(path, state)

        def account():
            progress = json.loads((args.output / 'progress.json').read_text())
            state['diagnostic_consumed_steps'] = progress['consumed_steps'] - 1703936
            state['diagnostic_collection_world_controls'] = progress['world_control_steps'] - 1081344
            state['diagnostic_evaluation_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            state['source_fresh'] = evaluate(SOURCE, args.output / 'source-fresh', SEEDS, state, path)
            for target in (16384, 32768):
                state.update(phase='training', active_training_output=str(args.output), updated_utc=utc())
                atomic_json(path, state)
                command = [sys.executable, '-u', '-m', 'boxer.striking_train', '--output', str(args.output),
                    '--resume', '--contact-ppo', '--demonstration-ppo', '--neutral-contact-ppo',
                    '--credit-horizon-ppo', '--training-gamma', '.999', '--retention-loss-weight', '1',
                    '--ceiling', str(1703936 + target), '--worlds', '8', '--unroll', '128', '--seed', '53']
                with (args.output / f'train-{target}.log').open('w') as out, (args.output / f'train-{target}-errors.log').open('w') as err:
                    subprocess.run(command, stdout=out, stderr=err, check=True)
                account()
                assert state['diagnostic_consumed_steps'] == state['diagnostic_collection_world_controls'] == target
                m = json.loads((checkpoint / 'latest.json').read_text())
                current, _ = load_checked(checkpoint, m['sha256'])
                validate_horizon_state(current)
                assert current['consumed_steps'] == current['optimized_steps'] == 1703936 + target
                assert int(current['optimizer']['actor_adam'][0].count) == 9096 + target // 1024 * 16
                assert int(current['optimizer']['critic_adam'][0].count) == 32768 + target // 1024 * 16
                assert sum(current['neutral_credits_by_corner']) == 65536 + target
                gate_model = args.output / f'gate-{target}' / 'checkpoint'
                gate_model.mkdir(parents=True, exist_ok=False)
                for filename in (m['file'], 'latest.json'):
                    shutil.copy2(checkpoint / filename, gate_model / filename)
                shutil.copy2(args.output / 'config.json', gate_model / 'config.json')
                state.update(phase='screen_evaluation', updated_utc=utc())
                atomic_json(path, state)
                screen = evaluate(gate_model, args.output / f'gate-{target}' / 'screen', [7, 23, 31, 47], state, path)
                complete = (screen['neutral-0']['pair_survivors'] >= 7 and screen['neutral-1']['pair_survivors'] == 8
                    and screen['selfplay']['pair_survivors'] >= 7
                    and all(len(screen[f'neutral-{role}']['own_surviving_scoring_seeds']) >= 2 for role in (0, 1)))
                state['gates'].append(dict(new_ppo_steps=target, consumed_steps=1703936 + target,
                    sha256=m['sha256'], screen=screen, passes_complete=complete))
                account()
                atomic_json(path, state)
                if min(p['pair_survivors'] for p in screen.values()) <= 3:
                    state['early_stop_reason'] = 'Severe survival regression: at most three surviving pairs in a full screen profile'
                    break
            state['final_fresh'] = evaluate(checkpoint, args.output / 'final-fresh', SEEDS, state, path)
            account()
            assert state['diagnostic_consumed_steps'] <= 32768 and state['diagnostic_evaluation_world_controls'] <= 48000
            state.update(status='completed', phase='awaiting_review', updated_utc=utc())
            atomic_json(path, state)
        except BaseException as error:
            account()
            state.update(status='failed', error=str(error), updated_utc=utc())
            atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
