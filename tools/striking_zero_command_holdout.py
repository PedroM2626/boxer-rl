"""One full diagnostic of the command-offset intervention, with a fresh baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_credit_horizon import validate_horizon_state
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_contact_compare import load_checked

SOURCE = Path('checkpoints/striking-validation/credit-horizon-continue-1703936/gate-32768')
SHA = '55976fc15b74b9997b4cb56cf13683f1f87315ac6d67c5cc966ab3c644dd2d73'
SEEDS = list(range(12000003, 12000007))


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'holdout.json'
        if path.exists():
            raise ValueError('Holdout exists; inspect before recovery')
        review = json.loads(Path('docs/results/striking-pilot/horizon-control-ablation-1736704/analysis.json').read_text())
        assert review['all_12_trials_complete'] and review['formal_world_controls_total'] == 834100
        saved, _ = load_checked(SOURCE, SHA)
        validate_horizon_state(saved)
        assert saved['consumed_steps'] == saved['optimized_steps'] == 1736704
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), diagnostic_output=str(args.output),
            source_sha256=SHA, fresh_seeds=SEEDS, maximum_world_controls=36000, diagnostic_world_controls=0,
            formal_world_controls_before_job=834100, diagnostic_ppo_retained=1679616,
            new_optimizer_updates=0, assistance=False, policy_adoption='none', promotion_eligible=False, checks=[])
        atomic_json(path, state)

        def account():
            state['diagnostic_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            for name, zero_commands, seeds in (
                ('unmodified-fresh', False, SEEDS),
                ('zero-commands-screen', True, [7, 23, 31, 47]),
                ('zero-commands-fresh', True, SEEDS)):
                folder = args.output / name
                folder.mkdir(exist_ok=False)
                for role in (0, 1, None):
                    profile = 'selfplay' if role is None else f'neutral-{role}'
                    state.update(active_evaluation=str(folder / profile), updated_utc=utc())
                    atomic_json(path, state)
                    command = [sys.executable, '-u', 'tools/evaluate_striking.py', '--checkpoint', str(SOURCE),
                        '--seeds', *map(str, seeds), '--output', str(folder / (profile + '.json')),
                        '--progress', str(folder / (profile + '-progress.json'))]
                    if zero_commands:
                        command += ['--zero-command-offsets']
                    if role is not None:
                        command += ['--neutral-partner', str(role)]
                    with (folder / (profile + '.log')).open('w') as out, (folder / (profile + '-errors.log')).open('w') as err:
                        subprocess.run(command, stdout=out, stderr=err, check=True)
                    assert not (folder / (profile + '-errors.log')).read_text().strip()
                    report = json.loads((folder / (profile + '.json')).read_text())
                    assert report['diagnostic_zero_command_offsets'] is zero_commands
                    assert not report['diagnostic_zero_arm_residuals'] and not report['certification']
                    assert report['striking_sha256'] == SHA
                    account()
                    assert state['diagnostic_world_controls'] <= 36000
                    atomic_json(path, state)
                complete, count = full_profiles(folder, seeds, SHA)
                state['checks'].append(dict(name=name, zero_command_offsets=zero_commands,
                    complete_criteria=complete, world_controls=count))
                atomic_json(path, state)
            state.update(status='completed', phase='awaiting_review', updated_utc=utc())
            atomic_json(path, state)
        except BaseException as error:
            account()
            state.update(status='failed', error=str(error), updated_utc=utc())
            atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
