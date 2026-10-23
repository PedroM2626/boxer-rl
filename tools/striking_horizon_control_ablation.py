"""Fixed latest-final closed-loop controller ablations, never promotion gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_credit_horizon import validate_horizon_state
from tools.striking_contact_compare import load_checked
from tools.striking_horizon_regression_trace import summarize_trial

SOURCE = Path('checkpoints/striking-validation/credit-horizon-continue-1703936/gate-32768')
SHA = '55976fc15b74b9997b4cb56cf13683f1f87315ac6d67c5cc966ab3c644dd2d73'


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'ablation.json'
        if path.exists():
            raise ValueError('Ablation exists; inspect before recovery')
        review = json.loads(Path('docs/results/striking-pilot/horizon-regression-trace-1736704/analysis.json').read_text())
        assert review['all_32_original_physical_trials_exact'] and review['formal_world_controls_total'] == 828100
        saved, _ = load_checked(SOURCE, SHA)
        validate_horizon_state(saved)
        assert saved['consumed_steps'] == saved['optimized_steps'] == 1736704
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), diagnostic_output=str(args.output),
            source_sha256=SHA, maximum_world_controls=6000, diagnostic_world_controls=0,
            formal_world_controls_before_job=828100, diagnostic_ppo_retained=1679616,
            new_optimizer_updates=0, assistance=False, policy_adoption='none', promotion_eligible=False,
            unchanged_rewards=True, checks=[])
        atomic_json(path, state)

        def account():
            state['diagnostic_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            for variant, flag, key in (
                ('zero_commands', '--zero-command-offsets', 'diagnostic_zero_command_offsets'),
                ('zero_arms', '--zero-arm-residuals', 'diagnostic_zero_arm_residuals')):
                for role, seeds in ((None, [11000003, 11000006]), (1, [11000006])):
                    profile = 'selfplay' if role is None else 'neutral-1'
                    folder = args.output / (variant + '-' + profile)
                    folder.mkdir(exist_ok=False)
                    state.update(active_evaluation=str(folder), updated_utc=utc())
                    atomic_json(path, state)
                    command = [sys.executable, '-u', 'tools/evaluate_striking.py', '--checkpoint', str(SOURCE),
                        '--seeds', *map(str, seeds), flag, '--output', str(folder / 'evaluation.json'),
                        '--progress', str(folder / 'evaluation-progress.json'), '--trace-directory', str(folder / 'traces')]
                    if role is not None:
                        command += ['--neutral-partner', str(role)]
                    with (folder / 'evaluation.log').open('w') as out, (folder / 'errors.log').open('w') as err:
                        subprocess.run(command, stdout=out, stderr=err, check=True)
                    assert not (folder / 'errors.log').read_text().strip()
                    report = json.loads((folder / 'evaluation.json').read_text())
                    assert report['striking_sha256'] == SHA and report[key] is True
                    assert not report['assistance'] and not report['certification'] and not report['sampled_policy']
                    rows = report['trials']
                    assert len(rows) == len(seeds) * 2 and {(t['seed'], t['swap']) for t in rows} == {(s, w) for s in seeds for w in (False, True)}
                    assert all(t['finite'] for t in rows)
                    summaries = [summarize_trial(t) for t in rows]
                    baseline = json.loads((Path('docs/results/striking-pilot/credit-horizon-continue-1703936/final-fresh') / (profile + '.json')).read_text())
                    reference = {(t['seed'], t['swap']): t for t in baseline['trials']}
                    comparisons = [dict(seed=t['seed'], swap=t['swap'],
                        baseline_survived=reference[t['seed'], t['swap']]['survived'],
                        ablated_survived=t['survived'],
                        baseline_scored=reference[t['seed'], t['swap']]['scored_by_corner'],
                        ablated_scored=t['scored_by_corner']) for t in rows]
                    state['checks'].append(dict(variant=variant, profile=profile, trials=summaries, comparisons=comparisons))
                    account()
                    assert state['diagnostic_world_controls'] <= 6000
                    atomic_json(path, state)
            state.update(status='completed', phase='awaiting_analysis', updated_utc=utc())
            atomic_json(path, state)
        except BaseException as error:
            account()
            state.update(status='failed', error=str(error), updated_utc=utc())
            atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
