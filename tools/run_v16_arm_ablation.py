"""Bounded saved-final diagnostic: remove learned arms in deterministic mirror play."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites

ROOT = Path('/home/pedro/boxing-runs/striking-v16-arm-ablation')
SOURCE = Path('checkpoints/striking-validation/contact-rebalance-v16/step-2097152')
SHA = '5bbf6a38aa237633bf1f84893587dd1c369d9bbe8849c171b5735f9d6920a8b5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--half-arm', action='store_true')
    args = parser.parse_args()
    root = Path('/home/pedro/boxing-runs/striking-v16-half-arm') if args.half_arm else ROOT
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (Path('runs/striking_diagnostic_supervisor.lock'), Path('runs/unassisted_boxing.lock'), root / '.lock'):
            fcntl.flock(stack.enter_context(path.open('a')), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'state.json').exists():
            raise ValueError('Existing attempts must be inspected before any retry')
        load_checked(SOURCE, SHA)
        before = 2234238 if args.half_arm else 2230238
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), source_sha256=SHA,
                     diagnostic_output=str(root), prerequisites=verify_prerequisites(), maximum_new_controls=4000,
                     formal_before=before, formal_ceiling=before + 4000, new_ppo_steps=0,
                     diagnostic_zero_arm_residuals=not args.half_arm,
                     diagnostic_arm_residual_scale=.5 if args.half_arm else 1., main_retained=476041216)
        atomic_json(root / 'state.json', state)
        try:
            cmd = [sys.executable, '-u', 'tools/evaluate_striking.py', '--checkpoint', str(SOURCE),
                   '--seeds', '34001003', '34001004', '34001005', '34001006', '--seconds', '10',
                   '--output', str(root / 'zero-arms.json'),
                   '--progress', str(root / 'zero-arms-progress.json'),
                   '--trace-directory', str(root / 'full'), '--punch-trace-directory', str(root / 'native')]
            cmd += ['--arm-residual-scale', '.5'] if args.half_arm else ['--zero-arm-residuals']
            with (root / 'zero-arms.log').open('w') as out, (root / 'zero-arms-errors.log').open('w') as err:
                subprocess.run(cmd, stdout=out, stderr=err, check=True)
            assert not (root / 'zero-arms-errors.log').read_text().strip()
            report = json.loads((root / 'zero-arms.json').read_text())
            assert report['striking_sha256'] == SHA and report['diagnostic_zero_arm_residuals'] == (not args.half_arm)
            assert report.get('diagnostic_arm_residual_scale', 1.) == (.5 if args.half_arm else 1.)
            assert not report['diagnostic_zero_command_offsets'] and not report['assistance']
            assert not report['sampled_policy'] and not report['certification']
            assert len(report['trials']) == 8 and all(r['finite'] for r in report['trials'])
            assert report['simulated_world_transitions'] <= 4000
            state.update(status='completed', finished_utc=utc())
        except BaseException as error:
            state.update(status='failed', error=str(error), updated_utc=utc())
            raise
        finally:
            progress = root / 'zero-arms-progress.json'
            state['new_controls'] = json.loads(progress.read_text())['simulated_world_transitions'] if progress.exists() else 0
            state['formal_total'] = state['formal_before'] + state['new_controls']
            atomic_json(root / 'state.json', state)


if __name__ == '__main__':
    main()
