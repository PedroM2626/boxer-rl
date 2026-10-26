"""Bounded deterministic v15 versus its actual frozen training opponent; no PPO."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from contextlib import ExitStack

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_fixed_opponent import fingerprint, FROZEN_FINGERPRINT
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites

ROOT = Path('/home/pedro/boxing-runs/striking-v15-opponent-gap')
FINAL = Path('checkpoints/striking-validation/contact-progress-v15/step-2097152')
OPPONENT = Path('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192')
ARCHIVE = Path('docs/results/striking-pilot/v15-opponent-gap')
FINAL_SHA = '9132ab4c2c86ed2b2958d551125b001370bf0f4e80f403490d33111ac3bb634a'
OPPONENT_SHA = '6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (Path('runs/striking_diagnostic_supervisor.lock'), Path('runs/unassisted_boxing.lock'), ROOT / '.lock'):
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'state.json').exists():
            raise ValueError('Existing diagnostic must be inspected; never silently repeat controls')
        final, _ = load_checked(FINAL, FINAL_SHA)
        opponent, _ = load_checked(OPPONENT, OPPONENT_SHA)
        assert fingerprint(dict(actor=opponent['actor'], norm=opponent['norm'])) == FROZEN_FINGERPRINT
        assert fingerprint(final['frozen_opponent']) == FROZEN_FINGERPRINT
        prerequisites = verify_prerequisites()
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), diagnostic_output=str(ROOT),
                     final_sha256=FINAL_SHA, opponent_sha256=OPPONENT_SHA,
                     frozen_opponent_fingerprint=FROZEN_FINGERPRINT, maximum_new_controls=8000,
                     formal_before=2183338, formal_ceiling=2191338, new_ppo_steps=0,
                     main_retained=476041216, prerequisites=prerequisites, profiles=[])
        atomic_json(ROOT / 'state.json', state)
        try:
            for corner in (0, 1):
                name = f'corner-{corner}'
                command = [sys.executable, '-u', 'tools/evaluate_striking.py', '--checkpoint', str(FINAL),
                           '--opponent-checkpoint', str(OPPONENT), '--evaluated-corner', str(corner),
                           '--seeds', '34000003', '34000004', '34000005', '34000006',
                           '--seconds', '10', '--output', str(ROOT / (name + '.json')),
                           '--progress', str(ROOT / (name + '-progress.json')),
                           '--trace-directory', str(ROOT / name / 'full'),
                           '--punch-trace-directory', str(ROOT / name / 'native')]
                state.update(active_profile=name, updated_utc=utc())
                atomic_json(ROOT / 'state.json', state)
                with (ROOT / (name + '.log')).open('w') as out, (ROOT / (name + '-errors.log')).open('w') as err:
                    subprocess.run(command, stdout=out, stderr=err, check=True)
                assert not (ROOT / (name + '-errors.log')).read_text().strip()
                report = json.loads((ROOT / (name + '.json')).read_text())
                assert report['striking_sha256'] == FINAL_SHA and report['opponent_sha256'] == OPPONENT_SHA
                assert report['evaluated_corner'] == corner and not report['sampled_policy'] and not report['assistance']
                assert all(row['finite'] for row in report['trials'])
                state['profiles'].append(dict(name=name, controls=report['simulated_world_transitions']))
            assert sum(p['controls'] for p in state['profiles']) <= 8000
            state.update(status='completed', finished_utc=utc())
        except BaseException as error:
            state.update(status='failed', error=str(error), updated_utc=utc())
            raise
        finally:
            # Durable attempted controls include unfinished evaluations on failure.
            state['new_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions']
                                        for p in ROOT.glob('*-progress.json'))
            state['formal_total'] = state['formal_before'] + state['new_controls']
            atomic_json(ROOT / 'state.json', state)


if __name__ == '__main__':
    main()
