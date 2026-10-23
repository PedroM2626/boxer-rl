"""Bounded passive matched trace of neutral-to-active transfer regression."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc
from boxer.striking_credit_horizon import validate_horizon_state
from tools.striking_contact_compare import load_checked


def summarize_trial(trial):
    """Report temporal associations only; traces alone do not identify causes."""
    contacts = [c for c in trial['first_observed_contacts']
        if {g.split('/')[0] for g in c['geoms']} == {'red', 'blue'}]
    first = min((c['elapsed'] for c in contacts), default=None)
    with np.load(trial['trace_file']) as arrays:
        assert all(np.isfinite(x).all() for x in arrays.values())
        assert hashlib.sha256(Path(trial['trace_file']).read_bytes()).hexdigest() == trial['trace_sha256']
        return dict(seed=trial['seed'], swap=trial['swap'], survived=trial['survived'],
            scored_by_corner=trial['scored_by_corner'], first_fall_time=trial['first_fall_time'],
            first_interbody_contact_time=first,
            fall_after_interbody_contact=(None if trial['survived'] or first is None else first <= trial['first_fall_time']),
            terminal_corners=trial['terminal_state']['corners'],
            minimum_pelvis_height=arrays['pelvis_height'].min(axis=0).tolist(),
            minimum_upright=arrays['upright'].min(axis=0).tolist(),
            mean_abs_command_offsets=np.abs(arrays['action'][:, :, :3]).mean(axis=0).tolist(),
            mean_abs_arm_residual=np.abs(arrays['arm_residual']).mean(axis=(0, 2)).tolist())


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'trace-review.json'
        if path.exists():
            raise ValueError('Regression trace exists; inspect before recovery')
        reference = Path('docs/results/striking-pilot/credit-horizon-continue-1703936')
        review = json.loads((reference / 'analysis.json').read_text())
        assert review['formal_world_controls_total'] == 813050 and review['diagnostic_ppo_total'] == 1679616
        jobs = [
            ('source', Path('checkpoints/striking-validation/credit-horizon-1671168/gamma_0999'),
                '22f4908af40b7c5269952d40a98f35630c2300d50d2bf8ba54c125f36cb6dd55', 'source-fresh'),
            ('final', Path('checkpoints/striking-validation/credit-horizon-continue-1703936/gate-32768'),
                '55976fc15b74b9997b4cb56cf13683f1f87315ac6d67c5cc966ab3c644dd2d73', 'final-fresh')]
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), diagnostic_output=str(args.output),
            maximum_world_controls=16000, diagnostic_world_controls=0,
            formal_world_controls_before_job=813050, diagnostic_ppo_retained=1679616,
            new_optimizer_updates=0, assistance=False, policy_adoption='none', promotion_eligible=False,
            unchanged_rewards=True, fresh_seeds=list(range(11000003, 11000007)), checks=[])
        atomic_json(path, state)

        def account():
            state['diagnostic_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            for variant, checkpoint, sha, sample in jobs:
                saved, _ = load_checked(checkpoint, sha)
                validate_horizon_state(saved)
                for role in (None, 1):
                    profile = 'selfplay' if role is None else 'neutral-1'
                    name = variant + '-' + profile
                    folder = args.output / name
                    folder.mkdir(exist_ok=False)
                    state.update(active_evaluation=str(folder), updated_utc=utc())
                    atomic_json(path, state)
                    command = [sys.executable, '-u', 'tools/evaluate_striking.py', '--checkpoint', str(checkpoint),
                        '--seeds', *[str(s) for s in state['fresh_seeds']], '--output', str(folder / 'evaluation.json'),
                        '--progress', str(folder / 'evaluation-progress.json'), '--trace-directory', str(folder / 'traces')]
                    if role is not None:
                        command += ['--neutral-partner', str(role)]
                    with (folder / 'evaluation.log').open('w') as out, (folder / 'errors.log').open('w') as err:
                        subprocess.run(command, stdout=out, stderr=err, check=True)
                    assert not (folder / 'errors.log').read_text().strip()
                    report = json.loads((folder / 'evaluation.json').read_text())
                    original = json.loads((reference / sample / (profile + '.json')).read_text())
                    old = {(t['seed'], t['swap']): t for t in original['trials']}
                    assert len(report['trials']) == 8 and report['striking_sha256'] == sha
                    assert {(t['seed'], t['swap']) for t in report['trials']} == set(old)
                    for trial in report['trials']:
                        assert all(trial.get(k) == v for k, v in old[trial['seed'], trial['swap']].items()), 'Passive trace changed original physical result'
                        assert trial['finite']
                    summaries = [summarize_trial(t) for t in report['trials']]
                    account()
                    assert state['diagnostic_world_controls'] <= 16000
                    state['checks'].append(dict(name=name, sha256=sha, all_original_physical_fields_exact=True, trials=summaries))
                    atomic_json(path, state)
            account()
            state.update(status='completed', phase='awaiting_analysis', updated_utc=utc())
            atomic_json(path, state)
        except BaseException as error:
            account()
            state.update(status='failed', error=str(error), updated_utc=utc())
            atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()
