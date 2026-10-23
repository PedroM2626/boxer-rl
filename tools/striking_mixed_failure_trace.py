"""Bounded passive source/final trace after mixed-profile confirmation stagnation."""
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
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked


def summarize_trial(trial):
    """Temporal associations only; no causal inference from an unmodified replay."""
    from tools.striking_horizon_regression_trace import summarize_trial as basic
    result=basic(trial)
    with np.load(trial['trace_file']) as arrays:
        scores=arrays['scored'];times=arrays['time_after'];first=result['first_interbody_contact_time']
        result['first_own_scoring_time']=[float(times[np.flatnonzero(scores[:,i]>0)[0]]) if np.any(scores[:,i]>0) else None for i in (0,1)]
        result['scored_before_terminal']=scores.sum(axis=0).tolist()
        result['terminal_command']=arrays['commands'][-1].tolist()
        result['last_half_second_mean_abs_command_offsets']=np.abs(arrays['action'][-25:,:,:3]).mean(axis=0).tolist()
        result['precontact_minimum_upright']=(arrays['upright'][arrays['time_after']<first].min(axis=0).tolist() if first is not None and np.any(arrays['time_after']<first) else None)
        result['cumulative_reward_terms']={k:arrays[k].sum(axis=0).tolist() for k in arrays.files if k.startswith('reward_')}
    return result


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='reuse only complete verified reports after a stopped diagnostic failure')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'trace-review.json'
        previous=None
        if path.exists():
            if not args.resume: raise ValueError('Failure trace exists; inspect before explicit resume')
            previous=json.loads(path.read_text())
            assert previous['status']=='failed', 'Only stopped failed diagnostics can resume'
            atomic_json(args.output/'failure-before-resume.json',previous)
        elif args.resume: raise ValueError('No failed diagnostic exists to resume')
        reference = Path('docs/results/striking-pilot/mixed-confirm-1851392')
        review = json.loads((reference / 'analysis.json').read_text())
        assert review['formal_world_controls_total'] == 1084975 and review['diagnostic_ppo_total'] == 1844480
        jobs = [
            ('source', Path('checkpoints/striking-validation/actor-rate-1835008/control/gate-16384'),
                'c6f7fbf4c5ef941c43b8a71be7f6f6680dd657d38e96a7ed112a0f2f8f79c04b', 'source-fresh'),
            ('final', Path('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192'),
                '6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a', 'final-fresh')]

        state = dict(status='running', pid=os.getpid(), started_utc=utc(), diagnostic_output=str(args.output),
            maximum_world_controls=24000, diagnostic_world_controls=0,
            formal_world_controls_before_job=1084975, diagnostic_ppo_retained=1844480,
            new_optimizer_updates=0, assistance=False, policy_adoption='none', promotion_eligible=False,
            unchanged_rewards=True, fresh_seeds=list(range(16000003, 16000007)), checks=[])
        if previous is not None:
            state.update(diagnostic_world_controls=previous['diagnostic_world_controls'], resumed_after_verified_failure=True, previous_error=previous['error'])
        atomic_json(path, state)

        def account():
            state['diagnostic_world_controls'] = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in args.output.rglob('*-progress.json'))

        try:
            for variant, checkpoint, sha, sample in jobs:
                saved, _ = load_checked(checkpoint, sha)
                require_ppo_checkpoint(saved)
                for role in (0, 1, None):
                    profile = 'selfplay' if role is None else f'neutral-{role}'
                    name = variant + '-' + profile
                    folder = args.output / name
                    folder.mkdir(exist_ok=args.resume)
                    state.update(active_evaluation=str(folder), updated_utc=utc())
                    atomic_json(path, state)
                    command = [sys.executable, '-u', 'tools/evaluate_striking.py', '--checkpoint', str(checkpoint),
                        '--seeds', *[str(s) for s in state['fresh_seeds']], '--output', str(folder / 'evaluation.json'),
                        '--progress', str(folder / 'evaluation-progress.json'), '--trace-directory', str(folder / 'traces')]
                    if role is not None:
                        command += ['--neutral-partner', str(role)]
                    if (folder/'evaluation.json').exists():
                        assert args.resume and json.loads((folder/'evaluation-progress.json').read_text())['status']=='completed'
                    else:
                        assert not (folder/'evaluation-progress.json').exists(), 'Partial physics attempts require separate investigation, not silent rerun'
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
                    assert state['diagnostic_world_controls'] <= 24000
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
