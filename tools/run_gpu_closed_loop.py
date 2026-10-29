"""Persistent bounded solver/backend comparison using unmodified punch measurements."""
from contextlib import ExitStack
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.result_archive import archive_tree
from boxer.curriculum_train import utc
from boxer.striking_punch_trace import describe_events
from tools.striking_contact_compare import load_checked
from tools.striking_pool_compare import verify_prerequisites

ROOT = Path('/home/pedro/boxing-runs/gpu-closed-loop-v1')
ARCHIVE = Path('docs/results/striking-pilot/gpu-closed-loop-v1')
SOURCE = Path('checkpoints/striking-validation/arm-effort-v17/step-2097152')
SHA = 'f311640d78941546938412f5802cff872e0a3611c392f49a3b4719eefaa1f26a'
SEEDS = [34002003, 34002004, 34002005, 34002006]


def valid(e):
    return bool(e['onset_relative_speed'] >= 1. and e['onset_forward_cosine'] >= .7
        and e['pre_max_displacement_m'] >= .12 and e['post_window_complete']
        and e['return_toward_pre_window_position_m'] is not None
        and e['return_toward_pre_window_position_m'] >= .06 and e['post_contact_seconds'] <= .10)


def review(report, trace_root=None):
    assert report['striking_sha256'] == SHA and not report['assistance'] and not report['sampled_policy']
    assert len(report['trials']) == 8
    assert {(r['seed'], r['swap']) for r in report['trials']} == {(s, w) for s in SEEDS for w in (False, True)}
    survivors, counts, reciprocal, own_seeds = 0, np.zeros(2, int), [], set()
    role = report['neutral_partner_learner_corner']
    label = 'selfplay' if role is None else f'neutral-{role}'
    def trace_path(original, family):
        archived = Path(trace_root)/label/family/Path(original).name if trace_root is not None else None
        return archived if archived is not None and archived.exists() else Path(original)
    for trial in report['trials']:
        assert trial['finite']
        full = trace_path(trial['trace_file'], 'full')
        assert hashlib.sha256(full.read_bytes()).hexdigest() == trial['trace_sha256']
        with np.load(full) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
        path = trace_path(trial['punch_trace_file'], 'native')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == trial['punch_trace_sha256']
        with np.load(path) as arrays:
            events = describe_events(arrays)
            assert events == trial['punch_measurements']
        cycles = np.array([sum(e['corner'] == c and valid(e) for e in events) for c in (0, 1)])
        if trial['survived']:
            survivors += 1; counts += cycles
            if role is not None and cycles[role] > 0: own_seeds.add(trial['seed'])
            if all(cycles > 0): reciprocal.append(dict(seed=trial['seed'], swap=trial['swap']))
    motion_pass = (len({r['seed'] for r in reciprocal}) >= 2 if role is None else len(own_seeds) >= 2)
    return dict(pair_survivors=survivors, surviving_valid_cycles=counts.tolist(),
                reciprocal_surviving_episodes=reciprocal, own_surviving_valid_seeds=sorted(own_seeds),
                passes_profile_diagnostic=survivors >= (8 if role == 1 else 7) and motion_pass,
                native_certification=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (ROOT/'.lock', Path('runs/striking_diagnostic_supervisor.lock'), Path('runs/unassisted_boxing.lock')):
            fcntl.flock(stack.enter_context(path.open('a')), fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous = None
        if (ROOT/'state.json').exists():
            if not args.resume: raise ValueError('Existing state requires investigation; no implicit diagnostic restart')
            previous = json.loads((ROOT/'state.json').read_text())
            assert previous['status'] == 'failed' and previous['source_sha256'] == SHA
            assert previous['maximum_new_controls'] == 36000 and previous['formal_before'] == 2271001
            # Resume only whole completed profiles. An interrupted physical child
            # needs a separately reviewed recovery, never a silent repeat.
            for progress in ROOT.rglob('*-progress.json'):
                assert json.loads(progress.read_text())['status'] == 'completed'
            assert sum(json.loads(p.read_text())['simulated_world_transitions'] for p in ROOT.rglob('*-progress.json')) == previous['new_controls']
            assert previous['formal_total'] == previous['formal_before']+previous['new_controls']
        load_checked(SOURCE, SHA)
        baseline = Path('docs/results/striking-pilot/arm-effort-v17/gates/fresh-final')
        state = dict(status='running', pid=os.getpid(), started_utc=utc(), diagnostic_output=str(ROOT),
            source_sha256=SHA, maximum_new_controls=36000, formal_before=2271001, formal_ceiling=2307001,
            main_retained=476041216, diagnostic_ppo_retained=9504000, new_ppo_steps=0,
            prerequisites=verify_prerequisites(), profiles=[], baseline={})
        for role in ('neutral-0', 'neutral-1', 'selfplay'):
            path = baseline/(role+'.json')
            state['baseline'][role] = dict(report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                **review(json.loads(path.read_text()), baseline))
        if previous is not None:
            state = previous
            for row in state['profiles']:
                folder = ROOT/row['arm']
                report = json.loads((folder/(row['profile']+'.json')).read_text())
                review(report, folder)
            state.setdefault('recoveries', []).append(dict(reviewed_utc=utc(), previous_error=state.get('error'),
                completed_profiles_retained=len(state['profiles']), decision='Resume remaining profiles only; no repeated physical controls'))
            state.update(status='running', pid=os.getpid(), updated_utc=utc())
            state.pop('error', None)
        atomic_json(ROOT/'state.json', state)
        def account():
            used = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in ROOT.rglob('*-progress.json'))
            assert used <= 36000
            state.update(new_controls=used, formal_total=2271001+used, updated_utc=utc())
            atomic_json(ROOT/'state.json', state)
        try:
            for name, backend, iterations, ls in (('native-30-20', 'native', 30, 20),
                    ('warp-3-5', 'warp', 3, 5), ('warp-30-20', 'warp', 30, 20)):
                for role in (0, 1, None):
                    label = 'selfplay' if role is None else f'neutral-{role}'
                    if any(p['arm'] == name and p['profile'] == label for p in state['profiles']): continue
                    folder = ROOT/name; folder.mkdir(exist_ok=True)
                    prefix = folder/label
                    command = [sys.executable, '-u', 'tools/evaluate_striking.py', '--checkpoint', str(SOURCE),
                        '--physics-backend', backend, '--solver-iterations', str(iterations),
                        '--solver-ls-iterations', str(ls), '--seeds', *map(str, SEEDS), '--seconds', '10',
                        '--output', str(prefix.with_suffix('.json')), '--progress', str(folder/(label+'-progress.json')),
                        '--trace-directory', str(prefix/'full'), '--punch-trace-directory', str(prefix/'native')]
                    if role is not None: command += ['--neutral-partner', str(role)]
                    state.update(active_arm=name, active_profile=label, updated_utc=utc())
                    atomic_json(ROOT/'state.json', state)
                    with (folder/(label+'.log')).open('w') as out, (folder/(label+'-errors.log')).open('w') as err:
                        child = subprocess.Popen(command, stdout=out, stderr=err)
                        state['child_pid'] = child.pid; atomic_json(ROOT/'state.json', state)
                        code = child.wait()
                    account()
                    if code or (folder/(label+'-errors.log')).read_text().strip():
                        raise RuntimeError(f'{name}/{label} failed: inspect stderr before any continuation')
                    report = json.loads(prefix.with_suffix('.json').read_text())
                    assert report['physics_backend'] == backend and report['diagnostic_solver_override']
                    state['profiles'].append(dict(arm=name, profile=label, **review(report, folder)))
                    account()
                    try:
                        archive_tree(folder, ARCHIVE/name)
                        atomic_json(ARCHIVE/'state.json', state)
                    except OSError as error:
                        state.setdefault('archive_pending', []).append(repr(error))
                        atomic_json(ROOT/'state.json', state)
            state.update(status='completed', completed_utc=utc(), decision='Review all arms before any GPU training or solver adoption')
        except BaseException as error:
            account(); state.update(status='failed', error=repr(error))
            raise
        finally:
            atomic_json(ROOT/'state.json', state)
            archive_tree(ROOT, ARCHIVE)


if __name__ == '__main__':
    main()
