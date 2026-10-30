"""Bounded source/final CUDA diagnostics matched to immutable native evaluations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
from contextlib import ExitStack

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_train import utc
from boxer.result_archive import archive_tree
from boxer.striking_punch_trace import describe_events
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_arm_effort_supervisor import independent_cycle
from tools.striking_pool_compare import verify_prerequisites

ROOT = Path('/home/pedro/boxing-runs/gpu-learning-transfer-v1')
ARCHIVE = Path('docs/results/striking-pilot/gpu-learning-transfer-v1')
PREVIOUS = Path('docs/results/striking-pilot/gpu-learning-v1')
FORMAL_BEFORE, MAX_CONTROLS = 2340326, 24000
SEEDS = [7, 23, 31, 47]
SOURCES = {
    'initial': ('baseline', '6432029087cb00e73e39bee3a974e07f47f598771bae05c519e1bcad616c2103'),
    'final': ('step-1048576', '639ccec99328b034d7d6c9f1ba08a7305a0edc3e67025aa01cedcc06cf979052')}
GIT = ['/mnt/c/Program Files/Git/cmd/git.exe', '-C', 'E:/boxer-agent']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def review(report, expected_sha, backend):
    assert report['striking_sha256'] == expected_sha and report['physics_backend'] == backend
    assert not report['assistance'] and not report['sampled_policy'] and not report['diagnostic_solver_override']
    assert report['physics_solver_iterations'] == 3 and report['physics_solver_ls_iterations'] == 5
    assert not report.get('diagnostic_profile_override') and report.get('diagnostic_arm_residual_scale', 1.) == 1.
    assert not report.get('diagnostic_zero_command_offsets') and not report.get('diagnostic_zero_arm_residuals')
    assert report.get('diagnostic_base_yaw_scale', 1.) == 1. and report.get('opponent_sha256') is None
    trials = report['trials']
    assert len(trials) == 8 and {(r['seed'], r['swap']) for r in trials} == {(s,w) for s in SEEDS for w in (False,True)}
    role = report['neutral_partner_learner_corner']
    counts = np.zeros(2, int); own = set(); reciprocal = set(); rows = []
    for row in trials:
        assert row['finite']
        native, full = Path(row['punch_trace_file']), Path(row['trace_file'])
        assert sha(native) == row['punch_trace_sha256'] and sha(full) == row['trace_sha256']
        with np.load(full) as arrays: assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
        with np.load(native) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            events = describe_events(arrays); assert events == row['punch_measurements']
        accepted = [e for e in events if independent_cycle(e)]
        per_corner = np.array([sum(e['corner'] == c for e in accepted) for c in (0,1)])
        counts += per_corner
        if row['survived']:
            if role is not None and per_corner[role]: own.add(row['seed'])
            if role is None and per_corner.all(): reciprocal.add(row['seed'])
        first = [c['elapsed'] for c in row.get('first_observed_contacts', [])
            if any(g.startswith('red/') for g in c['geoms']) and any(g.startswith('blue/') for g in c['geoms'])]
        rows.append(dict(seed=row['seed'], swap=row['swap'], survived=row['survived'], elapsed=row['elapsed'],
            fallen=row['fallen'], valid_cycles=per_corner.tolist(), first_interbody_contact=min(first) if first else None,
            terminal=row.get('terminal_state')))
    survivors = sum(row['survived'] for row in trials)
    return dict(pair_survivors=survivors, valid_cycles_by_corner=counts.tolist(),
        own_surviving_valid_seeds=sorted(own), reciprocal_surviving_valid_seeds=sorted(reciprocal),
        passes_profile_diagnostic=survivors >= (8 if role == 1 else 7) and len(reciprocal if role is None else own) >= 2,
        native_certification=False, conditions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); ROOT.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (ROOT/'.lock', Path('runs/striking_diagnostic_supervisor.lock'), Path('runs/unassisted_boxing.lock')):
            fcntl.flock(stack.enter_context(path.open('a')), fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = json.loads((PREVIOUS/'completed-review.json').read_text())
        assert complete['status'] == 'reviewed' and complete['formal_total'] == FORMAL_BEFORE
        assert complete['diagnostic_ppo_total'] == 10569984
        path = ROOT/'state.json'; child = [None]
        if path.exists():
            if not args.resume: raise ValueError('existing comparison needs explicit inspected resume')
            state = json.loads(path.read_text())
            assert state['formal_before'] == FORMAL_BEFORE and state['maximum_new_controls'] == MAX_CONTROLS
            assert state['sources'] == {k:list(v) for k,v in SOURCES.items()}
            for p in ROOT.rglob('*-progress.json'):
                assert json.loads(p.read_text())['status'] == 'completed', 'partial evaluation must be investigated before retry'
        else:
            if args.resume: raise ValueError('no existing comparison')
            state = dict(status='running', started_utc=utc(), diagnostic_output=str(ROOT),
                sources={k:list(v) for k,v in SOURCES.items()}, native_references={}, profiles=[],
                formal_before=FORMAL_BEFORE, maximum_new_controls=MAX_CONTROLS,
                formal_ceiling=FORMAL_BEFORE+MAX_CONTROLS, formal_total=FORMAL_BEFORE,
                main_retained=476041216, diagnostic_ppo_retained=10569984, new_ppo_steps=0,
                new_renderer_calls=0, prerequisites=verify_prerequisites())
            for label, (old_gate, expected_sha) in SOURCES.items():
                original = PREVIOUS/'gates'/old_gate/'checkpoint'
                assert sha(original/'policy.pkl') == expected_sha
                require_ppo_checkpoint(pickle.loads((original/'policy.pkl').read_bytes()))
                snapshot = ROOT/'sources'/label; archive_tree(original, snapshot)
                state['native_references'][label] = {}
                for role in ('neutral-0','neutral-1','selfplay'):
                    report_path = PREVIOUS/'gates'/old_gate/(role+'.json')
                    state['native_references'][label][role] = dict(report_sha256=sha(report_path),
                        source_report=str(report_path), new_controls=0,
                        **review(json.loads(report_path.read_text()), expected_sha, 'native'))
        state.update(status='running', pid=os.getpid())
        def record(enforce=True):
            used = sum(json.loads(p.read_text())['simulated_world_transitions'] for p in ROOT.rglob('*-progress.json'))
            ledgers = [json.loads(p.read_text()) for p in ROOT.rglob('*-gpu-ledger.json')]
            state['gpu_ledger'] = {k:sum(v[k] for v in ledgers) for k in
                ('gpu_world_substeps','gpu_forward_calls','native_reset_forward_calls','solver_limit_world_checks')}
            state['gpu_overflow_bits_seen'] = 0
            for value in ledgers: state['gpu_overflow_bits_seen'] |= value['overflow_bits_seen']
            state.update(new_controls=used, formal_total=FORMAL_BEFORE+used, updated_utc=utc())
            atomic_json(path, state)
            if enforce:
                assert used <= MAX_CONTROLS and state['gpu_ledger']['gpu_world_substeps'] <= 240006
                assert state['gpu_overflow_bits_seen'] & ~(512|1024) == 0
        def stop(signum,frame):
            if child[0] is not None and child[0].poll() is None: os.killpg(child[0].pid, signal.SIGTERM)
            raise InterruptedError('comparison stopped; preserve all child/physics ledgers before recovery')
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        def publish(label):
            try:
                archive_tree(ROOT, ARCHIVE)
                staged = subprocess.check_output(GIT+['diff','--cached','--name-only'], text=True).splitlines()
                if any(not p.startswith(str(ARCHIVE)+'/') for p in staged): raise ValueError('unrelated staged work blocks publication')
                subprocess.run(GIT+['add','--',str(ARCHIVE)], check=True)
                errors = [str(p) for p in ARCHIVE.rglob('*-errors.log')]
                if errors: subprocess.run(GIT+['add','-f','--',*errors], check=True)
                if subprocess.run(GIT+['diff','--cached','--quiet']).returncode:
                    subprocess.run(GIT+['commit','-m',f'Record matched GPU transfer {label}'], check=True)
                    subprocess.run(GIT+['push'], check=True)
                state.pop('publication_error',None)
            except Exception as error: state['publication_error'] = repr(error); record(enforce=False)
        record()
        try:
            for label, (_, expected_sha) in SOURCES.items():
                for role in (0,1,None):
                    name = 'selfplay' if role is None else f'neutral-{role}'
                    if any(r['arm']==label and r['role']==name for r in state['profiles']): continue
                    folder = ROOT/label; folder.mkdir(exist_ok=True)
                    if (folder/(name+'-gpu-ledger.json')).exists() or (folder/(name+'-progress.json')).exists():
                        raise ValueError('unreviewed prior physical attempt; no silent repeat')
                    command = [sys.executable,'-u','tools/evaluate_striking.py','--checkpoint',str(ROOT/'sources'/label),
                        '--physics-backend','warp','--seeds',*map(str,SEEDS),'--seconds','10',
                        '--output',str(folder/(name+'.json')),'--progress',str(folder/(name+'-progress.json')),
                        '--gpu-ledger',str(folder/(name+'-gpu-ledger.json')),
                        '--trace-directory',str(folder/name/'full'),'--punch-trace-directory',str(folder/name/'native')]
                    if role is not None: command += ['--neutral-partner',str(role)]
                    state.update(active_arm=label,active_role=name); record()
                    with (folder/(name+'.log')).open('w') as out, (folder/(name+'-errors.log')).open('w') as err:
                        child[0] = subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
                        state['child_pid'] = child[0].pid; record(); code = child[0].wait(); child[0] = None
                    record()
                    if code or (folder/(name+'-errors.log')).read_text().strip(): raise RuntimeError('GPU reference child failed/warned; inspect retained logs')
                    report = json.loads((folder/(name+'.json')).read_text())
                    ledger = json.loads((folder/(name+'-gpu-ledger.json')).read_text())
                    assert ledger['gpu_world_substeps'] == report['experimental_gpu_substeps']
                    assert ledger['gpu_forward_calls'] == report['experimental_gpu_forward_calls']
                    state['profiles'].append(dict(arm=label,role=name,sha256=expected_sha,
                        **review(report,expected_sha,'warp')))
                    record(); publish(label+'-'+name)
            assert verify_prerequisites() == state['prerequisites']
            state.update(status='completed', completed_utc=utc(), unused_controls_expire=MAX_CONTROLS-state['new_controls'],
                decision='Review matched source/final native/CUDA failure and punch results before any new PPO allocation; no promotion')
        except BaseException as error:
            state.update(status='failed',error=repr(error)); record(enforce=False); publish('failed'); raise
        record(); publish('completed')


if __name__ == '__main__': main()
