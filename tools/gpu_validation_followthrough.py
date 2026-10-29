"""Finish the declared validation sequence without overlapping physical jobs."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from boxer.checkpoint_io import atomic_json
from boxer.result_archive import archive_tree
from boxer.curriculum_train import utc
from tools.run_gpu_closed_loop import ROOT, ARCHIVE, review

QUEUE = Path('/home/pedro/boxing-runs/gpu-validation-followthrough')
GIT = ['/mnt/c/Program Files/Git/cmd/git.exe', '-C', 'E:\\boxer-agent']


def publish(paths, message, state):
    """Stage only completed owned results; preserve unrelated work and staging."""
    try:
        assert subprocess.check_output(GIT+['branch', '--show-current'], text=True).strip() == 'master'
        if subprocess.check_output(GIT+['diff', '--cached', '--name-only'], text=True).strip():
            raise RuntimeError('Existing staged changes; publication deferred without touching them')
        subprocess.run(GIT+['add', '--', *paths], check=True)
        if subprocess.run(GIT+['diff', '--cached', '--quiet']).returncode:
            subprocess.run(GIT+['commit', '-m', message], check=True)
        subprocess.run(GIT+['push'], check=True)
        state.setdefault('published', []).append(message)
    except (AssertionError, RuntimeError, subprocess.SubprocessError) as error:
        state.setdefault('publication_pending', []).append(repr(error))
    atomic_json(QUEUE/'state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous = None
        if (QUEUE/'state.json').exists():
            if not args.resume: raise ValueError('Existing coordinator requires inspection before retry')
            previous = json.loads((QUEUE/'state.json').read_text())
            assert previous['status'] == 'failed' and 'child_pid' not in previous, 'Legacy child recovery requires separate review'
        state = dict(status='waiting_for_comparison', pid=os.getpid(), started_utc=utc(),
            diagnostic_output=str(QUEUE), comparison_output=str(ROOT), legacy_output='/home/pedro/boxing-runs/legacy-punch-cycles-v1')
        if previous is not None:
            state['reviewed_recovery'] = dict(previous_error=previous['error'], decision='Wait for reviewed comparison resume; no legacy controls previously attempted')
            state['recoveries'] = previous.get('recoveries', [])+([previous['reviewed_recovery']] if 'reviewed_recovery' in previous else [])
        atomic_json(QUEUE/'state.json', state)
        try:
            while True:
                current = json.loads((ROOT/'state.json').read_text())
                if current['status'] == 'completed': break
                if current['status'] != 'running': raise RuntimeError('Comparison failed; inspect before continuation')
                cmdline = Path(f"/proc/{current['pid']}/cmdline")
                if not cmdline.exists() or b'run_gpu_closed_loop.py' not in cmdline.read_bytes():
                    raise RuntimeError('Comparison process stopped; no implicit restart')
                time.sleep(10)
            # The parent marks completion before its final archive copy. Wait for
            # its lock to release before reviewing/copying/publishing those files.
            with (ROOT/'.lock').open('a') as comparison_lock:
                fcntl.flock(comparison_lock, fcntl.LOCK_EX)
            profiles = []
            for arm in ('native-30-20', 'warp-3-5', 'warp-30-20'):
                for role in ('neutral-0', 'neutral-1', 'selfplay'):
                    report = json.loads((ROOT/arm/(role+'.json')).read_text())
                    profiles.append(dict(arm=arm, profile=role, **review(report, ROOT/arm)))
            completed = dict(status='independently_reviewed', reviewed_utc=utc(), profiles=profiles,
                formal_total=current['formal_total'], new_controls=current['new_controls'], new_ppo_steps=0,
                no_automatic_adoption=True, all_original_gates_retained=True,
                decision='Continue the declared own-engine legacy measurement; no unchanged CPU extension or unvalidated GPU PPO')
            atomic_json(ROOT/'completed-review.json', completed)
            archive_tree(ROOT, ARCHIVE)
            publish([str(ARCHIVE)], 'Review completed GPU backend comparison', state)
            state.update(status='running_legacy_comparison', formal_before_legacy=current['formal_total'])
            atomic_json(QUEUE/'state.json', state)
            with (QUEUE/'legacy.log').open('w') as out, (QUEUE/'legacy-errors.log').open('w') as err:
                child = subprocess.Popen([sys.executable, '-u', 'tools/evaluate_legacy_punch_cycles.py'], stdout=out, stderr=err)
                state['child_pid'] = child.pid; atomic_json(QUEUE/'state.json', state)
                if child.wait(): raise RuntimeError('Legacy comparison failed; inspect its durable ledger before recovery')
            legacy_root = Path(state['legacy_output'])
            legacy = json.loads((legacy_root/'state.json').read_text())
            assert legacy['status'] == 'completed' and len(legacy['episodes']) == 16
            archive_tree(legacy_root, Path('docs/results/striking-pilot/legacy-punch-cycles-v1'))
            state.update(status='completed_awaiting_training_design', completed_utc=utc(),
                formal_total=legacy['formal_total'], decision='Read both complete comparisons, implement and validate batched cycle/reset/reward parity before bounded GPU PPO')
            publish(['docs/results/striking-pilot/legacy-punch-cycles-v1'], 'Record retained legacy punch-cycle measurements', state)
        except BaseException as error:
            state.update(status='failed', error=repr(error)); raise
        finally:
            atomic_json(QUEUE/'state.json', state)
            archive_tree(QUEUE, Path('docs/results/striking-pilot/gpu-validation-followthrough'))


if __name__ == '__main__': main()
