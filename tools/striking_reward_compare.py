"""Bounded sequential reward comparison; diagnostic policies never enter main PPO.

Both arms clone exactly the same complete learning state. Only training reward
differs. Evaluation always uses the unchanged production StrikingEnv/scorecard.
Lineage checkpoint counters are not new main consumed work: count their deltas.
"""
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


def reach_bonus(distances, dt, both_up):
    """Training-only sustained reach signal, bounded to 0.25 reward per second.

    A close stationary guard can earn shaping but never a physical impact score.
    The experiment tests whether this signal improves contacts without falls.
    """
    distances = np.asarray(distances)
    if distances.shape != (2,) or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError('two finite nonnegative hand-target distances required')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite control interval required')
    return .25 * dt * np.exp(-(distances / .2)**2) if both_up else np.zeros(2)


def run_arm(args):
    import boxer.striking_train as training
    if args.arm == 'reach':
        base = training.StrikingEnv
        class ReachShaping(base):
            def step(self, action):
                obs, reward, done, outcome = super().step(action)
                reward += reach_bonus(self.hand_distances(), self.pair.dt, not outcome['done'])
                return obs, reward, done, outcome
        training.StrikingEnv = ReachShaping
    training.main(['--output', str(args.output), '--resume', '--ceiling', str(args.ceiling),
                   '--worlds', '4', '--unroll', '128'])


def compare(args):
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.compare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((args.source / 'latest.json').read_text())
        source_bytes = (args.source / manifest['file']).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != manifest['sha256']:
            raise ValueError('source checkpoint hash mismatch')
        if args.steps <= 0 or args.steps % 1024:
            raise ValueError('diagnostic budget must contain complete PPO blocks')
        state_path = args.output / 'comparison.json'
        if state_path.exists():
            raise ValueError('comparison already exists; inspect it before another launch')
        initial = manifest['consumed_steps']
        config = json.loads((args.source / 'config.json').read_text())
        if config['worlds'] != 4 or config['unroll'] != 128:
            raise ValueError('comparison requires inherited four-world 128-unroll profile')
        state = dict(status='running', started_utc=utc(), pid=os.getpid(),
            source=str(args.source), source_sha256=manifest['sha256'],
            main_consumed_steps_retained=initial, foundation_consumed_steps_retained=474992640,
            diagnostic_steps_per_arm=args.steps, diagnostic_steps_ceiling=2*args.steps,
            diagnostic_consumed_steps=0, diagnostic_output=str(args.output),
            physics_scorecard_unchanged=True, policy_adoption='none', arms={})
        atomic_json(state_path, state)
        try:
            for arm in ('baseline', 'reach'):
                folder = args.output / arm
                checkpoint = folder / 'checkpoint'
                checkpoint.mkdir(parents=True, exist_ok=False)
                (checkpoint / manifest['file']).write_bytes(source_bytes)
                atomic_json(checkpoint / 'latest.json', manifest)
                atomic_json(folder / 'config.json', config)
                atomic_json(folder / 'progress.json', dict(consumed_steps=initial))
                atomic_json(folder / 'diagnostic-profile.json', dict(arm=arm,
                    source_sha256=manifest['sha256'], lineage_counter_offset=initial,
                    training_reward='unchanged' if arm == 'baseline' else
                    'base + both_up * dt * 0.25 * exp(-(minimum_hand_target_center_distance/0.2)^2)',
                    evaluation_reward='unchanged', main_training=False,
                    tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
                state.update(active_arm=arm, phase='training', updated_utc=utc())
                atomic_json(state_path, state)
                with (folder / 'training.log').open('w') as stdout, (folder / 'errors.log').open('w') as stderr:
                    subprocess.run([sys.executable, '-u', str(Path(__file__)), '--arm', arm,
                        '--output', str(folder), '--ceiling', str(initial + args.steps)],
                        stdout=stdout, stderr=stderr, check=True)
                progress = json.loads((folder / 'progress.json').read_text())
                delta = progress['consumed_steps'] - initial
                if delta != args.steps or progress['optimized_steps'] != initial + args.steps:
                    raise ValueError('incomplete diagnostic arm; inspect counters before resuming')
                state['arms'][arm] = dict(status='evaluating', diagnostic_consumed_steps=delta)
                state['diagnostic_consumed_steps'] += delta
                state.update(phase='evaluation', updated_utc=utc())
                atomic_json(state_path, state)
                with (folder / 'evaluation.log').open('w') as stdout, (folder / 'evaluation-errors.log').open('w') as stderr:
                    subprocess.run([sys.executable, '-u', 'tools/evaluate_striking.py',
                        '--checkpoint', str(checkpoint), '--output', str(folder / 'evaluation.json')],
                        stdout=stdout, stderr=stderr, check=True)
                evaluation = json.loads((folder / 'evaluation.json').read_text())
                state['arms'][arm].update(status='completed',
                    evaluation_world_controls=evaluation['simulated_world_transitions'],
                    pair_survivors=sum(row['survived'] for row in evaluation['trials']),
                    scored_by_corner=np.sum([row['scored_by_corner'] for row in evaluation['trials']], axis=0).tolist(),
                    checkpoint=json.loads((checkpoint / 'latest.json').read_text()))
                atomic_json(state_path, state)
            state.update(status='completed', phase='awaiting_review', updated_utc=utc())
            atomic_json(state_path, state)
        except BaseException as error:
            # Include attempted controls, even when collection/update failed.
            state['diagnostic_consumed_steps'] = sum(max(0,
                json.loads(path.read_text())['consumed_steps'] - initial)
                for path in args.output.glob('*/progress.json'))
            state.update(status='failed', error=str(error), updated_utc=utc())
            atomic_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('checkpoints/striking-pilot/step-1048576'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=131072)
    parser.add_argument('--arm', choices=('baseline', 'reach'))
    parser.add_argument('--ceiling', type=int)
    args = parser.parse_args()
    run_arm(args) if args.arm else compare(args)


if __name__ == '__main__':
    main()
