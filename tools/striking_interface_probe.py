"""Read-only factorial arm-amplitude/range diagnostic; no policy optimization.

The production profile is 0.8 rad / 0.45 m. Other profiles are explicit
counterfactuals, never silently adopted settings. Every trial retains free
bodies, original first-fall rules and the same physical impact scorecard.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_train import utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('checkpoints/striking-pilot/step-1048576'))
    parser.add_argument('--scale', type=float)
    parser.add_argument('--distance', type=float)
    args = parser.parse_args()
    if args.scale is not None:
        if args.scale not in (.4, .8) or args.distance not in (.35, .45):
            raise ValueError('only predeclared diagnostic profiles allowed')
        import tools.evaluate_striking as evaluation
        evaluation.StrikingEnv.residual_scale = args.scale
        sys.argv = [sys.argv[0], '--checkpoint', str(args.source), '--output', str(args.output),
                    '--engagement-distance', str(args.distance)]
        evaluation.main()
        result = json.loads(args.output.read_text())
        result.update(diagnostic_arm_residual_scale=args.scale,
                      production_profile=bool(args.scale == .8 and args.distance == .45))
        atomic_json(args.output, result)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'probe.json'
    if state_path.exists():
        raise ValueError('probe already exists; inspect its state before another launch')
    source = json.loads((args.source / 'latest.json').read_text())
    if hashlib.sha256((args.source / source['file']).read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('source hash mismatch')
    state = dict(status='running', source=str(args.source), source_sha256=source['sha256'],
                 started_utc=utc(), optimization_transitions=0, profiles={},
                 expected_episodes_per_profile=8, maximum_world_controls=16000,
                 diagnostic_world_controls=0, policy_or_profile_adoption='none')
    atomic_json(state_path, state)
    try:
        for scale in (.8, .4):
            for distance in (.45, .35):
                name = f'scale-{scale:.2f}-range-{distance:.2f}'
                output = args.output / (name + '.json')
                state.update(active_profile=name, updated_utc=utc())
                atomic_json(state_path, state)
                with (args.output / (name + '.log')).open('w') as stdout, (args.output / (name + '-errors.log')).open('w') as stderr:
                    subprocess.run([sys.executable, '-u', str(Path(__file__)), '--source', str(args.source),
                        '--output', str(output), '--scale', str(scale), '--distance', str(distance)],
                        stdout=stdout, stderr=stderr, check=True)
                result = json.loads(output.read_text())
                rows = result['trials']
                state['diagnostic_world_controls'] += result['simulated_world_transitions']
                state['profiles'][name] = dict(pair_survivors=sum(x['survived'] for x in rows),
                    impacts=sum(sum(x['scored_by_corner']) for x in rows),
                    complete_episode_impacts=sum(sum(x['scored_by_corner']) for x in rows if x['survived']),
                    scoring_seeds=sorted({x['seed'] for x in rows if sum(x['scored_by_corner'])}),
                    diagnostic_world_controls=result['simulated_world_transitions'])
                atomic_json(state_path, state)
        state.update(status='completed', updated_utc=utc())
        atomic_json(state_path, state)
    except BaseException as error:
        state.update(status='failed', error=str(error), updated_utc=utc())
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
