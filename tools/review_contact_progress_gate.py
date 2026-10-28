"""Recheck an archived complete v15 gate without executing physics or policies."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_punch_trace import describe_events
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_contact_progress_supervisor import independent_cycle
from tools.striking_pool_compare import verify_prerequisites


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--rebalance-v16', action='store_true', help='review v16 without changing its thresholds')
    parser.add_argument('--arm-effort-v17', action='store_true', help='review v17 without changing its thresholds')
    args = parser.parse_args()
    assert not (args.rebalance_v16 and args.arm_effort_v17)
    assert args.label == 'fresh-final' or (args.label.startswith('step-') and args.label[5:].isdigit())
    experiment = 'arm-effort-v17' if args.arm_effort_v17 else 'contact-rebalance-v16' if args.rebalance_v16 else 'contact-progress-v15'
    offset = 7323648 if args.arm_effort_v17 else 5226496 if args.rebalance_v16 else 3129344
    archive = Path('docs/results/striking-pilot') / experiment
    folder = archive / 'gates' / args.label
    gate = json.loads((folder / 'motion-gate.json').read_text())
    prerequisites = verify_prerequisites()
    checkpoint_label = 'step-2097152' if args.label == 'fresh-final' else args.label
    cp = Path('checkpoints/striking-validation') / experiment / checkpoint_label
    manifest = json.loads((cp / 'latest.json').read_text())
    payload = cp / manifest['file']
    assert sha(payload) == manifest['sha256'] == gate['sha256']
    saved = pickle.loads(payload.read_bytes())
    require_ppo_checkpoint(saved)
    assert saved['consumed_steps'] == saved['optimized_steps'] == offset + gate['new_ppo_steps']
    original, controls = full_profiles(folder, gate['seeds'], gate['sha256'])
    assert original == gate['original_contact_gate'] and controls == gate['controls']
    motion, all_events, episodes = {}, [], []
    for role in (0, 1, None):
        name = 'selfplay' if role is None else f'neutral-{role}'
        assert not (folder / (name + '-errors.log')).read_text().strip()
        report = json.loads((folder / (name + '.json')).read_text())
        valid_seeds, reciprocal, total = set(), set(), np.zeros(2, int)
        for row in report['trials']:
            for field, digest in [('punch_trace_file', 'punch_trace_sha256'), ('trace_file', 'trace_sha256')]:
                # Published paths point at the live job; review immutable archive copies.
                subfolder = 'native' if field == 'punch_trace_file' else 'full'
                trace = folder / name / subfolder / Path(row[field]).name
                assert sha(trace) == row[digest]
                with np.load(trace) as raw:
                    assert all(np.isfinite(raw[k]).all() for k in raw.files)
                    if subfolder == 'native':
                        events = describe_events(raw)
                        assert events == row['punch_measurements']
            counts = np.zeros(2, int)
            for event in events:
                accepted = independent_cycle(event)
                all_events.append(dict(profile=name, seed=row['seed'], swap=row['swap'], **event, valid_cycle=accepted))
                if accepted:
                    assert event['native_contact_window_impulse_ns'] >= .5
                    assert event['onset_contact_closing'] >= .3 and event['onset_contact_advancing'] >= .2
                    counts[event['corner']] += 1
            total += counts
            if row['survived']:
                if role is not None and counts[role] > 0:
                    valid_seeds.add(row['seed'])
                if role is None and (counts > 0).all():
                    reciprocal.add(row['seed'])
            episodes.append(dict(profile=name, seed=row['seed'], swap=row['swap'], survived=row['survived'],
                                 valid_cycles_by_corner=counts.tolist(),
                                 termination={k: v for k, v in row.items() if 'fall' in k or 'termination' in k}))
        motion[name] = dict(valid_cycles_by_corner=total.tolist(), own_surviving_valid_seeds=sorted(valid_seeds),
                            reciprocal_surviving_valid_seeds=sorted(reciprocal),
                            passes=len(reciprocal if role is None else valid_seeds) >= 2)
    assert motion == gate['motion'] and all_events == gate['events']
    passed = original['passes_complete'] and all(m['passes'] for m in motion.values())
    assert passed == gate['passes_complete']
    result = dict(status='independently_reviewed_saved_full_gate', label=args.label, sha256=gate['sha256'],
                  new_ppo_steps=gate['new_ppo_steps'], controls=controls, original_contact_gate=original,
                  motion=motion, episodes=episodes, passes_complete=passed, prerequisites=prerequisites,
                  new_physics_controls=0, new_optimizer_updates=0, new_render_forwards=0,
                  native_errors_empty=True, all_saved_arrays_finite=True, checkpoint_and_trace_hashes_verified=True,
                  promotion=False, main_adoption=False,
                  limitation='Deterministic known-seed neutral practice does not certify reciprocal stable boxing or fresh-seed generalization.')
    (archive / (args.label + '-independent-review.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('label', 'motion', 'passes_complete', 'controls')}))


if __name__ == '__main__':
    main()
