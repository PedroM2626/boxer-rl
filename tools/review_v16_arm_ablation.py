"""Review both saved arms of the bounded v16 arm-residual comparison."""
import collections
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from boxer.striking_punch_trace import describe_events
from tools.striking_contact_rebalance_supervisor import independent_cycle
from tools.striking_pool_compare import verify_prerequisites
from tools.run_v16_arm_ablation import ROOT, SHA

ARCHIVE = Path('docs/results/striking-pilot/v16-arm-ablation')
BASELINE = Path('docs/results/striking-pilot/contact-rebalance-v16/gates/fresh-final')


def review_report(report, directory, expected_mask):
    assert report['striking_sha256'] == SHA and report['diagnostic_zero_arm_residuals'] == expected_mask
    assert not report['diagnostic_zero_command_offsets'] and not report['assistance'] and not report['sampled_policy']
    assert len(report['trials']) == 8
    assert {(r['seed'], r['swap']) for r in report['trials']} == {(s, w) for s in range(34001003, 34001007) for w in (False, True)}
    flags, episodes = collections.Counter(), []
    for row in report['trials']:
        assert row['finite']
        for field, digest, sub in [('trace_file', 'trace_sha256', 'full'), ('punch_trace_file', 'punch_trace_sha256', 'native')]:
            path = directory / sub / Path(row[field]).name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[digest]
            with np.load(path) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
                if sub == 'native':
                    events = describe_events(arrays)
                    assert events == row['punch_measurements']
        valid = [e for e in events if independent_cycle(e)]
        for i, fallen in enumerate(row['fallen']):
            if fallen:
                for key in ('low_pelvis', 'tilted', 'self_leg_contact'):
                    flags[key] += int(row['terminal_state']['corners'][i][key])
        cross = [c for c in row['first_observed_contacts']
                 if any(n.startswith('red/') for n in c['geoms']) and any(n.startswith('blue/') for n in c['geoms'])]
        episodes.append(dict(seed=row['seed'], swap=row['swap'], survived=row['survived'], fallen=row['fallen'],
                             elapsed=row['elapsed'], scored_by_corner=row['scored_by_corner'],
                             first_interbody_time=min((c['elapsed'] for c in cross), default=None),
                             valid_cycles_by_corner=[sum(e['corner'] == c for e in valid) for c in (0, 1)]))
    return dict(pair_survivors=sum(e['survived'] for e in episodes), flags=dict(flags),
                valid_cycles_by_corner=[sum(e['valid_cycles_by_corner'][c] for e in episodes) for c in (0, 1)],
                falls_before_interbody=sum(not e['survived'] and e['first_interbody_time'] is None for e in episodes),
                episodes=episodes)


def main():
    state = json.loads((ROOT / 'state.json').read_text())
    progress = json.loads((ROOT / 'zero-arms-progress.json').read_text())
    assert state['status'] == progress['status'] == 'completed' and state['new_ppo_steps'] == 0
    assert state['source_sha256'] == SHA and state['prerequisites'] == verify_prerequisites()
    assert not (ROOT / 'zero-arms-errors.log').read_text().strip()
    masked = json.loads((ROOT / 'zero-arms.json').read_text())
    baseline = json.loads((BASELINE / 'selfplay.json').read_text())
    assert state['new_controls'] == progress['simulated_world_transitions'] == masked['simulated_world_transitions'] <= 4000
    assert state['formal_total'] == 2230238 + state['new_controls']
    result = dict(status='independently_reviewed', source_sha256=SHA,
                  unmasked=review_report(baseline, BASELINE / 'selfplay', False),
                  masked=review_report(masked, ROOT, True), new_controls=state['new_controls'],
                  formal_total=state['formal_total'], new_ppo_steps=0, promotion=False, main_adoption=False,
                  limitation='Masking both actors changes the full trajectory; this is neither a promotable gate nor proof of a unique cause.')
    for sub in ('native', 'full'):
        shutil.copytree(ROOT / sub, ARCHIVE / sub, dirs_exist_ok=True)
    for name in ('state.json', 'zero-arms.json', 'zero-arms-progress.json', 'zero-arms.log', 'zero-arms-errors.log'):
        shutil.copy2(ROOT / name, ARCHIVE / name)
    (ARCHIVE / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('new_controls', 'formal_total')}))
    print(json.dumps({k: {x: v for x, v in result[k].items() if x != 'episodes'} for k in ('unmasked', 'masked')}))


if __name__ == '__main__':
    main()
