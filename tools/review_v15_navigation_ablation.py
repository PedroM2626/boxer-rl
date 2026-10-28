"""Independently review and archive the completed saved navigation ablation."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from boxer.striking_punch_trace import describe_events
from tools.striking_contact_progress_supervisor import independent_cycle
from tools.striking_pool_compare import verify_prerequisites

ROOT = Path('/home/pedro/boxing-runs/striking-v15-navigation-ablation')
ARCHIVE = Path('docs/results/striking-pilot/v15-navigation-ablation')
POLICY = '9132ab4c2c86ed2b2958d551125b001370bf0f4e80f403490d33111ac3bb634a'


def main():
    state = json.loads((ROOT / 'state.json').read_text())
    report = json.loads((ROOT / 'zero-commands.json').read_text())
    progress = json.loads((ROOT / 'zero-commands-progress.json').read_text())
    assert state['status'] == progress['status'] == 'completed'
    assert state['prerequisites'] == verify_prerequisites()
    assert report['striking_sha256'] == POLICY and report['diagnostic_zero_command_offsets']
    assert not report['diagnostic_zero_arm_residuals'] and not report['assistance'] and not report['sampled_policy']
    assert not report['certification'] and report['opponent_sha256'] is None
    assert len(report['trials']) == 8
    assert {(r['seed'], r['swap']) for r in report['trials']} == {(s, w) for s in range(34000003, 34000007) for w in (False, True)}
    assert not (ROOT / 'zero-commands-errors.log').read_text().strip()
    assert state['new_controls'] == progress['simulated_world_transitions'] == report['simulated_world_transitions'] <= 4000
    assert state['formal_total'] == 2188806 + state['new_controls']
    episodes, events = [], []
    for row in report['trials']:
        assert row['finite']
        for field, digest, sub in [('trace_file', 'trace_sha256', 'full'), ('punch_trace_file', 'punch_trace_sha256', 'native')]:
            path = ROOT / 'zero-commands' / sub / Path(row[field]).name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[digest]
            with np.load(path) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
                if sub == 'native':
                    described = describe_events(arrays)
                    assert described == row['punch_measurements']
        valid = [event for event in described if independent_cycle(event)]
        events.extend(valid)
        episodes.append(dict(seed=row['seed'], swap=row['swap'], survived=row['survived'],
                             fallen=row['fallen'], elapsed=row['elapsed'],
                             scored_by_corner=row['scored_by_corner'], terminal_state=row['terminal_state'],
                             valid_cycles_by_corner=[sum(e['corner'] == c for e in valid) for c in (0, 1)]))
    prior = json.loads(Path('docs/results/striking-pilot/contact-progress-v15/gates/fresh-final/motion-gate.json').read_text())
    assert prior['sha256'] == POLICY and prior['seeds'] == list(range(34000003, 34000007))
    result = dict(status='independently_reviewed', policy_sha256=POLICY, new_controls=state['new_controls'],
                  formal_total=state['formal_total'], masked_pair_survivors=sum(r['survived'] for r in episodes),
                  masked_valid_cycles=len(events), episodes=episodes,
                  unmasked_pair_survivors=prior['original_contact_gate']['profiles']['selfplay']['pair_survivors'],
                  unmasked_valid_cycles=sum(prior['motion']['selfplay']['valid_cycles_by_corner']),
                  promotion=False, main_adoption=False, new_ppo_steps=0,
                  limitation='Masking both actors is an intervention, not a complete gate or proof of isolated causation.')
    shutil.copytree(ROOT / 'zero-commands', ARCHIVE / 'zero-commands', dirs_exist_ok=True)
    for name in ('state.json', 'zero-commands.json', 'zero-commands-progress.json', 'zero-commands.log', 'zero-commands-errors.log'):
        shutil.copy2(ROOT / name, ARCHIVE / name)
    (ARCHIVE / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('new_controls', 'formal_total', 'masked_pair_survivors', 'masked_valid_cycles', 'unmasked_pair_survivors', 'unmasked_valid_cycles')}))


if __name__ == '__main__':
    main()
