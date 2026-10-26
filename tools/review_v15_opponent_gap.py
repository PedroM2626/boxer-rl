"""Archive and verify the bounded frozen-versus-current opponent comparison."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from boxer.striking_punch_trace import describe_events
from tools.striking_contact_progress_supervisor import independent_cycle
from tools.run_v15_opponent_gap import ROOT, ARCHIVE, FINAL_SHA, OPPONENT_SHA


def main():
    state = json.loads((ROOT / 'state.json').read_text())
    assert state['status'] == 'completed' and state['new_ppo_steps'] == 0
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    cases, consumed = [], 0
    for corner in (0, 1):
        name = f'corner-{corner}'
        report = json.loads((ROOT / (name + '.json')).read_text())
        assert report['striking_sha256'] == FINAL_SHA and report['opponent_sha256'] == OPPONENT_SHA
        assert report['evaluated_corner'] == corner and not report['sampled_policy'] and not report['assistance']
        assert report['neutral_partner_learner_corner'] is None
        assert len(report['trials']) == 8
        assert {(r['seed'], r['swap']) for r in report['trials']} == {(s, w) for s in range(34000003, 34000007) for w in (False, True)}
        assert not (ROOT / (name + '-errors.log')).read_text().strip()
        consumed += report['simulated_world_transitions']
        rows = []
        for row in report['trials']:
            assert row['finite']
            for field, digest, sub in [('trace_file', 'trace_sha256', 'full'), ('punch_trace_file', 'punch_trace_sha256', 'native')]:
                trace = ROOT / name / sub / Path(row[field]).name
                assert hashlib.sha256(trace.read_bytes()).hexdigest() == row[digest]
                with np.load(trace) as raw:
                    assert all(np.isfinite(raw[k]).all() for k in raw.files)
                    if sub == 'native':
                        events = describe_events(raw)
                        assert events == row['punch_measurements']
            valid = [e for e in events if independent_cycle(e)]
            rows.append(dict(seed=row['seed'], swap=row['swap'], survived=row['survived'],
                             fallen=row['fallen'], elapsed=row['elapsed'],
                             scored_by_corner=row['scored_by_corner'],
                             cycles_by_corner=[sum(e['corner'] == c for e in valid) for c in (0, 1)]))
        cases.append(dict(learner_corner=corner, pair_survivors=sum(r['survived'] for r in rows),
                          learner_falls=sum(r['fallen'][corner] for r in rows),
                          opponent_falls=sum(r['fallen'][1-corner] for r in rows),
                          learner_valid_cycles=sum(r['cycles_by_corner'][corner] for r in rows),
                          learner_surviving_valid_seeds=sorted({r['seed'] for r in rows if r['survived'] and r['cycles_by_corner'][corner]}),
                          episodes=rows))
        shutil.copytree(ROOT / name, ARCHIVE / name, dirs_exist_ok=True)
        for suffix in ('.json', '-progress.json', '.log', '-errors.log'):
            shutil.copy2(ROOT / (name + suffix), ARCHIVE / (name + suffix))
    assert consumed == state['new_controls'] <= 8000 and state['formal_total'] == 2183338 + consumed
    mirror = json.loads(Path('docs/results/striking-pilot/contact-progress-v15/gates/fresh-final/motion-gate.json').read_text())
    assert mirror['sha256'] == FINAL_SHA and mirror['seeds'] == list(range(34000003, 34000007))
    result = dict(status='reviewed', cases=cases, reused_mirror_contact=mirror['original_contact_gate']['profiles']['selfplay'],
                  reused_mirror_motion=mirror['motion']['selfplay'], new_controls=consumed,
                  formal_total=state['formal_total'], new_ppo_steps=0, main_adoption=False, promotion=False,
                  limitation='Opponent identity is isolated on these resets; no broader causal or convergence claim.')
    shutil.copy2(ROOT / 'state.json', ARCHIVE / 'state.json')
    (ARCHIVE / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('cases', 'new_controls', 'formal_total')}))


if __name__ == '__main__':
    main()
