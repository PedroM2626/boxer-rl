"""Review immutable legacy traces under the existing complete straight-cycle rules."""
import hashlib
import json
from pathlib import Path
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.striking_punch_trace import describe_events


def qualifies(event):
    return bool(event['onset_relative_speed'] >= 1. and event['onset_forward_cosine'] >= .7
        and event['pre_max_displacement_m'] >= .12 and event['post_window_complete']
        and event['return_toward_pre_window_position_m'] is not None
        and event['return_toward_pre_window_position_m'] >= .06 and event['post_contact_seconds'] <= .10)


def review_saved(root):
    root = Path(root)
    state = json.loads((root/'state.json').read_text())
    assert state['status'] == 'completed' and len(state['episodes']) == 16
    assert state['new_controls'] == sum(e['controls'] for e in state['episodes']) <= 7680
    assert state['formal_total'] == state['formal_before']+state['new_controls']
    assert state['gpu_world_substeps'] == sum(e['gpu_substeps'] for e in state['episodes'])
    groups = {}
    for trial in state['episodes']:
        path = root/trial['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == trial['trace_sha256']
        with np.load(path) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            assert trial['physics_dt'] == .004
            events = describe_events(arrays, native_timestep=trial['physics_dt'])
            assert events == trial['events']
        cycles = [sum(e['corner'] == c and qualifies(e) for e in events) for c in (0, 1)]
        assert cycles == trial['complete_cycles']
        key = f"{trial['run']}-spawn-{trial['separation']}"
        group = groups.setdefault(key, dict(episodes=0, complete_rounds_without_fall=0,
            valid_cycles=[0, 0], reciprocal_episodes_before_stop=0, reciprocal_complete_rounds=0,
            controls=0, terminal_reasons={}))
        group['episodes'] += 1; group['controls'] += trial['controls']
        full = trial['completed_480_controls'] and trial['common_posture_intact_at_stop'] and not trial['terminal_reasons']['fell']
        reciprocal = all(c > 0 for c in cycles)
        group['complete_rounds_without_fall'] += int(full)
        group['reciprocal_episodes_before_stop'] += int(reciprocal)
        group['reciprocal_complete_rounds'] += int(full and reciprocal)
        group['valid_cycles'] = [a+b for a, b in zip(group['valid_cycles'], cycles)]
        for reason, present in trial['terminal_reasons'].items():
            group['terminal_reasons'][reason] = group['terminal_reasons'].get(reason, 0)+int(present)
    assert len(groups) == 4 and all(g['episodes'] == 4 for g in groups.values())
    any_cycles = any(sum(g['valid_cycles']) for g in groups.values())
    decision = ('Retained policies demonstrate complete measured cycles in this reconstructed sample; compare separated transfer and stability before architecture selection.'
        if any_cycles else 'No complete straight cycles in this bounded reconstructed sample; this does not establish global infeasibility or invalidate every old contact.')
    atomic_json(root/'completed-review.json', dict(status='independently_reviewed', groups=groups,
        formal_total=state['formal_total'], new_controls=state['new_controls'], new_ppo_steps=0,
        unused_legacy_allowance_expired=7680-state['new_controls'], decision=decision,
        limitations=state['limitation'], native_current_gate_certification=False))
    rows = ['# Retained legacy punch-cycle measurements', '', decision, '',
        '| Run / spawn | Complete rounds without fall / 4 | Valid cycles by corner | Reciprocal complete rounds |',
        '| --- | --- | --- | --- |']
    for key, group in groups.items():
        rows.append(f"| {key} | {group['complete_rounds_without_fall']} | {group['valid_cycles']} | {group['reciprocal_complete_rounds']} |")
    rows += ['', state['limitation'], '',
        'Fist-only impacts, own-torso motion, travel and return use the existing criteria. Legacy 4 ms contact duration is explicit. These are not promotable current-model gates.',
        '', f"New formal controls: {state['new_controls']}; cumulative {state['formal_total']}. No learner updates or policy adoption."]
    (root/'README.md').write_text('\n'.join(rows)+'\n')


if __name__ == '__main__':
    review_saved('/home/pedro/boxing-runs/legacy-punch-cycles-v1')
