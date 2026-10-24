"""Independently verify and archive all reset groups and cross-policy matches."""
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import jax
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_control import promotion_ready
from boxer.striking_fixed_opponent import FROZEN_FINGERPRINT, fingerprint
from boxer.striking_train import require_ppo_checkpoint
from tools.review_striking_credit_horizon import full_profiles
from tools.striking_contact_compare import load_checked
from tools.striking_generalization_audit import SOURCE, SHA, OPPONENTS


def fallen_summary(rows):
    return dict(episodes=len(rows), pair_survivors=sum(t['survived'] for t in rows),
        fallen_by_corner=[sum(t['fallen'][c] for t in rows) for c in (0, 1)],
        fallen_by_swap=[[sum(t['fallen'][c] for t in rows if t['swap'] == s)
            for c in (0, 1)] for s in (False, True)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    live = parser.parse_args().input
    state = json.loads((live/'audit.json').read_text())
    assert state['status'] == 'completed' and len(state['independent_groups']) == 4 and len(state['cross_policy_checks']) == 4
    assert state['new_optimizer_updates'] == state['new_collection_world_controls'] == 0
    errors = list(live.rglob('*errors.log'))
    assert len(errors) == 16 and all(not p.read_text().strip() for p in errors)
    saved, _ = load_checked(SOURCE, SHA); require_ppo_checkpoint(saved)
    assert fingerprint(saved['frozen_opponent']) == FROZEN_FINGERPRINT
    assert saved['consumed_steps'] == saved['optimized_steps'] == 1916928
    assert saved['world_control_steps'] == 1294336
    assert int(saved['optimizer']['actor_adam'][0].count) == 12424
    assert int(saved['optimizer']['critic_adam'][0].count) == 36096
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(saved[k] for k in ('actor','critic','optimizer','norm','key'))))
    result = dict(status='reviewed', source_sha256=SHA, diagnostic_ppo_total=1918208,
        formal_world_controls_total=1351541, actual_new_evaluation_world_controls=55067,
        episodes=128, new_optimizer_updates=0, new_collection_world_controls=0,
        main_adoption=False, promotion=False, original_complete_pass_streak_retained=2,
        native_error_logs_empty=True, groups=[], cross_policy_checks=[], foundation_history_records={})
    controls = 0; profiles = {n: [] for n in ('neutral-0','neutral-1','selfplay')}
    for group in range(4):
        seeds = list(range(21000003+4*group, 21000007+4*group))
        full, n = full_profiles(live/f'group-{group}', seeds, SHA); controls += n
        assert full == state['independent_groups'][group]['full'] and not full['passes_complete']
        result['groups'].append(dict(group=group, seeds=seeds, full=full))
        for name in profiles:
            r = json.loads((live/f'group-{group}'/(name+'.json')).read_text())
            assert r['source_sha256'] == saved['foundation_sha256'] and r['actor_inputs'] == 148 and r['actions'] == 17
            profiles[name].extend(r['trials'])
    result['profile_totals'] = {name: fallen_summary(rows) for name, rows in profiles.items()}
    active = [t for t in profiles['selfplay'] if t['survived']]
    result['surviving_active_events_by_corner'] = [sum(t['scored_by_corner'][c] for t in active) for c in (0,1)]
    result['surviving_active_two_way_episodes'] = sum(all(v > 0 for v in t['scored_by_corner']) for t in active)
    for name, model, sha in OPPONENTS:
        other, _ = load_checked(model, sha); require_ppo_checkpoint(other)
        assert other['foundation_sha256'] == saved['foundation_sha256']
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(other[k] for k in ('actor','critic','optimizer','norm','key'))))
        for corner in (0,1):
            folder = live/name/f'corner-{corner}'
            r = json.loads((folder/'evaluation.json').read_text()); p = json.loads((folder/'evaluation-progress.json').read_text()); rows=r['trials']
            assert r['striking_sha256'] == p['striking_sha256'] == SHA and r['opponent_sha256'] == sha and r['evaluated_corner'] == corner
            assert r['source_sha256'] == saved['foundation_sha256'] and r['actor_inputs'] == 148 and r['actions'] == 17
            assert p['status'] == 'completed' and p['simulated_world_transitions'] == r['simulated_world_transitions']
            assert len(rows) == 8 and {(t['seed'],t['swap']) for t in rows} == {(s,w) for s in range(22000003,22000007) for w in (False,True)}
            assert all(t['finite'] for t in rows) and not r['assistance'] and not r['sampled_policy']
            assert not r['diagnostic_zero_command_offsets'] and not r['diagnostic_zero_arm_residuals'] and not r['diagnostic_profile_override']
            assert r['neutral_partner_learner_corner'] is None and r['diagnostic_approach_gain'] is None and r['diagnostic_base_yaw_scale'] == 1.
            survivors = [t for t in rows if t['survived']]
            summary = dict(opponent=name, opponent_sha256=sha, evaluated_corner=corner,
                pair_survivors=len(survivors), evaluated_corner_fallen=sum(t['fallen'][corner] for t in rows),
                opponent_corner_fallen=sum(t['fallen'][1-corner] for t in rows),
                surviving_events_by_corner=[sum(t['scored_by_corner'][i] for t in survivors) for i in (0,1)],
                surviving_two_way_episodes=sum(all(v > 0 for v in t['scored_by_corner']) for t in survivors))
            assert summary == state['cross_policy_checks'][len(result['cross_policy_checks'])]
            result['cross_policy_checks'].append(summary); controls += p['simulated_world_transitions']
    assert controls == state['diagnostic_evaluation_world_controls'] == 55067
    assert 1296474 + controls == result['formal_world_controls_total']
    for stage in ('balance','walk_easy','walk','approach'):
        h=json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert h == json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text()) and promotion_ready(h,stage)
        folder=Path('checkpoints/curriculum')/stage; m=json.loads((folder/'latest.json').read_text()); data=(folder/m['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == m['sha256'] and all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(data)))
        result['foundation_history_records'][stage]=len(h)
    folder=Path('checkpoints/striking-pilot/step-1048576'); m=json.loads((folder/'latest.json').read_text())
    assert hashlib.sha256((folder/m['file']).read_bytes()).hexdigest() == m['sha256'] == 'efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    history=json.loads(Path('docs/results/striking-pilot/fixed-opponent-original-gate-history.json').read_text())
    assert len(history['records']) == 5 and len({r['sha256'] for r in history['records']}) == 5
    assert [r['screen']['passes_complete'] for r in history['records']] == [False,False,False,True,True]
    result['decision'] = ('All four independent full groups fail. Preserve the latest actor and frozen adversary; no PPO extension, main adoption or promotion. Neutral failures include ten learner and three motor-only partner falls; some occur early at large separation. Before any further training change, run a bounded matched motor-floor diagnostic on all 16 existing fresh seeds (32 both-head-zero trials), plus exact passive full-head replays on four preselected failure seeds across all three profiles (24 trials). Maximum 28000 additional evaluation controls, zero training. Trace actual interbody contact and first-fall conditions; head masking is diagnostic only and must never replace original gates. Decide motor versus head/opponent coverage from the complete matched evidence, not favorable episodes.')
    out=Path('docs/results/striking-pilot/generalization-1916928')
    for path in live.rglob('*'):
        if path.is_file() and path.suffix == '.json':
            target=out/path.relative_to(live); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    for path in errors:
        target=out/path.relative_to(live).with_suffix('.txt')
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    atomic_json(out/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
