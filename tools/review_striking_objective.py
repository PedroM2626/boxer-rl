"""Independently account frozen-policy terminations, open rounds and reward gaps."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.striking_train import require_ppo_checkpoint
from boxer.striking_objective_audit import uncharged_pair_fall_cost
from tools.striking_contact_compare import load_checked
from tools.striking_objective_audit import SOURCE,SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    live=parser.parse_args().input;s=json.loads((live/'audit.json').read_text())
    assert s['status']=='completed' and s['simulated_world_transitions']==16384 and s['source_sha256']==SHA
    assert s['new_optimizer_updates']==s['new_collection_world_controls']==s['new_replay_presentations']==0 and not s['normalizer_updated']
    saved,m=load_checked(SOURCE,SHA);require_ppo_checkpoint(saved)
    assert hashlib.sha256((SOURCE/m['file']).read_bytes()).hexdigest()==SHA
    assert sum(sum(r) for r in s['controls_by_mode_and_corner'])==16384
    assert sum(r['controls'] for r in s['rounds'])+sum(s['open_round_controls'])==16384
    assert len(s['open_round_controls'])==8 and all(0<=n<500 for n in s['open_round_controls'])
    assert len({(r['world'],r['seed'],r['swap']) for r in s['rounds']})==len(s['rounds'])==33
    gaps=[]
    for r in s['rounds']:
        gap=uncharged_pair_fall_cost(r['fallen'],r['learner_corner']);gaps.append(gap)
        assert gap==r['uncharged_pair_fall_cost'] and r['counterfactual_terminal_reward']==r['original_terminal_reward']-gap
        assert r['own_fall_penalty']==5.*r['fallen'][r['learner_corner']]
        assert 1<=r['controls']<=500 and (any(r['fallen']) or r['timed_out'])
        assert np.isfinite(r['original_round_reward']) and np.isfinite(r['terminal_learner_action']).all()
    assert sum(gaps)==s['counterfactual_additional_pair_fall_cost']==5.
    assert np.isclose(sum(r['original_round_reward'] for r in s['rounds'])+sum(s['open_round_rewards']),s['cumulative_original_learner_reward'])
    groups=[]
    for active in (False,True):
        for corner in (0,1):
            rows=[r for r in s['rounds'] if r['active_opponent']==active and r['learner_corner']==corner]
            groups.append(dict(active_opponent=active,learner_corner=corner,rounds=len(rows),own_only_falls=sum(r['fallen'][corner] and not r['fallen'][1-corner] for r in rows),
                partner_only_falls=sum(r['fallen'][1-corner] and not r['fallen'][corner] for r in rows),simultaneous_falls=sum(all(r['fallen']) for r in rows),
                timeouts=sum(r['timed_out'] for r in rows),partner_only_without_valid_strikes=sum(r['uncharged_pair_fall_cost']>0 and sum(r['scored_by_corner'])==0 for r in rows)))
    result=dict(status='reviewed',source_sha256=SHA,diagnostic_ppo_total=1950976,formal_world_controls_total=1470942,
        actual_new_evaluation_world_controls=16384,completed_rounds=33,open_rounds=8,own_only_falls=4,partner_only_falls=1,
        simultaneous_falls=0,uncharged_pair_fall_cost=5.,groups=groups,new_optimizer_updates=0,new_collection_world_controls=0,
        main_adoption=False,promotion=False,source_checkpoint_unchanged=True,
        decision='One partner-only fall confirms the objective mismatch, but four own falls dominate and neutral sampled rounds all survive; no shared-fall reward change or PPO extension follows this small frozen-policy sample. Run one matched mean-action rollout on the same latest control, immutable opponent, eight worlds and identical initial reset schedule, at most 16384 evaluation controls and zero updates. Match complete rounds by world/seed/swap, report censored unmatched tails, and separate action sampling from seed/opponent coverage before another training intervention. Retain all source and accounting state; no original-gate replacement or promotion.')
    out=Path('docs/results/striking-pilot/objective-1933312');shutil.copy2(live/'audit.json',out/'completed-audit.json');atomic_json(out/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
