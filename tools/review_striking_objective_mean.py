"""Archive completed action-mode comparison with matched and censored round identities."""
import argparse,hashlib,json,pickle,shutil
from pathlib import Path
import jax,numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_control import promotion_ready
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_objective_audit import SOURCE,SHA
from tools.striking_contact_compare import load_checked


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    live=parser.parse_args().input;mean=json.loads((live/'audit.json').read_text())
    sampled=json.loads(Path('docs/results/striking-pilot/objective-1933312/completed-audit.json').read_text())
    saved,_=load_checked(SOURCE,SHA);require_ppo_checkpoint(saved)
    for s in (sampled,mean):
        assert s['status']=='completed' and s['source_sha256']==SHA and s['simulated_world_transitions']==16384
        assert s['new_optimizer_updates']==s['new_collection_world_controls']==s['new_replay_presentations']==0 and not s['normalizer_updated']
        assert sum(r['controls'] for r in s['rounds'])+sum(s['open_round_controls'])==16384
        assert sum(sum(r) for r in s['controls_by_mode_and_corner'])==16384
        assert len(s['rounds'])==33 and len(s['open_round_controls'])==8
        assert np.isclose(sum(r['original_round_reward'] for r in s['rounds'])+sum(s['open_round_rewards']),s['cumulative_original_learner_reward'])
    assert mean['sampled_learner'] is False and mean['formal_evaluation_world_controls_before_job']==1470942
    def key(r):return (r['world'],r['seed'],r['swap'],r['learner_corner'])
    a={key(r):r for r in sampled['rounds']};b={key(r):r for r in mean['rounds']}
    assert len(a)==len(b)==33 and len(a.keys()&b.keys())==32
    result=dict(status='reviewed',source_sha256=SHA,diagnostic_ppo_total=1950976,formal_world_controls_total=1487326,
        new_evaluation_world_controls=16384,matched_rounds=32,sampled_only_rounds=[a[k] for k in sorted(a.keys()-b.keys())],
        mean_only_rounds=[b[k] for k in sorted(b.keys()-a.keys())],open_rounds_per_arm=8,groups=[],matched_trials=[],
        new_optimizer_updates=0,new_collection_world_controls=0,main_adoption=False,promotion=False,foundation_history_records={})
    for active in (False,True):
        keys=sorted(k for k in a.keys()&b.keys() if a[k]['active_opponent']==active)
        group=dict(active_opponent=active,matched_rounds=len(keys))
        for name,rows in (('sampled',[a[k] for k in keys]),('mean',[b[k] for k in keys])):
            survivors=[r for r in rows if not any(r['fallen'])]
            group[name]=dict(pair_survivors=len(survivors),surviving_events_by_corner=[sum(r['scored_by_corner'][i] for r in survivors) for i in (0,1)],
                surviving_two_way_episodes=sum(all(v>0 for v in r['scored_by_corner']) for r in survivors))
        result['groups'].append(group)
        result['matched_trials'] += [dict(identity=list(k),sampled=a[k],mean=b[k]) for k in keys]
    for stage in ('balance','walk_easy','walk','approach'):
        h=json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text())
        assert h==json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text()) and promotion_ready(h,stage)
        folder=Path('checkpoints/curriculum')/stage;m=json.loads((folder/'latest.json').read_text());data=(folder/m['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==m['sha256'] and all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(data)))
        result['foundation_history_records'][stage]=len(h)
    history=json.loads(Path('docs/results/striking-pilot/fixed-opponent-original-gate-history.json').read_text())
    assert len(history['records'])==7 and history['pass_streak']==0
    p=Path('checkpoints/striking-pilot/step-1048576');m=json.loads((p/'latest.json').read_text())
    assert hashlib.sha256((p/m['file']).read_bytes()).hexdigest()==m['sha256']=='efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    result['decision']=('The action-mode gap is profile-dependent: matched neutral sampled/mean pair survival 16/16 versus 12/16, active 11/16 versus 14/16. Do not deploy sampling as a gate replacement, add shared-fall cost, revive posture shaping or restore an earlier actor. Permit one bounded on-policy training comparison from latest c47edf0b: half Gaussian standard deviation versus unchanged control, all 17 channels active, unchanged physics/rewards/motor/opponent. Preserve all actor/critic/Adam/normalizer/RNG/sampler leaves at the explicit sampling-profile transition; use the same shifted distribution for collection, likelihood ratios and entropy. Maximum 16384 credits per arm/32768 total, source and final untouched 24M full sets, original full screens at 8192/16384, maximum 84000 evaluation controls. Stop severe physical regression or invalid provenance/physics. Evaluate deterministic mean with the original independent gates; no automatic promotion, adoption or further allocation.')
    out=Path('docs/results/striking-pilot/objective-mean-1933312');shutil.copy2(live/'audit.json',out/'completed-audit.json');atomic_json(out/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('matched_trials','sampled_only_rounds','mean_only_rounds')},indent=2))


if __name__=='__main__':main()
