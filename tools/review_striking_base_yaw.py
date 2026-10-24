"""Verify complete base-yaw comparison and reject mixed physical improvement."""
import argparse,hashlib,json,pickle,shutil
from pathlib import Path
import jax,numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_control import promotion_ready
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_contact_compare import load_checked


def metrics(folder,seeds,sha,scale):
    result={};controls=0
    for role in (0,1,None):
        name='selfplay' if role is None else f'neutral-{role}'
        r=json.loads((folder/(name+'.json')).read_text());p=json.loads((folder/(name+'-progress.json')).read_text());rows=r['trials']
        assert r['striking_sha256']==p['striking_sha256']==sha and p['status']=='completed'
        assert r.get('diagnostic_base_yaw_scale',1.)==scale and not r['assistance'] and not r['sampled_policy']
        assert not r.get('diagnostic_zero_command_offsets',False) and not r.get('diagnostic_zero_arm_residuals',False) and r.get('diagnostic_approach_gain') is None and not r.get('diagnostic_profile_override',False)
        assert len(rows)==8 and {(t['seed'],t['swap']) for t in rows}=={(s,w) for s in seeds for w in (False,True)} and all(t['finite'] for t in rows)
        assert r['simulated_world_transitions']==p['simulated_world_transitions'];controls+=p['simulated_world_transitions']
        survived=[t for t in rows if t['survived']];own=sorted({t['seed'] for t in survived if (sum(t['scored_by_corner']) if role is None else t['scored_by_corner'][role])>0})
        result[name]=dict(pair_survivors=len(survived),own_surviving_scoring_seeds=own,surviving_events_by_corner=[sum(t['scored_by_corner'][i] for t in survived) for i in (0,1)],surviving_two_way_episodes=sum(all(v>0 for v in t['scored_by_corner']) for t in survived),meets_numeric_profile_thresholds=len(survived)>=(8 if role==1 else 7) and (role is None or len(own)>=2))
    return result,controls


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);live=parser.parse_args().input
    state=json.loads((live/'comparison.json').read_text());assert state['status']=='completed' and state['diagnostic_evaluation_world_controls']==33287 and len(state['checks'])==9
    assert all(not p.read_text().strip() for p in live.rglob('*errors.log'))
    out=Path('docs/results/striking-pilot/base-yaw-1859584');result=dict(status='reviewed',diagnostic_ppo_total=1844480,formal_world_controls_total=1139761,new_evaluation_world_controls=33287,new_optimizer_updates=0,main_adoption=False,promotion=False,native_error_logs_empty=True,foundation_history_records={},samples={})
    for stage in ('balance','walk_easy','walk','approach'):
        h=json.loads(Path(f'runs/curriculum/{stage}/gate_history.json').read_text());assert h==json.loads(Path(f'docs/results/curriculum/{stage}-gate_history.json').read_text()) and promotion_ready(h,stage)
        f=Path('checkpoints/curriculum')/stage;m=json.loads((f/'latest.json').read_text());b=(f/m['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==m['sha256'] and all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(pickle.loads(b)));result['foundation_history_records'][stage]=len(h)
    source=Path('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192');saved,_=load_checked(source,state['source_sha256']);require_ppo_checkpoint(saved)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((saved['actor'],saved['critic'],saved['optimizer'],saved['norm'])))
    f=Path('checkpoints/striking-pilot/step-1048576');m=json.loads((f/'latest.json').read_text());assert hashlib.sha256((f/m['file']).read_bytes()).hexdigest()==m['sha256']=='efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0'
    controls=0
    reference=Path(state['control_16m_reference']);result['samples']['control-16m'],_=metrics(reference,list(range(16000003,16000007)),state['source_sha256'],1.)
    for name,scale,seeds in [('modified-16m',.5,range(16000003,16000007)),('control-17m',1.,range(17000003,17000007)),('modified-17m',.5,range(17000003,17000007))]:
        row,n=metrics(live/name,list(seeds),state['source_sha256'],scale);controls+=n;result['samples'][name]=row
        assert not all(v['meets_numeric_profile_thresholds'] for v in row.values())
        shutil.copytree(live/name,out/name,dirs_exist_ok=True)
    assert controls==33287
    result['decision']='Reject global base-yaw scaling: neutral survival improves in both samples, but active survival regresses 6/8 to 5/8 on 16M and 7/8 to 5/8 on fresh 17M. Do not adopt controller or allocate training for it. Return to the unchanged version-6 latest final. Permit one matched bounded fixed-opponent versus current-opponent training comparison, 16384 learner controls per arm, preserving physics/actions/gates and every learned optimizer prefix. This tests a stationary opponent hypothesis, not a proven cause of instability. No automatic further allocation or promotion.'
    shutil.copy2(live/'comparison.json',out/'completed-comparison.json');atomic_json(out/'analysis.json',result);print(json.dumps(result))

if __name__=='__main__':main()
