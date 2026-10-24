"""Independently archive matched motor-floor outcomes and exact passive replays."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from boxer.checkpoint_io import atomic_json
from tools.striking_generalization_audit import SOURCE,SHA
from tools.striking_contact_compare import load_checked
from boxer.striking_train import require_ppo_checkpoint
from tools.striking_mixed_failure_trace import summarize_trial


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    live=parser.parse_args().input;state=json.loads((live/'audit.json').read_text())
    assert state['status']=='completed' and len(state['checks'])==4 and state['diagnostic_evaluation_world_controls']==23933
    assert state['new_optimizer_updates']==state['new_collection_world_controls']==0
    source,_=load_checked(SOURCE,SHA);require_ppo_checkpoint(source)
    reference=Path('docs/results/striking-pilot/generalization-1916928')
    previous=json.loads((reference/'analysis.json').read_text());assert previous['status']=='reviewed' and previous['formal_world_controls_total']==1351541
    out=Path('docs/results/striking-pilot/motor-floor-1916928')
    result=dict(status='reviewed',source_sha256=SHA,episodes=56,new_evaluation_world_controls=23933,
        formal_world_controls_total=1375474,diagnostic_ppo_total=1918208,new_optimizer_updates=0,
        new_collection_world_controls=0,main_adoption=False,promotion=False,checks=[],matched_profiles={},trace_manifest=[])
    controls=0;motor={}
    for check in state['checks']:
        name=check['name'];folder=live/name;r=json.loads((folder/'evaluation.json').read_text());p=json.loads((folder/'evaluation-progress.json').read_text())
        assert not (folder/'errors.log').read_text().strip() and p['status']=='completed'
        assert r['striking_sha256']==p['striking_sha256']==SHA and r['source_sha256']==source['foundation_sha256']
        assert r['simulated_world_transitions']==p['simulated_world_transitions'] and not r['assistance'] and not r['sampled_policy']
        assert r['diagnostic_base_yaw_scale']==1. and r['opponent_sha256'] is None and not r['diagnostic_profile_override']
        assert r['diagnostic_approach_gain'] is None and r['actor_inputs']==148 and r['actions']==17
        masked=name=='motor-only';assert r['diagnostic_zero_command_offsets']==r['diagnostic_zero_arm_residuals']==masked
        role={'neutral-0':0,'neutral-1':1}.get(name);assert r['neutral_partner_learner_corner']==role
        seeds=list(range(21000003,21000019)) if masked else [21000006,21000009,21000011,21000018]
        rows=r['trials'];assert len(rows)==len(seeds)*2 and {(t['seed'],t['swap']) for t in rows}=={(s,w) for s in seeds for w in (False,True)}
        original={} if masked else {(t['seed'],t['swap']):t for i in range(4) for t in json.loads((reference/f'group-{i}'/(name+'.json')).read_text())['trials']}
        summaries=[]
        for t in rows:
            assert t['finite']
            if not masked: assert all(t.get(k)==v for k,v in original[t['seed'],t['swap']].items())
            summary=summarize_trial(t);summaries.append(summary)
            trace=Path(t['trace_file']);assert hashlib.sha256(trace.read_bytes()).hexdigest()==t['trace_sha256']
            with np.load(trace) as arrays:
                assert all(np.isfinite(v).all() for v in arrays.values())
                if masked: assert np.count_nonzero(arrays['action'])==np.count_nonzero(arrays['arm_residual'])==0
            record=dict(profile=name,seed=t['seed'],swap=t['swap'],live_path=str(trace),sha256=t['trace_sha256'],bytes=trace.stat().st_size)
            if not t['survived']:
                dest=out/name/'fall-traces'/trace.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(trace,dest)
                record['repository_copy']=str(dest)
            result['trace_manifest'].append(record)
            if masked:motor[t['seed'],t['swap']]=t
        assert summaries==check['trials']
        falls=[t for t in summaries if not t['survived']]
        result['checks'].append(dict(name=name,episodes=len(rows),pair_survivors=len(rows)-len(falls),
            precontact_falls=sum(t['first_interbody_contact_time'] is None for t in falls),
            low_pelvis_falls=sum(any(c['low_pelvis'] for c in t['terminal_corners']) for t in falls),
            self_leg_contact_falls=sum(any(c['self_leg_contact'] for c in t['terminal_corners']) for t in falls),trials=summaries))
        controls+=p['simulated_world_transitions'];dest=out/name;dest.mkdir(parents=True,exist_ok=True)
        for filename in ('evaluation.json','evaluation-progress.json'):shutil.copy2(folder/filename,dest/filename)
        shutil.copy2(folder/'errors.log',dest/'errors.txt')
    for name in ('neutral-0','neutral-1','selfplay'):
        rows=[t for i in range(4) for t in json.loads((reference/f'group-{i}'/(name+'.json')).read_text())['trials']]
        result['matched_profiles'][name]=dict(episodes=32,full_head_pair_survivors=sum(t['survived'] for t in rows),
            motor_only_pair_survivors=sum(t['survived'] for t in motor.values()),
            failed_full_surviving_motor=[dict(seed=t['seed'],swap=t['swap']) for t in rows if not t['survived'] and motor[t['seed'],t['swap']]['survived']],
            surviving_full_failed_motor=[dict(seed=t['seed'],swap=t['swap']) for t in rows if t['survived'] and not motor[t['seed'],t['swap']]['survived']])
    assert controls==23933
    result['decision']=('Motor-only survival is 31/32; an inherited precontact self-leg failure remains. Exact passive replays show additional head-dependent precontact leg contact and pelvis collapses plus contact-associated failures. Masking is not adopted. Keep the motor, all learned action channels, original frozen opponent, unassisted physics and independent gates. Permit a single matched training-only posture-cost comparison from latest 670bed08: cost versus unchanged control, at most 16384 learner/world credits per arm, 32768 total; original full screens at 8192/16384 and fresh 23M source/finals, at most 84000 evaluation controls (168 episodes). Penalize low-pelvis/tilt margin only in the learner reward, not state/actions/termination. Retain all learned and optimizer/sampler state; stop severe regression or invalid physics. No main adoption, promotion or automatic further extension; final physical survival and surviving own strikes decide whether the cost helps.')
    shutil.copy2(live/'audit.json',out/'completed-audit.json');atomic_json(out/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trace_manifest','checks')},indent=2))


if __name__=='__main__':main()
