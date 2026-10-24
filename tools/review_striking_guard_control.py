"""Review both matched guard-control arms with no additional physical compute."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU comparison review required')
import numpy as np
from boxer.checkpoint_io import atomic_json,atomic_bytes
from boxer.curriculum_train import utc
from boxer.striking_punch_trace import describe_events
from boxer.jab_feasibility import PROFILE
from tools.striking_guard_control import BASE,LIMIT,SEEDS,MODES,REFERENCE,sha
from tools.striking_pool_compare import verify_prerequisites


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();root=args.input;state=json.loads((root/'audit.json').read_text())
    assert state['status']=='completed' and state['phase']=='awaiting_interpretation'
    assert state['profile']==PROFILE and state['formal_world_controls_before_job']==BASE
    assert state['maximum_world_controls']==LIMIT and state['formal_world_controls_ceiling']==BASE+LIMIT
    assert verify_prerequisites()==state['prerequisites']
    assert sha(REFERENCE/'audit.json')==state['source_audit_sha256']
    assert sha(REFERENCE/'analysis.json')==state['source_review_sha256']
    assert state['new_ppo_steps']==state['new_supervised_updates']==state['new_optimizer_updates']==0
    assert state['diagnostic_ppo_retained']==2163968 and state['cloned_forward_evaluations_retained']==4339
    assert state['main_project_consumed_retained']==476041216 and state['main_boxing_retained']==1048576
    assert not state['assistance'] and not state['learned_controller'] and not state['promotion'] and state['policy_adoption']=='none'
    assert not (root/'native-errors.log').read_text().strip()
    expected={(m,c,s,w) for m in MODES for c in (0,1) for s in SEEDS for w in (False,True)}
    assert len(state['trials'])==16 and {(r['mode'],r['corner'],r['seed'],r['swap']) for r in state['trials']}==expected
    count=sum(r['controls'] for r in state['trials'])
    assert count==state['diagnostic_world_controls']==json.loads((root/'progress.json').read_text())['diagnostic_world_controls']<=LIMIT
    assert state['formal_world_controls_total']==BASE+count
    source=json.loads((REFERENCE/'audit.json').read_text());summaries={};trials=[];baseline_states={}
    for mode in MODES:
        summaries[mode]=dict(episodes=0,pair_survivors=0,acting_falls=0,other_falls=0,contacts=0,
            acting_first_fall_flags={k:0 for k in ('low_pelvis','tilted','self_leg_contact')})
    clips=[];tracking=[]
    for row in state['trials']:
        c=row['corner'];key=(c,row['seed'],row['swap']);mode=row['mode'];summary=summaries[mode]
        reference=next(r for r in source['trials'] if (r['corner'],r['seed'],r['swap'])==key)
        original=REFERENCE/'full'/Path(reference['full_trace_file']).name
        assert sha(original)==reference['full_trace_sha256']==row['reference_full_trace_sha256']
        path=root/'full'/Path(row['full_trace_file']).name;assert sha(path)==row['full_trace_sha256']
        with np.load(path) as a,np.load(original) as b:
            assert all(np.isfinite(a[k]).all() for k in a.files) and len(a['qpos'])==row['controls']<=500
            np.testing.assert_array_equal(a['qpos_before'][1:],a['qpos'][:-1]);np.testing.assert_array_equal(a['qvel_before'][1:],a['qvel'][:-1])
            np.testing.assert_array_equal(a['external_force_norm'],0.)
            assert a['time_after'][-1]==row['elapsed']
            assert row['survived']==(not a['fallen'][-1].any()) and (not row['survived'] or row['controls']==500)
            assert not a['fallen'][:-1].any()
            for corner in (0,1):
                flags=row['terminal']['corners'][corner]
                assert bool(a['fallen'][-1,corner])==any(flags[k] for k in ('low_pelvis','tilted','self_leg_contact'))
                assert flags['pelvis_height']==a['pelvis_height'][-1,corner] and flags['upright']==a['upright'][-1,corner]
            n=int(np.flatnonzero(b['jab_active'])[0]);assert n==row['matched_approach_controls']
            for field in ('qpos_before','qpos','qvel_before','qvel'):
                np.testing.assert_array_equal(a[field][:n],b[field][:n])
            np.testing.assert_array_equal(a['scored'].sum(axis=0),row['scores'])
            np.testing.assert_array_equal(a['action'][:,:,:3],0.)
            np.testing.assert_array_equal(a['action'][:,1-c],0.)
            np.testing.assert_array_equal(a['action'][:,c,10:],0.)
            assert np.max(np.abs(a['action']))<=1.
            mask=a['jab_active'].astype(bool)
            if mode=='motor_only':
                np.testing.assert_array_equal(a['action'],0.);assert not mask.any()
                reset=(row['seed'],row['swap'])
                if reset in baseline_states:
                    np.testing.assert_array_equal(a['qpos'],baseline_states[reset])
                else:baseline_states[reset]=a['qpos'].copy()
            else:
                assert mask.any() and np.flatnonzero(mask)[0]==n and mask[n:].all()
                np.testing.assert_array_equal(a['jab_fraction'],0.)
                np.testing.assert_array_equal(a['jab_phase_code'][mask],1)
                # The motor action scale/pose are unchanged; compare the original
                # measured seven-arm filtered target against its stored baseline.
                actual=a['jab_baseline_target'][mask]+.8*a['arm_residual'][mask,c,:7]
                np.testing.assert_allclose(a['jab_filtered_target'][mask],actual,atol=2e-7,rtol=1e-6)
                clips.extend(a['jab_action_clipped'][mask].ravel().tolist());tracking.extend(a['jab_target_error_m'][mask].tolist())
        native=root/'native'/Path(row['native_trace_file']).name;assert sha(native)==row['native_trace_sha256']
        with np.load(native) as a:
            assert all(np.isfinite(a[k]).all() for k in a.files)
            np.testing.assert_allclose(np.diff(a['time_before']),.002,atol=1e-12,rtol=0)
            assert a['time_after'][-1]==row['elapsed'] and (row['controls']-1)*10<len(a['time_after'])<=row['controls']*10
            assert describe_events(a)==row['events'] and int(a['event'].sum())==sum(row['scores'])
        summary['episodes']+=1;summary['pair_survivors']+=int(row['survived']);summary['contacts']+=int(sum(row['scores']))
        for corner in (0,1):
            flags=row['terminal']['corners'][corner];fell=any(flags[k] for k in ('low_pelvis','tilted','self_leg_contact'))
            summary['acting_falls' if corner==c else 'other_falls']+=int(fell)
        for k in summary['acting_first_fall_flags']:summary['acting_first_fall_flags'][k]+=int(row['terminal']['corners'][c][k])
        trials.append({k:row[k] for k in ('case','mode','corner','seed','swap','controls','elapsed','survived','scores','terminal')})
    matched=[]
    for c in (0,1):
        for seed in SEEDS:
            for swap in (False,True):
                rows={m:next(r for r in trials if (r['mode'],r['corner'],r['seed'],r['swap'])==(m,c,seed,swap)) for m in MODES}
                old=next(r for r in source['trials'] if (r['corner'],r['seed'],r['swap'])==(c,seed,swap))
                matched.append(dict(corner=c,seed=seed,swap=swap,motor_survived=rows['motor_only']['survived'],
                    guard_survived=rows['guard_only']['survived'],jab_v1_survived=old['survived'],
                    motor_elapsed=rows['motor_only']['elapsed'],guard_elapsed=rows['guard_only']['elapsed'],jab_v1_elapsed=old['elapsed']))
    analysis=dict(status='reviewed',modes=summaries,matched_cases=matched,trials=trials,
        new_physical_controls=count,formal_world_controls_total=BASE+count,diagnostic_ppo_total=2163968,
        cloned_forward_evaluations_total=4339,new_optimizer_updates=0,promotion=False,main_adoption=False,
        guard_action_clipped_fraction=float(np.mean(clips)),guard_tracking_error_m=dict(median=float(np.median(tracking)),maximum=float(max(tracking))),
        limitations=['Motor-only corner labels duplicate four unique reset conditions; they are not eight independent samples.',
            'Two seeds and ten-second scripted episodes do not certify learned punching or generalization.',
            'Historical jab-v1 cases are reused with exact matched approach prefixes, not newly optimized policies.'],
        decision='Review matched motor/guard/cycle patterns before target redesign; no automatic PPO, imitation, promotion or unchanged retry.')
    archive=Path('docs/results/striking-pilot/guard-control-v1');archive.mkdir(parents=True,exist_ok=True)
    for folder in ('full','native'):shutil.copytree(root/folder,archive/folder,dirs_exist_ok=True)
    for filename in ('audit.json','progress.json','native-errors.log'):shutil.copy2(root/filename,archive/filename)
    for folder in (archive,root):
        atomic_json(folder/'analysis.json',analysis);atomic_bytes(folder/'completed-before-review.json',(folder/'audit.json').read_bytes())
        atomic_json(folder/'audit.json',dict(state,phase='independently_reviewed',reviewed_utc=utc()))
    print(json.dumps({k:analysis[k] for k in ('status','modes','new_physical_controls','guard_action_clipped_fraction')}))


if __name__=='__main__':main()
