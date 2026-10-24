"""Review every smooth-entry guard case without additional physical compute."""
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
from tools.striking_guard_ramp import BASE,LIMIT,SEEDS,MODES,REFERENCE,sha
from tools.striking_pool_compare import verify_prerequisites


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();root=args.input;state=json.loads((root/'audit.json').read_text())
    recovery=state['status']=='failed'
    if recovery:
        assert state['phase']=='smooth_guard_entry' and state['error']=='' and len(state['trials'])==8
        host_error=Path('runs/striking_guard_ramp_host_error.log').read_text()
        assert "assert len(state['trials'])==16 and verify_prerequisites()==prerequisites" in host_error
        assert host_error.rstrip().endswith('AssertionError')
    else:assert state['status']=='completed' and state['phase']=='awaiting_interpretation'
    assert state['entry_ramp_seconds']==2.
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
    assert len(state['trials'])==8 and {(r['mode'],r['corner'],r['seed'],r['swap']) for r in state['trials']}==expected
    count=sum(r['controls'] for r in state['trials'])
    assert count==state['diagnostic_world_controls']==json.loads((root/'progress.json').read_text())['diagnostic_world_controls']<=LIMIT
    if not recovery:assert state['formal_world_controls_total']==BASE+count
    source=json.loads((REFERENCE/'audit.json').read_text());summaries={};trials=[];baseline_states={}
    for mode in MODES:
        summaries[mode]=dict(episodes=0,pair_survivors=0,acting_falls=0,other_falls=0,contacts=0,
            acting_first_fall_flags={k:0 for k in ('low_pelvis','tilted','self_leg_contact')})
    clips=[];tracking=[]
    for row in state['trials']:
        c=row['corner'];key=(c,row['seed'],row['swap']);mode=row['mode'];summary=summaries[mode]
        reference=next(r for r in source['trials'] if r['mode']=='guard_only' and (r['corner'],r['seed'],r['swap'])==key)
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
                entry=a['ramp_entry_local'][n].copy()
                np.testing.assert_array_equal(a['ramp_entry_local'][mask],np.broadcast_to(entry,(mask.sum(),3)))
                clock=a['time_before'][mask]-a['time_before'][n]
                np.testing.assert_allclose(a['ramp_elapsed'][mask],clock,atol=1e-12,rtol=0)
                t=np.clip(clock/2.,0.,1.);np.testing.assert_allclose(a['ramp_fraction'][mask],t,atol=1e-12,rtol=0)
                blend=t*t*(3.-2.*t)
                local=entry+blend[:,None]*(np.asarray(PROFILE['guard_local'])-entry)
                world=a['torso_origin_before'][mask]+np.einsum('nij,nj->ni',a['torso_rotation_before'][mask],local)
                np.testing.assert_allclose(a['jab_guard_world'][mask],world,atol=1e-12,rtol=0)
                held_seconds=max(0.,row['elapsed']-float(a['time_before'][n])-2.)
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
            np.testing.assert_allclose(entry,a['local_position'][n*10,c,0],atol=1e-12,rtol=0)
            assert describe_events(a)==row['events'] and int(a['event'].sum())==sum(row['scores'])
        summary['episodes']+=1;summary['pair_survivors']+=int(row['survived']);summary['contacts']+=int(sum(row['scores']))
        for corner in (0,1):
            flags=row['terminal']['corners'][corner];fell=any(flags[k] for k in ('low_pelvis','tilted','self_leg_contact'))
            summary['acting_falls' if corner==c else 'other_falls']+=int(fell)
        for k in summary['acting_first_fall_flags']:summary['acting_first_fall_flags'][k]+=int(row['terminal']['corners'][c][k])
        trials.append(dict({k:row[k] for k in ('case','mode','corner','seed','swap','controls','elapsed','survived','scores','terminal')},held_final_guard_seconds=held_seconds))
    historical=json.loads((REFERENCE/'analysis.json').read_text())
    matched=[]
    for row in trials:
        prior=next(r for r in historical['matched_cases'] if (r['corner'],r['seed'],r['swap'])==(row['corner'],row['seed'],row['swap']))
        matched.append(dict(prior,ramp_survived=row['survived'],ramp_elapsed=row['elapsed'],held_final_guard_seconds=row['held_final_guard_seconds']))
    analysis=dict(status='reviewed',modes=summaries,matched_cases=matched,trials=trials,
        new_physical_controls=count,formal_world_controls_total=BASE+count,diagnostic_ppo_total=2163968,
        cloned_forward_evaluations_total=4339,new_optimizer_updates=0,promotion=False,main_adoption=False,
        guard_action_clipped_fraction=float(np.mean(clips)),guard_tracking_error_m=dict(median=float(np.median(tracking)),maximum=float(max(tracking))),
        limitations=['Motor-only corner labels duplicate four unique reset conditions; they are not eight independent samples.',
            'Two seeds and ten-second scripted episodes do not certify learned punching or generalization.',
            'Historical motor/static-guard/jab cases are reused; ramp approach prefixes match exactly without new optimization.'],
        decision='Review matched ramp/static-guard outcomes and actual final-guard hold duration before another design or training allocation.')
    archive=Path('docs/results/striking-pilot/guard-ramp-v1');archive.mkdir(parents=True,exist_ok=True)
    for folder in ('full','native'):shutil.copytree(root/folder,archive/folder,dirs_exist_ok=True)
    for filename in ('audit.json','progress.json','native-errors.log'):shutil.copy2(root/filename,archive/filename)
    if recovery:
        shutil.copy2(Path('runs/striking_guard_ramp_host_error.log'),archive/'completion-error.log')
        for folder in (archive,root):atomic_bytes(folder/'failed-before-completion-recovery.json',(folder/'audit.json').read_bytes())
        analysis['completion_recovery']=dict(reason='Final assertion expected 16 trials although the ramp protocol configured eight.',
            new_physical_controls=0,new_optimizer_updates=0,all_eight_trials_independently_checked=True)
    for folder in (archive,root):
        atomic_json(folder/'analysis.json',analysis);atomic_bytes(folder/'completed-before-review.json',(folder/'audit.json').read_bytes())
        atomic_json(folder/'audit.json',dict(state,status='completed',phase='independently_reviewed',reviewed_utc=utc(),
            formal_world_controls_total=BASE+count,completion_recovery=analysis.get('completion_recovery')))
    print(json.dumps({k:analysis[k] for k in ('status','modes','new_physical_controls','guard_action_clipped_fraction')}))


if __name__=='__main__':main()
