"""Review both lateral-command arms under the unchanged complete physical criteria."""
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
from tools.striking_jab_cycle_audit import candidate
from tools.striking_lateral_command import BASE,LIMIT,SEEDS,MODES,REFERENCE,AMPLITUDES,sha
from tools.striking_pool_compare import verify_prerequisites


def expected_phase(elapsed):
    # Independently reconstruct the protocol clock, including the full-target dwell.
    g=PROFILE['guard_seconds'];e=PROFILE['extend_seconds'];r=PROFILE['retract_seconds'];h=.12
    duration=sum(PROFILE[k] for k in ('guard_seconds','extend_seconds','retract_seconds','rest_seconds'))+h
    t=elapsed%duration
    if t<g:return 'guard',0.
    if t<g+e:return 'extend',float((1-np.cos(np.pi*(t-g)/e))/2)
    if t<g+e+h:return 'extend',1.
    if t<g+e+h+r:return 'retract',float((1+np.cos(np.pi*(t-g-e-h)/r))/2)
    return 'rest',0.


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();root=args.input;state=json.loads((root/'audit.json').read_text())
    assert state['status']=='completed' and state['phase']=='awaiting_interpretation' and state['controller_design_version']==5 and state['terminal_dwell_seconds']==.12
    assert state['profile']==PROFILE and state['formal_world_controls_before_job']==BASE
    assert state['maximum_world_controls']==LIMIT and state['formal_world_controls_ceiling']==BASE+LIMIT
    assert verify_prerequisites()==state['prerequisites']
    assert sha(Path('docs/results/striking-pilot/arc-path-v4/analysis.json'))==state['source_accounting_review_sha256']
    assert sha(REFERENCE/'audit.json')==state['source_audit_sha256']
    assert sha(REFERENCE/'analysis.json')==state['source_review_sha256']
    assert state['new_ppo_steps']==state['new_supervised_updates']==state['new_optimizer_updates']==0
    assert state['diagnostic_ppo_retained']==2163968 and state['cloned_forward_evaluations_retained']==6349 and state['surface_distance_queries_retained']==960
    assert state['main_project_consumed_retained']==476041216 and state['main_boxing_retained']==1048576
    assert not state['assistance'] and not state['learned_controller'] and not state['promotion'] and state['policy_adoption']=='none'
    assert not (root/'native-errors.log').read_text().strip()
    expected={(m,c,s,w) for m in MODES for c in (0,1) for s in SEEDS for w in (False,True)}
    assert len(state['trials'])==16 and {(r['mode'],r['corner'],r['seed'],r['swap']) for r in state['trials']}==expected
    count=sum(r['controls'] for r in state['trials'])
    assert count==state['diagnostic_world_controls']==json.loads((root/'progress.json').read_text())['diagnostic_world_controls']<=LIMIT
    assert state['formal_world_controls_total']==BASE+count
    source=json.loads((REFERENCE/'audit.json').read_text());summaries={};trials=[]
    for mode in MODES:
        summaries[mode]=dict(episodes=0,pair_survivors=0,acting_falls=0,other_falls=0,contacts=0,
            acting_first_fall_flags={k:0 for k in ('low_pelvis','tilted','self_leg_contact')})
    clips=[];tracking=[]
    for row in state['trials']:
        c=row['corner'];key=(c,row['seed'],row['swap']);mode=row['mode'];summary=summaries[mode]
        reference=next(r for r in source['trials'] if r['mode']=='terminal_dwell' and (r['corner'],r['seed'],r['swap'])==key)
        original=REFERENCE/'full'/Path(reference['full_trace_file']).name
        assert sha(original)==reference['full_trace_sha256']==row['reference_full_trace_sha256']
        path=root/'full'/Path(row['full_trace_file']).name;assert sha(path)==row['full_trace_sha256']
        assert row['lateral_action']==AMPLITUDES[mode]
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
            n=int(np.flatnonzero(b['cycle_phase'])[0]);assert n==row['matched_prefix_controls']
            assert row['matched_prefix_kind']=='first_lateral_guard'
            activation=int(np.flatnonzero(b['cycle_phase'])[0])
            for field in ('qpos_before','qpos','qvel_before','qvel'):
                np.testing.assert_array_equal(a[field][:n],b[field][:n])
            np.testing.assert_array_equal(a['scored'].sum(axis=0),row['scores'])
            np.testing.assert_array_equal(a['action'][:,:,0],0.)
            np.testing.assert_array_equal(a['action'][:,:,2],0.)
            expected_commands=np.clip(a['base_commands']+a['action'][:,:,:3]*np.array([.15,.15,.25]),[-.35,-.25,-.5],[.55,.25,.5])
            np.testing.assert_array_equal(a['commands'],expected_commands)
            np.testing.assert_array_equal(a['action'][:,1-c],0.)
            np.testing.assert_array_equal(a['action'][:,c,10:],0.)
            assert np.max(np.abs(a['action']))<=1.
            mask=a['jab_active'].astype(bool)
            clock=a['cycle_elapsed'];codes=a['cycle_phase'];expected_codes=np.zeros(len(clock),int);fractions=np.zeros(len(clock))
            np.testing.assert_allclose(clock[activation:],a['time_before'][activation:]-a['time_before'][activation],atol=1e-12,rtol=0)
            for i in range(activation,len(clock)):
                phase,fraction=expected_phase(float(clock[i]))
                expected_codes[i]={'guard':1,'extend':2,'retract':3,'rest':4}[phase];fractions[i]=fraction
            np.testing.assert_array_equal(codes,expected_codes);np.testing.assert_array_equal(mask,codes==2)
            np.testing.assert_array_equal(a['action'][codes!=2,:,3:],0.)
            expected_lateral=np.where((codes==1)|(codes==2),AMPLITUDES[mode],0.)
            np.testing.assert_array_equal(a['action'][:,c,1],expected_lateral)
            np.testing.assert_allclose(a['jab_fraction'][mask],fractions[mask],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(a['jab_phase_code'][mask],2)
            np.testing.assert_array_equal(a['lateral_action'],AMPLITUDES[mode])
            f=a['jab_fraction'][mask]
            target=a['jab_guard_world'][mask]+f[:,None]*(a['jab_strike_world'][mask]-a['jab_guard_world'][mask])
            np.testing.assert_allclose(a['jab_target_world'][mask],target,atol=1e-12,rtol=0)
            actual=a['jab_baseline_target'][mask]+.8*a['arm_residual'][mask,c,:7]
            np.testing.assert_allclose(a['jab_filtered_target'][mask],actual,atol=2e-7,rtol=1e-6)
            duration=sum(PROFILE[k] for k in ('guard_seconds','extend_seconds','retract_seconds','rest_seconds'))+.12
            hold=(codes==2)&(np.mod(clock,duration)>=PROFILE['guard_seconds']+PROFILE['extend_seconds'])
            np.testing.assert_array_equal(a['terminal_dwell'],hold)
            reference_native=root/'native'/Path(row['native_trace_file']).name
            seen=set()
            with np.load(reference_native) as raw:
                for i in np.flatnonzero(mask):
                    cycle=int(clock[i]//duration)
                    if cycle not in seen:
                        entry=raw['local_position'][i*10,c,0].copy();seen.add(cycle)
                    np.testing.assert_allclose(a['cycle_entry_local'][i],entry,atol=1e-12,rtol=0)
                    world=a['torso_origin_before'][i]+a['torso_rotation_before'][i]@entry
                    np.testing.assert_allclose(a['jab_guard_world'][i],world,atol=1e-12,rtol=0)
            clips.extend(a['jab_action_clipped'][mask].ravel().tolist());tracking.extend(a['jab_target_error_m'][mask].tolist())
        native=root/'native'/Path(row['native_trace_file']).name;assert sha(native)==row['native_trace_sha256']
        with np.load(native) as a:
            assert all(np.isfinite(a[k]).all() for k in a.files)
            np.testing.assert_allclose(np.diff(a['time_before']),.002,atol=1e-12,rtol=0)
            assert a['time_after'][-1]==row['elapsed'] and (row['controls']-1)*10<len(a['time_after'])<=row['controls']*10
            measured=[dict(e,jab_cycle_candidate=candidate(e,c)) for e in describe_events(a)]
            assert measured==row['events'] and int(a['event'].sum())==sum(row['scores'])
            candidates=sum(e['jab_cycle_candidate'] for e in measured)
            assert candidates==row['candidate_events']
            assert row['feasible_episode']==(row['survived'] and row['controls']==500 and candidates>=2)
        summary['episodes']+=1;summary['pair_survivors']+=int(row['survived']);summary['contacts']+=int(sum(row['scores']))
        for corner in (0,1):
            flags=row['terminal']['corners'][corner];fell=any(flags[k] for k in ('low_pelvis','tilted','self_leg_contact'))
            summary['acting_falls' if corner==c else 'other_falls']+=int(fell)
        for k in summary['acting_first_fall_flags']:summary['acting_first_fall_flags'][k]+=int(row['terminal']['corners'][c][k])
        trials.append({k:row[k] for k in ('case','mode','lateral_action','corner','seed','swap','controls','elapsed','survived','scores','terminal','candidate_events','feasible_episode')})
    historical=json.loads((REFERENCE/'analysis.json').read_text());matched=[]
    for row in trials:
        prior=next(r for r in historical['matched_cases'] if (r['corner'],r['seed'],r['swap'])==(row['corner'],row['seed'],row['swap']))
        matched.append(dict(prior,mode=row['mode'],lateral_action=row['lateral_action'],lateral_survived=row['survived'],lateral_elapsed=row['elapsed'],candidate_events=row['candidate_events'],feasible_episode=row['feasible_episode']))
    analysis=dict(status='reviewed',candidate_events=sum(r['candidate_events'] for r in trials),feasible_episodes=sum(r['feasible_episode'] for r in trials),modes=summaries,matched_cases=matched,trials=trials,
        new_physical_controls=count,formal_world_controls_total=BASE+count,diagnostic_ppo_total=2163968,
        cloned_forward_evaluations_total=6349,surface_distance_queries_total=960,new_optimizer_updates=0,promotion=False,main_adoption=False,
        extension_action_clipped_fraction=float(np.mean(clips)),extension_tracking_error_m=dict(median=float(np.median(tracking)),maximum=float(max(tracking))),
        limitations=['Motor-only corner labels duplicate four unique reset conditions; they are not eight independent samples.',
            'Two seeds and ten-second scripted episodes do not certify learned punching or generalization.',
            'Historical guard/cycle cases are reused with exact approach prefixes; this remains a scripted diagnostic, not a learned boxer.'],
        decision='Interpret both complete lateral command arms against the historical straight control; no automatic training or positive-demo adoption.')
    archive=Path('docs/results/striking-pilot/lateral-command-v5');archive.mkdir(parents=True,exist_ok=True)
    for folder in ('full','native'):shutil.copytree(root/folder,archive/folder,dirs_exist_ok=True)
    for filename in ('audit.json','progress.json','native-errors.log'):shutil.copy2(root/filename,archive/filename)
    for folder in (archive,root):
        atomic_json(folder/'analysis.json',analysis);atomic_bytes(folder/'completed-before-review.json',(folder/'audit.json').read_bytes())
        atomic_json(folder/'audit.json',dict(state,phase='independently_reviewed',reviewed_utc=utc()))
    print(json.dumps({k:analysis[k] for k in ('status','modes','new_physical_controls','extension_action_clipped_fraction')}))


if __name__=='__main__':main()
