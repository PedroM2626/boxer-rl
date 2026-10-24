"""Verify root/leg failure evidence and cloned FK results without extra physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU workspace review required')
import mujoco
import numpy as np
from boxer.checkpoint_io import atomic_json,atomic_bytes
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import utc
from boxer.paired_foundation import PairedFoundation
from tools.striking_jab_failure_workspace import BASE,LIMIT,GEOMETRY_LIMIT,REFERENCE
from tools.striking_pool_compare import verify_prerequisites


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();root=args.input;state=json.loads((root/'audit.json').read_text())
    if (root/'review-progress.json').exists():raise ValueError('review exists; retain cloned evaluation accounting before recovery')
    assert state['status']=='completed' and state['diagnostic_world_controls']==LIMIT
    assert state['formal_world_controls_before_job']==BASE and state['maximum_world_controls']==LIMIT
    assert not (root/'native-errors.log').read_text().strip()
    assert verify_prerequisites()==state['prerequisites'] and len(state['trials'])==16
    assert hashlib.sha256((REFERENCE/'audit.json').read_bytes()).hexdigest()==state['reference_audit_sha256']
    assert hashlib.sha256((REFERENCE/'analysis.json').read_bytes()).hexdigest()==state['reference_review_sha256']
    source=json.loads((REFERENCE/'audit.json').read_text());snapshots=json.loads((root/'snapshots.json').read_text())
    assert len(snapshots)==state['geometry_snapshot_count']<=32 and len(state['workspace'])==2*len(snapshots)
    assert sum(w['cloned_forward_evaluations'] for w in state['workspace'])==state['cloned_forward_evaluations']<=GEOMETRY_LIMIT
    causes={side:{kind:0 for kind in ('low_pelvis','tilted','self_leg_contact')} for side in ('acting','other')}
    clip_count=active_values=outside_count=0;errors=[];phase_errors={};trials=[]
    foundation=UnassistedG1('approach',impl='jax');pair=PairedFoundation(foundation)
    arm=np.array([i for i in range(29) if 'left_' in foundation.mj_model.actuator(i).name
        and any(s in foundation.mj_model.actuator(i).name for s in ('shoulder','elbow','wrist'))])
    assert len(arm)==7
    full_files={}
    for row in state['trials']:
        expected=next(r for r in source['trials'] if (r['corner'],r['seed'],r['swap'])==(row['corner'],row['seed'],row['swap']))
        assert row['elapsed']==expected['elapsed'] and row['controls']==expected['controls'] and row['scores']==expected['scored_by_corner']
        assert row['all_original_native_arrays_equal'] and row['survived']==expected['survived']
        path=root/'full'/Path(row['full_trace_file']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['full_trace_sha256']
        with np.load(path) as a:
            assert len(a['qpos'])==row['controls'] and all(np.isfinite(a[k]).all() for k in a.files)
            np.testing.assert_array_equal(a['qpos_before'][1:],a['qpos'][:-1])
            np.testing.assert_array_equal(a['qvel_before'][1:],a['qvel'][:-1])
            np.testing.assert_array_equal(a['external_force_norm'],0.)
            for c in (0,1):
                flags=row['terminal']['corners'][c]
                assert bool(a['fallen'][-1,c])==any(flags[k] for k in ('low_pelvis','tilted','self_leg_contact'))
                assert flags['pelvis_height']==a['pelvis_height'][-1,c] and flags['upright']==a['upright'][-1,c]
                side='acting' if c==row['corner'] else 'other'
                for k in causes[side]:causes[side][k]+=int(flags[k])
            mask=a['jab_active'].astype(bool);c=row['corner']
            clip_count+=int(a['jab_action_clipped'][mask].sum());active_values+=int(mask.sum())*7
            outside_count+=int((np.abs(a['jab_unbounded_request'][mask])>1).sum())
            residual_error=np.abs(a['jab_filtered_target'][mask]-a['jab_desired_target'][mask])
            errors.extend(residual_error.ravel().tolist())
            actual=pair.pose[arm]+pair.action_scale*a['applied_actions'][mask,c][:,arm]
            np.testing.assert_allclose(a['jab_filtered_target'][mask],actual,atol=2e-7,rtol=1e-6)
            for phase in (1,2,3,4):
                values=a['jab_target_error_m'][a['jab_phase_code']==phase]
                phase_errors.setdefault(str(phase),[]).extend(values.tolist())
        full_files[row['case']]=path
        trials.append(dict(case=row['case'],terminal=row['terminal'],elapsed=row['elapsed'],controls=row['controls']))
    # Check each saved best pose with one fresh cloned FK call. No solver rerun,
    # mj_step, live root/leg constraint or unrecorded physical trial is involved.
    review_geometry=0
    for w in state['workspace']:
        snapshot=next(s for s in snapshots if s['case']==w['case'] and s['kind']==w['snapshot_kind'])
        assert w['control_index']==snapshot['control_index']
        with np.load(full_files[w['case']]) as a:
            i=w['control_index']
            np.testing.assert_array_equal(a['qpos_before'][i],snapshot['qpos'])
            np.testing.assert_array_equal(a['jab_baseline_target'][i],snapshot['baseline'])
            np.testing.assert_array_equal(a['jab_'+w['target_kind']+'_world'][i],w['target_world'])
        x=np.asarray(w['best_targets']);lower=np.asarray(w['lower']);upper=np.asarray(w['upper'])
        assert np.all(x>=lower-1e-12) and np.all(x<=upper+1e-12)
        c=snapshot['corner'];q=pair.maps[c]['q'][arm];joints=pair.model.actuator_trnid[pair.maps[c]['act'][arm],0]
        np.testing.assert_array_equal(w['joint_coordinates'],q)
        np.testing.assert_array_equal(lower,np.maximum(np.asarray(snapshot['baseline'])-.8,pair.model.jnt_range[joints,0]))
        np.testing.assert_array_equal(upper,np.minimum(np.asarray(snapshot['baseline'])+.8,pair.model.jnt_range[joints,1]))
        assert not w['global_infeasibility_proven'] and not w['dynamic_feasibility_proven'] and w['physical_controls']==0
        review_geometry+=1
        assert state['cloned_forward_evaluations']+review_geometry<=GEOMETRY_LIMIT
        atomic_json(root/'review-progress.json',dict(cloned_forward_evaluations=review_geometry,new_physical_controls=0,updated_utc=utc()))
        clone=mujoco.MjData(pair.model);clone.qpos[:]=snapshot['qpos'];clone.qpos[q]=x
        mujoco.mj_kinematics(pair.model,clone);mujoco.mj_comPos(pair.model,clone)
        pos=clone.geom_xpos[pair.fists[c,0]]
        np.testing.assert_allclose(pos,w['best_fist_world'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.linalg.norm(pos-np.asarray(w['target_world'])),w['best_error_m'],atol=1e-12,rtol=0)
    geometry={}
    for pose in ('guard','extend_mid'):
        for target in ('guard','strike'):
            e=np.array([w['best_error_m'] for w in state['workspace'] if w['snapshot_kind']==pose and w['target_kind']==target])
            geometry[pose+'/'+target]=dict(count=len(e),minimum_m=float(e.min()),median_m=float(np.median(e)),maximum_m=float(e.max()))
    result=dict(status='reviewed',episodes=16,all_original_native_arrays_reproduced=True,native_errors_empty=True,
        new_physical_controls=LIMIT,formal_world_controls_total=BASE+LIMIT,diagnostic_ppo_total=2163968,
        cloned_forward_evaluations_producer=state['cloned_forward_evaluations'],cloned_forward_evaluations_review=review_geometry,
        cloned_forward_evaluations_total=state['cloned_forward_evaluations']+review_geometry,new_optimizer_updates=0,
        main_adoption=False,promotion=False,first_fall_flags=causes,
        active_arm_channel_values=active_values,action_clipped_fraction=clip_count/max(active_values,1),
        desired_outside_static_target_envelope_fraction=outside_count/max(active_values,1),
        absolute_filtered_target_request_error_rad=dict(median=float(np.median(errors)),maximum=float(max(errors))),
        phase_tracking_error_m={k:dict(median=float(np.median(v)),maximum=float(max(v))) for k,v in phase_errors.items() if v},
        static_geometry=geometry,trials=trials,
        decision='Do not repeat v1 or train on it. Guard endpoints are geometrically matched in the sampled target boxes; early full-extension endpoints are not matched by this local search. Only the acting body triggers first-fall predicates. Isolate static guard versus cycle dynamics with a prospective matched control before assigning causality or redesigning the target.',
        limitations=['Two local static solver starts do not prove global or dynamic reachability/infeasibility.',
            'Full-extension geometry at early guard/mid-extension snapshots is not the commanded interpolated target at every instant.',
            'Static geometry ignores collision safety, balance, moving motor baselines and velocity/torque limits.',
            'First-fall flags and saturation are observations, not a causal isolation of every failure.'])
    archive=Path('docs/results/striking-pilot/jab-failure-workspace-v1');archive.mkdir(parents=True,exist_ok=True)
    shutil.copytree(root/'full',archive/'full',dirs_exist_ok=True)
    for filename in ('audit.json','progress.json','snapshots.json','review-progress.json','native-errors.log'):shutil.copy2(root/filename,archive/filename)
    atomic_json(archive/'analysis.json',result);atomic_json(root/'analysis.json',result)
    for folder in (archive,root):
        atomic_bytes(folder/'completed-before-review.json',(folder/'audit.json').read_bytes())
        atomic_json(folder/'audit.json',dict(state,phase='independently_reviewed',reviewed_utc=utc(),
            cloned_forward_evaluations_review=review_geometry,continuation_decision=result['decision']))
    print(json.dumps({k:result[k] for k in ('status','first_fall_flags','action_clipped_fraction','formal_world_controls_total')}))


if __name__=='__main__':main()
