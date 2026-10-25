"""Audit compiled arm metadata and saved joint tracking without stepping physics."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
import mujoco
import numpy as np
from boxer.curriculum_env import UnassistedG1
from boxer.paired_foundation import build_pair
from boxer import scene
from tools.striking_pool_compare import verify_prerequisites


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ['JAX_PLATFORMS']=='cpu'
    prerequisites=verify_prerequisites()
    foundation=UnassistedG1('approach',impl='jax')
    model=build_pair(foundation)
    legacy=scene.build_model(2.,visual=False)
    metadata=[];maps=[]
    for side in scene.SIDES:
        ids=np.array([model.actuator(side+foundation.mj_model.actuator(i).name).id for i in range(29)])
        joints=model.actuator_trnid[ids,0];q=model.jnt_qposadr[joints];v=model.jnt_dofadr[joints]
        maps.append((ids,q,v))
        for i in range(29):
            name=foundation.mj_model.actuator(i).name
            if not any(word in name for word in ('shoulder','elbow','wrist')):continue
            a=int(ids[i]);old=legacy.actuator(side+name).id
            metadata.append(dict(side=side,name=name,index=i,qpos_index=int(q[i]),qvel_index=int(v[i]),
                position_gain=float(model.actuator_gainprm[a,0]),velocity_gain=float(-model.actuator_biasprm[a,2]),
                legacy_position_gain=float(legacy.actuator_gainprm[old,0]),
                ctrl_limited=bool(model.actuator_ctrllimited[a]),ctrl_range=model.actuator_ctrlrange[a].tolist(),
                force_limited=bool(model.actuator_forcelimited[a]),force_range=model.actuator_forcerange[a].tolist(),
                joint_range=model.jnt_range[joints[i]].tolist(),default_pose=float(foundation._default_pose[i])))
    studies=('terminal-dwell-v3','lateral-command-v5','guard-step-v7');trials=[];groups={}
    for study in studies:
        root=Path('docs/results/striking-pilot')/study;audit=json.loads((root/'audit.json').read_text())
        assert audit['phase']=='independently_reviewed' and audit['motor_sha256']==prerequisites['foundation']['approach']['sha256']
        for row in audit['trials']:
            path=root/'full'/Path(row['full_trace_file']).name;assert digest(path)==row['full_trace_sha256']
            c=row['corner'];ids,q,v=maps[c]
            indices=np.array([r['index'] for r in metadata if r['side']==scene.SIDES[c] and r['name'].startswith('left_')])
            assert len(indices)==7
            with np.load(path) as a:
                assert all(np.isfinite(a[k]).all() for k in a.files)
                mask=a['jab_active'].astype(bool);assert mask.any()
                requested=np.asarray(foundation._default_pose)[None,:]+float(foundation._config.action_scale)*a['applied_actions'][:,c]
                bounded=np.where(model.actuator_ctrllimited[ids],np.clip(requested,model.actuator_ctrlrange[ids,0],model.actuator_ctrlrange[ids,1]),requested)
                np.testing.assert_allclose(a['jab_filtered_target'][mask],requested[mask][:,indices],atol=2e-7,rtol=1e-6)
                records=[]
                for local,i in enumerate(indices):
                    name=foundation.mj_model.actuator(int(i)).name
                    error=bounded[mask,i]-a['qpos_before'][mask,q[i]]
                    next_error=bounded[mask,i]-a['qpos'][mask,q[i]]
                    requested_error=a['jab_desired_target'][mask,local]-a['qpos_before'][mask,q[i]]
                    entries=dict(name=name,median_abs_target_error_rad=float(np.median(abs(error))),
                        p95_abs_target_error_rad=float(np.quantile(abs(error),.95)),
                        median_abs_after_control_error_rad=float(np.median(abs(next_error))),
                        median_abs_jacobian_correction_rad=float(np.median(abs(requested_error))),
                        fraction_actuator_ctrl_clipped=float(np.mean(abs(bounded[mask,i]-requested[mask,i])>1e-9)),
                        median_abs_joint_speed=float(np.median(abs(a['qvel_before'][mask,v[i]]))))
                    records.append(entries);groups.setdefault((study,name),[]).append((error,next_error))
                trials.append(dict(study=study,case=row['case'],active_controls=int(mask.sum()),full_trace_sha256=digest(path),joints=records))
    summary=[]
    for (study,name),values in groups.items():
        errors=np.concatenate([x[0] for x in values]);after=np.concatenate([x[1] for x in values])
        summary.append(dict(study=study,name=name,active_joint_samples=len(errors),
            median_abs_target_error_rad=float(np.median(abs(errors))),p95_abs_target_error_rad=float(np.quantile(abs(errors),.95)),
            median_abs_after_control_error_rad=float(np.median(abs(after)))))
    assert verify_prerequisites()==prerequisites
    result=dict(status='reviewed_metadata_and_saved_arrays',prerequisites=prerequisites,
        new_physical_controls=0,new_cloned_forward_queries=0,new_surface_queries=0,new_optimizer_updates=0,
        model_inspection='Compiled models and joint tables only; no MjData, mj_forward, mj_kinematics, mj_step or search in this audit.',
        formal_world_controls_retained=2073802,diagnostic_ppo_retained=2163968,main_project_consumed_retained=476041216,
        cloned_queries_retained=6349,surface_queries_retained=960,metadata=metadata,trials=trials,summary=summary,
        limitations=['Target error does not isolate collision, actuator or feedback causes.',
            'Legacy and current action interfaces/scales are different contracts; gain differences do not alone prove the cause of failed punches.',
            'No learned training or physical feasibility test is performed.'])
    root=Path('docs/results/striking-pilot/arm-contract-v1');root.mkdir(parents=True,exist_ok=True)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(trials=len(trials),summary=summary)))


if __name__=='__main__':main()
