"""Read saved v14 motion/reward/exploration evidence; no new physics or learning."""
import hashlib,json,os,pickle
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np


def main():
    root=Path('docs/results/striking-pilot/motion-cycle-v14');rows=[]
    for label in ('baseline','step-524288','step-1048576','fresh-final'):
        folder=root/'gates'/label;vel=[];contact=[];distance=[];dense_total=0.;components={};cases=[]
        for role in ('neutral-0','neutral-1','selfplay'):
            report=json.loads((folder/(role+'.json')).read_text())
            corners=[0,1] if role=='selfplay' else [int(role[-1])]
            for trial in report['trials']:
                full=folder/role/'full'/Path(trial['trace_file']).name
                native=folder/role/'native'/Path(trial['punch_trace_file']).name
                assert hashlib.sha256(full.read_bytes()).hexdigest()==trial['trace_sha256']
                assert hashlib.sha256(native.read_bytes()).hexdigest()==trial['punch_trace_sha256']
                with np.load(full) as archive:a={k:archive[k] for k in archive.files}
                with np.load(native) as archive:b={k:archive[k] for k in archive.files}
                assert all(np.isfinite(v).all() for v in (*a.values(),*b.values()))
                late=a['time_after']>=3.;native_late=b['time_before']>=3.
                if late.any():distance.extend(a['hand_distance'][late][:,corners].ravel())
                vel.extend(np.linalg.norm(b['local_velocity'][native_late][:,corners],axis=-1).ravel())
                contact.extend(b['present'][native_late][:,corners].ravel())
                v=b['local_velocity'][:,corners];speed=np.linalg.norm(v,axis=-1)
                quality=np.clip(v[...,0],0,1)*np.clip(b['toward_nearest_target'][:,corners]/np.maximum(speed,1e-9),0,1)
                quality=np.max(quality,axis=2)
                group=np.arange(len(quality))//10
                for c in range(len(corners)):
                    q=np.bincount(group,weights=quality[:,c])/np.bincount(group)
                    dense_total+=float((.5*.02*q).sum())
                for name in ('facing','engagement','progress','fall','residual'):
                    components[name]=components.get(name,0.)+float(a['reward_'+name][:,corners].sum())
                cases.append(dict(role=role,seed=trial['seed'],swap=trial['swap'],late_controls=int(late.sum()),
                    full_sha256=trial['trace_sha256'],native_sha256=trial['punch_trace_sha256']))
        checkpoint=Path('checkpoints/striking-validation/motion-cycle-v14')/label
        manifest=json.loads((checkpoint/'latest.json').read_text());payload=(checkpoint/manifest['file']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
        saved=pickle.loads(payload);std=np.exp(np.clip(saved['actor']['params']['log_std'],-3.,-.5))
        rows.append(dict(label=label,late_window_seconds=3.,native_hand_speed_quantiles=np.quantile(vel,[.5,.95,1.]).tolist(),
            late_legal_contact_substep_fraction=float(np.mean(contact)),late_min_hand_target_center_distance_mean=float(np.mean(distance)),
            replayed_forward_preparation_reward=dense_total,base_reward_component_sums=components,
            saved_policy_std=std.tolist(),cases=cases))
    result=dict(status='reviewed_saved_arrays',new_physical_controls=0,new_optimizer_updates=0,new_geometry_queries=0,profiles=rows,
        observation='V14 forward preparation can pay before a hand reaches a target. Mean speed and target-center distance increase while legal contacts decrease on these saved evaluations.',
        limitation='Associations do not isolate a unique causal mechanism. Policy standard deviations are saved distribution parameters, not proof of effective exploration.',
        decision='Test explicit target-distance potential and graded same-hand physical contact progress, preserving complete valid-cycle acceptance and all physical gates.')
    out=Path('docs/results/striking-pilot/contact-progress-v15');out.mkdir(parents=True,exist_ok=True)
    (out/'proposal-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('label','native_hand_speed_quantiles','late_legal_contact_substep_fraction','late_min_hand_target_center_distance_mean','replayed_forward_preparation_reward')} for r in rows]))


if __name__=='__main__':main()
