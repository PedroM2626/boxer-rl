"""Arm reach and frozen-motor opposition diagnostics, never policy training.

Static candidates use forward kinematics only; the separate constant-command
trials keep both roots free and terminate at the first fall. No scripted strike
cycle, pose playback, or assistance enters the training environment.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params, utc
from boxer.striking_env import StrikingEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/striking-diagnosis/reach.json'))
    parser.add_argument('--samples',type=int,default=4000)
    args=parser.parse_args()
    if args.samples<1:raise ValueError('positive sample count required')
    foundation=UnassistedG1('approach',impl='jax')
    params,source=load_policy_params(Path('checkpoints/curriculum/approach'))
    env=StrikingEnv(foundation,params)
    env.reset(seed=31)
    transitions=0
    for _ in range(150):
        _,_,done,outcome=env.step(np.zeros((2,17)));transitions+=1
        if done:raise RuntimeError('neutral settling fell; reject reach setup')
    pair=env.pair;model=pair.model
    candidate_data=mujoco.MjData(model)
    candidate_data.qpos[:]=pair.data.qpos
    baseline_qpos=pair.data.qpos.copy()
    baseline_hand_distance=env.hand_distances().tolist()
    roots=[baseline_qpos[m['root']:m['root']+2] for m in pair.maps]
    baseline_root_distance=float(np.linalg.norm(roots[1]-roots[0]))
    actuators=pair.maps[0]['act'][env.arm_indices]
    joints=pair.maps[0]['q'][env.arm_indices]
    base_target=pair.data.ctrl[actuators].copy()
    low,high=model.actuator_ctrlrange[actuators].T
    rng=np.random.default_rng(67)
    variants={}
    for mode in ('filtered_initial_noise','full_residual_box'):
        distances=[];overlaps=0;best=None;best_distance=float('inf')
        for _ in range(args.samples):
            action=(np.tanh(rng.normal(0.,np.exp(-1.9)*np.sqrt(.2/1.8),14))
                    if mode=='filtered_initial_noise' else rng.uniform(-1,1,14))
            candidate_data.qpos[:]=baseline_qpos
            candidate_data.qpos[joints]=np.clip(base_target+.8*action,low,high)
            mujoco.mj_forward(model,candidate_data)
            distance=min(np.linalg.norm(candidate_data.geom_xpos[f]-candidate_data.geom_xpos[t])
                         for f in pair.fists[0] for t in pair.targets[1])
            overlap=any(any(int(f) in c.geom and any(int(t) in c.geom for t in pair.targets[1])
                            for f in pair.fists[0]) for c in candidate_data.contact)
            distances.append(distance);overlaps+=int(overlap)
            if distance<best_distance:best_distance=distance;best=action.copy()
        variants[mode]=dict(samples=args.samples,minimum_centre_distance=float(min(distances)),
            median_centre_distance=float(np.median(distances)),legal_surface_overlap_candidates=overlaps,
            best_residual_action=best.tolist(),kinematic_only=True)
        print(mode,json.dumps(variants[mode]),flush=True)
    # Hold the same settled pose and common residual samples while varying only
    # root separation in forward kinematics. These are workspace tests, not a
    # controller holding roots or a physical trial with teleported opponents.
    sweep=[]
    common=rng.uniform(-1,1,(args.samples,14))
    direction=(roots[1]-roots[0])/baseline_root_distance
    blue_root=pair.maps[1]['root']
    for separation in (.4,.45,.5,.55,.6):
        overlaps=0;minimum=float('inf')
        for action in common:
            candidate_data.qpos[:]=baseline_qpos
            candidate_data.qpos[blue_root:blue_root+2] += direction*(separation-baseline_root_distance)
            candidate_data.qpos[joints]=np.clip(base_target+.8*action,low,high)
            mujoco.mj_forward(model,candidate_data)
            distance=min(np.linalg.norm(candidate_data.geom_xpos[f]-candidate_data.geom_xpos[t])
                         for f in pair.fists[0] for t in pair.targets[1])
            minimum=min(minimum,distance)
            overlaps+=int(any(any(int(f) in c.geom and any(int(t) in c.geom for t in pair.targets[1])
                                 for f in pair.fists[0]) for c in candidate_data.contact))
        result=dict(root_separation=separation,samples=args.samples,
                    legal_surface_overlap_candidates=overlaps,minimum_centre_distance=float(minimum))
        sweep.append(result);print('range',json.dumps(result),flush=True)
    probes=[]
    left_elbow=np.zeros(14);left_elbow[3]=1.
    for name,target in [('neutral',np.zeros(14)),('half_left_elbow',left_elbow*.5),
                        ('full_left_elbow',left_elbow),
                        ('best_kinematic',np.array(variants['full_residual_box']['best_residual_action']))]:
        env.reset(seed=31)
        baseline=[]
        for _ in range(150):
            _,_,done,outcome=env.step(np.zeros((2,17)));transitions+=1
            baseline.append(outcome['applied_actions'][0,env.arm_indices]*.5)
            if done:raise RuntimeError('controlled settling mismatch')
        baseline=np.mean(baseline[-25:],axis=0)
        commands=np.zeros((2,17));commands[0,3:]=target
        records=[];hits=np.zeros(2);raw=np.zeros(2)
        for step in range(150):
            _,_,done,outcome=env.step(commands);transitions+=1
            requested=.8*outcome['arm_residual'][0]
            effective=.5*outcome['applied_actions'][0,env.arm_indices]
            base=effective-requested
            observed=pair.data.qpos[pair.maps[0]['q'][env.arm_indices]]-pair.pose[env.arm_indices]
            records.append(dict(requested=requested.copy(),base=base.copy(),effective=effective.copy(),observed=observed.copy()))
            hits+=outcome['scored'];raw+=outcome['legal_contact_impulse']
            if done:break
        steady=records[-min(25,len(records)):]
        means={k:np.mean([r[k] for r in steady],axis=0) for k in records[0]}
        opposition=-float((means['base']-baseline)@means['requested'])/max(float(means['requested']@means['requested']),1e-9)
        result=dict(name=name,survived=not env.pair.done,held_control_steps=len(records),
            final_hand_distances=env.hand_distances().tolist(),scored=hits.tolist(),raw_legal_impulses=raw.tolist(),
            motor_opposition_projection=opposition,
            mean_requested_radians=means['requested'].tolist(),
            mean_frozen_motor_radians=means['base'].tolist(),
            mean_effective_target_radians=means['effective'].tolist(),
            mean_observed_joint_offset=means['observed'].tolist())
        probes.append(result);print('probe',json.dumps(result),flush=True)
    atomic_json(args.output,dict(kind='striking_reach_diagnostic',updated_utc=utc(),
        foundation_sha256=source['sha256'],assistance=False,optimization_transitions=0,
        simulated_world_transitions=transitions,forward_kinematic_samples=args.samples*7,
        baseline_root_separation=baseline_root_distance,separation_sweep=sweep,
        baseline_hand_distances=baseline_hand_distance,static_variants=variants,
        free_body_probes=probes,limitations='Kinematic overlap is not a learned or physically executed punch; motor opposition is a controlled projection, not proof of a sole cause.'))


if __name__=='__main__':main()
