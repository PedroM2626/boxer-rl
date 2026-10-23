"""Ineligible target-directed arm diagnostic, never a learned boxing controller.

Only original bounded high-level actions enter StrikingEnv.step. Jacobians are
read-only; all roots/legs/contacts and the frozen motor remain physically active.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params, utc
from boxer.striking_env import StrikingEnv
from boxer.striking_train import Actor, normalize


def directed_action(env, action, corner, target, advance=False):
    """One damped geometric correction through the original residual/filter box."""
    action=np.asarray(action).copy()
    pair=env.pair
    roots=[pair.data.qpos[m['root']:m['root']+2] for m in pair.maps]
    distance=float(np.linalg.norm(roots[1]-roots[0]))
    if advance:
        action[corner,0]=np.clip((distance-.5)*2.,0.,1.)
    if distance>.75 or np.isfinite(pair.scorecard.last_score[corner,0]):
        return action,False
    indices=np.array([i for i in env.arm_indices if
        'left_' in env.pair.foundation.mj_model.actuator(int(i)).name])
    if len(indices)!=7:raise ValueError('seven left arm joints required')
    mapping=pair.maps[corner]
    fist=int(pair.fists[corner,0])
    target_id=pair.model.geom(pair.maps[1-corner]['side']+target+'_collision').id
    jac=np.zeros((3,pair.model.nv));rot=np.zeros_like(jac)
    mujoco.mj_jacGeom(pair.model,pair.data,jac,rot,fist)
    jac=jac[:,mapping['v'][indices]]
    error=pair.data.geom_xpos[target_id]-pair.data.geom_xpos[fist]
    delta=jac.T@np.linalg.solve(jac@jac.T+.05**2*np.eye(3),error)
    delta*=min(1.,.06/max(float(np.max(np.abs(delta))),1e-12))
    desired=pair.data.qpos[mapping['q'][indices]]+delta
    # Predict the same frozen motor evaluation that the real step will use.
    motor,_=env.motor({'state':jp.array(pair.observation(env.commands(action))),
                       'privileged_state':jp.zeros((2,216))},jax.random.PRNGKey(0))
    baseline=pair.pose[indices]+pair.action_scale*np.asarray(motor)[corner,indices]
    slots=np.array([int(np.flatnonzero(env.arm_indices==i)[0]) for i in indices])
    requested=np.clip((desired-baseline)/env.residual_scale,-1.,1.)
    action[corner,3+slots]=np.clip((requested-.8*env.residual[corner,slots])/.2,-1.,1.)
    return action,True


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,23,31,47])
    parser.add_argument('--target',choices=('head','torso'))
    parser.add_argument('--advance-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    targets=(args.target,) if args.target else ('head','torso')
    advances=(True,) if args.advance_only else (False,True)
    trial_count=len(targets)*len(advances)*2*len(args.seeds)*2
    if len(set(args.seeds))!=len(args.seeds) or trial_count>64:
        raise ValueError('unique seeds and at most 64 complete diagnostic trials required')
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'feasibility.json'
        if path.exists():raise ValueError('feasibility run exists; inspect before relaunch')
        if jax.devices()[0].platform!='cpu':raise ValueError('CPU diagnostic required')
        cp=Path('checkpoints/striking-validation/partner-duration-1441792')
        manifest=json.loads((cp/'latest.json').read_text());payload=(cp/manifest['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:raise ValueError('source checksum mismatch')
        saved=pickle.loads(payload)
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
                tuple(saved[k] for k in ('actor','critic','optimizer','norm','key')))):
            raise FloatingPointError('nonfinite source learning state')
        foundation=UnassistedG1('approach',impl='jax')
        params,source=load_policy_params(Path('checkpoints/curriculum/approach'))
        if source['sha256']!=saved['foundation_sha256']:raise ValueError('motor provenance mismatch')
        env=StrikingEnv(foundation,params,engagement_distance=.45,engagement_reward_distance=.45)
        actor=Actor();policy=jax.jit(lambda obs:jp.tanh(actor.apply(saved['actor'],obs)[0]))
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),
            source_sha256=manifest['sha256'],motor_sha256=source['sha256'],optimization_transitions=0,
            promotion_eligible=False,learned_controller=False,assistance=False,policy_adoption='none',
            diagnostic_world_controls=0,maximum_world_controls=trial_count*500,total_trials=trial_count,
            seeds=args.seeds,targets=list(targets),advance_profiles=list(advances),trials=[])
        atomic_json(path,state)
        try:
            for target in targets:
                for advance in advances:
                    for corner in (0,1):
                        for seed in args.seeds:
                            for swap in (False,True):
                                label=f'{target}-advance-{int(advance)}-role-{corner}-seed-{seed}-swap-{int(swap)}'
                                state.update(active_trial=label,updated_utc=utc());atomic_json(path,state)
                                obs=env.reset(seed,swap);hits=np.zeros(2);active=0;records=[];events=[]
                                minimum=env.hand_distances()[corner]
                                target_id=env.pair.model.geom(env.pair.maps[1-corner]['side']+target+'_collision').id
                                def selected_distance():
                                    return float(np.linalg.norm(env.pair.data.geom_xpos[target_id]-
                                        env.pair.data.geom_xpos[env.pair.fists[corner,0]]))
                                selected_minimum=selected_distance()
                                for step in range(500):
                                    action=np.asarray(policy(jp.array(normalize(obs,saved['norm'])))).copy()
                                    action[1-corner]=0.
                                    action,used=directed_action(env,action,corner,target,advance)
                                    records.append((obs[corner].copy(),action[corner].copy()))
                                    state['diagnostic_world_controls']+=1
                                    atomic_json(args.output/'progress.json',dict(
                                        diagnostic_world_controls=state['diagnostic_world_controls'],active_trial=label,
                                        attempted_control=step+1,optimization_transitions=0,updated_utc=utc()))
                                    previous_scores=env.pair.scorecard.last_score.copy()
                                    obs,_,done,outcome=env.step(action)
                                    hits+=outcome['scored'];active+=int(used)
                                    minimum=min(minimum,env.hand_distances()[corner])
                                    selected_minimum=min(selected_minimum,selected_distance())
                                    if np.any(outcome['scored']):
                                        recent=env.pair.scorecard.last_score>previous_scores
                                        events.append(dict(elapsed=env.pair.elapsed,scored_by_corner=outcome['scored'].tolist(),
                                            hands=[dict(corner=i,hand=h,time=float(env.pair.scorecard.last_score[i,h]),
                                                contact_window_impulse=float(env.pair.scorecard.impulse[i,h]))
                                                for i in range(2) for h in range(2) if recent[i,h]]))
                                    if done:break
                                left_score=bool(np.isfinite(env.pair.scorecard.last_score[corner,0]))
                                row=dict(target=target,advance=advance,learner_corner=corner,seed=seed,swap=swap,
                                    elapsed=env.pair.elapsed,controls=step+1,survived=not env.pair.done,
                                    fallen=outcome['fallen'].tolist(),finite=bool(np.isfinite(obs).all()),
                                    scored_by_corner=hits.tolist(),directed_left_fist_scored=left_score,
                                    active_correction_controls=active,minimum_hand_target_distance=float(minimum),
                                    minimum_selected_fist_target_distance=selected_minimum,impact_events=events)
                                if row['survived'] and left_score:
                                    np.savez_compressed(args.output/(label+'.npz'),
                                        observation=np.array([x[0] for x in records]),action=np.array([x[1] for x in records]))
                                    row['diagnostic_trajectory']=label+'.npz'
                                state['trials'].append(row);atomic_json(path,state)
                                print(json.dumps(row),flush=True)
            state.update(status='completed',phase='awaiting_review',updated_utc=utc());atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
