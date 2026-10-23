"""Independent first-fall evaluation of the unassisted striking interface."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import contract, load_policy_params, utc
from boxer.striking_env import StrikingEnv
from boxer.striking_train import Actor, normalize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=Path('runs/striking-validation/evaluation.json'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,23,31,47])
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--video',type=Path)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0:
        raise ValueError('positive finite duration required')
    foundation=UnassistedG1('approach',impl='jax')
    params,source=load_policy_params(Path('checkpoints/curriculum/approach'))
    if contract(foundation)!=source['contract']:raise ValueError('motor contract mismatch')
    env=StrikingEnv(foundation,params)
    residual_hash=None
    if args.checkpoint:
        manifest=json.loads((args.checkpoint/'latest.json').read_text())
        payload=(args.checkpoint/manifest['file']).read_bytes()
        residual_hash=hashlib.sha256(payload).hexdigest()
        if residual_hash!=manifest['sha256']:raise ValueError('striking checksum mismatch')
        saved=pickle.loads(payload)
        if saved['foundation_sha256']!=source['sha256']:raise ValueError('striking motor mismatch')
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
                tuple(saved[k] for k in ('actor','critic','optimizer','norm','key')))):
            raise FloatingPointError('non-finite striking checkpoint')
        actor=Actor()
        policy=jax.jit(lambda observation:jp.tanh(actor.apply(saved['actor'],observation)[0]))
        modes=['trained']
    else:
        modes=['neutral','exploration']
    results=[];transitions=0;frames=[]
    for mode in modes:
        for seed in args.seeds:
            for swap in (False,True):
                obs=env.reset(seed,swap)
                rng=np.random.default_rng(seed+int(swap)*1009)
                hits=np.zeros(2);impulses=np.zeros(2)
                minimum_hand_distance=env.hand_distances().copy()
                residual_activity=np.zeros(2)
                max_residual=np.zeros(2)
                for step in range(round(args.seconds/env.pair.dt)):
                    if mode=='trained':action=np.asarray(policy(jp.array(normalize(obs,saved['norm']))))
                    elif mode=='exploration':action=np.tanh(rng.normal(0.,np.exp(-1.9),(2,17)))
                    else:action=np.zeros((2,17))
                    obs,reward,done,outcome=env.step(action);transitions+=1
                    hits+=outcome['scored'];impulses+=outcome['legal_contact_impulse']
                    minimum_hand_distance=np.minimum(minimum_hand_distance,env.hand_distances())
                    residual_activity+=np.mean(np.abs(outcome['arm_residual']),axis=1)
                    max_residual=np.maximum(max_residual,np.max(np.abs(outcome['arm_residual']),axis=1))
                    if args.video and mode==modes[0] and seed==args.seeds[0] and not swap and step%2==0:
                        frames.append(env.pair.data.qpos.copy())
                    if done:break
                result=dict(mode=mode,seed=seed,swap=swap,survived=not env.pair.done,
                    elapsed=env.pair.elapsed,final_distance=outcome['distance'],
                    scored_by_corner=hits.tolist(),raw_legal_impulse=impulses.tolist(),
                    fallen=outcome['fallen'].tolist(),finite=bool(np.isfinite(obs).all()),
                    minimum_hand_target_distance=minimum_hand_distance.tolist(),
                    mean_abs_arm_residual=(residual_activity/(step+1)).tolist(),
                    max_abs_arm_residual=max_residual.tolist())
                results.append(result);print(json.dumps(result),flush=True)
    if frames:
        from tools.evaluate_paired_foundation import render
        render(env.pair,frames,args.video,caption='Unassisted striking pilot | Physical contacts | Evaluation')
    atomic_json(args.output,dict(kind='striking_interface_evaluation',updated_utc=utc(),
        assistance=False,source_sha256=source['sha256'],striking_sha256=residual_hash,
        simulated_world_transitions=transitions,optimization_transitions=0,
        scorecard_version=1,episode_seconds=args.seconds,trials=results,
        certification=False))


if __name__=='__main__':
    main()
