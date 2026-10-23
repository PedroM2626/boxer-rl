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
    parser.add_argument('--engagement-distance',type=float,
                        help='explicit diagnostic profile override; impact criteria remain fixed')
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0:
        raise ValueError('positive finite duration required')
    foundation=UnassistedG1('approach',impl='jax')
    params,source=load_policy_params(Path('checkpoints/curriculum/approach'))
    if contract(foundation)!=source['contract']:raise ValueError('motor contract mismatch')
    distance,reward_distance=.58,.6
    manifest=None
    config={}
    if args.checkpoint:
        manifest=json.loads((args.checkpoint/'latest.json').read_text())
        distance=manifest.get('engagement_distance',distance)
        reward_distance=manifest.get('engagement_reward_distance',reward_distance)
        config_path=args.checkpoint/'config.json'
        if not config_path.exists():config_path=args.checkpoint.parent/'config.json'
        if config_path.exists():
            config=json.loads(config_path.read_text())
            for key in ('engagement_distance','engagement_reward_distance','actor_inputs','actions','body_residual_radians'):
                if key in manifest and key in config and manifest[key]!=config[key]:
                    raise ValueError('checkpoint and configuration engagement profiles disagree')
            distance=config.get('engagement_distance',.58)
            reward_distance=config.get('engagement_reward_distance',.6)
    if args.engagement_distance is not None:
        distance=reward_distance=args.engagement_distance
    inputs=config.get('actor_inputs',(manifest or {}).get('actor_inputs',146))
    actions=config.get('actions',(manifest or {}).get('actions',17))
    body_bound=config.get('body_residual_radians',(manifest or {}).get('body_residual_radians',0.))
    if (inputs,actions)==(146,17):
        if body_bound!=0.:raise ValueError('legacy interface cannot carry body residual authority')
        env=StrikingEnv(foundation,params,engagement_distance=distance,
                       engagement_reward_distance=reward_distance)
    elif (inputs,actions)==(161,32):
        from boxer.striking_body import WholeBodyStrikingEnv
        env=WholeBodyStrikingEnv(foundation,params,engagement_distance=distance,
            engagement_reward_distance=reward_distance,body_residual_radians=body_bound)
    else:
        raise ValueError('unsupported striking interface dimensions')
    residual_hash=None
    if args.checkpoint:
        payload=(args.checkpoint/manifest['file']).read_bytes()
        residual_hash=hashlib.sha256(payload).hexdigest()
        if residual_hash!=manifest['sha256']:raise ValueError('striking checksum mismatch')
        saved=pickle.loads(payload)
        if saved['foundation_sha256']!=source['sha256']:raise ValueError('striking motor mismatch')
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
                tuple(saved[k] for k in ('actor','critic','optimizer','norm','key')))):
            raise FloatingPointError('non-finite striking checkpoint')
        actor=Actor(action_size=actions)
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
                impact_events=[]
                minimum_hand_distance=env.hand_distances().copy()
                residual_activity=np.zeros(2)
                max_residual=np.zeros(2)
                for step in range(round(args.seconds/env.pair.dt)):
                    if mode=='trained':action=np.asarray(policy(jp.array(normalize(obs,saved['norm']))))
                    elif mode=='exploration':action=np.tanh(rng.normal(0.,np.exp(-1.9),(2,actions)))
                    else:action=np.zeros((2,actions))
                    previous_scores=env.pair.scorecard.last_score.copy()
                    obs,reward,done,outcome=env.step(action);transitions+=1
                    hits+=outcome['scored'];impulses+=outcome['legal_contact_impulse']
                    if np.any(outcome['scored']):
                        recent=env.pair.scorecard.last_score > previous_scores
                        hands=[dict(corner=i,hand=h,
                                    score_time=float(env.pair.scorecard.last_score[i,h]),
                                    contact_window_impulse=float(env.pair.scorecard.impulse[i,h]))
                               for i in range(2) for h in range(2) if recent[i,h]]
                        impact_events.append(dict(control_step=step,elapsed=env.pair.elapsed,
                            scored_by_corner=outcome['scored'].tolist(),hands=hands,
                            fallen=outcome['fallen'].tolist()))
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
                result['impact_events']=impact_events
                results.append(result);print(json.dumps(result),flush=True)
    if frames:
        from tools.evaluate_paired_foundation import render
        render(env.pair,frames,args.video,caption='Unassisted striking pilot | Physical contacts | Evaluation')
    atomic_json(args.output,dict(kind='striking_interface_evaluation',updated_utc=utc(),
        assistance=False,source_sha256=source['sha256'],striking_sha256=residual_hash,
        simulated_world_transitions=transitions,optimization_transitions=0,
        scorecard_version=1,episode_seconds=args.seconds,trials=results,
        engagement_distance=distance,engagement_reward_distance=reward_distance,
        actor_inputs=inputs,actions=actions,body_residual_radians=body_bound,
        diagnostic_profile_override=args.engagement_distance is not None,
        certification=False))


if __name__=='__main__':
    main()
