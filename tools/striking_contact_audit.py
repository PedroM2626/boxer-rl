"""Fixed-policy replays and explicitly ineligible contact/body interventions.

Actual baseline trials must reproduce before intervention. Counterfactuals
replay recorded high-level actions while the learned motor stays reactive.
They never modify production gates, train a policy, or become boxing evidence.
"""
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
from boxer.curriculum_train import inference, load_policy_params, utc
from boxer.striking_body import WholeBodyStrikingEnv
from boxer.striking_train import Actor, normalize
from boxer.paired_foundation import PairedFoundation


def incoming_legal_pairs(pair, victim):
    """The four explicit incoming fist/head-or-torso pair names, for diagnostics."""
    fists=set(map(int,pair.fists[1-victim]));targets=pair.targets[victim]
    ids=[i for i,(a,b) in enumerate(zip(pair.model.pair_geom1,pair.model.pair_geom2))
         if (a in fists and b in targets) or (b in fists and a in targets)]
    if len(ids)!=4:raise ValueError('exactly four incoming legal-surface pairs required')
    return [(pair.model.geom(pair.model.pair_geom1[i]).name,
             pair.model.geom(pair.model.pair_geom2[i]).name) for i in ids]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdouts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    state_path=args.output/'audit.json'
    if state_path.exists():raise ValueError('audit already exists; preserve its accounting')
    foundation=UnassistedG1('approach',impl='jax')
    params,source=load_policy_params(Path('checkpoints/curriculum/approach'))
    motor=jax.jit(inference(foundation,params))
    state=dict(status='running',started_utc=utc(),optimization_transitions=0,
        counterfactual_physics_interventions=True,counterfactuals_evaluation_eligible=False,
        diagnostic_world_controls=0,policy_adoption='none',cases=[])
    atomic_json(state_path,state)
    try:
        for arm in ('frozen_body','adaptive_body'):
            folder=Path('checkpoints/striking-validation/body-comparison-1048576')/arm
            manifest=json.loads((folder/'latest.json').read_text());payload=(folder/manifest['file']).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:raise ValueError('source hash mismatch')
            saved=pickle.loads(payload)
            if saved['foundation_sha256']!=source['sha256']:raise ValueError('motor provenance mismatch')
            actor=Actor(action_size=32)
            distribution=jax.jit(lambda obs:actor.apply(saved['actor'],obs))
            mean_policy=jax.jit(lambda obs:jp.tanh(actor.apply(saved['actor'],obs)[0]))
            for mode in ('trained','sampled'):
                report=json.loads((args.holdouts/(arm+'-'+mode+'.json')).read_text())
                for expected in report['trials']:
                    if not sum(expected['scored_by_corner']):continue
                    def make_env(bound,excluded=()):
                        return WholeBodyStrikingEnv(foundation,params,motor=motor,
                            pair=PairedFoundation(foundation,diagnostic_excluded_pairs=excluded),
                            engagement_distance=.45,engagement_reward_distance=.45,
                            body_residual_radians=bound)
                    env=make_env(manifest['body_residual_radians'])
                    obs=env.reset(expected['seed'],expected['swap'])
                    rng=np.random.default_rng(expected['seed']+1009*int(expected['swap']))
                    plan=[];hits=np.zeros(2);impulse=np.zeros(2)
                    for step in range(500):
                        normalized=jp.array(normalize(obs,saved['norm']))
                        if mode=='sampled':
                            mean,log_std=map(np.asarray,distribution(normalized))
                            action=np.tanh(mean+np.exp(log_std)*rng.normal(size=mean.shape))
                        else:action=np.asarray(mean_policy(normalized))
                        plan.append(action.copy())
                        obs,_,done,outcome=env.step(action);state['diagnostic_world_controls']+=1
                        hits+=outcome['scored'];impulse+=outcome['legal_contact_impulse']
                        if done:break
                    if (not np.array_equal(hits,expected['scored_by_corner'])
                        or not np.array_equal(impulse,expected['raw_legal_impulse'])
                        or env.pair.elapsed!=expected['elapsed']
                        or (not env.pair.done)!=expected['survived']):
                        raise ValueError('baseline replay mismatch; do not interpret interventions')
                    case=dict(arm=arm,mode=mode,seed=expected['seed'],swap=expected['swap'],
                        source_sha256=manifest['sha256'],baseline_matches=True,
                        baseline=expected,high_level_controls=len(plan),counterfactuals=[])
                    interventions=[]
                    if not expected['survived']:
                        interventions.extend(('no_incoming_legal_forces',int(victim))
                            for victim in np.flatnonzero(expected['fallen']))
                    if arm=='adaptive_body':interventions.append(('zero_body_authority',None))
                    for kind,victim in interventions:
                        suppressed=incoming_legal_pairs(env.pair,victim) if victim is not None else []
                        replay=make_env(0. if kind=='zero_body_authority' else manifest['body_residual_radians'],suppressed)
                        replay.reset(expected['seed'],expected['swap'])
                        cf_hits=np.zeros(2)
                        for count,action in enumerate(plan,1):
                            _,_,done,outcome=replay.step(action);state['diagnostic_world_controls']+=1
                            cf_hits+=outcome['scored']
                            if done:break
                        case['counterfactuals'].append(dict(kind=kind,victim=victim,
                            evaluation_eligible=False,fixed_high_level_action_replay=True,
                            suppressed_pairs=suppressed,survived_observed_control_horizon=not replay.pair.done,
                            elapsed=replay.pair.elapsed,controls=count,
                            baseline_control_horizon_seconds=len(plan)*replay.pair.dt,
                            scored_by_corner=cf_hits.tolist(),fallen=outcome['fallen'].tolist()))
                    state['cases'].append(case);atomic_json(state_path,state)
                    print(json.dumps({k:case[k] for k in ('arm','mode','seed','swap','counterfactuals')}),flush=True)
        state.update(status='completed',updated_utc=utc());atomic_json(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(state_path,state)
        raise


if __name__=='__main__':main()
