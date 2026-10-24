"""Bounded frozen-policy rollout audit of actual training-distribution terminations."""
import argparse,hashlib,json,os
from contextlib import ExitStack
from pathlib import Path
import jax,jax.numpy as jp,numpy as np
from boxer.checkpoint_io import atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import load_policy_params,contract,utc
from boxer.striking_train import Actor,normalize,require_ppo_checkpoint
from boxer.striking_fixed_opponent import ContactFixedMixedStrikingEnv,opponent_state,FROZEN_FINGERPRINT,fingerprint
from boxer.striking_mixed_partner import MIXED_ACTIVE_WORLDS
from boxer.striking_objective_audit import uncharged_pair_fall_cost
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/posture-1916928/control/gate-16384')
SHA='c47edf0bb3abf0ad56259269d255a33fd100709f39a3abe79f314b0f49fc6be2'


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mean-policy',action='store_true',help='read-only matched deterministic continuation of the reviewed sampled audit')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for p in (args.output/'.supervisor.lock',Path('runs/unassisted_boxing.lock')):
            lock=stack.enter_context(p.open('w'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'audit.json'
        if path.exists():raise ValueError('Objective audit exists; inspect before recovery')
        review=json.loads(Path('docs/results/striking-pilot/posture-1916928/analysis.json').read_text())
        assert review['status']=='reviewed' and review['diagnostic_ppo_total']==1950976 and review['formal_world_controls_total']==1454558
        formal_offset=1454558
        if args.mean_policy:
            previous=json.loads(Path('docs/results/striking-pilot/objective-1933312/analysis.json').read_text())
            assert previous['status']=='reviewed' and previous['formal_world_controls_total']==1470942
            formal_offset=1470942
        saved,manifest=load_checked(SOURCE,SHA);require_ppo_checkpoint(saved)
        assert saved['consumed_steps']==saved['optimized_steps']==1933312 and not saved.get('posture_cost_ppo')
        assert fingerprint(saved['frozen_opponent'])==FROZEN_FINGERPRINT
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(tuple(saved[k] for k in ('actor','critic','optimizer','norm','key'))))
        state=dict(status='running',pid=os.getpid(),started_utc=utc(),diagnostic_output=str(args.output),source_sha256=SHA,
            maximum_evaluation_world_controls=16384,simulated_world_transitions=0,formal_evaluation_world_controls_before_job=formal_offset,
            sampled_learner=not args.mean_policy,
            diagnostic_ppo_retained=1950976,new_optimizer_updates=0,new_collection_world_controls=0,new_replay_presentations=0,
            normalizer_updated=False,assistance=False,unchanged_actions_physics_rewards=True,policy_adoption='none',promotion=False,
            main_consumed_retained=1048576,main_project_consumed_retained=476041216,
            controls_by_mode_and_corner=[[0,0],[0,0]],rounds=[],cumulative_original_learner_reward=0.,counterfactual_additional_pair_fall_cost=0.)
        atomic_json(path,state)
        try:
            foundation=UnassistedG1('approach',impl='jax');params,motor=load_policy_params(Path('checkpoints/curriculum/approach'))
            assert contract(foundation)==motor['contract'] and motor['sha256']==saved['foundation_sha256']
            profile={k:saved[k] for k in ('engagement_distance','engagement_reward_distance')}
            envs=[ContactFixedMixedStrikingEnv(foundation,params,active_opponent=i in MIXED_ACTIVE_WORLDS,**profile) for i in range(8)]
            for env in envs[1:]:env.motor=envs[0].motor
            actor=Actor();other_params,other_norm=opponent_state(saved,saved['actor'],saved['norm'])
            @jax.jit
            def sample(obs,key):
                mean,std=actor.apply(saved['actor'],obs)
                if args.mean_policy:return jp.tanh(mean)
                return jp.tanh(mean+jp.exp(std)*jax.random.normal(key,mean.shape))
            @jax.jit
            def other_action(obs):return jp.tanh(actor.apply(other_params,obs)[0])
            key=saved['key'];episode=np.full(8,1933312//8,dtype=np.int64)
            obs=np.stack([env.reset(53+1009*i+int(episode[i]),swap=bool(i%2)) for i,env in enumerate(envs)]).reshape(8,148)
            round_rewards=np.zeros(8);round_scores=np.zeros((8,2));round_controls=np.zeros(8,dtype=int)
            starts=[dict(seed=53+1009*i+int(episode[i]),swap=bool(i%2)) for i in range(8)]
            for t in range(2048):
                key,sub=jax.random.split(key);actions=np.asarray(sample(jp.array(normalize(obs,saved['norm'])),sub))
                opponents=np.asarray(other_action(jp.array(normalize(np.stack([e.opponent_observation for e in envs]),other_norm))))
                following=[]
                for i,env in enumerate(envs):
                    state['simulated_world_transitions']+=1
                    state['controls_by_mode_and_corner'][int(env.active_opponent)][env.learner_corner]+=1
                    state.update(active_world=i,control_index=t,updated_utc=utc());atomic_json(path,state)
                    next_obs,reward,done,outcome=env.step(actions[i:i+1],opponents[i] if env.active_opponent else None)
                    assert np.isfinite(next_obs).all() and np.isfinite(reward).all()
                    gap=uncharged_pair_fall_cost(outcome['fallen'],env.learner_corner)
                    round_rewards[i]+=float(reward[0]);round_scores[i]+=outcome['scored'];round_controls[i]+=1
                    state['cumulative_original_learner_reward']+=float(reward[0]);state['counterfactual_additional_pair_fall_cost']+=gap
                    if done:
                        state['rounds'].append(dict(world=i,active_opponent=env.active_opponent,learner_corner=env.learner_corner,
                            **starts[i],controls=int(round_controls[i]),elapsed=env.env.pair.elapsed,fallen=outcome['fallen'].tolist(),
                            timed_out=outcome['timed_out'],scored_by_corner=round_scores[i].tolist(),original_round_reward=float(round_rewards[i]),
                            original_terminal_reward=float(reward[0]),own_fall_penalty=5.*float(outcome['fallen'][env.learner_corner]),
                            uncharged_pair_fall_cost=gap,counterfactual_terminal_reward=float(reward[0])-gap,
                            terminal_learner_action=actions[i].tolist()))
                        episode[i]+=1;starts[i]=dict(seed=53+1009*i+int(episode[i]),swap=bool(episode[i]%2))
                        next_obs=env.reset(**starts[i]);round_rewards[i]=0.;round_scores[i]=0.;round_controls[i]=0
                    following.append(next_obs)
                obs=np.asarray(following).reshape(8,148)
            assert state['simulated_world_transitions']==16384
            assert hashlib.sha256((SOURCE/manifest['file']).read_bytes()).hexdigest()==SHA
            state.update(status='completed',phase='awaiting_review',updated_utc=utc(),open_round_controls=round_controls.tolist(),
                open_round_scores=round_scores.tolist(),open_round_rewards=round_rewards.tolist())
            atomic_json(path,state)
        except BaseException as error:
            state.update(status='failed',error=str(error),updated_utc=utc());atomic_json(path,state);raise


if __name__=='__main__':main()
