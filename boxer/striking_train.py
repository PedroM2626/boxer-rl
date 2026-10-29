"""Bounded native-physics self-play PPO over the certified frozen motor policy.

JAX stays on CPU. Both corners learn one shared opponent-aware residual policy.
Foundation steps, boxing learner steps and physical world steps remain distinct.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import shutil
import time

import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np
import optax

from boxer.checkpoint_io import atomic_bytes, atomic_json
from boxer.curriculum_env import UnassistedG1
from boxer.curriculum_train import contract, load_policy_params, utc
from boxer.curriculum_control import promotion_ready
from boxer.striking_env import StrikingEnv


class Actor(nn.Module):
    action_size: int = 17
    sampling_log_std_offset: float = 0.
    @nn.compact
    def __call__(self, obs):
        x=nn.tanh(nn.Dense(128)(obs))
        x=nn.tanh(nn.Dense(128)(x))
        mean=nn.Dense(self.action_size,kernel_init=nn.initializers.zeros)(x)
        log_std=self.param('log_std',nn.initializers.constant(-1.9),(self.action_size,))
        bounded=jp.clip(log_std,-3.,-.5)
        if self.sampling_log_std_offset:bounded=bounded+self.sampling_log_std_offset
        return mean,bounded


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x=nn.tanh(nn.Dense(128)(obs))
        x=nn.tanh(nn.Dense(128)(x))
        return nn.Dense(1)(x)[...,0]


def log_probability(raw, mean, log_std):
    return -.5*jp.sum(((raw-mean)/jp.exp(log_std))**2 + 2*log_std + np.log(2*np.pi),axis=-1)


def advantages(rewards, dones, values, bootstrap, gamma=.99, lam=.95):
    """Time-major GAE; both first falls and the round clock cut bootstrapping."""
    result=np.zeros_like(rewards)
    carry=np.zeros_like(bootstrap)
    following=bootstrap
    for t in reversed(range(len(rewards))):
        continuation=1.-dones[t]
        delta=rewards[t]+gamma*following*continuation-values[t]
        carry=delta+gamma*lam*continuation*carry
        result[t]=carry
        following=values[t]
    return result,result+values


def normalize(obs, norm):
    return np.clip((obs-norm['mean'])/np.sqrt(norm['var']+.01),-10.,10.).astype(np.float32)


def update_norm(norm, obs):
    n=len(obs); delta=obs.mean(axis=0)-norm['mean'];total=norm['count']+n
    variance=(norm['var']*norm['count']+obs.var(axis=0)*n
              +delta**2*norm['count']*n/total)/total
    return dict(mean=norm['mean']+delta*n/total,var=variance,count=total)


LEGACY_ENGAGEMENT = dict(engagement_distance=.58, engagement_reward_distance=.6)


def engagement_profile(previous=None, override=None):
    profile={key:(previous or {}).get(key,value) for key,value in LEGACY_ENGAGEMENT.items()}
    if override is not None:
        profile={key:override for key in profile}
    if not all(np.isfinite(value) and value>0 for value in profile.values()):
        raise ValueError('engagement distances must be finite and positive')
    return profile


def validate_resume_config(previous, current, explicit_profile_change=False):
    expected=dict(previous,**engagement_profile(previous))
    ignored=set(LEGACY_ENGAGEMENT) if explicit_profile_change else set()
    if {k:v for k,v in expected.items() if k not in ignored} != {
            k:v for k,v in current.items() if k not in ignored}:
        raise ValueError('resume requires the exact profile except an explicitly reviewed engagement change')


def require_ppo_checkpoint(saved):
    if saved.get('training_kind','ppo')!='ppo' or saved.get('ppo_optimizer_compatible') is False:
        raise ValueError('non-PPO checkpoint requires an explicit validated optimizer migration before PPO resume')
    if 'gpu_batched_ppo' in saved or 'gpu_collection_profile' in saved or 'gpu_collection_transition' in saved:
        from boxer.striking_gpu_profile import validate_gpu_metadata
        validate_gpu_metadata(saved)
    if 'motion_cycle_transition' in saved or 'motion_reward_profile' in saved or saved.get('motion_cycle_ppo'):
        from boxer.striking_motion_reward import validate_motion_metadata
        validate_motion_metadata(saved)
    if 'contact_progress_transition' in saved or 'contact_progress_profile' in saved or saved.get('contact_progress_ppo'):
        from boxer.striking_contact_progress import validate_progress_metadata
        validate_progress_metadata(saved)
    if 'contact_rebalance_transition' in saved or 'contact_rebalance_profile' in saved or saved.get('contact_rebalance_ppo'):
        from boxer.striking_contact_rebalance import validate_rebalance_metadata
        validate_rebalance_metadata(saved)
    if 'arm_effort_transition' in saved or 'arm_effort_profile' in saved or saved.get('arm_effort_ppo'):
        from boxer.striking_arm_effort import validate_arm_effort_metadata
        validate_arm_effort_metadata(saved)
    kind=saved.get('optimizer_kind')
    if kind is not None:
        from boxer.striking_ppo_warmstart import OPTIMIZER_KIND,validate_split_checkpoint
        if kind!=OPTIMIZER_KIND:raise ValueError('unsupported optimizer format')
        validate_split_checkpoint(saved)
        if saved.get('credit_horizon_ppo'):
            from boxer.striking_credit_horizon import validate_horizon_state
            validate_horizon_state(saved)
        if saved.get('fixed_opponent_ppo'):
            from boxer.striking_fixed_opponent import validate_fixed_metadata
            validate_fixed_metadata(saved)
        if saved.get('posture_cost_ppo'):
            from boxer.striking_posture_cost import validate_posture_metadata
            validate_posture_metadata(saved)
        if 'sampling_log_std_offset' in saved or saved.get('reduced_noise_ppo'):
            from boxer.striking_sampling_profile import validate_sampling_metadata
            validate_sampling_metadata(saved)
        if 'kl_guard_target' in saved or saved.get('kl_guard_ppo'):
            from boxer.striking_kl_guard import validate_guard_metadata
            validate_guard_metadata(saved)
        if 'training_gae_lambda' in saved or 'long_gae_transition' in saved or saved.get('long_gae_ppo'):
            from boxer.striking_gae_profile import validate_gae_metadata
            validate_gae_metadata(saved)
        if ('opponent_pool' in saved or 'opponent_pool_transition' in saved
                or saved.get('pool_opponent_ppo') or saved.get('opponent_mode')=='mixed_pool_mean'):
            from boxer.striking_opponent_pool import validate_pool_metadata
            validate_pool_metadata(saved)
        if saved.get('conservative_actor_ppo'):
            from boxer.striking_conservative_actor import validate_conservative_metadata
            validate_conservative_metadata(saved)
    elif 'optimizer' in saved:
        try:mu=saved['optimizer'][1][0].mu
        except (KeyError,IndexError,TypeError,AttributeError) as error:
            raise ValueError('invalid legacy joint optimizer; explicit optimizer migration required') from error
        if not isinstance(mu,tuple) or len(mu)!=2:
            raise ValueError('joint PPO optimizer required; explicit optimizer migration required')


def make_ppo_update(actor,critic,tx,anchor_weight=None,retention_weight=None,actor_enabled=True):
    if anchor_weight is not None:
        from boxer.striking_ppo_warmstart import anchor_loss
    if retention_weight is not None:
        from boxer.striking_retention import retention_loss
    @jax.jit
    def update(ap,vp,optimizer,batch):
        def loss(ap,vp):
            mean,log_std=actor.apply(ap,batch['obs'])
            ratio=jp.exp(log_probability(batch['raw'],mean,log_std)-batch['logp'])
            policy=-jp.mean(jp.minimum(ratio*batch['adv'],jp.clip(ratio,.8,1.2)*batch['adv']))
            value_loss=jp.mean((critic.apply(vp,batch['obs'])-batch['returns'])**2)
            entropy=jp.sum(log_std+.5*np.log(2*np.pi*np.e))
            result=policy+.25*value_loss-.001*entropy
            if anchor_weight is not None:
                result+=anchor_weight*anchor_loss(actor,ap,batch['anchor_obs'],batch['anchor_labels'])
            if retention_weight is not None:
                result+=retention_weight*retention_loss(actor,ap,batch['retention_obs'],batch['retention_labels'],batch['retention_weights'])
            return result
        loss_value,grads=jax.value_and_grad(loss,argnums=(0,1))(ap,vp)
        old_ap=ap
        if actor_enabled:updates,optimizer=tx.update(grads,optimizer,(ap,vp))
        else:updates,optimizer=tx.update(grads,optimizer,(ap,vp),update_actor=False)
        ap,vp=optax.apply_updates((ap,vp),updates)
        if not actor_enabled:ap=old_ap
        return ap,vp,optimizer,loss_value
    return update


def main(argv=None):
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/striking-pilot'))
    parser.add_argument('--foundation',type=Path,default=Path('checkpoints/curriculum/approach'))
    parser.add_argument('--ceiling',type=int,default=32768)
    parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--unroll',type=int,default=128)
    parser.add_argument('--seed',type=int,default=53)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--contact-ppo',action='store_true',help='explicit migrated 148-input split-Adam training')
    parser.add_argument('--anchor-loss-weight',type=float,choices=(0.,.02),default=0.)
    parser.add_argument('--demonstration-ppo',action='store_true')
    parser.add_argument('--retention-loss-weight',type=float,choices=(0.,1.),default=0.)
    parser.add_argument('--neutral-contact-ppo',action='store_true')
    parser.add_argument('--credit-horizon-ppo',action='store_true')
    parser.add_argument('--mixed-partner-ppo',action='store_true')
    parser.add_argument('--fixed-opponent-ppo',action='store_true')
    parser.add_argument('--posture-cost-ppo',action='store_true',help='explicit training-only posture margin cost')
    parser.add_argument('--reduced-noise-ppo',action='store_true',help='explicit half-standard-deviation on-policy sampling profile')
    parser.add_argument('--kl-guard-ppo',action='store_true',help='explicit actor-only full-rollout KL stop, critic keeps all updates')
    parser.add_argument('--long-gae-ppo',action='store_true',help='explicit lambda .99 over latest original fixed-opponent control')
    parser.add_argument('--contact-progress-ppo',action='store_true',help='v15 target-coupled potential and physical per-hand contact subgoals')
    parser.add_argument('--contact-rebalance-ppo',action='store_true',help='v16 reduced partial-contact payment with complete-cycle reward retained')
    parser.add_argument('--arm-effort-ppo',action='store_true',help='v17 smooth training-only cost for learned arm residuals')
    parser.add_argument('--motion-cycle-ppo',action='store_true',help='versioned complete-cycle reward with obsolete touch-teacher loss disabled')
    parser.add_argument('--pool-opponent-ppo',action='store_true',help='explicit immutable archived opponent pool over the latest long-GAE source')
    parser.add_argument('--conservative-actor-ppo',action='store_true')
    parser.add_argument('--actor-learning-rate',type=float,choices=(.0001,.00001),default=.0001)
    parser.add_argument('--update-audit-directory',type=Path,
                        help='passive update diagnostics on one explicit mixed-profile rollout only')
    parser.add_argument('--training-gamma',type=float,choices=(.99,.999),default=.99)
    parser.add_argument('--engagement-distance',type=float,
                        help='explicit reviewed change of approach stop and engagement reward peak')
    parser.add_argument('--publish-dir',type=Path,
                        help='optional repository mirror; durable per-step ledger stays in output')
    args=parser.parse_args(argv)
    if args.contact_ppo and (not args.resume or args.engagement_distance is not None):
        raise ValueError('contact PPO requires explicit migrated resume and inherited engagement profile')
    if not args.contact_ppo and args.anchor_loss_weight!=0.:
        raise ValueError('auxiliary loss requires explicit contact PPO')
    if args.demonstration_ppo and (not args.contact_ppo or args.anchor_loss_weight!=0.):
        raise ValueError('demonstration PPO requires contact resume without the rejected anchor-only profile')
    if not args.demonstration_ppo and args.retention_loss_weight!=0.:
        raise ValueError('retention loss requires explicit demonstration PPO')
    if args.neutral_contact_ppo and (not args.demonstration_ppo or args.retention_loss_weight!=(0. if args.motion_cycle_ppo else 1.)):
        raise ValueError('neutral contact task requires the reviewed demonstration-retention path')
    if args.credit_horizon_ppo and not args.neutral_contact_ppo:
        raise ValueError('credit-horizon comparison requires the reviewed neutral task')
    if not args.credit_horizon_ppo and args.training_gamma!=.99:
        raise ValueError('discount change requires explicit credit-horizon provenance')
    if args.mixed_partner_ppo and (not args.credit_horizon_ppo or args.training_gamma!=.999
            or args.worlds!=8 or args.unroll!=128):
        raise ValueError('mixed partner requires explicit gamma-0.999 lineage and eight 128-step worlds')
    if args.update_audit_directory is not None and not args.mixed_partner_ppo:
        raise ValueError('update audit requires the explicit mixed diagnostic path')
    if args.fixed_opponent_ppo and (not args.mixed_partner_ppo or args.conservative_actor_ppo):
        raise ValueError('fixed opponent requires explicit original-rate mixed lineage')
    if args.posture_cost_ppo and not args.fixed_opponent_ppo:
        raise ValueError('posture cost requires the explicit inherited fixed-opponent path')
    if args.reduced_noise_ppo and (not args.fixed_opponent_ppo or args.posture_cost_ppo):
        raise ValueError('reduced noise requires the original fixed-opponent reward path')
    if args.kl_guard_ppo and (not args.fixed_opponent_ppo or args.reduced_noise_ppo or args.posture_cost_ppo):
        raise ValueError('KL guard requires original covariance and fixed-opponent rewards')
    if args.long_gae_ppo and (not args.fixed_opponent_ppo or args.kl_guard_ppo or args.reduced_noise_ppo or args.posture_cost_ppo):
        raise ValueError('long GAE requires the original fixed-opponent update and reward path')
    if args.contact_progress_ppo and not args.motion_cycle_ppo:
        raise ValueError('contact progress requires the retained explicit motion-cycle task')
    if args.contact_rebalance_ppo and not args.contact_progress_ppo:
        raise ValueError('contact rebalance requires the inherited explicit contact-progress path')
    if args.arm_effort_ppo and not args.contact_rebalance_ppo:
        raise ValueError('arm effort requires the inherited contact-rebalance path')
    if args.motion_cycle_ppo and (not args.long_gae_ppo or args.pool_opponent_ppo or args.retention_loss_weight!=0.):
        raise ValueError('motion-cycle training requires original long-GAE fixed task and zero teacher loss')
    if args.pool_opponent_ppo and not args.long_gae_ppo:
        raise ValueError('opponent pool requires the explicit inherited long-GAE profile')
    if args.conservative_actor_ppo and (not args.mixed_partner_ppo or args.actor_learning_rate!=.00001):
        raise ValueError('conservative actor requires explicit mixed lineage and actor rate 1e-5')
    if not args.conservative_actor_ppo and args.actor_learning_rate!=.0001:
        raise ValueError('actor rate change requires explicit conservative provenance')
    if args.ceiling <= 0 or args.worlds <= 0 or args.unroll < 2:
        raise ValueError('positive ceiling/worlds and unroll >=2 required')
    if jax.devices()[0].platform!='cpu':
        raise ValueError('native boxing pilot requires JAX_PLATFORMS=cpu')
    args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for path in (args.output/'.lock',Path('runs/unassisted_boxing.lock')):
            lock=stack.enter_context(path.open('w'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            return train(args)
        except BaseException as error:
            state_path=args.output/'state.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else {}
            atomic_json(state_path,dict(state,status='failed',error=str(error),updated_utc=utc()))
            if args.publish_dir:
                args.publish_dir.mkdir(parents=True,exist_ok=True)
                shutil.copy2(state_path,args.publish_dir/'state.json')
            raise


def train(args):
    config_path=args.output/'config.json'
    previous=json.loads(config_path.read_text()) if config_path.exists() else None
    profile=engagement_profile(previous,args.engagement_distance)
    foundation=UnassistedG1('approach',impl='jax')
    for stage in ('balance','walk_easy','walk','approach'):
        source=json.loads((args.foundation.parent/stage/'latest.json').read_text())
        history=json.loads((Path('docs/results/curriculum')/(stage+'-gate_history.json')).read_text())
        if (not promotion_ready(history,stage)
                or history[-1]['checkpoint_sha256']!=source['sha256']
                or history[-1]['stage_steps']!=source['stage_steps']):
            raise ValueError('all four complete current foundation certifications required')
    motor_params,metadata=load_policy_params(args.foundation)
    if contract(foundation)!=metadata['contract']:
        raise ValueError('frozen motor contract mismatch')
    contact_ppo=getattr(args,'contact_ppo',False)
    demonstration_ppo=getattr(args,'demonstration_ppo',False)
    neutral_contact_ppo=getattr(args,'neutral_contact_ppo',False)
    credit_horizon_ppo=getattr(args,'credit_horizon_ppo',False)
    mixed_partner_ppo=getattr(args,'mixed_partner_ppo',False)
    conservative_actor_ppo=getattr(args,'conservative_actor_ppo',False)
    fixed_opponent_ppo=getattr(args,'fixed_opponent_ppo',False)
    posture_cost_ppo=getattr(args,'posture_cost_ppo',False)
    reduced_noise_ppo=getattr(args,'reduced_noise_ppo',False)
    kl_guard_ppo=getattr(args,'kl_guard_ppo',False)
    long_gae_ppo=getattr(args,'long_gae_ppo',False)
    pool_opponent_ppo=getattr(args,'pool_opponent_ppo',False)
    motion_cycle_ppo=getattr(args,'motion_cycle_ppo',False)
    contact_progress_ppo=getattr(args,'contact_progress_ppo',False)
    contact_rebalance_ppo=getattr(args,'contact_rebalance_ppo',False)
    arm_effort_ppo=getattr(args,'arm_effort_ppo',False)
    env_class=StrikingEnv
    if contact_ppo:
        from boxer.striking_contact import ContactAwareStrikingEnv
        env_class=ContactAwareStrikingEnv
        if neutral_contact_ppo:
            from boxer.striking_partner import ContactPartnerStrikingEnv
            env_class=ContactPartnerStrikingEnv
        if mixed_partner_ppo:
            from boxer.striking_mixed_partner import ContactMixedStrikingEnv,MIXED_ACTIVE_WORLDS
            env_class=ContactMixedStrikingEnv
            if fixed_opponent_ppo:
                from boxer.striking_fixed_opponent import ContactFixedMixedStrikingEnv
                env_class=ContactFixedMixedStrikingEnv
                if pool_opponent_ppo:
                    from boxer.striking_opponent_pool import ContactPoolMixedStrikingEnv
                    env_class=ContactPoolMixedStrikingEnv
    observation_size, action_size = env_class.observation_size, env_class.action_size
    learners_per_world = getattr(env_class, 'learners_per_world', 2)
    if learners_per_world not in (1, 2):
        raise ValueError('one or two learners per paired world required')
    interface_metadata = dict(actor_inputs=observation_size, actions=action_size)
    if learners_per_world != 2:
        interface_metadata.update(learners_per_world=learners_per_world,
                                  opponent_mode=env_class.opponent_mode)
    if action_size == 32:
        interface_metadata['body_residual_radians'] = StrikingEnv.body_residual_radians
    config=dict(version=1,foundation_sha256=metadata['sha256'],foundation_main_steps=metadata['total_steps'],
        worlds=args.worlds,unroll=args.unroll,seed=args.seed,**interface_metadata,
        arm_residual_radians=.8,command_offsets=[.15,.15,.25],assistance=False,
        dt=.02,physics_dt=.002,round_steps=500,learning_rate=.0001,epochs=4,
        minibatches=4,gamma=.99,gae_lambda=.95,scorecard_version=1,**profile)
    if contact_ppo:
        from boxer.striking_ppo_warmstart import OPTIMIZER_KIND,SplitAdam
        if previous is None:raise ValueError('migrated contact configuration required')
        interface_metadata.update(contact_history=True,learners_per_world=learners_per_world,
            opponent_mode=getattr(env_class,'opponent_mode','active_selfplay'),
            optimizer_kind=OPTIMIZER_KIND,anchor_loss_weight=args.anchor_loss_weight)
        config.update(interface_metadata,version=2)
        if demonstration_ppo:
            interface_metadata.update(demonstration_ppo=True,retention_loss_weight=args.retention_loss_weight)
            config.update(interface_metadata,version=3)
            if neutral_contact_ppo:
                interface_metadata['neutral_contact_ppo']=True
                config.update(interface_metadata,version=4)
                if credit_horizon_ppo:
                    interface_metadata.update(credit_horizon_ppo=True,training_gamma=args.training_gamma)
                    config.update(interface_metadata,version=5,gamma=args.training_gamma)
                    if mixed_partner_ppo:
                        interface_metadata.update(mixed_partner_ppo=True,active_world_indices=list(MIXED_ACTIVE_WORLDS))
                        config.update(interface_metadata,version=6)
                        if conservative_actor_ppo:
                            interface_metadata.update(conservative_actor_ppo=True,actor_learning_rate=args.actor_learning_rate)
                            config.update(interface_metadata,version=7)
                        if fixed_opponent_ppo:
                            from boxer.striking_fixed_opponent import FROZEN_FINGERPRINT
                            interface_metadata.update(fixed_opponent_ppo=True,fixed_opponent_fingerprint=FROZEN_FINGERPRINT)
                            config.update(interface_metadata,version=8)
                            if posture_cost_ppo:
                                from boxer.striking_posture_cost import PROFILE
                                interface_metadata.update(posture_cost_ppo=True,posture_cost_profile=PROFILE)
                                config.update(interface_metadata,version=9)
                            if reduced_noise_ppo:
                                from boxer.striking_sampling_profile import OFFSET
                                interface_metadata.update(reduced_noise_ppo=True,sampling_log_std_offset=OFFSET)
                                config.update(interface_metadata,version=10)
                            if kl_guard_ppo:
                                from boxer.striking_kl_guard import TARGET
                                interface_metadata.update(kl_guard_ppo=True,kl_guard_target=TARGET)
                                config.update(interface_metadata,version=11)
                            if long_gae_ppo:
                                interface_metadata.update(long_gae_ppo=True,training_gae_lambda=.99)
                                config.update(interface_metadata,version=12,gae_lambda=.99)
                                if pool_opponent_ppo:
                                    interface_metadata.update(pool_opponent_ppo=True)
                                    config.update(interface_metadata,version=13)
    if motion_cycle_ppo:
        from boxer.striking_motion_reward import PROFILE as MOTION_PROFILE
        interface_metadata.update(motion_cycle_ppo=True,motion_reward_profile=MOTION_PROFILE)
        config.update(interface_metadata,version=14)
    if contact_progress_ppo:
        from boxer.striking_contact_progress import PROFILE as PROGRESS_PROFILE
        interface_metadata.update(contact_progress_ppo=True,contact_progress_profile=PROGRESS_PROFILE)
        config.update(interface_metadata,version=15)
    if contact_rebalance_ppo:
        from boxer.striking_contact_rebalance import PROFILE as REBALANCE_PROFILE
        interface_metadata.update(contact_rebalance_ppo=True,contact_rebalance_profile=REBALANCE_PROFILE)
        config.update(interface_metadata,version=16)
    if arm_effort_ppo:
        from boxer.striking_arm_effort import PROFILE as ARM_EFFORT_PROFILE
        interface_metadata.update(arm_effort_ppo=True,arm_effort_profile=ARM_EFFORT_PROFILE)
        config.update(interface_metadata,version=17)
    checkpoint_path=args.output/'checkpoint/latest.json'
    if previous:
        if not args.resume:
            raise ValueError('existing pilot requires --resume with the exact same configuration')
        validate_resume_config(previous,config,args.engagement_distance is not None)
    atomic_json(config_path,config)
    actor,critic=Actor(action_size=action_size,sampling_log_std_offset=interface_metadata.get('sampling_log_std_offset',0.)),Critic()
    key=jax.random.PRNGKey(args.seed)
    ap=actor.init(key,jp.zeros((1,observation_size)));vp=critic.init(jax.random.fold_in(key,1),jp.zeros((1,observation_size)))
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0001))
    if contact_ppo:tx=SplitAdam(actor_learning_rate=getattr(args,'actor_learning_rate',.0001))
    optimizer=tx.init((ap,vp))
    norm=dict(mean=np.zeros(observation_size),var=np.ones(observation_size),count=1e-4)
    consumed=optimized=world_controls=0
    continuation={}
    if args.resume and checkpoint_path.exists():
        manifest=json.loads(checkpoint_path.read_text())
        payload=(checkpoint_path.parent/manifest['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:
            raise ValueError('boxing checkpoint checksum mismatch')
        saved=pickle.loads(payload)
        require_ppo_checkpoint(saved)
        if saved.get('gpu_batched_ppo'):
            raise ValueError('GPU checkpoint requires the explicit GPU collector; native resume cannot silently change backend')
        if contact_progress_ppo != saved.get('contact_progress_ppo',False):
            raise ValueError('contact progress checkpoint and explicit training reward must agree')
        if contact_rebalance_ppo != saved.get('contact_rebalance_ppo',False):
            raise ValueError('contact rebalance checkpoint and explicit training reward must agree')
        if arm_effort_ppo != saved.get('arm_effort_ppo',False):
            raise ValueError('arm effort checkpoint and explicit training reward must agree')
        if motion_cycle_ppo != saved.get('motion_cycle_ppo',False):
            raise ValueError('motion-cycle checkpoint and explicit training reward must agree')
        if demonstration_ppo != saved.get('demonstration_ppo',False):
            raise ValueError('explicit demonstration checkpoint and training path must agree')
        if neutral_contact_ppo != saved.get('neutral_contact_ppo',False):
            raise ValueError('explicit neutral checkpoint and task must agree')
        if credit_horizon_ppo != saved.get('credit_horizon_ppo',False):
            raise ValueError('explicit credit-horizon checkpoint and training path must agree')
        if mixed_partner_ppo != saved.get('mixed_partner_ppo',False):
            raise ValueError('explicit mixed checkpoint and training task must agree')
        if fixed_opponent_ppo != saved.get('fixed_opponent_ppo',False):
            raise ValueError('fixed opponent checkpoint and training path must agree')
        if posture_cost_ppo != saved.get('posture_cost_ppo',False):
            raise ValueError('posture-cost checkpoint and explicit training reward must agree')
        if reduced_noise_ppo != saved.get('reduced_noise_ppo',False):
            raise ValueError('sampling profile checkpoint and explicit on-policy distribution must agree')
        if kl_guard_ppo != saved.get('kl_guard_ppo',False):
            raise ValueError('KL guard checkpoint and explicit update protocol must agree')
        if long_gae_ppo != saved.get('long_gae_ppo',False):
            raise ValueError('long GAE checkpoint and explicit advantage profile must agree')
        if pool_opponent_ppo != saved.get('pool_opponent_ppo',False):
            raise ValueError('opponent pool checkpoint and explicit training task must agree')
        if conservative_actor_ppo != saved.get('conservative_actor_ppo',False):
            raise ValueError('explicit conservative checkpoint and training path must agree')
        if conservative_actor_ppo and saved['actor_learning_rate']!=args.actor_learning_rate:
            raise ValueError('checkpoint and requested actor rate disagree')
        if credit_horizon_ppo and saved['training_gamma']!=args.training_gamma:
            raise ValueError('checkpoint and requested discount disagree')
        if demonstration_ppo and saved['retention_loss_weight']!=args.retention_loss_weight:
            raise ValueError('retention checkpoint and requested coefficient disagree')
        if contact_ppo != (saved.get('optimizer_kind')=='split_adam_v1'):
            raise ValueError('optimizer format requires the matching explicit training path')
        for key in ('actor_inputs','actions','contact_history','learners_per_world','opponent_mode','anchor_loss_weight'):
            default=2 if key=='learners_per_world' else False if key=='contact_history' else None
            if key in saved and saved[key]!=interface_metadata.get(key,default):
                raise ValueError('checkpoint and training interface disagree')
        if saved['foundation_sha256']!=metadata['sha256']:
            raise ValueError('boxing motor provenance mismatch')
        ap,vp,optimizer,norm,key=(saved[k] for k in ('actor','critic','optimizer','norm','key'))
        optimized=saved['optimized_steps']
        progress_path=args.output/'progress.json'
        progress=json.loads(progress_path.read_text())
        consumed=max(saved['consumed_steps'],progress['consumed_steps'])
        # Older two-learner payloads used an implicit physical counter. Preserve
        # that prefix when an explicitly cloned diagnostic changes learner count.
        world_controls=max(saved.get('world_control_steps',saved['consumed_steps']//2),
                           progress.get('world_control_steps',progress['consumed_steps']//2))
        if contact_ppo:continuation=dict(saved)
    elif args.resume:
        if contact_ppo:raise ValueError('explicit migrated contact checkpoint required')
        # A failure before the first PPO update still consumed simulation.
        # Keep that ledger when recreating the original neutral head.
        progress=json.loads((args.output/'progress.json').read_text())
        consumed=progress['consumed_steps']
        world_controls=progress.get('world_control_steps',consumed//learners_per_world)
    rows=args.worlds*learners_per_world
    block=rows*args.unroll
    if block%4:
        raise ValueError('rollout must divide into four minibatches')
    update_audit=None
    if getattr(args,'update_audit_directory',None) is not None:
        if not mixed_partner_ppo or args.ceiling-consumed!=block:
            raise ValueError('passive update diagnostic is bounded to exactly one mixed rollout')
        from boxer.striking_update_audit import UpdateAudit
        update_audit=UpdateAudit(actor,log_probability,args.update_audit_directory)
    stopping=[False]
    def stop(signum,frame):
        stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if contact_ppo:
        from boxer.striking_anchor import load_anchor_candidates
        anchor_data,anchor_manifest=load_anchor_candidates()
        if anchor_manifest['sha256']!=continuation['anchor_dataset_sha256']:
            raise ValueError('auxiliary anchor provenance mismatch')
        aux_rng=np.random.default_rng();aux_rng.bit_generator.state=continuation['aux_sampler_state']
        if demonstration_ppo:
            from boxer.striking_retention import load_retention_data,validate_retention_state,retention_batch
            retention_data,retention_provenance=load_retention_data()
            validate_retention_state(continuation,retention_provenance)
            retention_rng=np.random.default_rng();retention_rng.bit_generator.state=continuation['retention_sampler_state']

    @jax.jit
    def act(parameters, obs, rng):
        mean,log_std=actor.apply(parameters,obs)
        raw=mean+jp.exp(log_std)*jax.random.normal(rng,mean.shape)
        return jp.tanh(raw),raw,log_probability(raw,mean,log_std)

    @jax.jit
    def value(parameters,obs):
        return critic.apply(parameters,obs)

    @jax.jit
    def mean_action(parameters,obs):
        return jp.tanh(actor.apply(parameters,obs)[0])

    update=make_ppo_update(actor,critic,tx,args.anchor_loss_weight if contact_ppo else None,
        args.retention_loss_weight if demonstration_ppo else None)
    if kl_guard_ppo:
        from boxer.striking_kl_guard import mean_rollout_kl
        critic_update=make_ppo_update(actor,critic,tx,args.anchor_loss_weight,args.retention_loss_weight,actor_enabled=False)
        predict_distribution=jax.jit(lambda params,observations:actor.apply(params,observations))
        guard_steps=list(continuation['kl_guard_actor_steps_by_rollout'])
        last_guard={}

    envs=[env_class(foundation,motor_params,**profile,
        **({'active_opponent':i in MIXED_ACTIVE_WORLDS} if mixed_partner_ppo else {})) for i in range(args.worlds)]
    # Share the already compiled frozen network across worlds.
    for env in envs[1:]:env.motor=envs[0].motor
    episode=np.full(args.worlds,consumed//rows,dtype=np.int64)
    initial_episode_index=consumed//rows
    obs=np.stack([env.reset(args.seed+1009*i+int(episode[i]),swap=bool(i%2)) for i,env in enumerate(envs)]).reshape(rows,observation_size)
    if motion_cycle_ppo:
        from boxer.striking_motion_reward import MotionCycleMonitor,replace_contact_payment
        monitor_class=MotionCycleMonitor
        if contact_progress_ppo:
            from boxer.striking_contact_progress import ContactProgressMonitor
            monitor_class=ContactProgressMonitor
        motion_monitors=[monitor_class() for env in envs]
        for env,monitor in zip(envs,motion_monitors):
            monitor.capture_completed_events=True
            env.env.pair.diagnostic_punch_trace=monitor
    if pool_opponent_ppo:
        from boxer.striking_opponent_pool import (restore_sampler, draw_opponent,
            validate_pool_ledger, pool_predictor, OFFSET)
        pool_frontier=progress if progress.get('consumed_steps',0)>continuation['consumed_steps'] else continuation
        pool_credits=validate_pool_ledger(pool_frontier['pool_credits_by_opponent_and_corner'],
                                        pool_frontier['mixed_credits_by_mode_and_corner']).copy()
        pool_rng=restore_sampler(pool_frontier['pool_sampler_state'])
        pool_assignments=np.asarray(pool_frontier['pool_assignments'],dtype=np.int64).copy()
        if consumed>OFFSET:
            pool_assignments=np.asarray([draw_opponent(pool_rng,env.active_opponent) for env in envs])
        predict_pool=pool_predictor(actor,continuation['opponent_pool'])
    state=dict(status='running',pid=os.getpid(),started_utc=utc(),ceiling=args.ceiling,
        foundation_steps=metadata['total_steps'],foundation_sha256=metadata['sha256'],
        assistance=False,optimizer_saved=True,physics_state_saved=False,
        discarded_collected_steps=consumed-optimized,**interface_metadata,**profile)
    started=time.monotonic()
    session_start=consumed
    completed_rounds=falls=0
    neutral_credits=np.zeros(2,dtype=np.int64)
    mixed_credits=np.zeros((2,2),dtype=np.int64)
    if neutral_contact_ppo:
        neutral_credits=np.maximum(np.asarray(continuation['neutral_credits_by_corner'],dtype=np.int64),
            np.asarray(progress.get('neutral_credits_by_corner',[0,0]),dtype=np.int64))
    if mixed_partner_ppo:
        mixed_credits=np.maximum(np.asarray(continuation['mixed_credits_by_mode_and_corner'],dtype=np.int64),
            np.asarray(progress.get('mixed_credits_by_mode_and_corner',[[0,0],[0,0]]),dtype=np.int64))
    scored=np.zeros(2)
    motion_cycles=np.asarray(progress.get('motion_cycles_by_corner',[0.,0.]),float) if motion_cycle_ppo else np.zeros(2)
    motion_own=float(progress.get('motion_learner_cycles_total',0.)) if motion_cycle_ppo else 0.
    motion_other=float(progress.get('motion_opponent_cycles_total',0.)) if motion_cycle_ppo else 0.
    progress_potential=0.;progress_contact=0.
    motion_dense=0.
    posture_cost_sum=0.
    arm_effort_sum=0.
    def record(status='running',**extra):
        progress=dict(consumed_steps=consumed,optimized_steps=optimized,
            world_control_steps=world_controls,combined_project_consumed_steps=metadata['total_steps']+consumed,
            updated_utc=utc(),rounds=completed_rounds,fall_rounds=falls,scored_by_corner=scored.tolist(),
            actor_steps_per_second=(consumed-session_start)/max(time.monotonic()-started,1e-6),**extra)
        if contact_ppo:
            progress['combined_foundation_and_lineage_steps']=progress.pop('combined_project_consumed_steps')
            progress.update(actor_adam_count=int(optimizer['actor_adam'][0].count),
                critic_adam_count=int(optimizer['critic_adam'][0].count))
        if neutral_contact_ppo:progress['neutral_credits_by_corner']=neutral_credits.tolist()
        if mixed_partner_ppo:progress['mixed_credits_by_mode_and_corner']=mixed_credits.tolist()
        if pool_opponent_ppo:
            validate_pool_ledger(pool_credits,mixed_credits)
            progress.update(pool_credits_by_opponent_and_corner=pool_credits.tolist(),
                pool_sampler_state=pool_rng.bit_generator.state,pool_assignments=pool_assignments.tolist())
        if motion_cycle_ppo:progress.update(motion_cycles_by_corner=motion_cycles.tolist(),motion_learner_cycles_total=motion_own,motion_opponent_cycles_total=motion_other,session_motion_dense_reward=motion_dense)
        if contact_progress_ppo:progress.update(session_target_potential_reward=progress_potential,session_physical_contact_progress_reward=progress_contact)
        if posture_cost_ppo:progress['session_posture_cost']=posture_cost_sum
        if arm_effort_ppo:progress['session_arm_effort_cost']=arm_effort_sum
        if kl_guard_ppo:progress.update(kl_guard_actor_steps_total=sum(guard_steps),last_kl_guard=last_guard)
        atomic_json(args.output/'progress.json',progress)
        atomic_json(args.output/'state.json',dict(state,status=status,**progress))
    def publish():
        if args.publish_dir is None:
            return
        if args.publish_dir.resolve()==args.output.resolve():
            raise ValueError('publish directory must differ from the durable output')
        args.publish_dir.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','state.json','progress.json'):
            atomic_bytes(args.publish_dir/name,(args.output/name).read_bytes())
        if checkpoint_path.exists():
            manifest=json.loads(checkpoint_path.read_text())
            destination=args.publish_dir/'checkpoint'
            atomic_bytes(destination/manifest['file'],(checkpoint_path.parent/manifest['file']).read_bytes())
            atomic_json(destination/'latest.json',manifest)
    record()
    publish()
    while consumed+block <= args.ceiling and not stopping[0]:
        storage={k:[] for k in ('obs','raw','logp','reward','done','value')}
        for t in range(args.unroll):
            normalized=normalize(obs,norm)
            key,sub=jax.random.split(key)
            action,raw,logp=act(ap,jp.array(normalized),sub)
            predicted=np.asarray(value(vp,jp.array(normalized)))
            actions=np.asarray(action).reshape(args.worlds,learners_per_world,action_size)
            if mixed_partner_ppo:
                other_obs=np.stack([env.opponent_observation for env in envs])
                if pool_opponent_ppo:
                    opponent_actions=np.asarray(predict_pool(jp.asarray(other_obs),jp.asarray(pool_assignments)))
                else:
                    from boxer.striking_fixed_opponent import opponent_state
                    opponent_params,opponent_norm=opponent_state(continuation,ap,norm)
                    opponent_actions=np.asarray(mean_action(opponent_params,jp.array(normalize(other_obs,opponent_norm))))
            rewards=[];dones=[];following=[]
            for i,env in enumerate(envs):
                consumed+=learners_per_world
                world_controls+=1
                if mixed_partner_ppo:mixed_credits[int(env.active_opponent),env.learner_corner]+=1
                elif neutral_contact_ppo:neutral_credits[env.learner_corner]+=1
                if pool_opponent_ppo and env.active_opponent:
                    pool_credits[pool_assignments[i],env.learner_corner]+=1
                # Count attempted physics even if a later world/update fails.
                record(phase='collecting')
                if motion_cycle_ppo:motion_monitors[i].control_policy_optimized_steps=optimized
                if mixed_partner_ppo:
                    next_obs,reward,done,outcome=env.step(actions[i],
                        opponent_actions[i] if env.active_opponent else None)
                else:next_obs,reward,done,outcome=env.step(actions[i])
                if motion_cycle_ppo:
                    cycles,dense=motion_monitors[i].consume(terminal=done)
                    reward=replace_contact_payment(reward,outcome,cycles,dense,env.learner_corner)
                    from boxer.striking_event_evidence import save_completed_evidence
                    save_completed_evidence(motion_monitors[i],args.output/'motion-event-evidence',checkpoint_path.parent,
                        consumed=consumed,world=i,learner_corner=env.learner_corner,
                        reset_seed=args.seed+1009*i+int(episode[i]),
                        swap=bool(i%2) if episode[i]==initial_episode_index else bool(episode[i]%2),
                        active_opponent=env.active_opponent,foundation_sha256=metadata['sha256'])
                    if contact_progress_ppo:
                        parts=motion_monitors[i].last_components
                        progress_potential+=float(parts['target_potential'][env.learner_corner])
                        progress_contact+=float(parts['physical_contact_progress'][env.learner_corner])
                        if contact_rebalance_ppo:
                            from boxer.striking_contact_rebalance import correction
                            reward=reward+correction(parts,env.learner_corner)
                    motion_cycles+=cycles;motion_own+=float(cycles[env.learner_corner]);motion_other+=float(cycles[1-env.learner_corner]);motion_dense+=float(dense[env.learner_corner])
                if posture_cost_ppo:
                    from boxer.striking_posture_cost import learner_cost
                    cost=learner_cost(env);posture_cost_sum+=cost
                    reward=reward-cost
                if arm_effort_ppo:
                    from boxer.striking_arm_effort import effort_cost
                    cost=effort_cost(outcome,env.learner_corner,env.env.pair.dt)
                    arm_effort_sum+=cost;reward=reward-cost
                rewards.extend(reward);dones.extend([done]*learners_per_world)
                scored+=outcome['scored']
                if done:
                    completed_rounds+=1;falls+=int(outcome['done'])
                    episode[i]+=1
                    next_obs=env.reset(args.seed+1009*i+int(episode[i]),swap=bool(episode[i]%2))
                    if motion_cycle_ppo:
                        motion_monitors[i]=monitor_class()
                        motion_monitors[i].capture_completed_events=True
                        env.env.pair.diagnostic_punch_trace=motion_monitors[i]
                    if pool_opponent_ppo:pool_assignments[i]=draw_opponent(pool_rng,env.active_opponent)
                following.append(next_obs)
            for k,v in dict(obs=obs.copy(),raw=np.asarray(raw),logp=np.asarray(logp),
                reward=np.asarray(rewards),done=np.asarray(dones,dtype=np.float32),value=predicted).items():
                storage[k].append(v)
            obs=np.asarray(following).reshape(rows,observation_size)
        rollout={k:np.asarray(v) for k,v in storage.items()}
        bootstrap=np.asarray(value(vp,jp.array(normalize(obs,norm))))
        adv,returns=advantages(rollout['reward'],rollout['done'],rollout['value'],bootstrap,
            gamma=config['gamma'],lam=config['gae_lambda'])
        flat_obs=rollout['obs'].reshape(block,observation_size)
        batch=dict(obs=normalize(flat_obs,norm),raw=rollout['raw'].reshape(block,action_size),
            logp=rollout['logp'].reshape(block),adv=((adv-adv.mean())/(adv.std()+1e-8)).reshape(block),
            returns=returns.reshape(block))
        losses=[]
        if kl_guard_ppo:
            guard_observations=jp.array(batch['obs'])
            guard_old=tuple(np.asarray(x) for x in predict_distribution(ap,guard_observations))
            actor_active=True;executed_actor_steps=0
        if update_audit is not None:
            update_audit.begin(ap,batch,np.tile([int(i in MIXED_ACTIVE_WORLDS) for i in range(args.worlds)],args.unroll))
        for epoch in range(4):
            key,sub=jax.random.split(key)
            order=np.asarray(jax.random.permutation(sub,block))
            for minibatch_index,indices in enumerate(np.split(order,4)):
                minibatch={k:jp.array(v[indices]) for k,v in batch.items()}
                if contact_ppo:
                    ai=aux_rng.integers(0,len(anchor_data['observation']),64)
                    minibatch.update(anchor_obs=jp.array(normalize(anchor_data['observation'][ai],norm)),
                        anchor_labels=jp.array(anchor_data['action'][ai]))
                    if demonstration_ppo:minibatch.update(retention_batch(retention_data,retention_rng,norm))
                selected_update=update
                if kl_guard_ppo:
                    kl=mean_rollout_kl(guard_old,predict_distribution(ap,guard_observations))
                    if kl>config['kl_guard_target']:actor_active=False
                    selected_update=update if actor_active else critic_update
                    executed_actor_steps+=int(actor_active)
                ap,vp,optimizer,loss=selected_update(ap,vp,optimizer,minibatch)
                losses.append(float(loss))
                if update_audit is not None:update_audit.observe(ap,epoch,minibatch_index,loss=float(loss))
        if contact_ppo and not np.isfinite(losses).all():raise FloatingPointError('nonfinite contact PPO loss')
        if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((ap,vp,optimizer))):
            raise FloatingPointError('non-finite boxing update; prior checkpoint retained')
        norm=update_norm(norm,flat_obs)
        if kl_guard_ppo:
            guard_steps.append(executed_actor_steps)
            last_guard=dict(actor_steps=executed_actor_steps,critic_steps=16,
                kl_before_normalizer=mean_rollout_kl(guard_old,predict_distribution(ap,guard_observations)),
                kl_after_normalizer=mean_rollout_kl(guard_old,predict_distribution(ap,jp.array(normalize(flat_obs,norm)))))
        if update_audit is not None:update_audit.after_normalizer(ap,normalize(flat_obs,norm))
        optimized+=block
        saved=dict(continuation,actor=ap,critic=vp,optimizer=optimizer,norm=norm,key=key,
            optimized_steps=optimized,consumed_steps=consumed,world_control_steps=world_controls,
            foundation_sha256=metadata['sha256'],**interface_metadata,**profile)
        if contact_ppo:
            if kl_guard_ppo:saved['kl_guard_actor_steps_by_rollout']=list(guard_steps)
            saved['aux_sampler_state']=aux_rng.bit_generator.state
            if neutral_contact_ppo:saved['neutral_credits_by_corner']=neutral_credits.tolist()
            if mixed_partner_ppo:saved['mixed_credits_by_mode_and_corner']=mixed_credits.tolist()
            if pool_opponent_ppo:
                saved.update(pool_credits_by_opponent_and_corner=pool_credits.tolist(),
                    pool_sampler_state=pool_rng.bit_generator.state,pool_assignments=pool_assignments.tolist())
            if demonstration_ppo:
                saved['retention_sampler_state']=retention_rng.bit_generator.state
                validate_retention_state(saved,retention_provenance)
            require_ppo_checkpoint(saved)
        payload=pickle.dumps(jax.device_get(saved),protocol=pickle.HIGHEST_PROTOCOL)
        name=f'step-{consumed}.pkl'
        atomic_bytes(checkpoint_path.parent/name,payload)
        checkpoint_metadata=dict(interface_metadata)
        if contact_ppo:checkpoint_metadata.update(training_kind='ppo',ppo_optimizer_compatible=True,migration_version=1)
        atomic_json(checkpoint_path,dict(file=name,sha256=hashlib.sha256(payload).hexdigest(),
            consumed_steps=consumed,optimized_steps=optimized,world_control_steps=world_controls,
            foundation_sha256=metadata['sha256'],**checkpoint_metadata,**profile))
        record(phase='updated',mean_loss=float(np.mean(losses)))
        publish()
        print(json.dumps(json.loads((args.output/'progress.json').read_text())),flush=True)
    record(status='stopped' if stopping[0] else 'budget_reached',phase='stopped')
    publish()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
