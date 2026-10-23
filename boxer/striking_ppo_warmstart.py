"""Versioned BC-to-PPO migration with independently counted Adam histories."""
import copy

import jax
import jax.numpy as jp
import numpy as np
import optax

OPTIMIZER_KIND='split_adam_v1'


class SplitAdam:
    """One joint gradient clip, followed by two independently counted Adam states."""
    def __init__(self,actor_learning_rate=.0001):
        self.clip=optax.clip_by_global_norm(1.)
        self.adam=optax.adam(.0001)
        self.actor_adam=optax.adam(actor_learning_rate)

    def init(self,params):
        return dict(actor_adam=self.actor_adam.init(params[0]),critic_adam=self.adam.init(params[1]))

    def update(self,grads,state,params):
        clipped,_=self.clip.update(grads,optax.EmptyState())
        actor,sa=self.actor_adam.update(clipped[0],state['actor_adam'],params[0])
        critic,sv=self.adam.update(clipped[1],state['critic_adam'],params[1])
        return (actor,critic),dict(actor_adam=sa,critic_adam=sv)


def anchor_loss(actor,params,obs,labels):
    prediction=jp.tanh(actor.apply(params,obs)[0])
    channels=jp.array([2.]*3+[4.]*7+[1.]*7)
    return jp.mean((prediction-labels)**2*channels)


def validate_split_checkpoint(saved):
    neutral=saved.get('neutral_contact_ppo') is True
    mixed=saved.get('mixed_partner_ppo') is True
    expected=(1,'mixed_current_mean') if mixed else (1,'neutral_motor') if neutral else (2,'active_selfplay')
    if (saved.get('training_kind')!='ppo' or saved.get('ppo_optimizer_compatible') is not True or
            saved.get('optimizer_kind')!=OPTIMIZER_KIND or saved.get('migration_version')!=1 or
            (saved.get('actor_inputs'),saved.get('actions'))!=(148,17) or not saved.get('contact_history') or
            (saved.get('learners_per_world'),saved.get('opponent_mode'))!=expected):
        raise ValueError('invalid explicit split-Adam PPO interface')
    if mixed:
        from boxer.striking_mixed_partner import validate_mixed_metadata
        if not neutral:raise ValueError('mixed task requires preserved neutral lineage')
        validate_mixed_metadata(saved)
    if neutral:
        transition=saved.get('neutral_partner_transition',{})
        sha=transition.get('source_sha256','')
        credits=np.asarray(saved.get('neutral_credits_by_corner',[]))
        if (transition.get('consumed_steps')!=1638400 or transition.get('world_control_steps')!=1015808 or
                transition.get('actor_count')!=8072 or transition.get('critic_count')!=31744 or
                transition.get('optimizer_reset') is not False or len(sha)!=64 or
                any(c not in '0123456789abcdef' for c in sha) or
                credits.shape!=(2,) or credits.dtype.kind not in 'iu' or np.any(credits<0) or
                credits.sum()!=(1736704 if mixed else saved['consumed_steps'])-1638400):
            raise ValueError('explicit neutral task transition and exact learner-credit accounting required')
    migration=saved.get('optimizer_migration',{})
    if migration.get('actor_count_at_migration')!=5000 or migration.get('critic_count_at_migration',-1)<0:
        raise ValueError('reviewed optimizer prefixes required')
    for name in ('actor_source_sha256','critic_optimizer_source_sha256'):
        value=migration.get(name,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('missing optimizer source provenance')
    if set(saved.get('optimizer',{}))!={'actor_adam','critic_adam'}:
        raise ValueError('split optimizer states required')
    for name,params in (('actor',saved['actor']),('critic',saved['critic'])):
        state=saved['optimizer'][name+'_adam']
        if len(state)!=2 or not hasattr(state[0],'count'):raise ValueError('Adam state format mismatch')
        count=np.asarray(state[0].count)
        prefix=migration.get(name+'_count_at_migration')
        if neutral:prefix=max(prefix or 0,8072 if name=='actor' else 31744)
        if count.shape or count.dtype.kind not in 'iu' or prefix is None or int(count)<prefix:
            raise ValueError('optimizer count lost or invalid')
        for moments in (state[0].mu,state[0].nu):
            if jax.tree_util.tree_structure(moments)!=jax.tree_util.tree_structure(params):
                raise ValueError('optimizer parameter tree mismatch')
            for a,b in zip(jax.tree_util.tree_leaves(moments),jax.tree_util.tree_leaves(params)):
                if np.shape(a)!=np.shape(b) or not np.isfinite(a).all():raise ValueError('optimizer leaf mismatch')
        if any(np.any(x<0) for x in jax.tree_util.tree_leaves(state[0].nu)):
            raise ValueError('negative second moment')
    for name in ('bc_sampler_state','aux_sampler_state'):
        if name not in saved:raise ValueError('missing preserved sampler state')
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(saved[name])
    if saved.get('sampler_state')!=saved['bc_sampler_state']:
        raise ValueError('BC sampler provenance changed')
    if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(
            tuple(saved[k] for k in ('actor','critic','norm','key')))):
        raise FloatingPointError('nonfinite migrated state')
    if (saved['norm']['mean'].shape!=(148,) or saved['norm']['var'].shape!=(148,) or
            np.any(saved['norm']['var']<0) or saved['norm']['count']<migration.get('normalizer_count_at_migration',np.inf)):
        raise ValueError('normalizer shape/count/provenance mismatch')


def migrate_supervised_state(saved,original,actor_sha,critic_sha):
    """Keep the latest actor/Adam and recover only the unchanged critic's Adam."""
    # The reviewed source is the next completed generation of that BC interface.
    if (saved.get('training_kind')!='supervised_motor_anchor' or saved.get('supervised_updates')!=5000 or
            int(saved['optimizer'][1][0].count)!=5000 or saved.get('anchor_training_arm')!='zero_anchor' or
            saved.get('ppo_optimizer_compatible') is not False or 'sampler_state' not in saved):
        raise ValueError('reviewed latest BC5000 treatment required')
    if original.get('training_kind','ppo')!='ppo' or original['norm']['mean'].shape!=(146,):
        raise ValueError('original joint-Adam PPO critic source required')
    def pad(old,template):
        old=np.asarray(old);template=np.asarray(template)
        if old.shape==template.shape:return old.copy()
        if old.shape!=(146,128) or template.shape!=(148,128):raise ValueError('unexpected critic expansion')
        return np.r_[old,np.zeros((2,128),dtype=old.dtype)]
    expected=jax.tree_util.tree_map(pad,original['critic'],saved['critic'])
    for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(saved['critic'])):
        np.testing.assert_array_equal(a,b)
    for name in ('mean','var'):np.testing.assert_array_equal(original['norm'][name],saved['norm'][name][:146])
    if original['norm']['count']!=saved['norm']['count']:raise ValueError('critic normalizer source mismatch')
    for name in ('consumed_steps','optimized_steps','world_control_steps','foundation_sha256'):
        if saved[name]!=original[name]:raise ValueError('inherited critic lineage mismatch')
    old=original['optimizer'][1][0]
    if not isinstance(old.mu,tuple) or len(old.mu)!=2:raise ValueError('joint critic optimizer required')
    mu=jax.tree_util.tree_map(pad,old.mu[1],saved['critic'])
    nu=jax.tree_util.tree_map(pad,old.nu[1],saved['critic'])
    critic=old._replace(mu=mu,nu=nu)
    result=copy.deepcopy(saved)
    result.update(training_kind='ppo',ppo_optimizer_compatible=True,optimizer_kind=OPTIMIZER_KIND,migration_version=1,
        optimizer=dict(actor_adam=copy.deepcopy(saved['optimizer'][1]),
                       critic_adam=(critic,copy.deepcopy(original['optimizer'][1][1]))),
        learners_per_world=2,opponent_mode='active_selfplay',
        bc_sampler_state=copy.deepcopy(saved['sampler_state']),aux_sampler_state=copy.deepcopy(saved['sampler_state']),
        optimizer_migration=dict(actor_source_sha256=actor_sha,critic_optimizer_source_sha256=critic_sha,
            actor_count_at_migration=5000,critic_count_at_migration=int(old.count),
            normalizer_count_at_migration=float(saved['norm']['count']),
            source_learners_per_world=saved['learners_per_world'],source_opponent_mode=saved['opponent_mode']))
    validate_split_checkpoint(result)
    return result


def contact_config(source,weight):
    keys=('foundation_sha256','foundation_main_steps','worlds','unroll','seed','arm_residual_radians',
          'command_offsets','assistance','dt','physics_dt','round_steps','learning_rate','epochs','minibatches',
          'gamma','gae_lambda','scorecard_version','engagement_distance','engagement_reward_distance')
    if weight not in (0.,.02):raise ValueError('only reviewed auxiliary weights allowed')
    result={k:source[k] for k in keys}
    if result['engagement_distance']!=.45 or result['engagement_reward_distance']!=.45:
        raise ValueError('reviewed engagement profile required')
    result.update(version=2,actor_inputs=148,actions=17,contact_history=True,learners_per_world=2,
        opponent_mode='active_selfplay',optimizer_kind=OPTIMIZER_KIND,anchor_loss_weight=weight)
    return result
