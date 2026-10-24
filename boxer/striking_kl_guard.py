"""Explicit actor-only rollout KL stop with an immutable source and Adam ledger."""
import copy
import numpy as np
from boxer.striking_policy_diagnostics import gaussian_kl

SOURCE_SHA='97af578363d9a8e43c2331546d07a85437e44a02a4673b50d7c1035352f09721'
TARGET=.02
TRANSITION=dict(source_sha256=SOURCE_SHA,consumed_steps=1949696,world_control_steps=1327104,
    actor_count=12936,critic_count=36608,optimizer_reset=False,critic_steps_per_rollout=16)


def validate_guard_metadata(saved):
    counts=np.asarray(saved.get('kl_guard_actor_steps_by_rollout',[]))
    delta=saved['optimized_steps']-1949696
    if (saved.get('kl_guard_ppo') is not True or not saved.get('fixed_opponent_ppo') or saved.get('reduced_noise_ppo')
            or 'kl_guard_actor_steps_by_rollout' not in saved
            or saved.get('posture_cost_ppo') or saved.get('kl_guard_target')!=TARGET
            or saved.get('kl_guard_transition')!=TRANSITION or saved['consumed_steps']<1949696
            or delta<0 or delta%1024 or counts.shape!=(delta//1024,)
            or (counts.size and (counts.dtype.kind not in 'iu' or np.any(counts<1) or np.any(counts>16)))
            or int(saved['optimizer']['actor_adam'][0].count)!=12936+int(counts.sum())):
        raise ValueError('explicit KL guard source, retained optimizer and exact actor-step ledger required')


def expected_actor_count(saved):
    validate_guard_metadata(saved)
    return 12936+sum(saved['kl_guard_actor_steps_by_rollout'])


def mean_rollout_kl(old,new):
    value=float(gaussian_kl(*old,*new).sum(axis=1).mean())
    if not np.isfinite(value):raise FloatingPointError('nonfinite actor rollout KL')
    return value


def prepare_guard_state(saved,sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha!=SOURCE_SHA or saved.get('kl_guard_ppo') or saved.get('reduced_noise_ppo') or saved.get('posture_cost_ppo')
            or saved['consumed_steps']!=saved['optimized_steps'] or saved['optimized_steps']!=1949696
            or saved['world_control_steps']!=1327104 or int(saved['optimizer']['actor_adam'][0].count)!=12936
            or int(saved['optimizer']['critic_adam'][0].count)!=36608):
        raise ValueError('latest completed original-covariance control required without rollback')
    result=copy.deepcopy(saved)
    result.update(kl_guard_ppo=True,kl_guard_target=TARGET,kl_guard_transition=copy.deepcopy(TRANSITION),kl_guard_actor_steps_by_rollout=[])
    require_ppo_checkpoint(result);return result


def guard_config(source):
    if source['version']!=8 or not source['fixed_opponent_ppo']:raise ValueError('inherited version-8 profile required')
    return dict(source,version=11,kl_guard_ppo=True,kl_guard_target=TARGET)
