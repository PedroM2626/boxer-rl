"""Training-only smooth arm-effort cost, with full action authority and physical gates."""
import copy
import numpy as np

SOURCE = 'checkpoints/striking-validation/contact-rebalance-v16/step-2097152'
SOURCE_SHA = '5bbf6a38aa237633bf1f84893587dd1c369d9bbe8849c171b5735f9d6920a8b5'
PROFILE = dict(version=1, kind='bounded_arm_effort', weight_per_second=2.,
               training_reward_only=True, action_authority_unchanged=True, full_cycle_bonus=10.)
TRANSITION = dict(source_sha256=SOURCE_SHA, consumed_steps=7323648, optimized_steps=7323648,
                  world_control_steps=6701056, actor_count=96904, critic_count=120576,
                  optimizer_reset=False, parent_version=16, new_version=17)


def effort_cost(outcome, corner, dt):
    residual = np.asarray(outcome['arm_residual'], dtype=float)
    if (corner not in (0, 1) or residual.shape != (2, 14) or not np.isfinite(residual).all()
            or not np.isfinite(dt) or dt <= 0 or np.any(np.abs(residual) > 1.00001)):
        raise ValueError('bounded finite per-corner arm residuals and positive dt required')
    return float(PROFILE['weight_per_second'] * dt * np.mean(residual[corner] ** 2))


def validate_arm_effort_metadata(saved):
    if (saved.get('arm_effort_ppo') is not True or saved.get('arm_effort_profile') != PROFILE
            or saved.get('arm_effort_transition') != TRANSITION or not saved.get('contact_rebalance_ppo')
            or saved['optimized_steps'] < TRANSITION['optimized_steps'] or saved['consumed_steps'] < saved['optimized_steps']):
        raise ValueError('explicit v16 source and arm-effort provenance required')


def prepare_arm_effort_state(saved, sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha != SOURCE_SHA or saved.get('arm_effort_ppo') or not saved.get('contact_rebalance_ppo')
            or saved['consumed_steps'] != saved['optimized_steps'] or saved['optimized_steps'] != 7323648
            or saved['world_control_steps'] != 6701056 or int(saved['optimizer']['actor_adam'][0].count) != 96904
            or int(saved['optimizer']['critic_adam'][0].count) != 120576):
        raise ValueError('actual completed v16 final required without optimizer reset or rollback')
    result = copy.deepcopy(saved)
    result.update(arm_effort_ppo=True, arm_effort_profile=copy.deepcopy(PROFILE),
                  arm_effort_transition=copy.deepcopy(TRANSITION))
    require_ppo_checkpoint(result)
    return result


def arm_effort_config(source):
    if source['version'] != 16 or not source.get('contact_rebalance_ppo'):
        raise ValueError('completed v16 profile required')
    return dict(source, version=17, arm_effort_ppo=True, arm_effort_profile=copy.deepcopy(PROFILE))
