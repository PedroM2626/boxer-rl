"""Versioned task change to one learned corner against a free neutral motor."""
import copy

from boxer.striking_retention import validate_retention_state


def prepare_neutral_state(saved,source_sha):
    validate_retention_state(saved,saved['retention_provenance'])
    if (saved['consumed_steps']!=saved['optimized_steps'] or saved['consumed_steps']!=1638400 or
            saved['world_control_steps']!=1015808 or saved['retention_loss_weight']!=1. or
            int(saved['optimizer']['actor_adam'][0].count)!=8072 or
            int(saved['optimizer']['critic_adam'][0].count)!=31744 or saved.get('neutral_contact_ppo')):
        raise ValueError('latest completed active-retention generation required')
    result=copy.deepcopy(saved)
    result.update(neutral_contact_ppo=True,learners_per_world=1,opponent_mode='neutral_motor',
        neutral_credits_by_corner=[0,0],neutral_partner_transition=dict(source_sha256=source_sha,
            consumed_steps=1638400,world_control_steps=1015808,actor_count=8072,critic_count=31744,
            optimizer_reset=False))
    validate_retention_state(result,result['retention_provenance'])
    return result


def neutral_config(source):
    if source['version']!=3 or source['retention_loss_weight']!=1. or source['worlds']!=4:
        raise ValueError('reviewed active retention profile required')
    return dict(source,version=4,neutral_contact_ppo=True,learners_per_world=1,
        opponent_mode='neutral_motor',worlds=8)
