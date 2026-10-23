"""Explicit matched discount-horizon profiles over the same latest learned state."""
import copy

from boxer.striking_retention import validate_retention_state


def validate_horizon_state(saved):
    validate_retention_state(saved,saved['retention_provenance'])
    transition=saved.get('credit_horizon_transition',{})
    sha=transition.get('source_sha256','')
    if (saved.get('credit_horizon_ppo') is not True or saved.get('neutral_contact_ppo') is not True or
            saved.get('training_gamma') not in (.99,.999) or
            transition.get('old_gamma')!=.99 or transition.get('new_gamma')!=saved['training_gamma'] or
            transition.get('consumed_steps')!=1671168 or transition.get('world_control_steps')!=1048576 or
            transition.get('actor_count')!=8584 or transition.get('critic_count')!=32256 or
            transition.get('optimizer_reset') is not False or len(sha)!=64 or
            any(c not in '0123456789abcdef' for c in sha) or
            int(saved['optimizer']['actor_adam'][0].count)<8584 or
            int(saved['optimizer']['critic_adam'][0].count)<32256):
        raise ValueError('explicit matched horizon provenance and retained optimizer prefixes required')


def prepare_horizon_state(saved,source_sha,gamma):
    validate_retention_state(saved,saved['retention_provenance'])
    if (gamma not in (.99,.999) or not saved.get('neutral_contact_ppo') or
            saved['consumed_steps']!=saved['optimized_steps'] or saved['consumed_steps']!=1671168 or
            saved['world_control_steps']!=1048576 or saved['retention_loss_weight']!=1. or
            int(saved['optimizer']['actor_adam'][0].count)!=8584 or
            int(saved['optimizer']['critic_adam'][0].count)!=32256):
        raise ValueError('latest completed neutral-retention actor required for horizon comparison')
    result=copy.deepcopy(saved)
    result.update(credit_horizon_ppo=True,training_gamma=gamma,
        credit_horizon_transition=dict(source_sha256=source_sha,old_gamma=.99,new_gamma=gamma,
            consumed_steps=1671168,world_control_steps=1048576,actor_count=8584,critic_count=32256,
            optimizer_reset=False))
    validate_horizon_state(result)
    return result


def horizon_config(source,gamma):
    if source['version']!=4 or source['gamma']!=.99 or source['worlds']!=8 or gamma not in (.99,.999):
        raise ValueError('reviewed version-4 source and bounded gamma choices required')
    return dict(source,version=5,credit_horizon_ppo=True,training_gamma=gamma,gamma=gamma)
