"""Explicit actor-only step-size change preserving every learned/Adam prefix."""
import copy

SOURCE_SHA='8c99c2dfe21cd6c94a90ed31c487620118df5b0d0fdfe430eabe2471745a31c2'


def validate_conservative_metadata(saved):
    expected=dict(source_sha256=SOURCE_SHA,consumed_steps=1835008,world_control_steps=1212416,
        actor_count=11144,critic_count=34816,old_actor_learning_rate=.0001,
        new_actor_learning_rate=.00001,critic_learning_rate=.0001,optimizer_reset=False)
    if (saved.get('conservative_actor_ppo') is not True or not saved.get('mixed_partner_ppo')
            or saved.get('actor_learning_rate')!=.00001 or saved.get('conservative_actor_transition')!=expected
            or saved['consumed_steps']<1835008 or saved['optimized_steps']<1835008
            or int(saved['optimizer']['actor_adam'][0].count)<11144
            or int(saved['optimizer']['critic_adam'][0].count)<34816):
        raise ValueError('explicit conservative actor source, rate and retained optimizer prefixes required')


def prepare_conservative_state(saved,sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha!=SOURCE_SHA or saved.get('conservative_actor_ppo') or not saved.get('mixed_partner_ppo')
            or saved['consumed_steps']!=saved['optimized_steps'] or saved['consumed_steps']!=1835008
            or saved['world_control_steps']!=1212416
            or int(saved['optimizer']['actor_adam'][0].count)!=11144
            or int(saved['optimizer']['critic_adam'][0].count)!=34816):
        raise ValueError('latest fully evaluated mixed final required; no source rollback')
    result=copy.deepcopy(saved)
    result.update(conservative_actor_ppo=True,actor_learning_rate=.00001,
        conservative_actor_transition=dict(source_sha256=sha,consumed_steps=1835008,
            world_control_steps=1212416,actor_count=11144,critic_count=34816,
            old_actor_learning_rate=.0001,new_actor_learning_rate=.00001,
            critic_learning_rate=.0001,optimizer_reset=False))
    require_ppo_checkpoint(result)
    return result


def conservative_config(source):
    if source['version']!=6 or source['learning_rate']!=.0001 or not source.get('mixed_partner_ppo'):
        raise ValueError('unchanged reviewed mixed source configuration required')
    return dict(source,version=7,conservative_actor_ppo=True,actor_learning_rate=.00001)
