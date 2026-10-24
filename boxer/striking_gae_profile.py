"""Explicit longer GAE trace, retaining the latest original-control optimizer."""
import copy

SOURCE_SHA = '24e23288b073843637d5210a5a5d4a03f3383c47580fe004832cfd0d3a50d423'
TRANSITION = dict(source_sha256=SOURCE_SHA, consumed_steps=1966080,
    world_control_steps=1343488, actor_count=13192, critic_count=36864,
    old_lambda=.95, new_lambda=.99, optimizer_reset=False)


def validate_gae_metadata(saved):
    if (saved.get('long_gae_ppo') is not True or saved.get('training_gae_lambda') != .99
            or saved.get('long_gae_transition') != TRANSITION
            or not saved.get('fixed_opponent_ppo') or saved.get('kl_guard_ppo')
            or saved.get('posture_cost_ppo') or saved.get('reduced_noise_ppo')
            or saved['optimized_steps'] < 1966080 or saved['consumed_steps'] < 1966080):
        raise ValueError('explicit latest original-control GAE transition required')


def prepare_gae_state(saved, sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha != SOURCE_SHA or saved.get('long_gae_ppo') or saved.get('kl_guard_ppo')
            or saved.get('posture_cost_ppo') or saved.get('reduced_noise_ppo')
            or saved['consumed_steps'] != saved['optimized_steps'] or saved['optimized_steps'] != 1966080
            or saved['world_control_steps'] != 1343488
            or int(saved['optimizer']['actor_adam'][0].count) != 13192
            or int(saved['optimizer']['critic_adam'][0].count) != 36864):
        raise ValueError('latest completed control required without optimizer reset or rollback')
    result = copy.deepcopy(saved)
    result.update(long_gae_ppo=True, training_gae_lambda=.99,
        long_gae_transition=copy.deepcopy(TRANSITION))
    require_ppo_checkpoint(result)
    return result


def gae_config(source):
    if source['version'] != 8 or source['gae_lambda'] != .95 or source['gamma'] != .999:
        raise ValueError('inherited original fixed-opponent profile required')
    return dict(source, version=12, long_gae_ppo=True, training_gae_lambda=.99, gae_lambda=.99)
