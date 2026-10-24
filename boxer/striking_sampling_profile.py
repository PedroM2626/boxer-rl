"""Explicit on-policy covariance profile without resetting any learned leaves."""
import copy,math

SOURCE_SHA='c47edf0bb3abf0ad56259269d255a33fd100709f39a3abe79f314b0f49fc6be2'
OFFSET=math.log(.5)
TRANSITION=dict(source_sha256=SOURCE_SHA,consumed_steps=1933312,world_control_steps=1310720,
    actor_count=12680,critic_count=36352,optimizer_reset=False,standard_deviation_scale=.5)


def validate_sampling_metadata(saved):
    if (saved.get('reduced_noise_ppo') is not True or not saved.get('fixed_opponent_ppo') or saved.get('posture_cost_ppo')
            or saved.get('sampling_log_std_offset')!=OFFSET or saved.get('sampling_profile_transition')!=TRANSITION
            or saved['consumed_steps']<1933312 or saved['optimized_steps']<1933312):
        raise ValueError('explicit on-policy covariance profile and retained source required')


def sampling_offset(saved):
    if 'sampling_log_std_offset' in saved or saved.get('reduced_noise_ppo'):
        validate_sampling_metadata(saved);return OFFSET
    return 0.


def prepare_sampling_state(saved,sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha!=SOURCE_SHA or saved.get('reduced_noise_ppo') or saved.get('posture_cost_ppo')
            or saved['consumed_steps']!=saved['optimized_steps'] or saved['consumed_steps']!=1933312
            or saved['world_control_steps']!=1310720 or int(saved['optimizer']['actor_adam'][0].count)!=12680
            or int(saved['optimizer']['critic_adam'][0].count)!=36352):
        raise ValueError('latest completed control required without rollback or optimizer reset')
    result=copy.deepcopy(saved)
    result.update(reduced_noise_ppo=True,sampling_log_std_offset=OFFSET,sampling_profile_transition=copy.deepcopy(TRANSITION))
    require_ppo_checkpoint(result);return result


def sampling_config(source):
    if source['version']!=8 or not source['fixed_opponent_ppo']:raise ValueError('inherited version-8 profile required')
    return dict(source,version=10,reduced_noise_ppo=True,sampling_log_std_offset=OFFSET)
