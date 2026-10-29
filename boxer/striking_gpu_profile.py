"""Explicit collection-backend transition, preserving the complete v17 task/state."""
import copy

SOURCE = 'checkpoints/striking-validation/arm-effort-v17/step-2097152'
SOURCE_SHA = 'f311640d78941546938412f5802cff872e0a3611c392f49a3b4719eefaa1f26a'
PROFILE = dict(version=1, collection_backend='batched_mujoco_warp', reward_version=17,
    timestep=.002, control_dt=.02, solver_iterations=3, line_search_iterations=5,
    worlds=8, unroll=128, learners_per_world=1, full_random_resets=True,
    first_fall_termination=True, assistance=False, objective_unchanged=True)
TRANSITION = dict(source_sha256=SOURCE_SHA, consumed_steps=9420800, optimized_steps=9420800,
    world_control_steps=8798208, actor_count=129672, critic_count=153344,
    optimizer_reset=False, rng_reset=False, normalizer_reset=False, previous_backend='native_mujoco')
BATCH_SOURCE = 'checkpoints/striking-validation/gpu-batched-integration-v1'
BATCH_SOURCE_SHA = 'be1b5a8d64d88ac5ae2a266f1b975dc79ac39f90f95df1082d438fc86d5327e9'
BATCH_PROFILE = dict(PROFILE, version=2, worlds=128, packed_readback=True,
    epochs=4, minibatch_rows=256, minibatches_per_epoch=64, updates_per_rollout=256,
    active_pattern_worlds=8, normalizer_update_credits=16384)
BATCH_TRANSITION = dict(source_sha256=BATCH_SOURCE_SHA, consumed_steps=9421824,
    optimized_steps=9421824, world_control_steps=8799232, actor_count=129688,
    critic_count=153360, old_worlds=8, new_worlds=128, optimizer_reset=False,
    rng_reset=False, normalizer_reset=False, changed_normalizer_update_frequency=True)


def validate_gpu_metadata(saved):
    profile = saved.get('gpu_collection_profile')
    scaling = 'gpu_batch_transition' in saved
    if (saved.get('gpu_batched_ppo') is not True or profile != (BATCH_PROFILE if scaling else PROFILE)
            or saved.get('gpu_collection_transition') != TRANSITION or not saved.get('arm_effort_ppo')
            or saved['consumed_steps'] < TRANSITION['consumed_steps']
            or saved['optimized_steps'] < TRANSITION['optimized_steps']
            or saved['consumed_steps'] < saved['optimized_steps']):
        raise ValueError('explicit source-v17 batched backend transition and retained state required')
    if scaling and (saved['gpu_batch_transition'] != BATCH_TRANSITION
            or saved['optimized_steps'] < BATCH_TRANSITION['optimized_steps']
            or saved['world_control_steps'] < BATCH_TRANSITION['world_control_steps']):
        raise ValueError('explicit latest-GPU source and exact batch/update profile required')


def prepare_gpu_state(saved, sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha != SOURCE_SHA or saved.get('gpu_batched_ppo') or not saved.get('arm_effort_ppo')
            or saved['consumed_steps'] != TRANSITION['consumed_steps']
            or saved['optimized_steps'] != TRANSITION['optimized_steps']
            or saved['world_control_steps'] != TRANSITION['world_control_steps']
            or int(saved['optimizer']['actor_adam'][0].count) != TRANSITION['actor_count']
            or int(saved['optimizer']['critic_adam'][0].count) != TRANSITION['critic_count']):
        raise ValueError('actual latest v17 final required, without rollback or state reset')
    result = copy.deepcopy(saved)
    result.update(gpu_batched_ppo=True, gpu_collection_profile=copy.deepcopy(PROFILE),
                  gpu_collection_transition=copy.deepcopy(TRANSITION))
    require_ppo_checkpoint(result)
    return result


def prepare_scaled_state(saved, sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha != BATCH_SOURCE_SHA or saved.get('gpu_collection_profile') != PROFILE
            or 'gpu_batch_transition' in saved
            or any(saved[key] != BATCH_TRANSITION[key] for key in
                   ('consumed_steps', 'optimized_steps', 'world_control_steps'))
            or int(saved['optimizer']['actor_adam'][0].count) != BATCH_TRANSITION['actor_count']
            or int(saved['optimizer']['critic_adam'][0].count) != BATCH_TRANSITION['critic_count']):
        raise ValueError('reviewed latest GPU integration final required; no rollback or reset')
    result = copy.deepcopy(saved)
    result.update(gpu_collection_profile=copy.deepcopy(BATCH_PROFILE),
                  gpu_batch_transition=copy.deepcopy(BATCH_TRANSITION))
    require_ppo_checkpoint(result)
    return result
