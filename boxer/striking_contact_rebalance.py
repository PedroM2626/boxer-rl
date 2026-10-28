"""Explicit v16 partial-contact reward rebalance, retaining v15 source provenance."""
import copy
import numpy as np

SOURCE = 'checkpoints/striking-validation/contact-progress-v15/step-2097152'
SOURCE_SHA = '9132ab4c2c86ed2b2958d551125b001370bf0f4e80f403490d33111ac3bb634a'
PROFILE = dict(version=1, kind='partial_contact_rebalance', original_contact_bonus=2.,
               adjusted_contact_bonus=.25, full_cycle_bonus=10., training_reward_only=True)
TRANSITION = dict(source_sha256=SOURCE_SHA, consumed_steps=5226496, optimized_steps=5226496,
                  world_control_steps=4603904, actor_count=64136, critic_count=87808,
                  optimizer_reset=False, parent_version=15, new_version=16)


def correction(parts, corner):
    if corner not in (0, 1):
        raise ValueError('learner corner must be 0 or 1')
    contact = np.asarray(parts['physical_contact_progress'], dtype=float)
    if contact.shape != (2,) or not np.isfinite(contact).all():
        raise ValueError('finite original per-corner contact payments required')
    return float((PROFILE['adjusted_contact_bonus'] / PROFILE['original_contact_bonus'] - 1.) *
                 (contact[corner] - contact[1-corner]))


def validate_rebalance_metadata(saved):
    if (saved.get('contact_rebalance_ppo') is not True or saved.get('contact_rebalance_profile') != PROFILE
            or saved.get('contact_rebalance_transition') != TRANSITION or not saved.get('contact_progress_ppo')
            or saved['optimized_steps'] < TRANSITION['optimized_steps'] or saved['consumed_steps'] < saved['optimized_steps']):
        raise ValueError('explicit v15 source and partial-contact rebalance provenance required')


def prepare_rebalance_state(saved, sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha != SOURCE_SHA or saved.get('contact_rebalance_ppo') or not saved.get('contact_progress_ppo')
            or saved['consumed_steps'] != saved['optimized_steps'] or saved['optimized_steps'] != 5226496
            or saved['world_control_steps'] != 4603904 or int(saved['optimizer']['actor_adam'][0].count) != 64136
            or int(saved['optimizer']['critic_adam'][0].count) != 87808):
        raise ValueError('actual completed v15 final required without optimizer reset or rollback')
    result = copy.deepcopy(saved)
    result.update(contact_rebalance_ppo=True, contact_rebalance_profile=copy.deepcopy(PROFILE),
                  contact_rebalance_transition=copy.deepcopy(TRANSITION))
    require_ppo_checkpoint(result)
    return result


def rebalance_config(source):
    if source['version'] != 15 or not source.get('contact_progress_ppo'):
        raise ValueError('completed v15 profile required')
    return dict(source, version=16, contact_rebalance_ppo=True, contact_rebalance_profile=copy.deepcopy(PROFILE))
