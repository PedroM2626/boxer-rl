"""Explicit one-learner mixed-opponent task with retained training provenance."""
import copy

import numpy as np

from boxer.striking_partner import ContactPartnerStrikingEnv

SOURCE_SHA = '55976fc15b74b9997b4cb56cf13683f1f87315ac6d67c5cc966ab3c644dd2d73'
MIXED_ACTIVE_WORLDS = (1, 2, 5, 6)


class ContactMixedStrikingEnv(ContactPartnerStrikingEnv):
    """Route one learner and an optional current-mean opponent into original physics."""
    opponent_mode = 'mixed_current_mean'

    def __init__(self, *args, active_opponent=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_opponent = bool(active_opponent)

    def reset(self, seed=0, swap=False):
        self.learner_corner = int(seed % 2)
        obs = self.env.reset(seed, swap)
        self.opponent_observation = obs[1-self.learner_corner].copy()
        return obs[self.learner_corner:self.learner_corner+1]

    def step(self, actions, opponent_action=None):
        actions = np.asarray(actions)
        if actions.shape != (1, 17) or not np.isfinite(actions).all():
            raise ValueError('one finite learner action row required')
        paired = np.zeros((2, 17), dtype=actions.dtype)
        i = self.learner_corner
        paired[i] = actions[0]
        if self.active_opponent:
            other = np.asarray(opponent_action)
            if other.shape != (17,) or not np.isfinite(other).all():
                raise ValueError('active world requires the finite current-mean opponent action')
            paired[1-i] = other
        elif opponent_action is not None:
            raise ValueError('neutral world must retain zero opponent residuals')
        obs, reward, done, outcome = self.env.step(paired)
        self.opponent_observation = obs[1-i].copy()
        outcome.update(learner_corner=i, active_opponent=self.active_opponent)
        return obs[i:i+1], reward[i:i+1], done, outcome


def validate_mixed_metadata(saved):
    """No recursive state validation: called by the shared split-Adam validator."""
    transition = saved.get('mixed_partner_transition', {})
    credits = np.asarray(saved.get('mixed_credits_by_mode_and_corner', []))
    delta = saved['consumed_steps'] - 1736704
    optimized = saved['optimized_steps'] - 1736704
    if (saved.get('mixed_partner_ppo') is not True or saved.get('credit_horizon_ppo') is not True
            or saved.get('training_gamma') != .999 or saved.get('retention_loss_weight') != 1.
            or transition != dict(source_sha256=SOURCE_SHA, consumed_steps=1736704,
                world_control_steps=1114112, actor_count=9608, critic_count=33280,
                neutral_credits_by_corner=[50229, 48075], optimizer_reset=False,
                active_worlds=4, neutral_worlds=4, active_world_indices=list(MIXED_ACTIVE_WORLDS),
                opponent_policy='current_learner_mean_per_rollout')
            or saved.get('active_world_indices') != list(MIXED_ACTIVE_WORLDS)
            or saved.get('neutral_credits_by_corner') != [50229, 48075]
            or credits.shape != (2, 2) or credits.dtype.kind not in 'iu' or np.any(credits < 0)
            or delta < 0 or credits.sum() != delta or saved['world_control_steps'] != 1114112 + delta
            or optimized < 0 or optimized > delta or optimized % 1024
            or int(saved['optimizer']['actor_adam'][0].count) != 9608 + optimized // 1024 * 16
            or int(saved['optimizer']['critic_adam'][0].count) != 33280 + optimized // 1024 * 16):
        raise ValueError('explicit mixed task, optimizer and mode/corner prefixes required')


def prepare_mixed_state(saved, sha):
    from boxer.striking_credit_horizon import validate_horizon_state
    validate_horizon_state(saved)
    if (sha != SOURCE_SHA or saved.get('mixed_partner_ppo') or saved['consumed_steps'] != 1736704
            or saved['optimized_steps'] != 1736704 or saved['world_control_steps'] != 1114112
            or saved['training_gamma'] != .999
            or int(saved['optimizer']['actor_adam'][0].count) != 9608
            or int(saved['optimizer']['critic_adam'][0].count) != 33280):
        raise ValueError('latest completed gamma-0.999 final required without rollback')
    result = copy.deepcopy(saved)
    result.update(mixed_partner_ppo=True, opponent_mode='mixed_current_mean', active_world_indices=list(MIXED_ACTIVE_WORLDS),
        mixed_credits_by_mode_and_corner=[[0, 0], [0, 0]],
        mixed_partner_transition=dict(source_sha256=sha, consumed_steps=1736704,
            world_control_steps=1114112, actor_count=9608, critic_count=33280,
            neutral_credits_by_corner=[50229, 48075], optimizer_reset=False,
            active_worlds=4, neutral_worlds=4, active_world_indices=list(MIXED_ACTIVE_WORLDS),
            opponent_policy='current_learner_mean_per_rollout'))
    validate_horizon_state(result)
    return result


def mixed_config(source):
    if (source['version'] != 5 or source['gamma'] != .999 or source['worlds'] != 8
            or source['unroll'] != 128 or source['opponent_mode'] != 'neutral_motor'):
        raise ValueError('reviewed latest neutral profile required')
    return dict(source, version=6, mixed_partner_ppo=True, opponent_mode='mixed_current_mean',
        active_world_indices=list(MIXED_ACTIVE_WORLDS))
