"""Diagnostic whole-body residual authority over the original learned motor.

The first 146 inputs and 17 outputs retain their original meaning. Fifteen
extra outputs adjust the twelve leg and three waist joint targets, never roots.
All additional outputs start with zero mean. This is not a main-policy upgrade.
"""
import copy

import jax
import jax.numpy as jp
import numpy as np
import optax

from boxer.striking_env import StrikingEnv


class WholeBodyStrikingEnv(StrikingEnv):
    observation_size = 161
    action_size = 32
    body_residual_radians = .1

    def __init__(self, *args, body_residual_radians=None, **kwargs):
        if body_residual_radians is not None:
            self.body_residual_radians = body_residual_radians
        if self.body_residual_radians not in (0., .1):
            raise ValueError('only the predeclared zero/0.1-rad diagnostic authority is allowed')
        super().__init__(*args, **kwargs)
        self.body_indices = np.array([i for i in range(29) if i not in self.arm_indices])
        if len(self.body_indices) != 15:
            raise ValueError('twelve leg and three waist joints required')
        original_step = self.pair.step
        def step_with_body_residual(actions):
            actual = np.asarray(actions).copy()
            actual[:, self.body_indices] += (
                self.body_residual_radians / self.pair.action_scale * self.body_residual)
            return original_step(actual)
        self.pair.step = step_with_body_residual

    def reset(self, seed=0, swap=False):
        self.body_residual = np.zeros((2, 15))
        self.previous_body = np.zeros((2, 15))
        return super().reset(seed, swap)

    def observation(self):
        return np.concatenate([super().observation(), self.previous_body], axis=1)

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (2, 32) or not np.isfinite(actions).all() or np.any(np.abs(actions)>1.00001):
            raise ValueError('finite bounded (2, 32) actions required')
        self.body_residual = .8 * self.body_residual + .2 * actions[:, 17:]
        self.previous_body = actions[:, 17:].copy()
        obs, reward, done, outcome = super().step(actions[:, :17])
        # Base reports its arm-adjusted motor output; include actual body targets.
        outcome['applied_actions'] = outcome['applied_actions'].copy()
        outcome['applied_actions'][:, self.body_indices] += (
            self.body_residual_radians / self.pair.action_scale * self.body_residual)
        outcome['body_residual'] = self.body_residual.copy()
        return obs, reward, done, outcome


def expand_learning_state(saved):
    """Append zero weights/moments, retaining every old parameter and Adam count."""
    from boxer.striking_train import Actor, Critic
    if saved['norm']['mean'].shape != (146,):
        raise ValueError('migration requires the original 146-input striking source')
    ap = Actor(action_size=32).init(jax.random.PRNGKey(0), jp.zeros((1, 161)))
    vp = Critic().init(jax.random.PRNGKey(1), jp.zeros((1, 161)))
    def pad(old, template, fill=0.):
        old, template = np.asarray(old), np.asarray(template)
        if old.shape == template.shape:
            return old.copy()
        if len(old.shape) != len(template.shape) or any(a>b for a,b in zip(old.shape,template.shape)):
            raise ValueError('unexpected source parameter/optimizer shape')
        result = np.full(template.shape, fill, dtype=old.dtype)
        result[tuple(slice(0,n) for n in old.shape)] = old
        return result
    def pad_parameter(path, old, template):
        return pad(old, template, -1.9 if getattr(path[-1], 'key', None) == 'log_std' else 0.)
    ap, vp = jax.tree_util.tree_map_with_path(pad_parameter,
        (saved['actor'], saved['critic']), (ap, vp))
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.0001))
    optimizer = jax.tree_util.tree_map(pad, saved['optimizer'], tx.init((ap, vp)))
    result = copy.deepcopy(saved)
    result.update(actor=ap, critic=vp, optimizer=optimizer, actor_inputs=161, actions=32)
    result['norm'] = dict(mean=np.r_[saved['norm']['mean'], np.zeros(15)],
        var=np.r_[saved['norm']['var'], np.ones(15)], count=saved['norm']['count'])
    return result
