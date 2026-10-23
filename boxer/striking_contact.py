"""Actual first-impact history and explicit supervised-state migration."""
import copy

import jax
import jax.numpy as jp
import numpy as np
import optax

from boxer.striking_env import StrikingEnv


class ContactAwareStrikingEnv(StrikingEnv):
    observation_size = 148

    def observation(self):
        return np.concatenate([super().observation(),
            np.isfinite(self.pair.scorecard.last_score).astype(np.float32)], axis=1)


def expand_supervised_state(saved):
    """Retain BC Adam, critic, RNG and lineage; append only zero input weights."""
    from boxer.striking_train import Actor, Critic
    if (saved.get('training_kind') != 'supervised_demonstration' or
            saved['norm']['mean'].shape != (146,)):
        raise ValueError('original supervised 146-input source required')
    ap = Actor().init(jax.random.PRNGKey(0), jp.zeros((1, 148)))
    vp = Critic().init(jax.random.PRNGKey(1), jp.zeros((1, 148)))

    def pad(old, template):
        old, template = np.asarray(old), np.asarray(template)
        if old.shape == template.shape:
            return old.copy()
        if old.shape != (146, 128) or template.shape != (148, 128):
            raise ValueError('unexpected supervised parameter or optimizer shape')
        return np.concatenate([old, np.zeros((2,128), dtype=old.dtype)])

    ap, vp = jax.tree_util.tree_map(pad, (saved['actor'], saved['critic']), (ap, vp))
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.0001))
    optimizer = jax.tree_util.tree_map(pad, saved['optimizer'], tx.init(ap))
    result = copy.deepcopy(saved)
    result.update(actor=ap, critic=vp, optimizer=optimizer, actor_inputs=148,
        actions=17, contact_history=True, training_kind='supervised_contact_state',
        ppo_optimizer_compatible=False)
    result['norm'] = dict(mean=np.r_[saved['norm']['mean'], np.zeros(2, np.float32)],
        var=np.r_[saved['norm']['var'], np.ones(2, np.float32)], count=saved['norm']['count'])
    return result


def demonstration_flags(trial, rows=500, dt=.02):
    """An event becomes visible at the following pre-action control observation."""
    flags = np.zeros((rows,2), np.float32)
    times = np.arange(rows)*dt
    for event in trial['impact_events']:
        for hand in event['hands']:
            if hand['corner'] == trial['learner_corner']:
                flags[times + 1e-9 >= event['elapsed'], hand['hand']] = 1.
    return flags


def recover_sampler(updates=2000, rows=7500, batch=256):
    """Reproduce the documented old sampler prefix without drawing another batch."""
    rng = np.random.default_rng(313)
    for _ in range(updates):
        rng.integers(0, rows, batch)
    return rng
