"""Version-13 archived opponents with independent statistics and physical credits."""
import copy

import jax
import jax.numpy as jp
import numpy as np

from boxer.striking_fixed_opponent import fingerprint
from boxer.striking_mixed_partner import ContactMixedStrikingEnv

SOURCE = 'checkpoints/striking-validation/gae-consolidate-1998848/gate-65536'
SOURCE_SHA = 'bea8147f6a13dd2d473a558f2e2ac68d3492e3136d2899e2d9daf139364bb870'
OFFSET = 2064384
WORLD_OFFSET = 1441792
ACTIVE_PREFIX = 163840
ACTIVE_CORNER_PREFIX = np.array([85379, 78461])
SAMPLER_SEED = 32000053
SPECS = (
    ('checkpoints/striking-validation/mixed-confirm-1851392/gate-8192',
     '6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a',
     '158d7b3dbdccc0a0aadec27d7d39b714677191d35b1d8a68759f12e5472873b0'),
    ('checkpoints/striking-validation/gae-1966080/long_gae/gate-16384',
     '95dbdff01def598cdc907e93f1dc7bceb2f1d6b58bc22f488a1936afa86553f9',
     '0938b48ad5a635b8ae8fe2050581ac145ab50764f408a586dbeaf50e1665d214'),
    (SOURCE, SOURCE_SHA, 'fd982c1683940bda4e98da3dfb1a50f3712429d9c8d6b70e7d2bef4b78ebcad8'),
)
TRANSITION = dict(source_sha256=SOURCE_SHA, consumed_steps=OFFSET,
    world_control_steps=WORLD_OFFSET, actor_count=14728, critic_count=38400,
    optimizer_reset=False, sampler_seed=SAMPLER_SEED, sampling='uniform_per_active_episode',
    active_credit_prefix=ACTIVE_PREFIX,
    opponents=[dict(path=p, sha256=s, fingerprint=f) for p,s,f in SPECS])


class ContactPoolMixedStrikingEnv(ContactMixedStrikingEnv):
    opponent_mode = 'mixed_pool_mean'


def credit_array(value):
    result = np.asarray(value)
    if result.shape != (len(SPECS), 2) or result.dtype.kind not in 'iu' or np.any(result < 0):
        raise ValueError('nonnegative integer per-opponent/corner credits required')
    return result.astype(np.int64)


def validate_pool_ledger(credits, mixed):
    result = credit_array(credits)
    mixed = np.asarray(mixed)
    if mixed.shape != (2, 2) or mixed.dtype.kind not in 'iu' or np.any(mixed < 0):
        raise ValueError('integer mixed-role ledger required')
    if int(result.sum()) != int(mixed[1].sum()) - ACTIVE_PREFIX:
        raise ValueError('pool credits must conserve every attempted active control')
    if not np.array_equal(result.sum(axis=0), mixed[1]-ACTIVE_CORNER_PREFIX):
        raise ValueError('pool credits must retain the exact historical corner prefix')
    return result


def restore_sampler(state):
    rng = np.random.default_rng()
    try:
        rng.bit_generator.state = copy.deepcopy(state)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError('valid persistent pool sampler required') from error
    return rng


def draw_opponent(rng, active):
    return int(rng.integers(len(SPECS))) if active else -1


def validate_pool_metadata(saved):
    if (saved.get('pool_opponent_ppo') is not True or saved.get('opponent_mode') != 'mixed_pool_mean'
            or saved.get('opponent_pool_transition') != TRANSITION
            or not saved.get('long_gae_ppo') or not saved.get('fixed_opponent_ppo')
            or saved.get('training_gae_lambda') != .99
            or saved['optimized_steps'] < OFFSET or saved['consumed_steps'] < OFFSET):
        raise ValueError('explicit latest-source pool transition required')
    pool = saved.get('opponent_pool', [])
    if len(pool) != len(SPECS):
        raise ValueError('all three immutable opponent snapshots required')
    for snapshot, (_, sha, expected) in zip(pool, SPECS):
        if set(snapshot) != {'source_sha256', 'actor', 'norm'} or snapshot['source_sha256'] != sha:
            raise ValueError('archived opponent provenance mismatch')
        if fingerprint(dict(actor=snapshot['actor'], norm=snapshot['norm'])) != expected:
            raise ValueError('archived opponent actor/normalizer changed')
        norm = snapshot['norm']
        if (np.shape(norm['mean']) != (148,) or np.shape(norm['var']) != (148,)
                or np.any(np.asarray(norm['var']) < 0) or norm['count'] <= 0):
            raise ValueError('valid independent opponent normalization required')
    validate_pool_ledger(saved.get('pool_credits_by_opponent_and_corner'),
                         saved['mixed_credits_by_mode_and_corner'])
    restore_sampler(saved.get('pool_sampler_state'))
    assignments = np.asarray(saved.get('pool_assignments'))
    if (assignments.shape != (8,) or assignments.dtype.kind not in 'iu'
            or np.any(assignments < -1) or np.any(assignments >= len(SPECS))
            or np.any(assignments[[0,3,4,7]] != -1)
            or np.any(assignments[[1,2,5,6]] < 0)):
        raise ValueError('eight-world pool assignment provenance required')


def prepare_pool_state(saved, sha, snapshots):
    """One metadata-only transition; never repeat on ordinary resumes."""
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha != SOURCE_SHA or saved.get('pool_opponent_ppo')
            or saved['consumed_steps'] != saved['optimized_steps'] or saved['optimized_steps'] != OFFSET
            or saved['world_control_steps'] != WORLD_OFFSET
            or int(saved['optimizer']['actor_adam'][0].count) != 14728
            or int(saved['optimizer']['critic_adam'][0].count) != 38400):
        raise ValueError('latest completed source required without rollback or optimizer reset')
    result = copy.deepcopy(saved)
    rng = np.random.default_rng(SAMPLER_SEED)
    assignments = [draw_opponent(rng, i in (1,2,5,6)) for i in range(8)]
    result.update(pool_opponent_ppo=True, opponent_mode='mixed_pool_mean',
        opponent_pool_transition=copy.deepcopy(TRANSITION), opponent_pool=copy.deepcopy(snapshots),
        pool_credits_by_opponent_and_corner=np.zeros((3,2), dtype=np.int64).tolist(),
        pool_sampler_state=copy.deepcopy(rng.bit_generator.state), pool_assignments=assignments)
    require_ppo_checkpoint(result)
    return result


def pool_config(source):
    if (source['version'] != 12 or source['gae_lambda'] != .99 or source['gamma'] != .999
            or source['opponent_mode'] != 'mixed_frozen_mean'):
        raise ValueError('inherited original version-12 long-GAE profile required')
    return dict(source, version=13, pool_opponent_ppo=True, opponent_mode='mixed_pool_mean')


def pool_predictor(actor, snapshots):
    """Vectorize routed means; each archived actor uses its own frozen statistics."""
    params = jax.tree_util.tree_map(lambda *xs: jp.stack(xs), *(s['actor'] for s in snapshots))
    means = jp.stack([jp.asarray(s['norm']['mean']) for s in snapshots])
    variances = jp.stack([jp.asarray(s['norm']['var']) for s in snapshots])

    @jax.jit
    def predict(observations, assignments):
        indices = jp.maximum(assignments, 0)  # neutral rows are computed but never applied
        selected = jax.tree_util.tree_map(lambda x: x[indices], params)
        normalized = jp.clip((observations-means[indices])/jp.sqrt(variances[indices]+.01), -10, 10)
        return jax.vmap(lambda p,o: jp.tanh(actor.apply(p,o)[0]))(selected, normalized)
    return predict
