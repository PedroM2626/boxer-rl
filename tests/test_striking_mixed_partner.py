"""Mixed task routing, exact warm state and consumed-credit provenance."""
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer.striking_contact import ContactAwareStrikingEnv
from boxer.striking_mixed_partner import ContactMixedStrikingEnv, prepare_mixed_state, mixed_config, SOURCE_SHA, MIXED_ACTIVE_WORLDS
from boxer.striking_train import require_ppo_checkpoint, validate_resume_config, main
from tools.striking_contact_compare import load_checked
from test_paired_foundation import foundation

SOURCE = Path('checkpoints/striking-validation/credit-horizon-continue-1703936/gate-32768')


@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('seed,swap', [(22, False), (23, True)])
def test_routes_exact_one_learner_and_opponent_actions_into_original_free_physics(foundation, active, seed, swap):
    motor = lambda obs, key: (jp.zeros((2, 29)), {})
    env = ContactMixedStrikingEnv(foundation, None, motor=motor, active_opponent=active)
    ref = ContactAwareStrikingEnv(foundation, None, motor=motor)
    i = seed % 2
    np.testing.assert_array_equal(env.reset(seed, swap), ref.reset(seed, swap)[i:i+1])
    learner = np.linspace(-.1, .1, 17)[None]
    other = np.linspace(.05, -.05, 17)
    paired = np.zeros((2, 17)); paired[i] = learner[0]
    if active: paired[1-i] = other
    for _ in range(3):
        obs, reward, done, out = env.step(learner, other if active else None)
        expected, rewards, stop, _ = ref.step(paired)
        np.testing.assert_array_equal(obs, expected[i:i+1])
        np.testing.assert_array_equal(env.opponent_observation, expected[1-i])
        np.testing.assert_array_equal(reward, rewards[i:i+1])
        np.testing.assert_array_equal(env.env.pair.data.qpos, ref.pair.data.qpos)
        assert done == stop and out['active_opponent'] is active
        assert env.learners_per_world == 1
    assert env.env.pair.model.neq == 0 and not np.any(env.env.pair.model.body_gravcomp)
    assert not np.any(env.env.pair.data.qfrc_applied) and not np.any(env.env.pair.data.xfrc_applied)
    with pytest.raises(ValueError):
        env.step(learner, None if active else other)


def test_transition_preserves_all_learned_optimizer_sampler_and_consumed_state():
    source, _ = load_checked(SOURCE, SOURCE_SHA)
    prepared = prepare_mixed_state(source, SOURCE_SHA)
    require_ppo_checkpoint(prepared)
    for key in source.keys() - {'opponent_mode'}:
        for a, b in zip(jax.tree_util.tree_leaves(source[key]), jax.tree_util.tree_leaves(prepared[key])):
            np.testing.assert_equal(a, b)
    assert prepared['neutral_credits_by_corner'] == [50229, 48075]
    assert prepared['mixed_credits_by_mode_and_corner'] == [[0, 0], [0, 0]]
    old = json.loads((SOURCE / 'config.json').read_text())
    cfg = mixed_config(old)
    assert cfg['version'] == 6 and cfg['worlds'] * cfg['unroll'] == 1024
    with pytest.raises(ValueError): validate_resume_config(old, cfg)
    with pytest.raises(ValueError): prepare_mixed_state(source, '0' * 64)
    with pytest.raises(ValueError): main(['--mixed-partner-ppo'])


@pytest.mark.parametrize('mutation', ['credits', 'old_prefix', 'optimizer', 'transition', 'schedule'])
def test_mixed_resume_rejects_erased_or_fabricated_credits_and_prefixes(mutation):
    source, _ = load_checked(SOURCE, SOURCE_SHA)
    bad = prepare_mixed_state(source, SOURCE_SHA)
    if mutation == 'credits': bad['mixed_credits_by_mode_and_corner'] = [[1, 0], [0, 0]]
    elif mutation == 'old_prefix': bad['neutral_credits_by_corner'] = [50230, 48074]
    elif mutation == 'transition': del bad['mixed_partner_transition']
    elif mutation == 'schedule': bad['active_world_indices'] = [0, 2, 4, 6]
    else:
        opt = bad['optimizer']['actor_adam']
        bad['optimizer']['actor_adam'] = (opt[0]._replace(count=opt[0].count-1), opt[1])
    with pytest.raises(ValueError): require_ppo_checkpoint(bad)


def test_initial_opponent_schedule_does_not_conflict_with_learner_role_parity():
    for active in (False, True):
        worlds = [i for i in range(8) if (i in MIXED_ACTIVE_WORLDS) == active]
        assert len(worlds) == 4 and sum((53 + 1009*i + 1736704//8) % 2 for i in worlds) == 2


def test_completed_updates_and_discarded_attempts_keep_separate_exact_prefixes():
    source, _ = load_checked(SOURCE, SOURCE_SHA)
    saved = prepare_mixed_state(source, SOURCE_SHA)
    saved.update(consumed_steps=1736704+1028, optimized_steps=1736704+1024,
        world_control_steps=1114112+1028, mixed_credits_by_mode_and_corner=[[257,257],[257,257]])
    for name in ('actor_adam', 'critic_adam'):
        opt = saved['optimizer'][name]
        saved['optimizer'][name] = (opt[0]._replace(count=opt[0].count+16), opt[1])
    require_ppo_checkpoint(saved)
    # Four attempted but unoptimized controls are retained, not invented Adam updates.
    bad = copy.deepcopy(saved); bad['mixed_credits_by_mode_and_corner'] = [[256,256],[256,256]]
    with pytest.raises(ValueError): require_ppo_checkpoint(bad)
    bad = copy.deepcopy(saved); bad['world_control_steps'] -= 4
    with pytest.raises(ValueError): require_ppo_checkpoint(bad)
