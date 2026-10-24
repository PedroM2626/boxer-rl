"""Reward-only safety margin and lossless ordinary-resume provenance."""
import copy,json
import jax
import numpy as np
import pytest
from boxer.striking_posture_cost import margin_cost,learner_cost,prepare_posture_state,posture_config,SOURCE_SHA
from boxer.striking_train import require_ppo_checkpoint,main
from tools.striking_contact_compare import load_checked
from tools.striking_generalization_audit import SOURCE


def test_cost_is_dt_scaled_bounded_and_zero_above_margins():
    assert margin_cost(.74,.99,.02)==0.
    assert margin_cost(.55,.6,.02)==pytest.approx(.04)
    assert margin_cost(.2,.1,.02)==pytest.approx(.04)
    assert margin_cost(.615,.75,.02)==pytest.approx(.01)
    assert margin_cost(.615,.75,.01)==pytest.approx(.005)
    for values in ((float('nan'),1,.02),(.7,float('inf'),.02),(.7,1,0)):
        with pytest.raises(ValueError):margin_cost(*values)


def test_transition_retains_every_original_leaf_and_rejects_bad_resume():
    saved,_=load_checked(SOURCE,SOURCE_SHA);new=prepare_posture_state(saved,SOURCE_SHA)
    require_ppo_checkpoint(new)
    for key in saved:
        for a,b in zip(jax.tree_util.tree_leaves(saved[key]),jax.tree_util.tree_leaves(new[key])):np.testing.assert_equal(a,b)
    assert posture_config(json.loads((SOURCE/'config.json').read_text()))['version']==9
    for key in ('posture_cost_profile','posture_cost_transition'):
        bad=copy.deepcopy(new);del bad[key]
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    with pytest.raises(ValueError):prepare_posture_state(new,SOURCE_SHA)
    bad=copy.deepcopy(saved);bad['optimized_steps']-=1024
    with pytest.raises(ValueError):prepare_posture_state(bad,SOURCE_SHA)
    with pytest.raises(ValueError):main(['--posture-cost-ppo'])


def test_native_cost_observation_does_not_modify_physical_state(monkeypatch):
    from boxer.curriculum_env import UnassistedG1
    from boxer.striking_fixed_opponent import ContactFixedMixedStrikingEnv
    foundation=UnassistedG1(); motor=lambda obs,key:(np.zeros((2,29)),{})
    env=ContactFixedMixedStrikingEnv(foundation,None,motor=motor,engagement_distance=.45,engagement_reward_distance=.45)
    env.reset(21000006,False);env.step(np.zeros((1,17)))
    pair=env.env.pair;before=[x.copy() for x in (pair.data.qpos,pair.data.qvel,pair.data.ctrl,pair.data.qfrc_applied,pair.data.xfrc_applied)]
    cost=learner_cost(env);assert np.isfinite(cost) and cost>=0
    for old,new in zip(before,(pair.data.qpos,pair.data.qvel,pair.data.ctrl,pair.data.qfrc_applied,pair.data.xfrc_applied)):np.testing.assert_array_equal(old,new)
    assert pair.model.neq==0 and not np.any(pair.model.body_gravcomp)
