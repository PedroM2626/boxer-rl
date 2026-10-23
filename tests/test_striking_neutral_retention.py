import copy
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer.striking_partner import ContactPartnerStrikingEnv
from boxer.striking_contact import ContactAwareStrikingEnv
from boxer.striking_neutral_retention import prepare_neutral_state,neutral_config
from boxer.striking_train import main,require_ppo_checkpoint,validate_resume_config
from tools.striking_contact_compare import load_checked
from test_paired_foundation import foundation

SOURCE=Path('checkpoints/striking-validation/retention-1605632/gate-32768')
SHA='76b83b4c1bd0a66b439c24296366b557fda02a0fecf7d801b67e670472587c45'


@pytest.mark.parametrize('seed,swap',[(22,False),(23,True)])
def test_contact_partner_routes_exact_real_learner_observations_rewards_and_physics(foundation,seed,swap):
    motor=lambda obs,key:(jp.zeros((2,29)),{})
    env=ContactPartnerStrikingEnv(foundation,None,motor=motor)
    reference=ContactAwareStrikingEnv(foundation,None,motor=motor)
    i=seed%2
    np.testing.assert_array_equal(env.reset(seed,swap),reference.reset(seed,swap)[i:i+1])
    assert env.observation_size==148 and env.learners_per_world==1
    action=np.linspace(-.1,.1,17)[None];paired=np.zeros((2,17));paired[i]=action[0]
    for _ in range(3):
        obs,reward,done,out=env.step(action);expected,rewards,stop,_=reference.step(paired)
        np.testing.assert_array_equal(obs,expected[i:i+1]);np.testing.assert_array_equal(reward,rewards[i:i+1])
        np.testing.assert_array_equal(env.env.pair.data.qpos,reference.pair.data.qpos)
        assert done==stop and out['learner_corner']==i
        np.testing.assert_array_equal(env.env.previous[1-i],0.)
    assert env.env.pair.model.neq==0 and not np.any(env.env.pair.model.body_gravcomp)


def test_neutral_transition_keeps_every_learned_state_and_rejects_credit_or_profile_loss():
    saved,_=load_checked(SOURCE,SHA);prepared=prepare_neutral_state(saved,SHA)
    for key in saved.keys()-{'learners_per_world','opponent_mode'}:
        for a,b in zip(jax.tree_util.tree_leaves(saved[key]),jax.tree_util.tree_leaves(prepared[key])):
            np.testing.assert_equal(a,b)
    require_ppo_checkpoint(prepared)
    assert prepared['neutral_credits_by_corner']==[0,0]
    bad=copy.deepcopy(prepared);bad['neutral_credits_by_corner']=[1,0]
    with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    bad=copy.deepcopy(prepared);del bad['neutral_partner_transition']
    with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    bad=copy.deepcopy(prepared);bad['learners_per_world']=2
    with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    old=json.loads((SOURCE/'config.json').read_text());cfg=neutral_config(old)
    assert cfg['worlds']*cfg['learners_per_world']*cfg['unroll']==1024
    with pytest.raises(ValueError):validate_resume_config(old,cfg)
    with pytest.raises(ValueError):main(['--neutral-contact-ppo'])
