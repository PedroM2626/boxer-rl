import copy
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from boxer.striking_credit_horizon import prepare_horizon_state,horizon_config
from boxer.striking_train import require_ppo_checkpoint,validate_resume_config,advantages,main
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/neutral-retention-1638400/gate-32768')
SHA='63c06464fa0ce0ec238d349e4fac559c454a323f235fc8630a5bf599753ce8e9'


def test_matched_horizons_keep_all_existing_learned_state_and_reject_implicit_changes():
    source,_=load_checked(SOURCE,SHA)
    for gamma in (.99,.999):
        prepared=prepare_horizon_state(source,SHA,gamma);require_ppo_checkpoint(prepared)
        for key in source:
            for a,b in zip(jax.tree_util.tree_leaves(source[key]),jax.tree_util.tree_leaves(prepared[key])):
                np.testing.assert_equal(a,b)
        bad=copy.deepcopy(prepared);bad['training_gamma']=.5
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
        bad=copy.deepcopy(prepared);del bad['credit_horizon_transition']
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    config=json.loads((SOURCE/'config.json').read_text())
    with pytest.raises(ValueError):validate_resume_config(horizon_config(config,.99),horizon_config(config,.999))
    with pytest.raises(ValueError):main(['--training-gamma','.999'])
    with pytest.raises(ValueError):main(['--credit-horizon-ppo'])


def test_configured_credit_reaches_late_rewards_and_still_stops_at_real_done():
    reward=np.zeros((5,1),np.float32);reward[-1]=2.
    values=np.zeros_like(reward);done=np.zeros_like(reward)
    a,_=advantages(reward,done,values,np.zeros(1),gamma=.99,lam=1.)
    b,_=advantages(reward,done,values,np.zeros(1),gamma=.999,lam=1.)
    np.testing.assert_allclose(a[0],2*.99**4,rtol=1e-6)
    np.testing.assert_allclose(b[0],2*.999**4,rtol=1e-6)
    assert b[0]>a[0]
    done[1]=1
    cut,_=advantages(reward,done,values,np.zeros(1),gamma=.999,lam=1.)
    np.testing.assert_array_equal(cut[:2],0.)
