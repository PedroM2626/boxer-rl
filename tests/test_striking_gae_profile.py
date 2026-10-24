import copy
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from boxer.striking_gae_profile import SOURCE_SHA, prepare_gae_state, gae_config
from boxer.striking_train import advantages, require_ppo_checkpoint, validate_resume_config, main
from tools.audit_striking_terminal_credit import fragmented_advantages
from tools.striking_contact_compare import load_checked

SOURCE = Path('checkpoints/striking-validation/kl-guard-1949696/control/gate-16384')


def test_long_gae_preserves_latest_source_and_rejects_implicit_profile_changes():
    source, _ = load_checked(SOURCE, SOURCE_SHA)
    prepared = prepare_gae_state(source, SOURCE_SHA)
    require_ppo_checkpoint(prepared)
    for key in source:
        for a,b in zip(jax.tree_util.tree_leaves(source[key]),jax.tree_util.tree_leaves(prepared[key])):
            np.testing.assert_equal(a,b)
    for key in ('long_gae_transition','training_gae_lambda','long_gae_ppo'):
        bad=copy.deepcopy(prepared);del bad[key]
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    for key,value in (('training_gae_lambda',.95),('kl_guard_ppo',True)):
        bad=copy.deepcopy(prepared);bad[key]=value
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    with pytest.raises(ValueError):prepare_gae_state(prepared,SOURCE_SHA)
    with pytest.raises(ValueError):prepare_gae_state(source,'0'*64)
    config=json.loads((SOURCE/'config.json').read_text());changed=gae_config(config)
    assert changed['version']==12 and changed['gae_lambda']==.99 and config['gae_lambda']==.95
    with pytest.raises(ValueError):validate_resume_config(config,changed)
    for args in (['--long-gae-ppo'],['--long-gae-ppo','--fixed-opponent-ppo','--kl-guard-ppo']):
        with pytest.raises(ValueError):main(args)


def test_lambda_extends_direct_fall_credit_but_preserves_episode_and_fragment_cuts():
    rewards=np.zeros((150,2));rewards[100,0]=-5.;rewards[149,1]=2.
    dones=np.zeros((150,2));dones[100]=1.;dones[149]=1.
    zero=np.zeros_like(rewards)
    short,_=advantages(rewards,dones,zero,np.array([100.,100.]),gamma=.999,lam=.95)
    long,_=advantages(rewards,dones,zero,np.array([100.,100.]),gamma=.999,lam=.99)
    assert abs(long[50,0])>abs(short[50,0])*7
    assert long[100,1]==0. and long[149,1]==2.  # no next episode or terminal bootstrap leakage
    fragment=fragmented_advantages(rewards,dones,zero,zero,.99)
    assert fragment[127,1]==0. and fragment[128,1]>0.  # later fragment's direct reward cannot leak backward
