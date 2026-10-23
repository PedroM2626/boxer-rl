import copy
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer.striking_conservative_actor import prepare_conservative_state,conservative_config,SOURCE_SHA
from boxer.striking_ppo_warmstart import SplitAdam
from boxer.striking_train import require_ppo_checkpoint,validate_resume_config,main
from tools.striking_contact_compare import load_checked

SOURCE=Path('checkpoints/striking-validation/mixed-partner-continue-1769472/gate-65536')


def test_explicit_transition_preserves_every_old_leaf_and_rejects_implicit_rates():
    source,_=load_checked(SOURCE,SOURCE_SHA);candidate=prepare_conservative_state(source,SOURCE_SHA)
    require_ppo_checkpoint(candidate)
    for key in source:
        for a,b in zip(jax.tree_util.tree_leaves(source[key]),jax.tree_util.tree_leaves(candidate[key])):
            np.testing.assert_equal(a,b)
    old=json.loads((SOURCE/'config.json').read_text());new=conservative_config(old)
    assert new['learning_rate']==.0001 and new['actor_learning_rate']==.00001
    with pytest.raises(ValueError):validate_resume_config(old,new)
    with pytest.raises(ValueError):main(['--actor-learning-rate','.00001'])
    with pytest.raises(ValueError):main(['--conservative-actor-ppo'])
    for key in ('actor_learning_rate','conservative_actor_transition'):
        bad=copy.deepcopy(candidate);del bad[key]
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)


def test_only_actor_step_scales_with_identical_gradients_and_retained_moments():
    source,_=load_checked(SOURCE,SOURCE_SHA)
    params=(jax.tree_util.tree_map(jp.array,source['actor']),jax.tree_util.tree_map(jp.array,source['critic']))
    gradients=jax.tree_util.tree_map(lambda x:jp.ones_like(x)*.01,params)
    normal,state_a=SplitAdam().update(gradients,source['optimizer'],params)
    slow,state_b=SplitAdam(actor_learning_rate=.00001).update(gradients,source['optimizer'],params)
    for a,b in zip(jax.tree_util.tree_leaves(normal[0]),jax.tree_util.tree_leaves(slow[0])):
        np.testing.assert_allclose(b,a*.1,rtol=2e-6,atol=1e-10)
    for a,b in zip(jax.tree_util.tree_leaves(normal[1]),jax.tree_util.tree_leaves(slow[1])):
        np.testing.assert_array_equal(a,b)
    for a,b in zip(jax.tree_util.tree_leaves(state_a),jax.tree_util.tree_leaves(state_b)):
        np.testing.assert_array_equal(a,b)
