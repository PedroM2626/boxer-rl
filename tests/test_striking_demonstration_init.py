"""Supervised initialization must not silently reuse its optimizer as PPO."""
import jax
import jax.numpy as jp
import numpy as np
import pytest
from boxer.striking_train import Actor, require_ppo_checkpoint
from tools.striking_demonstration_init import behavior_loss


def test_supervised_optimizer_is_rejected_by_ppo_resume():
    require_ppo_checkpoint({})
    require_ppo_checkpoint({'training_kind':'ppo'})
    with pytest.raises(ValueError,match='optimizer migration'):
        require_ppo_checkpoint({'training_kind':'supervised_demonstration'})


def test_behavior_cloning_leaves_exploration_parameter_unsupervised():
    actor=Actor();obs=jp.zeros((2,146));params=actor.init(jax.random.PRNGKey(7),obs)
    gradient=jax.grad(lambda p:behavior_loss(actor,p,obs,jp.ones((2,17))*.2,jp.ones(2)))(params)
    np.testing.assert_array_equal(gradient['params']['log_std'],np.zeros(17))
    assert np.isfinite(behavior_loss(actor,params,obs,jp.zeros((2,17)),jp.ones(2)))
