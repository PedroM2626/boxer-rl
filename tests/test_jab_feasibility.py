"""A scripted cycle must retract, repeat and stay inside the original channels."""
import jax.numpy as jp
import numpy as np
import pytest

from boxer.jab_feasibility import JabFeasibility,cycle_fraction
from boxer.striking_env import StrikingEnv
from test_paired_foundation import foundation


def test_cycle_returns_and_repeats_instead_of_stopping_after_first_contact():
    assert cycle_fraction(.1)==('guard',0.)
    assert cycle_fraction(.44)[0]=='extend'
    assert cycle_fraction(.60)[0]=='retract'
    assert cycle_fraction(.9)==('rest',0.)
    for t in (.1,.44,.6,.9):
        a=cycle_fraction(t);b=cycle_fraction(t+1.08)
        assert a[0]==b[0];np.testing.assert_allclose(a[1],b[1],atol=1e-14)
    with pytest.raises(ValueError):cycle_fraction(float('nan'))


@pytest.mark.parametrize('corner',[0,1])
def test_jab_is_a_bounded_action_probe_without_root_or_leg_actuation(foundation,corner):
    motor=lambda obs,key:(jp.zeros((2,29)),{})
    env=StrikingEnv(foundation,None,motor=motor,engagement_distance=.45,engagement_reward_distance=.45)
    env.reset(7);controller=JabFeasibility(corner);controller.started=0.
    env.pair.elapsed=.44
    before=env.pair.data.qpos.copy()
    action,phase=controller.action(env)
    assert phase=='extend' and action.shape==(2,17) and np.max(np.abs(action))<=1.
    np.testing.assert_array_equal(action[1-corner],0.)
    np.testing.assert_array_equal(action[corner,:3],0.)
    np.testing.assert_array_equal(action[corner,10:],0.)  # seven right-arm slots untouched
    np.testing.assert_array_equal(env.pair.data.qpos,before)
    env.pair.scorecard.last_score[corner,0]=.4
    repeated,_=controller.action(env)
    np.testing.assert_array_equal(repeated,action)  # no one-contact early exit
    assert env.pair.model.neq==0 and not np.any(env.pair.model.body_gravcomp)
    assert not np.any(env.pair.data.qfrc_applied) and not np.any(env.pair.data.xfrc_applied)
