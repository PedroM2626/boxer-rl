"""A scripted cycle must retract, repeat and stay inside the original channels."""
import jax.numpy as jp
import numpy as np
import pytest

from boxer.jab_feasibility import JabFeasibility,GuardOnly,GuardRamp,MotorGuardJab,cycle_fraction
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


@pytest.mark.parametrize('corner',[0,1])
def test_motor_guard_jab_leaves_learned_guard_and_recovery_to_motor(foundation,corner):
    env=StrikingEnv(foundation,None,motor=lambda obs,key:(jp.zeros((2,29)),{}))
    env.reset(7);controller=MotorGuardJab(corner);controller.started=0.
    for t in (.1,.6,.9,1.18):
        env.pair.elapsed=t;action,phase=controller.action(env)
        np.testing.assert_array_equal(action,0.);assert phase!='extend'
    env.pair.elapsed=.44;before=env.pair.data.qpos.copy()
    action,phase=controller.action(env);assert phase=='extend'
    np.testing.assert_array_equal(action[1-corner],0.);np.testing.assert_array_equal(action[corner,:3],0.)
    np.testing.assert_array_equal(action[corner,10:],0.);assert np.max(np.abs(action))<=1.
    np.testing.assert_array_equal(env.pair.data.qpos,before)
    env.pair.scorecard.last_score[corner,0]=.43
    same,_=controller.action(env);np.testing.assert_array_equal(same,action)
    env.pair.scorecard.last_score[corner,0]=.4
    repeated,_=controller.action(env)
    np.testing.assert_array_equal(repeated,action)  # no one-contact early exit
    assert env.pair.model.neq==0 and not np.any(env.pair.model.body_gravcomp)
    assert not np.any(env.pair.data.qfrc_applied) and not np.any(env.pair.data.xfrc_applied)


def test_guard_only_matches_original_guard_but_never_extends(foundation):
    motor=lambda obs,key:(jp.zeros((2,29)),{})
    env=StrikingEnv(foundation,None,motor=motor)
    env.reset(7);original=JabFeasibility(0);guard=GuardOnly(0)
    original.started=guard.started=0.;env.pair.elapsed=.1
    expected,_=original.action(env);actual,phase=guard.action(env)
    np.testing.assert_array_equal(actual,expected);assert phase=='guard'
    for elapsed in (.44,.60,.9,1.52,9.9):
        env.pair.elapsed=elapsed
        action,phase=guard.action(env)
        np.testing.assert_array_equal(action,expected);assert phase=='guard'
        assert original.phase_fraction(elapsed)==cycle_fraction(elapsed)


def test_guard_ramp_starts_at_actual_fist_and_ends_at_original_guard(foundation):
    env=StrikingEnv(foundation,None,motor=lambda obs,key:(jp.zeros((2,29)),{}))
    env.reset(7);ramp=GuardRamp(0);ramp.started=0.
    before=env.pair.data.qpos.copy();fist=env.pair.data.geom_xpos[env.pair.fists[0,0]].copy()
    start=ramp.guard_target(env);np.testing.assert_allclose(start,fist,atol=1e-14)
    env.pair.elapsed=1.;middle=ramp.guard_target(env)
    env.pair.elapsed=2.;end=ramp.guard_target(env)
    np.testing.assert_allclose(end,GuardOnly(0).guard_target(env),atol=1e-14)
    np.testing.assert_allclose(middle,(start+end)/2,atol=1e-14)
    env.pair.elapsed=9.;np.testing.assert_allclose(ramp.guard_target(env),end,atol=1e-14)
    np.testing.assert_array_equal(env.pair.data.qpos,before)
