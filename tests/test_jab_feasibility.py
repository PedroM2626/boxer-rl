"""A scripted cycle must retract, repeat and stay inside the original channels."""
import jax.numpy as jp
import numpy as np
import pytest

from boxer.jab_feasibility import JabFeasibility,GuardOnly,GuardRamp,MotorGuardJab,MotorGuardDwell,MotorGuardArc,cycle_fraction
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


def test_terminal_dwell_preserves_outbound_motion_and_releases_to_motor(foundation):
    original=MotorGuardJab(0);dwell=MotorGuardDwell(0)
    for t in (0.,.1,.36,.44,.52):assert dwell.phase_fraction(t)==original.phase_fraction(t)
    for t in (.54,.60,.64):assert dwell.phase_fraction(t)==('extend',1.)
    assert dwell.phase_fraction(.66)[0]=='retract'
    assert dwell.phase_fraction(.95)==('rest',0.)
    np.testing.assert_allclose(dwell.cycle_duration(),1.2)
    env=StrikingEnv(foundation,None,motor=lambda obs,key:(jp.zeros((2,29)),{}))
    env.reset(7);original.started=dwell.started=0.;env.pair.elapsed=.44
    a,_=original.action(env);b,_=dwell.action(env);np.testing.assert_array_equal(a,b)
    entry=dwell.entry_local.copy();env.pair.elapsed=.60
    b,phase=dwell.action(env);assert phase=='extend';np.testing.assert_array_equal(dwell.entry_local,entry)
    env.pair.elapsed=.66;b,phase=dwell.action(env);assert phase=='retract';np.testing.assert_array_equal(b,0.)
    env.pair.elapsed=1.64;dwell.action(env);assert dwell.extension_cycle==1


@pytest.mark.parametrize('corner',[0,1])
def test_arc_keeps_endpoints_and_zero_arc_matches_original_actions(foundation,corner):
    env=StrikingEnv(foundation,None,motor=lambda obs,key:(jp.zeros((2,29)),{}));env.reset(7)
    dwell=MotorGuardDwell(corner);zero=MotorGuardArc(corner,0.);arc=MotorGuardArc(corner,.06)
    dwell.started=zero.started=arc.started=0.;env.pair.elapsed=.44
    a,_=dwell.action(env);b,_=zero.action(env);np.testing.assert_array_equal(a,b)
    guard=np.array([1.,2.,3.]);target=np.array([2.,3.,4.])
    for f in (0.,1.):np.testing.assert_array_equal(arc.path_target(env,guard,target,f),dwell.path_target(env,guard,target,f))
    rotation=env.pair.data.site(env.pair.maps[corner]['side']+'imu_in_torso').xmat.reshape(3,3)
    np.testing.assert_allclose(arc.path_target(env,guard,target,.5)-(guard+target)/2,rotation[:,1]*.06,atol=1e-14)
    action,_=arc.action(env);np.testing.assert_array_equal(action[1-corner],0.)
    np.testing.assert_array_equal(action[corner,:3],0.);np.testing.assert_array_equal(action[corner,10:],0.)


@pytest.mark.parametrize('corner',[0,1])
@pytest.mark.parametrize('direction',[-1,1])
def test_side_step_uses_existing_command_before_arm_baseline(foundation,corner,direction):
    from boxer.jab_feasibility import MotorGuardSideStep
    seen=[]
    def motor(obs,key):
        seen.append(np.asarray(obs['state'])[:,9:12].copy())
        return jp.zeros((2,29)),{}
    env=StrikingEnv(foundation,None,motor=motor);env.reset(7)
    controller=MotorGuardSideStep(corner,direction)
    action,phase=controller.action(env)
    assert phase=='approach';np.testing.assert_array_equal(action,0.)
    controller.started=0.;before=env.pair.data.qpos.copy()
    for t,active in ((.1,True),(.44,True),(.6,True),(.7,False),(1.,False)):
        env.pair.elapsed=t;action,phase=controller.action(env)
        assert action[corner,1]==(direction if active else 0)
        np.testing.assert_array_equal(action[1-corner],0.)
        np.testing.assert_array_equal(action[corner,[0,2]],0.)
        np.testing.assert_array_equal(action[corner,10:],0.)
        if phase=='extend':np.testing.assert_allclose(seen[-1],env.commands(action),atol=1e-7)
        else:np.testing.assert_array_equal(action[:,3:],0.)
        np.testing.assert_array_equal(env.pair.data.qpos,before)
    assert env.pair.model.neq==0 and not np.any(env.pair.model.body_gravcomp)
    assert not np.any(env.pair.data.qfrc_applied) and not np.any(env.pair.data.xfrc_applied)
