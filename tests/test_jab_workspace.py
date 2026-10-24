"""Cloned geometry cannot alter live physics or become a physical feasibility pass."""
import jax.numpy as jp
import numpy as np
import pytest

from boxer.striking_env import StrikingEnv
from boxer.jab_feasibility import JabFeasibility
from boxer.jab_workspace import MotorTap,controller_measurements,GeometryBudget,static_target_search,left_arm
from test_paired_foundation import foundation


def test_clone_recovers_known_pose_without_moving_live_roots_or_joints(foundation):
    env=StrikingEnv(foundation,None,motor=lambda obs,key:(jp.zeros((2,29)),{}))
    p=env.pair;indices,_=left_arm(env,0);q=p.maps[0]['q'][indices]
    baseline=p.data.qpos[q].copy();target=p.data.geom_xpos[p.fists[0,0]].copy()
    before=p.data.qpos.copy();velocity=p.data.qvel.copy()
    budget=GeometryBudget(100)
    result=static_target_search(p,0,before,baseline,target,budget)
    assert result['best_error_m']<1e-7 and result['physical_controls']==0
    assert not result['dynamic_feasibility_proven'] and not result['global_infeasibility_proven']
    np.testing.assert_array_equal(p.data.qpos,before);np.testing.assert_array_equal(p.data.qvel,velocity)
    with pytest.raises(RuntimeError):static_target_search(p,0,before,baseline,target,GeometryBudget(0))


def test_motor_observer_and_telemetry_leave_controller_actions_identical(foundation):
    motor=lambda obs,key:(jp.zeros((2,29)),{})
    a=StrikingEnv(foundation,None,motor=motor);b=StrikingEnv(foundation,None,motor=MotorTap(motor))
    a.reset(73);b.reset(73);ca=JabFeasibility(0);cb=JabFeasibility(0)
    ca.started=cb.started=0.;a.pair.elapsed=b.pair.elapsed=.44
    x,phase=ca.action(a);y,other=cb.action(b)
    before=b.pair.data.qpos.copy()
    measurements=controller_measurements(b,cb,y,other,b.motor.last)
    np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(b.pair.data.qpos,before)
    assert measurements['active'] and measurements['phase_code']==2 and phase==other
