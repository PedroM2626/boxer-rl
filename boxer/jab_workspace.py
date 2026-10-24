"""Read-only controller telemetry and bounded static geometry on cloned states."""
import mujoco
import numpy as np
from scipy.optimize import least_squares

from boxer.jab_feasibility import PROFILE,cycle_fraction

PHASES={'approach':0,'guard':1,'extend':2,'retract':3,'rest':4}


class MotorTap:
    """Observe exactly the motor result requested by the original controller."""
    def __init__(self,motor):self.motor=motor;self.last=None;self.calls=0
    def __call__(self,*args,**kwargs):
        result=self.motor(*args,**kwargs)
        self.last=np.asarray(result[0]).copy();self.calls+=1
        return result


def left_arm(env,corner):
    indices=np.array([i for i in env.arm_indices if 'left_' in env.pair.foundation.mj_model.actuator(int(i)).name])
    if len(indices)!=7:raise ValueError('original seven left-arm channels required')
    slots=np.array([int(np.flatnonzero(env.arm_indices==i)[0]) for i in indices])
    return indices,slots


def controller_measurements(env,controller,action,phase,motor_result):
    """Recompute request/clipping evidence without writing live physics state."""
    p=env.pair;c=controller.corner;m=p.maps[c];indices,slots=left_arm(env,c)
    empty=np.zeros(7)
    if phase=='approach':
        return dict(phase_code=np.int64(0),active=np.bool_(False),fraction=0.,
            guard_world=np.zeros(3),strike_world=np.zeros(3),target_world=np.zeros(3),
            baseline_target=empty.copy(),desired_target=empty.copy(),filtered_target=empty.copy(),
            unbounded_request=empty.copy(),action_clipped=np.zeros(7,bool),target_error_m=0.)
    assert motor_result is not None
    actual,fraction=cycle_fraction(p.elapsed-controller.started);assert actual==phase
    site=p.data.site(m['side']+'imu_in_torso')
    guard=site.xpos+site.xmat.reshape(3,3)@np.asarray(PROFILE['guard_local'])
    foe=p.model.geom(p.maps[1-c]['side']+'torso_collision').id
    strike=p.data.geom_xpos[foe]+[0,0,PROFILE['target_torso_world_height_offset']]
    target=guard+fraction*(strike-guard);fist=int(p.fists[c,0])
    jac=np.zeros((3,p.model.nv));rotation=np.zeros_like(jac)
    mujoco.mj_jacGeom(p.model,p.data,jac,rotation,fist);jac=jac[:,m['v'][indices]]
    delta=jac.T@np.linalg.solve(jac@jac.T+PROFILE['damping']**2*np.eye(3),target-p.data.geom_xpos[fist])
    delta*=min(1.,PROFILE['joint_correction_cap']/max(float(np.max(np.abs(delta))),1e-12))
    desired=p.data.qpos[m['q'][indices]]+delta
    baseline=p.pose[indices]+p.action_scale*motor_result[c,indices]
    requested=(desired-baseline)/env.residual_scale
    before_clip=(np.clip(requested,-1,1)-.8*env.residual[c,slots])/.2
    np.testing.assert_array_equal(action[c,3+slots],np.clip(before_clip,-1,1))
    filtered=baseline+env.residual_scale*(.8*env.residual[c,slots]+.2*action[c,3+slots])
    return dict(phase_code=np.int64(PHASES[phase]),active=np.bool_(True),fraction=float(fraction),
        guard_world=guard.copy(),strike_world=strike.copy(),target_world=target.copy(),
        baseline_target=baseline.copy(),desired_target=desired.copy(),filtered_target=filtered.copy(),
        unbounded_request=requested.copy(),action_clipped=np.abs(before_clip)>1.,
        target_error_m=float(np.linalg.norm(target-p.data.geom_xpos[fist])))


class GeometryBudget:
    def __init__(self,limit=20000):self.limit=limit;self.used=0
    def consume(self):
        if self.used>=self.limit:raise RuntimeError('cloned kinematic evaluation ceiling reached')
        self.used+=1


def static_target_search(pair,corner,qpos,baseline,target,budget):
    """Local geometry only: not dynamic reachability, balance or collision safety.

    Roots/legs/other joints are held at a recorded pose in a disposable MjData.
    Only kinematics/com-position callbacks run; no mj_step or live state writes.
    """
    model=pair.model;m=pair.maps[corner]
    indices=np.array([i for i in range(29) if 'left_' in pair.foundation.mj_model.actuator(i).name
                      and any(s in pair.foundation.mj_model.actuator(i).name for s in ('shoulder','elbow','wrist'))])
    assert len(indices)==7
    q=m['q'][indices];v=m['v'][indices];joints=model.actuator_trnid[m['act'][indices],0]
    if not np.all(model.jnt_limited[joints]):raise ValueError('limited original arm joints required')
    lower=np.maximum(np.asarray(baseline)-.8,model.jnt_range[joints,0])
    upper=np.minimum(np.asarray(baseline)+.8,model.jnt_range[joints,1])
    if np.any(lower>=upper):raise ValueError('empty motor-centered target box')
    clone=mujoco.MjData(model);clone.qpos[:]=qpos;clone.qvel[:]=0.
    fist=int(pair.fists[corner,0]);cached=None;position=None
    def update(x):
        nonlocal cached,position
        if cached is None or not np.array_equal(cached,x):
            budget.consume();clone.qpos[q]=x
            mujoco.mj_kinematics(model,clone);mujoco.mj_comPos(model,clone)
            position=clone.geom_xpos[fist].copy();cached=np.asarray(x).copy()
            if not np.isfinite(position).all():raise FloatingPointError('nonfinite cloned geometry')
    def fun(x):update(x);return position-target
    def jac(x):
        update(x);j=np.zeros((3,model.nv));r=np.zeros_like(j)
        mujoco.mj_jacGeom(model,clone,j,r,fist);return j[:,v]
    before=budget.used;attempts=[]
    for label,start in (('recorded',np.asarray(qpos)[q]),('motor',np.asarray(baseline))):
        result=least_squares(fun,np.clip(start,lower,upper),jac=jac,bounds=(lower,upper),
            max_nfev=40,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        error=float(np.linalg.norm(fun(result.x)))
        attempts.append(dict(start=label,error_m=error,targets=result.x.tolist(),
            function_evaluations=int(result.nfev),solver_status=int(result.status)))
    best=min(attempts,key=lambda a:a['error_m']);fun(np.asarray(best['targets']))
    return dict(best_error_m=best['error_m'],best_targets=best['targets'],lower=lower.tolist(),upper=upper.tolist(),
        target_world=np.asarray(target).tolist(),best_fist_world=position.tolist(),
        joint_coordinates=q.tolist(),attempts=attempts,cloned_forward_evaluations=budget.used-before,
        physical_controls=0,global_infeasibility_proven=False,dynamic_feasibility_proven=False)
