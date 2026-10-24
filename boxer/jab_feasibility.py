"""Scripted physical feasibility probe only; never a learned boxing policy."""
import jax
import jax.numpy as jp
import mujoco
import numpy as np

PROFILE=dict(version=1,guard_seconds=.35,extend_seconds=.18,retract_seconds=.25,
    rest_seconds=.30,engage_distance=.75,guard_local=[.16,.16,.12],
    target_torso_world_height_offset=.10,joint_correction_cap=.16,
    damping=.05,arm_residual_radians=.8,filter_old=.8,filter_new=.2)


def cycle_fraction(elapsed):
    """Continue cycles after contacts, with explicit return to the guard target."""
    if not np.isfinite(elapsed) or elapsed<0:raise ValueError('nonnegative finite elapsed time required')
    duration=sum(PROFILE[k] for k in ('guard_seconds','extend_seconds','retract_seconds','rest_seconds'))
    t=elapsed%duration
    g=PROFILE['guard_seconds'];e=PROFILE['extend_seconds'];r=PROFILE['retract_seconds']
    if t<g:return 'guard',0.
    if t<g+e:return 'extend',float((1-np.cos(np.pi*(t-g)/e))/2)
    if t<g+e+r:return 'retract',float((1+np.cos(np.pi*(t-g-e)/r))/2)
    return 'rest',0.


class JabFeasibility:
    def __init__(self,corner):
        if corner not in (0,1):raise ValueError('one physical corner required')
        self.corner=corner;self.started=None

    def phase_fraction(self,elapsed):
        return cycle_fraction(elapsed)

    def guard_target(self,env):
        site=env.pair.data.site(env.pair.maps[self.corner]['side']+'imu_in_torso')
        return site.xpos+site.xmat.reshape(3,3)@np.asarray(PROFILE['guard_local'])

    def action(self,env):
        pair=env.pair;corner=self.corner
        roots=[pair.data.qpos[m['root']:m['root']+2] for m in pair.maps]
        distance=np.linalg.norm(roots[1]-roots[0]);result=np.zeros((2,17))
        if self.started is None:
            if distance>PROFILE['engage_distance']:return result,'approach'
            self.started=pair.elapsed
        phase,fraction=self.phase_fraction(pair.elapsed-self.started)
        mapping=pair.maps[corner];guard=self.guard_target(env)
        foe=pair.model.geom(pair.maps[1-corner]['side']+'torso_collision').id
        target=pair.data.geom_xpos[foe]+np.array([0,0,PROFILE['target_torso_world_height_offset']])
        desired_position=guard+fraction*(target-guard)
        indices=np.array([i for i in env.arm_indices if 'left_' in pair.foundation.mj_model.actuator(int(i)).name])
        if len(indices)!=7:raise ValueError('seven original left-arm channels required')
        fist=int(pair.fists[corner,0]);jac=np.zeros((3,pair.model.nv));rot=np.zeros_like(jac)
        mujoco.mj_jacGeom(pair.model,pair.data,jac,rot,fist)
        jac=jac[:,mapping['v'][indices]]
        delta=jac.T@np.linalg.solve(jac@jac.T+PROFILE['damping']**2*np.eye(3),desired_position-pair.data.geom_xpos[fist])
        delta*=min(1.,PROFILE['joint_correction_cap']/max(float(np.max(np.abs(delta))),1e-12))
        desired=pair.data.qpos[mapping['q'][indices]]+delta
        motor,_=env.motor({'state':jp.array(pair.observation(env.commands(result))),
            'privileged_state':jp.zeros((2,216))},jax.random.PRNGKey(0))
        baseline=pair.pose[indices]+pair.action_scale*np.asarray(motor)[corner,indices]
        slots=np.array([int(np.flatnonzero(env.arm_indices==i)[0]) for i in indices])
        requested=np.clip((desired-baseline)/env.residual_scale,-1,1)
        result[corner,3+slots]=np.clip((requested-.8*env.residual[corner,slots])/.2,-1,1)
        if not np.isfinite(result).all():raise FloatingPointError('invalid bounded jab diagnostic action')
        return result,phase


class GuardOnly(JabFeasibility):
    """Matched diagnostic: retain the exact guard target without an arm cycle."""
    def phase_fraction(self,elapsed):
        if not np.isfinite(elapsed) or elapsed<0:raise ValueError('nonnegative finite elapsed time required')
        return 'guard',0.


class GuardRamp(GuardOnly):
    """Diagnostic two-second smooth entry into the identical final guard point."""
    ramp_seconds=2.

    def __init__(self,corner):
        super().__init__(corner);self.entry_local=None

    def guard_target(self,env):
        if self.started is None:raise ValueError('guard engagement must precede target calculation')
        pair=env.pair;site=pair.data.site(pair.maps[self.corner]['side']+'imu_in_torso')
        rotation=site.xmat.reshape(3,3)
        if self.entry_local is None:
            self.entry_local=rotation.T@(pair.data.geom_xpos[pair.fists[self.corner,0]]-site.xpos)
        t=float(np.clip((pair.elapsed-self.started)/self.ramp_seconds,0.,1.))
        blend=t*t*(3.-2.*t)
        local=self.entry_local+blend*(np.asarray(PROFILE['guard_local'])-self.entry_local)
        return site.xpos+rotation@local


class MotorGuardJab(JabFeasibility):
    """Diagnostic: extend from the learned guard, return through the motor/filter."""
    def __init__(self,corner):
        super().__init__(corner);self.entry_local=None;self.extension_cycle=None

    def cycle_duration(self):
        return sum(PROFILE[k] for k in ('guard_seconds','extend_seconds','retract_seconds','rest_seconds'))

    def action(self,env):
        pair=env.pair
        if self.started is None:
            roots=[pair.data.qpos[m['root']:m['root']+2] for m in pair.maps]
            if np.linalg.norm(roots[1]-roots[0])>PROFILE['engage_distance']:
                return np.zeros((2,17)),'approach'
            self.started=pair.elapsed
        elapsed=pair.elapsed-self.started;phase,_=self.phase_fraction(elapsed)
        if phase!='extend':return np.zeros((2,17)),phase
        cycle=int(elapsed//self.cycle_duration())
        if self.extension_cycle!=cycle:
            self.extension_cycle=cycle
            site=pair.data.site(pair.maps[self.corner]['side']+'imu_in_torso')
            self.entry_local=site.xmat.reshape(3,3).T@(pair.data.geom_xpos[pair.fists[self.corner,0]]-site.xpos)
        return super().action(env)

    def guard_target(self,env):
        if self.entry_local is None:raise ValueError('capture the learned guard before extension')
        site=env.pair.data.site(env.pair.maps[self.corner]['side']+'imu_in_torso')
        return site.xpos+site.xmat.reshape(3,3)@self.entry_local


class MotorGuardDwell(MotorGuardJab):
    """Diagnostic: retain the terminal extension target for another 120 ms."""
    dwell_seconds=.12

    def cycle_duration(self):return super().cycle_duration()+self.dwell_seconds

    def phase_fraction(self,elapsed):
        if not np.isfinite(elapsed) or elapsed<0:raise ValueError('nonnegative finite elapsed time required')
        t=elapsed%self.cycle_duration();g=PROFILE['guard_seconds'];e=PROFILE['extend_seconds'];r=PROFILE['retract_seconds']
        if t<g:return 'guard',0.
        if t<g+e:return 'extend',float((1-np.cos(np.pi*(t-g)/e))/2)
        if t<g+e+self.dwell_seconds:return 'extend',1.
        if t<g+e+self.dwell_seconds+r:
            return 'retract',float((1+np.cos(np.pi*(t-g-e-self.dwell_seconds)/r))/2)
        return 'rest',0.
