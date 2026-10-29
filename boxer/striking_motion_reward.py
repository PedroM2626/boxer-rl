"""Explicit valid-cycle reward transition; physical gates remain independent."""
import copy
import numpy as np
from boxer.striking_punch_trace import PunchTrace,describe_events

SOURCE='checkpoints/striking-validation/pool-2064384/control/gate-16384'
SOURCE_SHA='3a34aa85f0e9cf1ecb178c02f5fe945f8dd17f45dc1b323208b6083e2091b501'
PROFILE=dict(version=1,kind='straight_punch_cycle_training',cycle_bonus=10.,forward_motion_rate=.5,
    retained_native_samples=650,onset_speed=1.,forward_cosine=.7,pre_travel=.12,
    return_travel=.06,max_contact_seconds=.10,post_window_seconds=.5,
    legacy_contact_payment=0.,retention_loss_weight=0.,either_hand=True)
TRANSITION=dict(source_sha256=SOURCE_SHA,consumed_steps=2080768,optimized_steps=2080768,
    world_control_steps=1458176,actor_count=14984,critic_count=38656,
    old_retention_loss_weight=1.,new_retention_loss_weight=0.,optimizer_reset=False)


def cycle_qualifies(e):
    return bool(e['onset_relative_speed']>=1. and e['onset_forward_cosine']>=.7
        and e['pre_max_displacement_m']>=.12 and e['post_window_complete']
        and e['return_toward_pre_window_position_m'] is not None
        and e['return_toward_pre_window_position_m']>=.06 and e['post_contact_seconds']<=.10)


class MotionCycleMonitor(PunchTrace):
    """Bounded native history; pay each measured completed cycle once, per hand."""
    def __init__(self):
        super().__init__();self.paid=set();self.pending_events=set();self.new_rows=0
        self.capture_completed_events=False;self.completed_evidence=[];self.control_policy_optimized_steps=-1

    def before_substep(self,pair,positions,velocities,torso_motion):
        super().before_substep(pair,positions,velocities,torso_motion)
        if self.capture_completed_events:
            self.pending.update(evidence_qpos_before=pair.data.qpos.copy(),evidence_qvel_before=pair.data.qvel.copy(),
                evidence_ctrl=pair.data.ctrl.copy(),policy_optimized_steps=np.int64(self.control_policy_optimized_steps),
                applied_force_norm=np.array(np.linalg.norm(pair.data.qfrc_applied)+np.linalg.norm(pair.data.xfrc_applied)))

    def take_completed_evidence(self):
        rows=self.completed_evidence;self.completed_evidence=[];return rows

    def _capture_events(self,row):
        for corner,hand in zip(*np.nonzero(row['event'])):
            self.pending_events.add((int(corner),int(hand),float(row['time_after'])))

    def append_row(self,row):
        super().append_row(row);self.new_rows+=1
        self._capture_events(row)
        if len(self.rows)>PROFILE['retained_native_samples']:del self.rows[0]

    def consume(self,dt=.02,terminal=False):
        if not self.rows or not self.new_rows:return np.zeros(2),np.zeros(2)
        recent=self.rows[-self.new_rows:]
        new=np.asarray([r['local_velocity'] for r in recent])
        speed=np.linalg.norm(new,axis=-1)
        toward=np.asarray([r['toward_nearest_target'] for r in recent])
        quality=np.clip(new[...,0],0.,1.)*np.clip(toward/np.maximum(speed,1e-9),0.,1.)
        dense=PROFILE['forward_motion_rate']*dt*np.mean(np.max(quality,axis=2),axis=0)
        cycles=np.zeros(2);now=float(self.rows[-1]['time_after'])
        ready={k for k in self.pending_events if now>=k[2]+.5}
        # Most controls have no mature impact: do not rebuild 650-row arrays then.
        if ready:
            arrays=self.arrays()
            for event in describe_events(arrays):
                key=(event['corner'],event['hand'],event['event_time'])
                if key in ready and event['post_window_complete']:
                    self.pending_events.discard(key)
                    if key not in self.paid:
                        self.paid.add(key)
                        if cycle_qualifies(event):
                            cycles[event['corner']]+=1
                            if self.capture_completed_events:
                                self.completed_evidence.append(dict(event=dict(event),arrays={k:v.copy() for k,v in arrays.items()}))
        self.paid={k for k in self.paid if k[2]>=now-2.}
        self.new_rows=0
        return cycles,dense


def replace_contact_payment(reward,outcome,cycles,dense,corner):
    scored=np.asarray(outcome['scored'])
    # Remove both old contact payments; retain fall, approach and other terms.
    return np.asarray(reward)-2.*scored[corner]+2.*scored[1-corner]+10.*(cycles[corner]-cycles[1-corner])+dense[corner]


def validate_motion_metadata(saved):
    if (saved.get('motion_cycle_ppo') is not True or saved.get('motion_reward_profile')!=PROFILE
            or saved.get('motion_cycle_transition')!=TRANSITION or saved.get('retention_loss_weight')!=0.
            or not saved.get('long_gae_ppo') or not saved.get('fixed_opponent_ppo')
            or saved.get('pool_opponent_ppo') or saved['optimized_steps']<TRANSITION['optimized_steps']
            or saved['consumed_steps']<saved['optimized_steps']):
        raise ValueError('explicit motion-cycle source, reward, zero teacher loss and retained optimizer required')


def prepare_motion_state(saved,sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha!=SOURCE_SHA or saved.get('motion_cycle_ppo') or saved.get('pool_opponent_ppo')
            or saved['consumed_steps']!=saved['optimized_steps'] or saved['consumed_steps']!=2080768
            or saved['world_control_steps']!=1458176 or saved['retention_loss_weight']!=1.
            or int(saved['optimizer']['actor_adam'][0].count)!=14984
            or int(saved['optimizer']['critic_adam'][0].count)!=38656):
        raise ValueError('latest completed fixed-control final required, without rollback')
    result=copy.deepcopy(saved)
    result.update(motion_cycle_ppo=True,motion_reward_profile=copy.deepcopy(PROFILE),
        motion_cycle_transition=copy.deepcopy(TRANSITION),retention_loss_weight=0.)
    require_ppo_checkpoint(result)
    return result


def motion_config(source):
    if source['version']!=12 or source['retention_loss_weight']!=1. or source.get('pool_opponent_ppo'):
        raise ValueError('latest original fixed-control profile required')
    return dict(source,version=14,motion_cycle_ppo=True,motion_reward_profile=copy.deepcopy(PROFILE),retention_loss_weight=0.)
