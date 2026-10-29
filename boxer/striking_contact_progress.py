"""Target-coupled shaping and graded physical contact progress over v14."""
import copy
import numpy as np
from boxer.striking_motion_reward import MotionCycleMonitor
from boxer.striking_punch_trace import describe_events

SOURCE='checkpoints/striking-validation/motion-cycle-v14/step-1048576'
SOURCE_SHA='be59681ddf896daeb32cbb276c64ce1e6d0254a912a3c582bff32d9018b35ab5'
PROFILE=dict(version=1,kind='target_coupled_contact_progress',gamma=.999,
    potential_scale=2.,distance_scale_m=.3,physical_progress_bonus=2.,
    replaces_forward_velocity_preparation=True,complete_cycle_bonus_unchanged=10.,
    original_physical_gates_unchanged=True,retention_loss_weight=0.)
TRANSITION=dict(source_sha256=SOURCE_SHA,consumed_steps=3129344,optimized_steps=3129344,
    world_control_steps=2506752,actor_count=31368,critic_count=55040,
    optimizer_reset=False,old_profile_version=14,new_profile_version=15)


def target_distances(pair,positions):
    return np.asarray([[min(np.linalg.norm(positions[int(f)]-positions[int(t)])
        for t in pair.targets[1-c]) for f in pair.fists[c]] for c in (0,1)])


def potential(distances):
    d=np.asarray(distances)
    if d.shape!=(2,2) or not np.isfinite(d).all() or np.any(d<0):
        raise ValueError('finite nonnegative per-corner/hand target distances required')
    return np.exp(-d/PROFILE['distance_scale_m']).sum(axis=1)


def graded_contact(event):
    # A subgoal payment, never a declaration that a full punch has passed.
    return float(np.clip(event['onset_relative_speed'],0,1)
        *np.clip(event['onset_forward_cosine']/.7,0,1)
        *np.clip(event['pre_max_displacement_m']/.12,0,1))


class ContactProgressMonitor(MotionCycleMonitor):
    def __init__(self):
        super().__init__();self.last_components=None;self.current_pair=None

    def before_substep(self,pair,positions,velocities,torso_motion):
        super().before_substep(pair,positions,velocities,torso_motion)
        self.pending['target_center_distance']=target_distances(pair,positions)
        self.current_pair=pair

    def consume(self,dt=.02,terminal=False,*,target_distances_after=None):
        if not self.rows or not self.new_rows:return np.zeros(2),np.zeros(2)
        recent=self.rows[-self.new_rows:]
        old_phi=potential(recent[0]['target_center_distance'])
        if terminal:
            new_phi=np.zeros(2)
        else:
            if target_distances_after is None:
                target_distances_after=target_distances(self.current_pair,self.current_pair.data.geom_xpos)
            new_phi=potential(target_distances_after)
        target_progress=PROFILE['potential_scale']*(PROFILE['gamma']*new_phi-old_phi)
        new_events={(int(c),int(h),float(r['time_after'])) for r in recent for c,h in zip(*np.nonzero(r['event']))}
        contact_progress=np.zeros(2)
        if new_events:
            for event in describe_events(self.arrays()):
                if (event['corner'],event['hand'],event['event_time']) in new_events:
                    contact_progress[event['corner']]+=PROFILE['physical_progress_bonus']*graded_contact(event)
        cycles,_=super().consume(dt,terminal=terminal)
        self.last_components=dict(target_potential=target_progress.copy(),physical_contact_progress=contact_progress.copy())
        return cycles,target_progress+contact_progress-contact_progress[::-1]


def validate_progress_metadata(saved):
    if (saved.get('contact_progress_ppo') is not True or saved.get('contact_progress_profile')!=PROFILE
            or saved.get('contact_progress_transition')!=TRANSITION or not saved.get('motion_cycle_ppo')
            or saved.get('retention_loss_weight')!=0. or saved['optimized_steps']<3129344
            or saved['consumed_steps']<saved['optimized_steps']):
        raise ValueError('explicit latest-v14 contact-progress transition and retained optimizer required')


def prepare_progress_state(saved,sha):
    from boxer.striking_train import require_ppo_checkpoint
    require_ppo_checkpoint(saved)
    if (sha!=SOURCE_SHA or saved.get('contact_progress_ppo') or not saved.get('motion_cycle_ppo')
            or saved['consumed_steps']!=saved['optimized_steps'] or saved['optimized_steps']!=3129344
            or saved['world_control_steps']!=2506752 or int(saved['optimizer']['actor_adam'][0].count)!=31368
            or int(saved['optimizer']['critic_adam'][0].count)!=55040):
        raise ValueError('actual completed v14 final required without rollback or optimizer reset')
    result=copy.deepcopy(saved);result.update(contact_progress_ppo=True,contact_progress_profile=copy.deepcopy(PROFILE),contact_progress_transition=copy.deepcopy(TRANSITION))
    require_ppo_checkpoint(result);return result


def progress_config(source):
    if source['version']!=14 or not source.get('motion_cycle_ppo') or source['retention_loss_weight']!=0.:
        raise ValueError('completed v14 profile required')
    return dict(source,version=15,contact_progress_ppo=True,contact_progress_profile=copy.deepcopy(PROFILE))
