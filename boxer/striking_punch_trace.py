"""Passive 2 ms fist kinematics and legacy impact-window observations."""
import io

import numpy as np

from boxer.checkpoint_io import atomic_bytes


class PunchTrace:
    """Read simulation state only; never classify a contact as a certified punch."""
    def __init__(self):
        self.rows=[]

    def before_substep(self, pair, positions, velocities, torso_motion):
        local_position=np.zeros((2,2,3));local_velocity=np.zeros_like(local_position)
        elbow=np.zeros((2,2));toward=np.zeros((2,2))
        for corner,m in enumerate(pair.maps):
            origin,velocity=torso_motion[corner]
            rotation=pair.data.site(m['side']+'imu_in_torso').xmat.reshape(3,3)
            targets=sorted(pair.targets[1-corner])
            for hand,fist in enumerate(pair.fists[corner]):
                offset=positions[fist]-origin
                relative=velocities[fist]-velocity[3:]-np.cross(velocity[:3],offset)
                local_position[corner,hand]=rotation.T@offset
                local_velocity[corner,hand]=rotation.T@relative
                delta=min((positions[t]-positions[fist] for t in targets),key=np.linalg.norm)
                toward[corner,hand]=relative@delta/max(np.linalg.norm(delta),1e-12)
                label=('left','right')[hand]
                joint=pair.model.joint(m['side']+label+'_elbow_joint').id
                elbow[corner,hand]=pair.data.qpos[pair.model.jnt_qposadr[joint]]
        self.pending=dict(time_before=float(pair.data.time),local_position=local_position,
            local_velocity=local_velocity,elbow=elbow,toward_nearest_target=toward,
            engaged_before=pair.scorecard.engaged.copy(),last_score_before=pair.scorecard.last_score.copy())

    def after_substep(self, pair, present, impulse, closing, advancing):
        row=self.pending;previous=row.pop('last_score_before')
        row.update(time_after=float(pair.data.time),present=present.copy(),impulse=impulse.copy(),
            contact_closing=closing.copy(),contact_advancing=advancing.copy(),
            event=pair.scorecard.last_score>previous,
            window_impulse=pair.scorecard.impulse.copy())
        self.rows.append(row);del self.pending

    def arrays(self):
        if not self.rows:raise ValueError('nonempty native punch trace required')
        result={k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
        if not all(np.isfinite(x).all() for x in result.values()):
            raise FloatingPointError('nonfinite passive punch trace')
        return result

    def save(self,path):
        data=io.BytesIO();np.savez_compressed(data,**self.arrays());atomic_bytes(path,data.getvalue())


def describe_events(arrays, pre_seconds=.25, post_seconds=.5, native_timestep=.002):
    """Descriptive windows, not prospective punch acceptance thresholds.

    Onset speed is measured before the solver at the contact's first substep.
    Retraction is approach back toward that onset's pre-window hand position;
    it does not assert that the reference position is a competent guard.
    """
    if not np.isfinite(native_timestep) or native_timestep <= 0:
        raise ValueError('positive measured native timestep required')
    times=arrays['time_before'];positions=arrays['local_position']
    velocities=arrays['local_velocity'];events=[]
    for index,corner,hand in zip(*np.nonzero(arrays['event'])):
        onset=index
        while onset>0 and arrays['engaged_before'][onset,corner,hand]:onset-=1
        time=float(times[onset]);event_time=float(arrays['time_after'][index])
        pre=np.where((times>=time-pre_seconds)&(times<=time))[0]
        post=np.where((times>=event_time)&(times<=event_time+post_seconds))[0]
        path=positions[pre,corner,hand]
        distance=np.linalg.norm(path-path[0],axis=1)
        reference=path[0]
        impact_distance=float(np.linalg.norm(positions[index,corner,hand]-reference))
        after=np.linalg.norm(positions[post,corner,hand]-reference,axis=1) if len(post) else np.array([])
        speed=np.linalg.norm(velocities[pre,corner,hand],axis=1)
        onset_velocity=velocities[onset,corner,hand];onset_speed=float(np.linalg.norm(onset_velocity))
        events.append(dict(corner=int(corner),hand=int(hand),onset_time=time,event_time=event_time,
            onset_relative_speed=onset_speed,
            onset_forward_cosine=float(onset_velocity[0]/max(onset_speed,1e-12)),
            onset_contact_closing=float(arrays['contact_closing'][onset,corner,hand]),
            onset_contact_advancing=float(arrays['contact_advancing'][onset,corner,hand]),
            pre_window_seconds=float(times[pre[-1]]-times[pre[0]]),
            pre_peak_relative_speed=float(speed.max()),
            pre_max_displacement_m=float(distance.max()),
            pre_path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
            pre_elbow_excursion_rad=float(np.ptp(arrays['elbow'][pre,corner,hand])),
            event_window_impulse_ns=float(arrays['window_impulse'][index,corner,hand]),
            native_contact_window_impulse_ns=float(arrays['impulse'][onset:index+1,corner,hand].sum()),
            post_window_seconds=float(times[post[-1]]-event_time) if len(post) else 0.,
            post_contact_seconds=float(arrays['present'][post,corner,hand].sum()*native_timestep),
            return_toward_pre_window_position_m=float(impact_distance-after.min()) if len(after) else None,
            post_window_complete=bool(len(post) and times[post[-1]]>=event_time+post_seconds-(native_timestep+.0001))))
    return events
