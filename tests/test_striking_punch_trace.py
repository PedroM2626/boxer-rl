"""Passive instrumentation preserves native dynamics; windows remain descriptive."""
import numpy as np

from boxer.paired_foundation import PairedFoundation
from boxer.striking_punch_trace import PunchTrace,describe_events
from test_paired_foundation import foundation


def test_native_trace_does_not_change_physics_actions_or_score(foundation,tmp_path):
    plain=PairedFoundation(foundation);observed=PairedFoundation(foundation)
    plain.reset(seed=73);observed.reset(seed=73)
    trace=PunchTrace();observed.diagnostic_punch_trace=trace
    rng=np.random.default_rng(99)
    for _ in range(10):
        action=rng.normal(0,.03,(2,29));a=plain.step(action);b=observed.step(action)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        for key in ('qpos','qvel','ctrl','qfrc_applied','xfrc_applied'):
            np.testing.assert_array_equal(getattr(plain.data,key),getattr(observed.data,key))
        np.testing.assert_array_equal(plain.scorecard.last_score,observed.scorecard.last_score)
    assert len(trace.rows)==100
    path=tmp_path/'native.npz';trace.save(path)
    with np.load(path) as saved:
        assert saved['local_position'].shape==(100,2,2,3)
        assert all(np.isfinite(saved[k]).all() for k in saved.files)
        assert describe_events(saved)==[]
    assert observed.model.neq==0 and not np.any(observed.model.body_gravcomp)


def test_descriptive_windows_find_onset_and_mark_censored_recovery():
    n=400;time=np.arange(n)*.002
    arrays=dict(time_before=time,time_after=time+.002,
        local_position=np.zeros((n,2,2,3)),local_velocity=np.zeros((n,2,2,3)),
        elbow=np.zeros((n,2,2)),present=np.zeros((n,2,2),bool),
        engaged_before=np.zeros((n,2,2),bool),event=np.zeros((n,2,2),bool),
        impulse=np.zeros((n,2,2)),window_impulse=np.zeros((n,2,2)),
        contact_closing=np.zeros((n,2,2)),contact_advancing=np.zeros((n,2,2)))
    arrays['local_position'][:,0,0,0]=time
    arrays['local_velocity'][:,0,0,0]=1.
    arrays['elbow'][:,0,0]=time*2
    arrays['present'][125:165,0,0]=True
    arrays['present'][200:225,0,0]=True
    arrays['engaged_before'][126:200,0,0]=True
    arrays['impulse'][125:165,0,0]=.013
    arrays['event'][164,0,0]=True
    arrays['window_impulse'][164,0,0]=.52
    arrays['contact_closing'][125,0,0]=.4
    arrays['contact_advancing'][125,0,0]=.3
    event,=describe_events(arrays)
    assert event['onset_time']==.25 and event['onset_relative_speed']==1.
    assert event['onset_forward_cosine']==1.
    np.testing.assert_allclose(event['pre_max_displacement_m'],.25)
    np.testing.assert_allclose(event['pre_elbow_excursion_rad'],.5)
    np.testing.assert_allclose(event['native_contact_window_impulse_ns'],.52)
    assert not event['post_window_complete']
    assert event['return_toward_pre_window_position_m']<0  # moving away, not recovery
    assert event['post_contact_seconds'] == .05
    assert describe_events(arrays, native_timestep=.004)[0]['post_contact_seconds'] == .1


def test_body_translation_and_rotation_are_not_measured_as_arm_motion(foundation):
    pair=PairedFoundation(foundation);trace=PunchTrace()
    positions=pair.data.geom_xpos.copy();velocities={};torso=[]
    motion=np.array([.2,-.3,.4,.5,.1,-.2])  # angular then linear velocity
    for corner,m in enumerate(pair.maps):
        origin=pair.data.site(m['side']+'imu_in_torso').xpos.copy()
        torso.append((origin,motion.copy()))
        for fist in pair.fists[corner]:
            velocities[fist]=motion[3:]+np.cross(motion[:3],positions[fist]-origin)
    trace.before_substep(pair,positions,velocities,torso)
    np.testing.assert_allclose(trace.pending['local_velocity'],0.,atol=1e-15)
    np.testing.assert_allclose(trace.pending['toward_nearest_target'],0.,atol=1e-15)
