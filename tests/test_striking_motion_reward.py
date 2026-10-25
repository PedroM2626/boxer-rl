"""A completed physical cycle can pay once; a moving touch cannot substitute."""
import copy,json
from pathlib import Path
import jax
import numpy as np
import pytest
from boxer.striking_motion_reward import (SOURCE,SOURCE_SHA,PROFILE,prepare_motion_state,
    motion_config,cycle_qualifies,MotionCycleMonitor,replace_contact_payment)
from boxer.striking_train import require_ppo_checkpoint,validate_resume_config,main
from tools.striking_contact_compare import load_checked
from test_strike_motion import event


def test_warm_start_preserves_latest_actor_critic_adam_rng_and_normalizer():
    source,_=load_checked(Path(SOURCE),SOURCE_SHA);prepared=prepare_motion_state(source,SOURCE_SHA)
    for key in source.keys()-{'retention_loss_weight'}:
        for a,b in zip(jax.tree_util.tree_leaves(source[key]),jax.tree_util.tree_leaves(prepared[key])):
            np.testing.assert_equal(a,b)
    assert prepared['retention_loss_weight']==0.
    require_ppo_checkpoint(prepared)
    for key in ('motion_reward_profile','motion_cycle_transition','motion_cycle_ppo'):
        bad=copy.deepcopy(prepared);del bad[key]
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    bad=copy.deepcopy(prepared);bad['retention_loss_weight']=1.
    with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    with pytest.raises(ValueError):prepare_motion_state(prepared,SOURCE_SHA)
    old=json.loads((Path(SOURCE)/'config.json').read_text());new=motion_config(old)
    assert new['version']==14 and old['retention_loss_weight']==1.
    with pytest.raises(ValueError):validate_resume_config(old,new)
    with pytest.raises(ValueError):main(['--motion-cycle-ppo'])


@pytest.mark.parametrize('key,bad',[('onset_relative_speed',.99),('onset_forward_cosine',.69),
    ('pre_max_displacement_m',.119),('return_toward_pre_window_position_m',.059),
    ('post_contact_seconds',.101),('post_window_complete',False)])
def test_each_component_required_and_either_hand_is_eligible(key,bad):
    e=event();e['hand']=1;assert cycle_qualifies(e)
    e[key]=bad;assert not cycle_qualifies(e)


def test_monitor_waits_for_recovery_and_does_not_pay_twice(monkeypatch):
    import boxer.striking_motion_reward as module
    monitor=MotionCycleMonitor();monitor.rows=[dict(local_velocity=np.ones((2,2,3)),toward_nearest_target=np.ones((2,2)),time_after=1.)];monitor.new_rows=1
    monitor.pending_events.add((0,0,.2))
    arrays=dict(local_velocity=np.ones((1,2,2,3)),toward_nearest_target=np.ones((1,2,2)),time_after=np.array([1.]))
    monkeypatch.setattr(monitor,'arrays',lambda:arrays)
    e=event();e.update(event_time=.2,post_window_complete=False)
    monkeypatch.setattr(module,'describe_events',lambda a:[e])
    cycles,dense=monitor.consume();np.testing.assert_array_equal(cycles,0.);assert (dense>=0).all()
    e['post_window_complete']=True;monitor.new_rows=1
    cycles,_=monitor.consume();np.testing.assert_array_equal(cycles,[1.,0.])
    monitor.new_rows=1;cycles,_=monitor.consume();np.testing.assert_array_equal(cycles,0.)
    monitor=MotionCycleMonitor();assert not monitor.paid and not monitor.rows


def test_legacy_contact_payment_removed_but_fall_term_retained():
    outcome=dict(scored=np.array([1.,0.]))
    reward=replace_contact_payment(np.array([-3.]),outcome,np.zeros(2),np.zeros(2),0)
    np.testing.assert_array_equal(reward,[-5.])
    reward=replace_contact_payment(np.array([-2.]),outcome,np.zeros(2),np.zeros(2),1)
    np.testing.assert_array_equal(reward,[0.])
    reward=replace_contact_payment(np.zeros(1),dict(scored=np.zeros(2)),np.array([1.,0.]),np.zeros(2),0)
    np.testing.assert_array_equal(reward,[10.])


def test_saved_moving_touch_does_not_pay_full_cycle_and_history_is_bounded():
    root=Path('docs/results/striking-pilot/lateral-command-v5')
    audit=json.loads((root/'audit.json').read_text());row=next(r for r in audit['trials'] if r['events'])
    monitor=MotionCycleMonitor();total=np.zeros(2)
    with np.load(root/'native'/Path(row['native_trace_file']).name) as archive:
        a={k:archive[k].copy() for k in archive.files}
        for i in range(len(a['time_before'])):
            monitor.rows.append({k:v[i].copy() for k,v in a.items()});monitor.new_rows+=1
            monitor._capture_events(monitor.rows[-1])
            if len(monitor.rows)>PROFILE['retained_native_samples']:del monitor.rows[0]
            if i%10==9:
                cycles,dense=monitor.consume();total+=cycles;assert np.isfinite(dense).all()
            assert len(monitor.rows)<=650
    np.testing.assert_array_equal(total,0.)


def test_complete_synthetic_native_cycle_pays_once_only_after_full_post_window():
    n=550;t=np.arange(n)*.002
    arrays=dict(time_before=t,time_after=t+.002,
        local_position=np.zeros((n,2,2,3)),local_velocity=np.zeros((n,2,2,3)),
        elbow=np.zeros((n,2,2)),present=np.zeros((n,2,2),bool),
        engaged_before=np.zeros((n,2,2),bool),event=np.zeros((n,2,2),bool),
        impulse=np.zeros((n,2,2)),window_impulse=np.zeros((n,2,2)),
        contact_closing=np.zeros((n,2,2)),contact_advancing=np.zeros((n,2,2)),
        toward_nearest_target=np.zeros((n,2,2)))
    arrays['local_position'][:,0,1,0]=np.where(t<=.328,t,np.maximum(.05,.328-2*(t-.328)))
    arrays['local_velocity'][:165,0,1,0]=1.
    arrays['engaged_before'][126:165,0,1]=True
    arrays['present'][125:165,0,1]=True
    arrays['event'][164,0,1]=True
    arrays['impulse'][125:165,0,1]=.013
    arrays['window_impulse'][164,0,1]=.52
    monitor=MotionCycleMonitor();total=np.zeros(2)
    for i in range(n):
        monitor.rows.append({k:v[i].copy() for k,v in arrays.items()});monitor.new_rows+=1
        monitor._capture_events(monitor.rows[-1])
        if i%10==9:
            cycles,_=monitor.consume();total+=cycles
            if t[i]<.826:np.testing.assert_array_equal(total,0.)
    np.testing.assert_array_equal(total,[1.,0.])


def test_independent_gate_matches_full_per_hand_conjunction():
    from tools.striking_motion_cycle_supervisor import independent_cycle
    rng=np.random.default_rng(909)
    for _ in range(200):
        e=event();e.update(hand=int(rng.integers(0,2)),
            onset_relative_speed=float(rng.uniform(0,3)),onset_forward_cosine=float(rng.uniform(-1,1)),
            pre_max_displacement_m=float(rng.uniform(0,.3)),post_window_complete=bool(rng.integers(0,2)),
            return_toward_pre_window_position_m=float(rng.uniform(-.1,.2)),post_contact_seconds=float(rng.uniform(0,.2)))
        assert independent_cycle(e)==cycle_qualifies(e)
