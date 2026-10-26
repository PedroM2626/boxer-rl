"""Target-coupled subgoals do not replace full physical-cycle acceptance."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import jax
import numpy as np
import pytest
from boxer.striking_contact_progress import (SOURCE,SOURCE_SHA,PROFILE,prepare_progress_state,progress_config,
    ContactProgressMonitor,potential,graded_contact)
from boxer.striking_train import require_ppo_checkpoint,validate_resume_config,main
from tools.striking_contact_compare import load_checked
from test_strike_motion import event


def test_latest_v14_migration_preserves_every_old_saved_field():
    source,_=load_checked(Path(SOURCE),SOURCE_SHA);prepared=prepare_progress_state(source,SOURCE_SHA)
    for key in source:
        for a,b in zip(jax.tree_util.tree_leaves(source[key]),jax.tree_util.tree_leaves(prepared[key])):
            np.testing.assert_equal(a,b)
    require_ppo_checkpoint(prepared)
    for key in ('contact_progress_ppo','contact_progress_profile','contact_progress_transition'):
        bad=copy.deepcopy(prepared);del bad[key]
        with pytest.raises(ValueError):require_ppo_checkpoint(bad)
    with pytest.raises(ValueError):prepare_progress_state(prepared,SOURCE_SHA)
    old=json.loads((Path(SOURCE)/'config.json').read_text());new=progress_config(old)
    assert old['version']==14 and new['version']==15
    with pytest.raises(ValueError):validate_resume_config(old,new)
    with pytest.raises(ValueError):main(['--contact-progress-ppo'])


def test_distance_potential_rewards_closing_and_terminal_has_no_positive_bonus():
    far=potential(np.full((2,2),.8));near=potential(np.full((2,2),.15))
    assert np.all(PROFILE['gamma']*near-far>0)
    assert np.all(PROFILE['gamma']*far-near<0)
    assert np.all(-near<0)
    for bad in (np.full((2,2),-.1),np.full((2,2),np.nan)):
        with pytest.raises(ValueError):potential(bad)


def test_discounted_potential_terms_telescope_with_zero_terminal():
    gamma=PROFILE['gamma'];phi=[potential(np.full((2,2),d)) for d in (.8,.3,.1)]+[np.zeros(2)]
    total=sum(gamma**t*(gamma*phi[t+1]-phi[t]) for t in range(3))
    np.testing.assert_allclose(total,-phi[0],atol=1e-12)


def test_same_hand_progress_is_graded_and_incomplete_progress_is_not_a_valid_cycle():
    from boxer.striking_motion_reward import cycle_qualifies
    assert graded_contact(event())==1.
    assert graded_contact(event(.3,.2))<.1
    sideways=event(3.,-.2);assert graded_contact(sideways)==0.
    e=event(complete=False);assert graded_contact(e)==1. and not cycle_qualifies(e)


def test_monitor_replaces_air_velocity_term_and_accounts_for_terminal(monkeypatch):
    import boxer.striking_contact_progress as module
    from boxer.striking_motion_reward import MotionCycleMonitor
    monitor=ContactProgressMonitor()
    monitor.rows=[dict(target_center_distance=np.full((2,2),.8),event=np.zeros((2,2),bool),time_after=.02)]
    monitor.new_rows=1;monitor.current_pair=SimpleNamespace(data=SimpleNamespace(geom_xpos=None))
    monkeypatch.setattr(module,'target_distances',lambda pair,pos:np.full((2,2),.15))
    def base(self,dt=.02,terminal=False):self.new_rows=0;return np.zeros(2),np.full(2,999.)
    monkeypatch.setattr(MotionCycleMonitor,'consume',base)
    cycles,reward=monitor.consume()
    np.testing.assert_allclose(reward,2*(PROFILE['gamma']*potential(np.full((2,2),.15))-potential(np.full((2,2),.8))))
    monitor.new_rows=1;cycles,reward=monitor.consume(terminal=True)
    np.testing.assert_allclose(reward,-2*potential(np.full((2,2),.8)))
    np.testing.assert_array_equal(monitor.consume()[1],0.)


def test_physical_progress_is_per_hand_once_and_is_not_air_movement(monkeypatch):
    import boxer.striking_contact_progress as module
    from boxer.striking_motion_reward import MotionCycleMonitor
    monitor=ContactProgressMonitor();mark=np.zeros((2,2),bool);mark[0,0]=True
    monitor.rows=[dict(target_center_distance=np.full((2,2),.3),event=mark,time_after=.2)]
    monitor.new_rows=1;monitor.current_pair=SimpleNamespace(data=SimpleNamespace(geom_xpos=None))
    monkeypatch.setattr(module,'target_distances',lambda pair,pos:np.full((2,2),.3))
    e=event(complete=False);e.update(event_time=.2)
    monkeypatch.setattr(monitor,'arrays',lambda:{})
    monkeypatch.setattr(module,'describe_events',lambda a:[e])
    def base(self,dt=.02,terminal=False):self.new_rows=0;return np.zeros(2),np.full(2,999.)
    monkeypatch.setattr(MotionCycleMonitor,'consume',base)
    cycles,reward=monitor.consume();np.testing.assert_array_equal(cycles,0.)
    phi_term=2*(PROFILE['gamma']-1)*potential(np.full((2,2),.3))
    np.testing.assert_allclose(reward,phi_term+[2.,-2.])
    np.testing.assert_array_equal(monitor.consume()[1],0.)
    monitor.rows[-1]['event']=np.zeros((2,2),bool);monitor.new_rows=1
    _,reward=monitor.consume();np.testing.assert_allclose(reward,phi_term)
