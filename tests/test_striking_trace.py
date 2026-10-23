"""Passive trace agreement and descriptive coverage without self leakage."""
import jax.numpy as jp
import numpy as np

from boxer.striking_contact import ContactAwareStrikingEnv
from boxer.striking_trace import StrikingTrace,terminal_details,root_distance
from boxer.striking_coverage import block_distances,leave_trajectory_out_baselines
from test_paired_foundation import foundation
from tools.evaluate_striking import ablate_action


def test_trace_is_read_only_with_identical_native_trajectory(foundation,tmp_path):
    def motor(obs,key):return jp.ones((2,29))*.02,{}
    plain=ContactAwareStrikingEnv(foundation,None,motor=motor)
    traced=ContactAwareStrikingEnv(foundation,None,motor=motor)
    a=plain.reset(23);b=traced.reset(23);trace=StrikingTrace(traced)
    initial_distance=root_distance(traced.pair)
    action=np.random.default_rng(2).uniform(-.1,.1,(2,17)).astype(np.float32)
    for _ in range(5):
        trace.before(traced,b,action)
        a,ra,da,oa=plain.step(action);b,rb,db,ob=traced.step(action)
        trace.after(traced,b,ob)
        for x,y in ((a,b),(ra,rb),(plain.pair.data.qpos,traced.pair.data.qpos),
                (plain.pair.data.qvel,traced.pair.data.qvel),(oa['scored'],ob['scored'])):
            np.testing.assert_array_equal(x,y)
        assert da==db
    file=tmp_path/'trace.npz';contacts=trace.save(file,traced)
    with np.load(file) as stored:
        assert stored['observation'].shape==(5,2,148)
        terms=[stored['reward_'+name] for name in ('scored','received','fall','progress','facing','engagement','residual')]
        np.testing.assert_allclose(sum(terms),stored['reward_base_total'],rtol=1e-6,atol=1e-8)
        assert stored['action'].dtype==np.float32
        np.testing.assert_array_equal(stored['action'],np.repeat(action[None],5,axis=0))
        np.testing.assert_array_equal(stored['next_observation'][-1],b)
        np.testing.assert_array_equal(stored['observation'][1:],stored['next_observation'][:-1])
        np.testing.assert_array_equal(stored['qpos'][-1],traced.pair.data.qpos)
        assert stored['root_distance_before'][0]==initial_distance
        assert stored['root_distance_after'][-1]==root_distance(traced.pair)
        np.testing.assert_array_equal(stored['root_distance_after'][:-1],stored['root_distance_before'][1:])
    assert contacts and all(0<c['elapsed']<=.1+1e-9 for c in contacts)
    assert traced.pair.model.neq==0 and not np.any(traced.pair.model.body_gravcomp)
    assert not np.any(traced.pair.data.qfrc_applied) and not np.any(traced.pair.data.xfrc_applied)
    details=terminal_details(traced)
    np.testing.assert_array_equal([c['low_pelvis'] or c['tilted'] or c['self_leg_contact']
        for c in details['corners']],traced.pair.fallen())


def test_block_rms_and_leave_whole_trajectory_out():
    data=np.repeat(np.array([0.,2.,10.,14.])[:,None],148,axis=1)
    roles=np.array([0,0,1,1]);ids=np.arange(4)
    baseline=leave_trajectory_out_baselines(data,roles,ids)
    for block in baseline['0']:assert baseline['0'][block]['q99']==2.
    for block in baseline['1']:assert baseline['1'][block]['q99']==4.
    d=block_distances(data[:1],data[1:2])
    for values in d.values():np.testing.assert_allclose(values,2.)


def test_channel_ablation_preserves_original_action_and_unselected_authority():
    action=np.arange(64,dtype=np.float32).reshape(2,32)/64
    original=action.copy()
    assert ablate_action(action) is action
    command=ablate_action(action,zero_commands=True)
    arm=ablate_action(action,zero_arms=True)
    both=ablate_action(action,True,True)
    np.testing.assert_array_equal(action,original)
    np.testing.assert_array_equal(command[:,:3],0.)
    np.testing.assert_array_equal(command[:,3:],original[:,3:])
    np.testing.assert_array_equal(arm[:,3:17],0.)
    np.testing.assert_array_equal(arm[:,:3],original[:,:3])
    np.testing.assert_array_equal(arm[:,17:],original[:,17:])
    np.testing.assert_array_equal(both[:,:17],0.)
    np.testing.assert_array_equal(both[:,17:],original[:,17:])
