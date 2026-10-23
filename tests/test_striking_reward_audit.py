import numpy as np

from boxer.striking_reward_audit import reward_terms


def test_actual_impact_rewards_cancel_jointly_and_fall_disables_all_shaping():
    parts=reward_terms([3,2],[False,False],[.1,-.1],[1,1],.45,np.ones((2,14)),.02,.45,True)
    assert (parts['scored']+parts['received']).sum()==0
    np.testing.assert_array_equal(parts['scored']+parts['received'],[2,-2])
    np.testing.assert_array_equal(parts['engagement'],[.003,.003])
    fallen=reward_terms([1,0],[False,True],[.1,.1],[1,1],.45,np.zeros((2,14)),.02,.45,False)
    for name in ('progress','facing','engagement','residual'):np.testing.assert_array_equal(fallen[name],[0,0])
    np.testing.assert_array_equal(sum(fallen.values()),[2,-7])
