"""Diagnostic arm covariance must preserve default draws and command channels."""
import numpy as np
from tools.evaluate_striking import arm_perturbation


def test_default_noise_is_bit_identical_to_original_stream():
    rng=np.random.default_rng(31);held=None
    for step in range(30):
        noise=rng.normal(size=(2,17));expected=noise.copy()
        actual,held=arm_perturbation(noise,held,step)
        np.testing.assert_array_equal(actual,expected)


def test_holding_arms_does_not_hold_commands_or_modify_body_noise():
    rng=np.random.default_rng(7);held=None
    for step in range(21):
        noise=rng.normal(size=(2,32));expected=noise.copy()
        if step%10==0:first=noise[:,3:17].copy()
        actual,held=arm_perturbation(noise,held,step,hold=10,scale=3.)
        np.testing.assert_array_equal(actual[:,:3],expected[:,:3])
        np.testing.assert_array_equal(actual[:,17:],expected[:,17:])
        np.testing.assert_array_equal(actual[:,3:17],3.*first)
