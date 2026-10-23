import numpy as np
import pytest

from tools.audit_striking_policy_drift import gaussian_kl, kl_summary


def test_gaussian_kl_is_directional_and_separates_command_and_arm_channels():
    a = np.zeros((2,17)); b = a.copy(); b[:,0] = 1.
    zero = np.zeros(17)
    np.testing.assert_array_equal(gaussian_kl(a,zero,a,zero),0.)
    stats = kl_summary((a,zero),(b,zero))
    assert stats['mean'] == stats['command_mean'] == .5 and stats['arm_mean'] == 0.
    np.testing.assert_allclose(gaussian_kl(a,zero,a,np.full(17,np.log(2))), np.log(2)-.375)
    assert not np.allclose(gaussian_kl(a,zero,a,np.full(17,np.log(2))),gaussian_kl(a,np.full(17,np.log(2)),a,zero))


def test_nonfinite_policy_probe_is_rejected():
    with pytest.raises(FloatingPointError): gaussian_kl(np.array([[np.nan]]),np.zeros(1),np.zeros((1,1)),np.zeros(1))
