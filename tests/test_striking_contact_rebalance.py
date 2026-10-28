"""Protect the bounded partial-contact reward change and its source contract."""
import numpy as np
import pytest
from boxer.striking_contact_rebalance import PROFILE, correction, rebalance_config


def test_correction_changes_only_net_partial_contact_payment():
    parts = {'physical_contact_progress': np.array([2., 1.]), 'target_potential': np.array([7., 8.])}
    # Existing v15 payment is +1 for corner 0 and -1 for corner 1. The v16
    # correction leaves 1/8 of that net payment and does not touch other terms.
    assert correction(parts, 0) == pytest.approx(-.875)
    assert correction(parts, 1) == pytest.approx(.875)
    assert PROFILE['full_cycle_bonus'] == 10.
    np.testing.assert_array_equal(parts['target_potential'], [7., 8.])


def test_rejects_nonfinite_inputs_and_wrong_source_configuration():
    with pytest.raises(ValueError):
        correction({'physical_contact_progress': [float('nan'), 1.]}, 0)
    with pytest.raises(ValueError):
        correction({'physical_contact_progress': [1., 1.]}, 2)
    with pytest.raises(ValueError):
        rebalance_config({'version': 14, 'contact_progress_ppo': False})
