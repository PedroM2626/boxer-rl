"""The diagnostic reports only the missing partner-fall cost; it changes no reward."""
import numpy as np
import pytest
from boxer.striking_objective_audit import uncharged_pair_fall_cost


@pytest.mark.parametrize('corner',[0,1])
def test_counterfactual_only_charges_a_partner_only_fall(corner):
    assert uncharged_pair_fall_cost([False,False],corner)==0.
    flags=np.array([False,False]);flags[corner]=True;before=flags.copy()
    assert uncharged_pair_fall_cost(flags,corner)==0.;np.testing.assert_array_equal(flags,before)
    assert uncharged_pair_fall_cost(~flags,corner)==5.
    assert uncharged_pair_fall_cost([True,True],corner)==0.


@pytest.mark.parametrize('flags,corner',[([0,1],0),([False],0),([False,True],2),([float('nan'),0],0)])
def test_invalid_physical_flags_are_rejected(flags,corner):
    with pytest.raises(ValueError):uncharged_pair_fall_cost(flags,corner)
