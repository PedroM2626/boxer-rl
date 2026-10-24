"""Independent impact scorecard rejects sustained contact and passive catches."""
import numpy as np
import pytest

from boxer.impact_scorecard import ImpactScorecard


def sample(card, time, contact=True, impulse=.6, closing=1., extension=1., up=True):
    present=np.zeros((2,2),bool); values=np.zeros((3,2,2))
    present[0,0]=contact
    values[:,0,0]=[impulse if contact else 0.,closing,extension]
    return card.update(present,*values,time,.002,up)


def test_real_onset_scores_once_and_separation_rearms():
    card=ImpactScorecard()
    np.testing.assert_array_equal(sample(card,0.),[1,0])
    for step in range(1,150):
        assert not sample(card,step*.002).any()
    for step in range(150,210):
        sample(card,step*.002,contact=False)
    np.testing.assert_array_equal(sample(card,.42),[1,0])


@pytest.mark.parametrize('closing,extension,up',[(0.,1.,True),(1.,0.,True),(1.,1.,False)])
def test_passive_contact_body_lean_and_fallen_round_cannot_score(closing,extension,up):
    card=ImpactScorecard()
    for step in range(100):
        assert not sample(card,step*.002,closing=closing,extension=extension,up=up).any()


def test_weak_contact_does_not_accumulate_forever_or_rearm_on_chatter():
    card=ImpactScorecard()
    for step in range(500):
        assert not sample(card,step*.002,impulse=.001,contact=step%4!=3).any()


def test_nonfinite_input_is_invalid():
    with pytest.raises(FloatingPointError):
        sample(ImpactScorecard(),0.,closing=np.nan)


def test_qualifying_contact_does_not_certify_a_boxing_punch():
    # A low-speed contact can meet the legacy contract over its 80 ms window.
    # No arm travel, extension/retraction or guard recovery is represented here.
    # This is a synthetic contract example, not a measured physical trajectory.
    card=ImpactScorecard();events=[]
    for step in range(40):
        events.append(sample(card,step*.002,impulse=.0128,closing=.31,extension=.21))
    np.testing.assert_array_equal(np.sum(events,axis=0),[1,0])
