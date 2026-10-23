"""Duration checks must stop regressions/stagnation without manufacturing passes."""
from tools.striking_partner_duration_job import review


def role(survivors=8, distance=.27, seeds=()):
    return dict(survivors=survivors,median_minimum_hand_distance=distance,
                own_surviving_scoring_seeds=list(seeds))


def test_closer_hands_are_progress_but_not_a_phase_pass():
    result=review([role(6,.25),role(8,.25)],[role(6),role()],.27,[],1,0)
    assert result['improved'] and result['stagnant_checks']==0
    assert not result['phase_pass'] and result['stop_reason'] is None


def test_survival_regression_stops_even_with_contacts():
    result=review([role(6,.2,(7,23)),role(7,.2,(7,23))],[role(6),role()],.27,[],0,0)
    assert result['stop_reason']=='survival_regression'


def test_two_stagnant_checks_stop_without_automatic_extension():
    result=review([role(6),role()],[role(6),role()],.27,[],1,0)
    assert result['stop_reason']=='two_stagnant_checks'


def test_two_consecutive_complete_role_passes_stop_for_review():
    result=review([role(7,.2,(7,23)),role(8,.2,(7,23))],
                  [role(6),role()],.27,[],0,1)
    assert result['phase_pass'] and result['stop_reason']=='two_complete_phase_passes'
