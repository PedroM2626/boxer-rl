"""Reference evidence cannot certify punches or combine unrelated hands."""
import pytest

from boxer.strike_motion import evaluate_motion


def event(speed=2.,forward=.9,complete=True):
    return dict(corner=0,hand=0,onset_relative_speed=speed,onset_forward_cosine=forward,
        pre_max_displacement_m=.2,pre_path_length_m=.24,pre_elbow_excursion_rad=.5,
        event_window_impulse_ns=.7,post_window_seconds=.5,post_contact_seconds=.02,
        post_window_complete=complete,return_toward_pre_window_position_m=.15)


def test_fast_sideways_and_slow_forward_hands_cannot_borrow_each_others_components():
    a=event(3.,.2);b=event(.4,1.);b['hand']=1
    result=evaluate_motion([a,b])
    assert result['both_reference_components_count']==0
    assert result['certified_punch_count'] is None and not result['certification']


def test_even_complete_reference_motion_is_not_punch_or_guard_certification():
    result=evaluate_motion([event()]);row,=result['events']
    assert row['both_reference_components_met']
    assert row['return_toward_reference_m']==.15
    assert not row['punch_certified'] and not row['guard_recovery_certified']
    result['profile']['speed_reference_m_s']=.01
    assert evaluate_motion([event(.2)])['both_reference_components_count']==0


def test_censored_return_is_unavailable_rather_than_failed_or_certified():
    row,=evaluate_motion([event(complete=False)])['events']
    assert row['return_toward_reference_m'] is None
    assert not row['recovery_observation_complete']


@pytest.mark.parametrize('key,value', [('onset_relative_speed',float('nan')),
    ('pre_max_displacement_m',-.1),('onset_forward_cosine',1.1)])
def test_invalid_measurement_rejected(key,value):
    e=event();e[key]=value
    with pytest.raises(ValueError):evaluate_motion([e])
