"""Passive trace summaries distinguish contact timing from a causal claim."""
import hashlib

import numpy as np
import pytest

from tools.striking_horizon_regression_trace import summarize_trial


def trial_fixture(tmp_path):
    path = tmp_path / 'trace.npz'
    np.savez(path, pelvis_height=np.full((2, 2), .75), upright=np.ones((2, 2)),
        action=np.zeros((2, 2, 17)), arm_residual=np.zeros((2, 2, 14)))
    return dict(seed=1, swap=False, survived=False, scored_by_corner=[0, 0],
        first_fall_time=2., first_observed_contacts=[dict(geoms=['floor', 'red/left_foot'], elapsed=0.),
            dict(geoms=['red/left_hand_collision', 'blue/torso_collision'], elapsed=1.)],
        trace_file=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        terminal_state=dict(corners=[{}, {}]))


@pytest.mark.parametrize('survived,fall,expected', [(False, 2., True), (False, .5, False), (True, None, None)])
def test_temporal_contact_association(tmp_path, survived, fall, expected):
    trial = trial_fixture(tmp_path)
    trial.update(survived=survived, first_fall_time=fall)
    summary = summarize_trial(trial)
    assert summary['first_interbody_contact_time'] == 1.
    assert summary['fall_after_interbody_contact'] is expected
    assert summary['minimum_pelvis_height'] == [.75, .75]


def test_floor_contact_is_not_an_opponent_contact(tmp_path):
    trial = trial_fixture(tmp_path)
    trial['first_observed_contacts'].pop()
    summary = summarize_trial(trial)
    assert summary['first_interbody_contact_time'] is None
    assert summary['fall_after_interbody_contact'] is None


def test_changed_trace_payload_rejected(tmp_path):
    trial = trial_fixture(tmp_path)
    trial['trace_sha256'] = '0' * 64
    with pytest.raises(AssertionError):
        summarize_trial(trial)
