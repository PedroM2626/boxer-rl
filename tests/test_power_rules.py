"""Power rule and arm fatigue, the same for both corners."""
import numpy as np

from boxer.paired_foundation import PUNCH_IMPULSE_ENV, PairedFoundation, Stamina
from test_paired_foundation import foundation


def test_the_power_rule_reaches_the_scorecard_but_not_the_block_count(foundation, monkeypatch):
    assert PairedFoundation(foundation).scorecard.minimum_impulse == .5
    monkeypatch.setenv(PUNCH_IMPULSE_ENV, '2')
    pair = PairedFoundation(foundation)
    assert pair.scorecard.minimum_impulse == 2.
    assert pair.blockcard.minimum_impulse == .5


def test_stamina_drains_with_arm_power_refills_at_rest_and_weakens_tired_arms(foundation):
    rule = Stamina(capacity=100., recovery=10., floor=.2)
    np.testing.assert_allclose(rule.update([110., 0.], 1.), [.2, 1.])     # red spent 100 J net
    np.testing.assert_allclose(rule.level, [0., 1.])
    rule.update([0., 0.], 5.)
    np.testing.assert_allclose(rule.level, [.5, 1.])
    np.testing.assert_allclose(rule.work, [110., 0.])
    pair = PairedFoundation(foundation)
    gains = [pair.model.actuator_gainprm[a, 0].copy() for a in pair.arm_actuators]
    pair.stamina = Stamina(capacity=1e-6, recovery=0., floor=.25)
    pair.reset(randomized=False)
    pair.step(np.zeros((2, 29)) + .3)
    assert (pair.stamina.level < 1).all()
    strength = pair.stamina.strength()
    for i, (a, g) in enumerate(zip(pair.arm_actuators, gains)):
        np.testing.assert_allclose(pair.model.actuator_gainprm[a, 0], g * strength[i])
    pair.reset(randomized=False)                     # a new bout: fresh arms
    for a, g in zip(pair.arm_actuators, gains):
        np.testing.assert_allclose(pair.model.actuator_gainprm[a, 0], g)
