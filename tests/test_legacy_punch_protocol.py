"""Old policies must retain the fight-state mechanics they learned."""
import pytest
pytest.importorskip('mujoco_warp')
from tools.evaluate_legacy_punch_cycles import historical_fight_state


def test_pre_damage_policy_does_not_inherit_new_damage_or_stun():
    old = historical_fight_state({'obs_size': 128})
    assert old['damage_cap'] > 480 and old['stun_steps'] == 0
    assert historical_fight_state({'obs_size': 131, 'damage_cap': 3., 'stun_steps': 10}) == {
        'damage_cap': 3., 'stun_steps': 10}
    with pytest.raises(ValueError):
        historical_fight_state({'obs_size': 131})
