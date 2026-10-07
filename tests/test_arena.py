"""Matchup accounting: who won, and whether the arena runs at all."""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from boxer import arena, policy


def _log(length, worlds, done_at=None, red_down_at=None, blue_down_at=None):
    done = np.zeros((length, worlds), dtype=np.float32)
    red = np.zeros((length, worlds), dtype=np.float32)
    blue = np.zeros((length, worlds), dtype=np.float32)
    for world, step in (done_at or {}).items():
        if step < length:
            done[step, world] = 1.0
    for world, step in (red_down_at or {}).items():
        if step < length:
            red[step, world] = 1.0
    for world, step in (blue_down_at or {}).items():
        if step < length:
            blue[step, world] = 1.0
    return done, red, blue


def test_a_knockdown_of_blue_is_a_win_for_a():
    done, red, blue = _log(10, 2, done_at={0: 4}, blue_down_at={0: 4})
    a_win, b_win, draw, finished, last = arena.outcomes(done, red, blue, 10)
    assert a_win[0] and not b_win[0] and not draw[0]
    assert finished[0] and last[0] == 4


def test_both_down_is_a_draw():
    done, red, blue = _log(10, 1, done_at={0: 3}, red_down_at={0: 3}, blue_down_at={0: 3})
    a_win, b_win, draw, finished, _ = arena.outcomes(done, red, blue, 10)
    assert draw[0] and not a_win[0] and not b_win[0]


def test_an_unfinished_round_is_undecided():
    done, red, blue = _log(10, 1)
    a_win, b_win, draw, finished, last = arena.outcomes(done, red, blue, 10)
    assert not finished[0]
    assert (a_win | b_win | draw).sum() == 0
    assert last[0] == 9


def test_only_the_terminal_step_counts():
    """A boxer that was down mid-round but recovered must not be scored down."""
    done, red, blue = _log(10, 1, done_at={0: 8}, blue_down_at={0: 2})
    a_win, _, _, _, _ = arena.outcomes(done, red, blue, 10)
    assert not a_win[0]

    done, red, blue = _log(10, 1, done_at={0: 8}, blue_down_at={0: 8})
    a_win, _, _, _, _ = arena.outcomes(done, red, blue, 10)
    assert a_win[0]


def test_play_reports_a_probability_distribution(level):
    passive = policy.passive(level.action_size)
    result = arena.play(level, passive, passive, rounds=4, length=20, seed=0)
    assert set(result) == {
        "a_win", "b_win", "draw", "undecided", "round_seconds",
        "hits_per_round", "heads_per_round", "damage_per_round",
        "a_hits_per_round", "b_hits_per_round",
        "a_heads_per_round", "b_heads_per_round", "dropped_rounds",
    }
    total = result["a_win"] + result["b_win"] + result["draw"] + result["undecided"]
    assert total == pytest.approx(1.0, abs=1e-6)
    assert all(np.isfinite(list(result.values())))
    assert result["dropped_rounds"] == 0.0, "a clean roll should discard nothing"


def test_corner_attribution_adds_up_to_the_pooled_mean(level):
    """The per-corner counts must be the two halves the pooled mean averages.

    A win is attributed by the pooled mean alone, so a boxer that only falls on
    top of the other one looks like a puncher. Splitting by corner is the only
    way to tell those apart, and the split has to be consistent with the mean.
    """
    jab = policy.jabber(
        level.action_size,
        pitch_adr=level.actuator_index("red/left_shoulder_pitch_joint"),
        roll_adr=level.actuator_index("red/left_shoulder_roll_joint"),
    )
    passive = policy.passive(level.action_size)
    result = arena.play(level, jab, passive, rounds=4, length=60, seed=2)
    assert result["a_hits_per_round"] + result["b_hits_per_round"] == pytest.approx(
        2 * result["hits_per_round"], abs=1e-5
    )
    assert result["a_heads_per_round"] + result["b_heads_per_round"] == pytest.approx(
        2 * result["heads_per_round"], abs=1e-5
    )
    assert result["a_hits_per_round"] >= result["b_hits_per_round"], (
        "the jabber attacks and the statue does not, so the attacker should land"
        " at least as many gated punches"
    )


def test_two_boxers_holding_guard_never_trade_punches(level):
    """The null hypothesis for the whole arena: nobody attacks, nothing lands."""
    passive = policy.passive(level.action_size)
    result = arena.play(level, passive, passive, rounds=8, length=40, seed=3)
    assert result["hits_per_round"] == 0.0
    assert result["a_hits_per_round"] == 0.0
    assert result["b_hits_per_round"] == 0.0
    assert result["damage_per_round"] == 0.0
    assert result["undecided"] == 1.0, "holding the guard should not end the round"


def test_matchup_pools_both_corner_assignments(level):
    passive = policy.passive(level.action_size)
    result = arena.matchup(
        level, "passive", "passive", rounds=4, length=20, seed=1, swap_sides=True
    )
    assert result["a_win"] == pytest.approx(result["b_win"])
