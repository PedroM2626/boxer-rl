"""The native gate picks checkpoints by native record, not by recency."""
import numpy as np

from boxer.curriculum_train import load_policy_params, save_policy
from boxer.tactics_gate import candidates, combine, score, stage


def record(wins, losses, draws, mine, theirs, bouts=16, knockdowns=0):
    return dict(bouts=bouts, knockdowns=knockdowns,
                tactics=dict(first_wins=wins, first_losses=losses, draws=draws,
                             first_clean_per_bout=mine, second_clean_per_bout=theirs))


def test_candidates_are_the_latest_steps_in_numeric_order_and_stage_loads(tmp_path):
    for step in (0, 5_000_000, 10_000_000, 15_000_000, 20_000_000):
        save_policy(tmp_path, (np.full(2, step, float),), dict(stage_steps=step, stage="tactics"))
    picked = candidates(tmp_path, 3)
    assert [p.name for p in picked] == ["step-10000000.pkl.json", "step-15000000.pkl.json",
                                        "step-20000000.pkl.json"]
    params, metadata = load_policy_params(stage(picked[0], tmp_path / "staged"))
    assert metadata["stage_steps"] == 10_000_000 and params[0][0] == 10_000_000


def test_score_prefers_record_then_punch_margin_and_combine_weights_bouts():
    assert score(record(8, 5, 3, 10, 12)) > score(record(7, 5, 4, 20, 5))
    assert score(record(7, 7, 2, 15, 12)) > score(record(7, 7, 2, 12, 12))
    total = combine(record(10, 5, 1, 16, 16, knockdowns=1), record(11, 18, 3, 13, 16, bouts=32, knockdowns=2))
    assert (total["first_wins"], total["first_losses"], total["draws"], total["bouts"]) == (21, 23, 4, 48)
    assert abs(total["first_clean_per_bout"] - 14.0) < 1e-9 and total["falls"] == 3
