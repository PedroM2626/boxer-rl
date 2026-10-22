"""Reward experiments cannot silently change the default or physical benchmark."""
import json
from types import SimpleNamespace

import pytest

from boxer import curriculum_train as trainer
from boxer import curriculum_eval as evaluator


@pytest.mark.parametrize("value", ["0", "-0.01", "nan", "inf"])
def test_invalid_tracking_width_is_rejected_before_training(tmp_path, monkeypatch, value):
    def forbidden():
        pytest.fail("invalid experiment must fail before JAX initialization")
    monkeypatch.setattr(trainer, "jax_compat", forbidden)
    with pytest.raises(ValueError, match="finite and positive"):
        trainer.main(["--output", str(tmp_path), "--tracking-sigma", value])


@pytest.mark.parametrize("override,expected", [(None, .25), ("0.01", .01)])
def test_training_override_is_recorded_and_does_not_change_gate(tmp_path, monkeypatch,
                                                              override, expected):
    monkeypatch.setattr(trainer, "jax_compat", lambda: None)
    monkeypatch.setattr(trainer, "UnassistedG1",
                        lambda stage, config: SimpleNamespace(_config=config))
    monkeypatch.setattr(trainer, "contract", lambda env: {})
    received = []
    monkeypatch.setattr(trainer.ppo, "train",
                        lambda **kwargs: received.append(kwargs["environment"]._config))
    flags = ["--stage", "walk_easy", "--output", str(tmp_path),
             "--steps", "50000", "--worlds", "1"]
    if override is not None:
        flags += ["--tracking-sigma", override]
    assert trainer.main(flags) == 0
    assert received[0].reward_config.tracking_sigma == expected
    recorded = json.loads((tmp_path / "config.json").read_text())
    assert recorded["environment"]["reward_config"]["tracking_sigma"] == expected

    class GateInspected(Exception):
        pass
    monkeypatch.setattr(evaluator, "load_policy_params",
                        lambda root: ((), dict(stage="walk_easy")))
    def inspect_gate(stage, config):
        assert config.reward_config.tracking_sigma == .25
        assert not config.push_config.enable
        assert config.njmax == 90
        raise GateInspected
    monkeypatch.setattr(evaluator, "UnassistedG1", inspect_gate)
    with pytest.raises(GateInspected):
        evaluator.evaluate(tmp_path)
