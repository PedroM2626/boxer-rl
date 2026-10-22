"""Exercise early PPO exit without restarting the optimizer for every gate."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from boxer import curriculum_train as trainer
from test_curriculum_control import gate


@pytest.mark.parametrize("passes", [True, False])
def test_gates_control_early_exit_without_restarting_live_ppo(tmp_path, monkeypatch, passes):
    monkeypatch.setattr(trainer, "jax_compat", lambda: None)
    monkeypatch.setattr(trainer, "UnassistedG1", lambda stage, config: SimpleNamespace(_config=config))
    monkeypatch.setattr(trainer, "contract", lambda env: {})
    def evaluate(command, **kwargs):
        checkpoint = tmp_path / "checkpoint"
        metadata = json.loads((checkpoint / "latest.json").read_text())
        result = gate(step=metadata["stage_steps"], passed=passes)
        result["checkpoint_sha256"] = metadata["sha256"]
        output = tmp_path / "gates" / f"step-{metadata['stage_steps']}.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(result))
    monkeypatch.setattr(trainer.subprocess, "run", evaluate)
    calls = []
    def fake_ppo(**kwargs):
        calls.append(kwargs)
        for step in (0, 10240, 20480, 30720, 40960):
            kwargs["policy_params_fn"](step, None, (np.array([step]), {}, {}))
            if step:
                kwargs["progress_fn"](step, {})
        assert not passes, "PPO should exit at the second passing checkpoint"
    monkeypatch.setattr(trainer.ppo, "train", fake_ppo)
    assert trainer.main(["--output", str(tmp_path), "--steps", "50000", "--worlds", "1",
                         "--adaptive-gates", "--gate-every", "10240"]) == 0
    assert len(calls) == 1  # One PPO invocation, Adam resident between evaluations.
    assert calls[0]["wrap_env_fn"].keywords["full_reset"]
    assert calls[0]["learning_rate"] == 1e-4
    result = json.loads((tmp_path / "training_result.json").read_text())
    expected = 10240 if passes else 40960
    assert result["status"] == ("promoted" if passes else "budget_reached")
    assert result["stage_steps"] == expected
    assert calls[0]["num_timesteps"] <= 50000
    if passes:
        assert result["stage_steps"] < calls[0]["num_timesteps"]
    assert json.loads((tmp_path / "checkpoint/latest.json").read_text())["stage_steps"] == expected
