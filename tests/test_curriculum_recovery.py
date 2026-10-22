"""Recovery keeps compute accounting, avoids repeated rollback, and freezes gates."""
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from boxer import curriculum_eval as evaluator
from boxer import curriculum_train as trainer
from boxer import curriculum_job as job


def test_rollback_is_one_time_and_keeps_consumed_counter(tmp_path, monkeypatch):
    own, seed = tmp_path / "run", tmp_path / "seed"
    trainer.save_policy(own / "checkpoint", (np.array([30000]), {}, {}),
                        dict(stage_steps=30000, stage="balance", contract={}))
    trainer.save_policy(seed, (np.array([10000]), {}, {}),
                        dict(stage_steps=10000, stage="balance", contract={}))
    monkeypatch.setattr(trainer, "jax_compat", lambda: None)
    monkeypatch.setattr(trainer, "UnassistedG1", lambda stage, config: SimpleNamespace(_config=config))
    monkeypatch.setattr(trainer, "contract", lambda env: {})
    restored = []
    def ppo(**kwargs):
        params = kwargs["restore_params"]
        restored.append(int(params[0][0]))
        kwargs["policy_params_fn"](0, None, params)
        step = kwargs["num_timesteps"]
        kwargs["policy_params_fn"](step, None, (params[0] + step, {}, {}))
    monkeypatch.setattr(trainer.ppo, "train", ppo)
    flags = ["--output", str(own), "--warm-start", "--restart-from", str(seed), "--worlds", "1"]
    assert trainer.main(flags + ["--steps", "80000"]) == 0
    first = json.loads((own / "checkpoint/latest.json").read_text())
    assert first["stage_steps"] == 70960
    assert restored == [10000]
    assert trainer.main(flags + ["--steps", "130000"]) == 0
    assert restored == [10000, 50960]  # Keep subsequent learning, do not reapply the seed.
    assert json.loads((own / "checkpoint/latest.json").read_text())["stage_steps"] > 70960
    consumed = json.loads((own / "checkpoint/latest.json").read_text())["stage_steps"]
    assert trainer.main(flags + ["--force-restart", "--steps", "180000"]) == 0
    assert restored[-1] == 10000
    assert json.loads((own / "checkpoint/latest.json").read_text())["stage_steps"] > consumed


def test_evaluation_disables_training_pushes(tmp_path, monkeypatch):
    class StopProbe(Exception):
        pass
    monkeypatch.setattr(evaluator, "load_policy_params", lambda root: ((), dict(stage="balance")))
    def inspect(stage, config):
        assert not config.push_config.enable
        assert config.noise_config.level == 0
        assert config.njmax == 90
        raise StopProbe
    monkeypatch.setattr(evaluator, "UnassistedG1", inspect)
    with pytest.raises(StopProbe):
        evaluator.evaluate(tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="Linux process-group supervisor")
def test_physics_overflow_stops_child_instead_of_continuing(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "ROOT", tmp_path)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Physics allocation overflow"):
        job.run_child([sys.executable, "-u", "-c",
                       "import time;print('nefc overflow - increase njmax',flush=True);time.sleep(20)"],
                      tmp_path / "child.log")
    assert time.monotonic() - started < 10
