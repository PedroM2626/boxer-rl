"""Long runs must survive an interrupted write without losing their last save."""
import json

import numpy as np
import pytest

from boxer import checkpoint_io as io


def payload(step):
    return ({"weights": np.arange(4)}, {}, {}, {"mean": np.zeros(2)}, step)


def test_checkpoint_round_trip_restores_rng_and_verifies_bytes(tmp_path):
    key = np.array([12, 34], np.uint32)
    path = io.save_checkpoint(tmp_path, payload(100), key)
    restored, state = io.load_checkpoint(tmp_path)
    assert restored[4] == 100 and state["rng_key"] == [12, 34]
    np.testing.assert_array_equal(restored[0]["weights"], np.arange(4))
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="verification failed"):
        io.load_checkpoint(tmp_path)


def test_failed_pointer_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    io.save_checkpoint(tmp_path, payload(100), [1, 2])
    replace = io.os.replace

    def interrupt(source, destination):
        if destination.name == "latest.txt":
            raise OSError("simulated power interruption")
        replace(source, destination)

    monkeypatch.setattr(io.os, "replace", interrupt)
    with pytest.raises(OSError):
        io.save_checkpoint(tmp_path, payload(200), [3, 4])
    restored, _ = io.load_checkpoint(tmp_path)
    assert restored[4] == 100
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_payload_is_readable_but_has_no_rng_state(tmp_path):
    import pickle
    (tmp_path / "latest.txt").write_text("100")
    (tmp_path / "step-100.pkl").write_bytes(pickle.dumps(payload(100)))
    restored, state = io.load_checkpoint(tmp_path)
    assert restored[4] == 100 and not state


def test_nonfinite_values_cannot_be_published_as_valid_progress(tmp_path):
    with pytest.raises(FloatingPointError, match="rewards"):
        io.require_finite(rewards=[1, np.nan])
    with pytest.raises(ValueError):
        io.atomic_json(tmp_path / "progress.json", {"loss": float("inf")})
    assert not (tmp_path / "progress.json").exists()
