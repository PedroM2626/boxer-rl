"""Atomic, verified training checkpoints, compatible with the older five-tuple."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


def atomic_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def require_finite(**arrays) -> None:
    for name, value in arrays.items():
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError(f"non-finite {name}; retaining the last valid checkpoint")


def save_checkpoint(directory: Path, payload: tuple, rng_key) -> Path:
    directory = Path(directory)
    step = int(payload[4])
    path = directory / f"step-{step}.pkl"
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    state = dict(format_version=1, step=step,
                 sha256=hashlib.sha256(data).hexdigest(),
                 rng_key=np.asarray(rng_key, dtype=np.uint32).tolist())
    atomic_bytes(path, data)
    atomic_json(path.with_suffix(".state.json"), state)
    # Publish the pointer only after both the weights and their state are durable.
    atomic_bytes(directory / "latest.txt", str(step).encode())
    return path


def load_checkpoint(directory: Path) -> tuple[tuple, dict]:
    directory = Path(directory)
    latest = (directory / "latest.txt").read_text().strip()
    if not latest.isdigit():
        raise ValueError(f"invalid checkpoint pointer in {directory}")
    path = directory / f"step-{int(latest)}.pkl"
    data = path.read_bytes()
    state_path = path.with_suffix(".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and (state.get("sha256") != hashlib.sha256(data).hexdigest()
                  or state.get("step") != int(latest)):
        raise ValueError(f"checkpoint verification failed: {path}")
    payload = pickle.loads(data)
    if len(payload) != 5 or int(payload[4]) != int(latest):
        raise ValueError(f"checkpoint payload does not match its pointer: {path}")
    return payload, state
