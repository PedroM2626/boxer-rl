#!/usr/bin/env bash
# How much of the "6 GB" card can JAX actually be given, right now?
#
# WSL2 virtualises the GPU. What nvidia-smi reports as free is not what the
# paravisor can back: an allocation that the host will not commit fails as
# CUDA_ERROR_OUT_OF_MEMORY even with the card idle. JAX asks for 75% of the card
# at start-up (4.5 GiB here), that request fails, and XLA then falls back to a
# smaller pool and carries on -- which is why training runs print the error and
# still finish. The number that matters is therefore not the start-up request but
# the largest allocation the device will serve once it is up, and that is what
# this measures.
#
# There is no knob for the start-up request in this build:
# `XLA_FLAGS=--xla_gpu_memory_limit_slop_factor=...` is rejected as an unknown
# flag, so the failing probe cannot be silenced, only worked around.
#
#   wsl bash tools/gpu_headroom.sh
set -uo pipefail
PY=/home/pedro/venvs/boxer313/bin/python
export PYTHONPATH=/mnt/e/boxer-agent
cd /mnt/e/boxer-agent

echo "=== device as JAX sees it ==="
"$PY" -c '
import jax
d = jax.devices()[0]
print(d.platform, getattr(d, "device_kind", ""), getattr(d, "memory_size", "unknown"), "bytes")
' 2>&1 | grep -Ev "^(external|=== Source)" | tail -2

echo "=== largest single allocation the device will serve ==="
"$PY" -c '
import jax, jax.numpy as jp
for gib in (0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
    n = int(gib * 2**30 / 4)
    try:
        x = jp.zeros(n, dtype=jp.float32)
        jax.block_until_ready(x)
        print(f"  {gib:>4} GiB: served")
        del x
    except Exception as exc:
        print(f"  {gib:>4} GiB: {type(exc).__name__}")
        break
' 2>&1 | grep "GiB:"
