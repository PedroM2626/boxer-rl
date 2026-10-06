#!/usr/bin/env bash
# Diagnoses the JAX/CUDA path inside WSL2 and writes a readable report.
#
#   wsl bash tools/wsl_gpu_check.sh
#
# Output goes to tools/gpu_report.txt in the repo (UTF-8, no progress bars), so
# it can be read from the Windows side without decoding WSL's console encoding.
set -uo pipefail

VENV="${VENV:-$HOME/venvs/boxer312}"
PY="$VENV/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT="$HERE/gpu_report.txt"

{
  echo "=== python ==="
  "$PY" --version 2>&1

  echo "=== gpu nodes (WSL2 CUDA needs /dev/dxcore) ==="
  ls -la /dev/dxcore /dev/dxg 2>&1
  echo "libcuda: $(ls /usr/lib/wsl/lib/libcuda.so* 2>/dev/null | head -1)"

  echo "=== nvidia-smi ==="
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1

  echo "=== installed cuda wheels ==="
  "$PY" -m pip list 2>/dev/null | grep -iE '^(jax|jaxlib|nvidia|brax|mujoco)' || \
    "$VENV/bin/uv" pip list --python "$PY" 2>/dev/null | grep -iE '^(jax|jaxlib|nvidia|brax|mujoco)'

  echo "=== import numpy ==="
  "$PY" -c 'import numpy; print(numpy.__version__)' 2>&1 | tail -3

  echo "=== import jaxlib ==="
  "$PY" -c 'import jaxlib; print("jaxlib", jaxlib.__version__)' 2>&1 | tail -5

  echo "=== jax devices, default platform ==="
  "$PY" -c 'import jax; print(jax.devices()); print("platform:", jax.default_backend())' 2>&1 | tail -20

  echo "=== jax devices, JAX_PLATFORMS=cuda ==="
  JAX_PLATFORMS=cuda "$PY" -c 'import jax; print(jax.devices()); print(jax.default_backend())' 2>&1 | tail -20

  echo "=== cuda runtime handshake ==="
  "$PY" - <<'PY' 2>&1 | tail -20
try:
    import ctypes
    cu = ctypes.CDLL("libcuda.so.1")
    print("cuInit:", cu.cuInit(0))
    n = ctypes.c_int()
    print("cuDeviceGetCount:", cu.cuDeviceGetCount(ctypes.byref(n)), "count:", n.value)
    if n.value:
        name = ctypes.create_string_buffer(100)
        print("cuDeviceGetName:", cu.cuDeviceGetName(name, 100, 0), name.value.decode())
except Exception as exc:
    print("FAILED:", type(exc).__name__, exc)
PY

  echo "=== gpu arrays on device ==="
  "$PY" - <<'PY' 2>&1 | tail -20
import jax, jax.numpy as jp, time
x = jp.ones((2048, 2048))
y = x @ x
jax.block_until_ready(y)
t = time.time()
for _ in range(20):
    y = x @ x
jax.block_until_ready(y)
print("matmul tflops-ish:", round(2 * 2048**3 * 20 / (time.time() - t) / 1e12, 2))
print("device:", y.device_buffer.device())
PY
} > "$REPORT" 2>&1

echo "wrote $REPORT"
