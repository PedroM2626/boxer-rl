#!/usr/bin/env bash
# One-shot GPU setup for the boxer project inside WSL2.
#
# Native Windows has no CUDA build of jaxlib, so training on the GPU runs in
# WSL2. This script needs no sudo: it bootstraps pip if the distro has none,
# installs uv in userspace, and builds a Python 3.12 venv.
#
# Python 3.12 matters: on 3.10 pip can only resolve brax 0.14.1, which depends
# on mujoco-mjx and pins jax to 0.6.2 -- and that jaxlib cannot load the cuDNN
# 9.27 wheels its own extra installs. Python 3.12 gets brax 0.14.2 + jax 0.11.x.
#
# cuda13, not cuda12: the WSL host driver reports "CUDA UMD Version: 13.3", and
# the CUDA 12 wheel set's libcusolver.so.11 raises SIGBUS from its ELF
# constructor on this pairing (libcudart, cublas, cublasLt and every cuDNN 9
# library load fine -- only cuSOLVER faults). The CUDA 13 set works, and the
# cu12 environment cannot even run on CPU because plugin discovery loads those
# libraries regardless of the requested platform.
#
# scipy is pinned: scipy 1.18's bundled OpenBLAS raises SIGBUS from
# scipy.linalg.blas inside WSL2. Because mujoco.mjx._src imports scipy.linalg,
# that one library takes down the entire MJX import -- on the CPU path too, so
# it does not look like a GPU problem and is easy to misdiagnose. 1.15.3 loads
# cleanly.
set -uo pipefail

VENV="${VENV:-$HOME/venvs/boxer313}"
BOOT="$HOME/venvs/bootstrap"
PROJECT="${PROJECT:-/mnt/e/boxer-agent}"

if [ ! -x "$BOOT/bin/uv" ]; then
  [ -d "$BOOT" ] || python3 -m venv --without-pip "$BOOT"
  if [ ! -x "$BOOT/bin/pip" ]; then
    curl -sSL -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
    "$BOOT/bin/python" /tmp/get-pip.py --quiet
  fi
  "$BOOT/bin/python" -m pip install --quiet uv
fi
UV="$BOOT/bin/uv"

"$UV" venv --python 3.12 --clear "$VENV"
"$UV" pip install --python "$VENV/bin/python" \
  'jax[cuda13]' 'scipy==1.15.3' brax mujoco mujoco-menagerie 'imageio[ffmpeg]' pytest

# Keep the repo's code importable from inside WSL without installing the package.
PROJECT="$PROJECT" "$VENV/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["PROJECT"])

import jax
print("jax", jax.__version__, "| devices:", jax.devices())
if not any(d.platform == "cuda" for d in jax.devices()):
    raise SystemExit("FAIL: no CUDA device visible")

# mujoco.mjx pulls in scipy.linalg, which is what has faulted on this WSL2 host.
from mujoco import mjx  # noqa: F401
print("mujoco.mjx loaded")

import jax.numpy as jp
from boxer import scene
from boxer.env import BoxingEnv

model = scene.build_model()
print(f"scene: {model.nbody} bodies, {model.npair} contact pairs, {model.nsensordata} sensor dims")

env = BoxingEnv()
state = env.reset(jax.random.PRNGKey(0))
state = jax.jit(env.step)(state, jp.zeros((2, env.action_size)))
print("gpu step ok:", tuple(state.obs.shape), "reward", state.reward)
print("ALL CHECKS PASSED")
PY

echo
echo "Train with:"
echo "  cd $PROJECT && PYTHONPATH=$PROJECT JAX_PLATFORMS=cuda,cpu $VENV/bin/python -m boxer.train"
echo
echo "Use JAX_PLATFORMS=cuda,cpu, not cuda alone: Brax's PPO uses jax.debug.callback"
echo "for its metrics logger, which needs a CPU device to place its inputs on."
