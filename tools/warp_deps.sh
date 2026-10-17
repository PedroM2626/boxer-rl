#!/usr/bin/env bash
# CPU-side training deps for the warp venv: the PPO learner runs on jax CPU
# while mujoco_warp owns the GPU, so the two never contend for the 6 GB card.
set -euo pipefail

/home/pedro/venvs/boxer/bin/uv pip install \
  --python /home/pedro/venvs/warp_test/bin/python \
  "jax[cpu]" flax optax

/home/pedro/venvs/warp_test/bin/python - <<'EOF'
import jax
assert jax.devices()[0].platform == "cpu", jax.devices()
import flax, optax
print(f"jax {jax.__version__} (cpu), flax {flax.__version__}, "
      f"optax {optax.__version__}")
EOF
