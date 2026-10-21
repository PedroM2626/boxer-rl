# Assisted self-play seed: 2,109,440 transitions

This bundle preserves the measured `runs/skills2` checkpoint before the long
training continuation. It contains the policy, critic, Adam state and running
observation statistics, plus the original environment/PPO configuration.
It uses explicit balance assistance and the six-action boxing controller.

Render it from the repository root in the Warp environment:

```bash
MUJOCO_GL=egl PYTHONPATH=. JAX_PLATFORMS=cpu \
  "$HOME/venvs/warp_test/bin/python" tools/warp_video.py \
  checkpoints/skills_seed_2m --length 600 --output runs/seed_replay.mp4
```

The payload uses Python pickle and should only be loaded from a trusted source.
The project's checkpoint readers also accept its explicit `step-2109440.pkl`
path. The original simulator state and random key were not saved by this older
checkpoint format; a continuation starts a new physical round.

Evaluation: [128 deterministic rounds](../../docs/results/assisted_selfplay_eval.json).
Video: [12-second self-play](../../fights/assisted_ppo_selfplay.mp4).
