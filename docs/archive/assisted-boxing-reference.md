# Archived assisted boxing: runnable sparring reference (2026-10-21)

Moved from the README on 2026-11-03, unchanged apart from link paths.

The stopped experiment continued the pinned 2.11M seed toward **200M total
learner transitions**. Completed milestones were 10M, 25M, 50M and 100M.
[Protocol and operations](../long-training-protocol.md) explain background
execution, recovery and automatic commits/pushes; the
[published experiment report](../skills-200m.md) records the interruption and completed milestones.

The `skills` environment adds a hierarchical controller: a shared PPO policy
chooses punches, head/body aim, advance, circle and guard for both G1 boxers.
The motor controller executes complete extension/retraction cycles. Hits require
an actual **fist-to-head or fist-to-torso contact above 50 N**, with approach
speed above 1 m/s. Forearms block but do not score; punches below the torso do
not score. Impacts are checked at every 4 ms physics substep, using the fist's
pre-impact velocity as well as its post-impact velocity.

**This mode has explicit physical balance assistance.** Bounded pelvis springs
keep the robots upright while PPO learns boxing decisions. The roots remain
free and compliant under impacts, but walking and balance are not learned in
this mode. The video labels the assistance. This is a practical sparring
curriculum, not evidence of an unassisted humanoid boxing policy.

The verified local run, `runs/skills2`, completed **2,109,440 learner
transitions**. In 128 deterministic evaluation rounds, the two sides landed
**3.36 / 3.59 clean punches per round**, including **0.53 / 0.48 head shots**,
with no falls. The delivered [12-second video](../../fights/assisted_ppo_selfplay.mp4)
shows five red and three blue clean hits. The evaluation contains one damage
knockout; the other rounds run out their clock. See the
[saved evaluation](../results/assisted_selfplay_eval.json).
The reproducible seed checkpoint is included in
[`checkpoints/skills_seed_2m`](../../checkpoints/skills_seed_2m/README.md), including
the policy, critic, optimizer and observation statistics.

The compact guard uses the G1's legal negative elbow range and keeps the
wrists within 0.15 rad. Old checkpoints keep their original guards and direct
joint controls; their results below describe a different experiment.

```bash
# Linux / WSL2, Python 3.12. Use a separate environment for Warp.
python3.12 -m venv "$HOME/venvs/warp_test"
"$HOME/venvs/warp_test/bin/pip" install -r requirements-warp.txt

# Reproduce the two-million-transition self-play run.
bash tools/train_skills.sh --output runs/my_sparring

# Render the learned policy in its own environment, without exploration noise.
MUJOCO_GL=egl PYTHONPATH=. JAX_PLATFORMS=cpu \
  "$HOME/venvs/warp_test/bin/python" tools/warp_video.py runs/my_sparring \
  --length 600 --output fights/my_sparring.mp4

# Compare both corners, exchange corners, and save the measured results.
PYTHONPATH=. JAX_PLATFORMS=cpu \
  "$HOME/venvs/warp_test/bin/python" -m boxer.warp_arena \
  runs/my_sparring runs/my_sparring --worlds 64 --steps 600 \
  --output runs/my_sparring/selfplay.json

# A controller-only baseline, clearly labelled as scripted.
MUJOCO_GL=egl PYTHONPATH=. JAX_PLATFORMS=cpu \
  "$HOME/venvs/warp_test/bin/python" tools/warp_video.py --scripted \
  --output fights/assisted_baseline.mp4

PYTHONPATH=. JAX_PLATFORMS=cpu "$HOME/venvs/warp_test/bin/python" \
  -m pytest tests/test_skills.py tests/test_warp_training.py -q
```

`BOXER_PYTHON` selects another interpreter for `train_skills.sh`. Repeat the
same training command with `--resume --total-steps <larger-total>` to continue.
The trainer rejects incompatible environment/network settings before replacing
the recorded configuration, and rejects an existing run unless `--resume` is
given. See [the assisted boxing notes](../assisted-boxing.md) for the control
contract and measured results. All code, logs, documentation and video labels
are in English.
