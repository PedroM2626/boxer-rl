# Unassisted G1 curriculum

## Decision and preserved experiment

On 2026-10-21 the user authorized stopping assisted sparring and implementing a
standing/walking curriculum. The assisted learner had reached **116,633,600
observed transitions**. Its last durable checkpoint, **115,712,000**, is
archived under `checkpoints/skills_200m/stopped-step-115712000/`, including the
actor, critic, optimizer, normalization and saved sampling key. Its 100M
evaluation remains the last evaluated assisted milestone. Interruption was a
requested change of experiment, not evidence that 200M steps were completed.

The assisted six-action policy holds the legs at posture targets and applies
pelvis springs. More training cannot turn those unavailable controls into learned
balance. The replacement starts a fresh 29-action policy, with a compact boxing
guard, and learns the legs and the rest of the body together.

## Physical control contract

`boxer/curriculum_env.py` subclasses the official MuJoCo Playground G1 joystick
environment. Every position actuator receives a policy action at 50 Hz; ten
2 ms physics substeps resolve each action. Actions offset the default pose by
up to 0.5 rad. The actor sees 103 inputs, including requested local velocity,
proprioception, previous actions and gait phase. The critic has 216 privileged
inputs. All 12 hip/knee/ankle actuators are controllable. Position servos create
physical joint torques; no servo moves the floating base directly.

There are no root welds, stabilizing pelvis springs, gravity compensation,
root tracking forces or prescribed gait playback. The training push feature is
disabled initially. Evaluation may inject a documented horizontal velocity
impulse; this is a disturbance, not assistance. A trial ends on the first fall
(pelvis below 0.55 m or torso up component below 0.60), the upstream leg-contact
termination, or non-finite state. Evaluation never resets a fallen trial and
counts it as surviving.

The upstream feet-only training model is used for locomotion. It does not yet
contain the two-boxer head/body/fist collision scene. Its physical floor
contacts and joint dynamics are real, but its locomotion results must not be
presented as boxing contact results.

## Stages and promotion gates

The supervisor runs each foundation stage for its full configured budget before
evaluating it. Default: **200M transitions per stage**, potentially 600M across
all three. One transition is one robot's 20 ms control step. PPO rounds budgets
up to whole training blocks; actual counters are recorded in every snapshot.
The two-boxer experiment has a different counting convention and is not added
to these counters. Early random-policy falls and short-term reward plateaus do
not cancel training.

| Stage | Training task | Gate, for every evaluation seed and command |
| --- | --- | --- |
| Balance | Zero requested translation/yaw, standing height and posture rewards, no stepping-phase reward | At least 95% survive 10 s, linear tracking error ≤0.15 m/s, yaw error ≤0.25 rad/s; same thresholds after a 0.2 m/s impulse on each horizontal axis at 4 s |
| Walk | Forward/backward/lateral/yaw command tracking, retaining the boxing guard | At least 90% survive 10 s; linear error ≤0.20 m/s and yaw error ≤0.25 rad/s; translation in each requested direction ≥45% of commanded speed ×10 s |
| Approach | A fixed navigation marker begins 1.8–2.2 m ahead; request velocity toward it and stop at 0.75 m | At least 90% survive and finish within 0.95 m; tracking errors as in walking |

The deterministic gate uses seeds **7 and 23**, with **32 independent reset
states per trial**. Walking tests standing, +0.40/-0.25 m/s forward/backward,
±0.20 m/s laterally and ±0.40 rad/s yaw. Velocity error excludes the first
second of settling; survival includes it. A failed direction cannot be hidden
by averaging successful directions. Distances come from simulated root qpos,
never from commanded velocity or animation. Failed trials contribute zero to
the required direction's displacement. Rewards do not decide promotion.

A failed gate after the full budget stops in `gate_not_passed` and preserves
the policy for inspection. There is no automatic promotion or endless restart.
A successful stage seeds the next stage's actor, critic and normalization.
Normalization uses a variance epsilon of 0.01 (standard-deviation floor 0.1).
Zero command dimensions in standing must not produce enormous normalized
inputs when walking commands are introduced. Adam and the sampling RNG are fresh at the stage change. This is deliberate
fine-tuning, not an optimizer-state resume.

## Transfer to two boxers

The final intended task is two free humanoids spawning around two metres apart,
approaching, defending and landing legal fist contacts while maintaining their
own balance. **This paired transfer is not implemented by the foundation job.**
The supervisor explicitly ends in `foundation_completed` after the approach
gate; it does not relaunch the assisted environment or call navigation boxing.

Before paired training, the locomotion actor must be integrated with the real
fist/head/torso collision scene and checked against the recorded joint order,
gains, default pose, action scaling and observation contract. The older
Menagerie boxing scene and the Playground feet-only model cannot be assumed
interchangeable. Evaluation must then measure real clean punches, falls,
footwork and engagement from the wider spawn distance, including swapped
corners and held-out seeds. A learned locomotion controller can be a lower
layer, but the old pelvis springs and fixed leg targets must remain disabled.

## Reproducible runtime and operation

The dedicated Linux/WSL environment is `$HOME/venvs/boxing_curriculum`; the
existing assisted Warp environment is not modified. Run:

```bash
bash tools/setup_curriculum.sh
```

`requirements-curriculum.txt` pins JAX CUDA, Brax, MuJoCo/MJX/Warp and the upstream
Playground commit. Asset setup fetches only the G1 directory at the pinned
Menagerie commit. Set `BOXER_CURRICULUM_MENAGERIE` to choose another cache path;
the environment rejects a different Git revision. Setup requires either `uv`
(also detected in the existing project venv) or Python 3.12 with venv/pip.
The 6 GB laptop uses 1,024 worlds by default, smaller than the official large-GPU configuration. CUDA
physics and PPO share one GPU. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`JAX_PLATFORMS=cuda,cpu`; the CPU device is needed for Brax callbacks.

Start a persistent hidden Windows WSL host from the repository:

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_curriculum.ps1 -Publish
```

The launcher protects against a duplicate live host and the supervisor holds an
exclusive Linux file lock. Publishing also holds a shared artifact-namespace
lock so a differently named run cannot overwrite the same published policies.
Keep Windows and WSL running. Outputs:

- `runs/curriculum/state.json`: supervisor PID, phase, stage and completion.
- `runs/curriculum/<stage>/progress.json` and `progress.jsonl`: actual counters,
  throughput and training metrics. The first compilation may take minutes.
- `runs/curriculum/<stage>/checkpoint/`: atomic, checksum-verified snapshots.
- `runs/curriculum/<stage>/train.log` and `eval.log`: full subprocess logs.
- `checkpoints/curriculum/<stage>/`: latest published policy and model contract.
- `docs/results/curriculum/`: published progress and completed gate evaluations.
- `fights/curriculum/`: deterministic videos after a stage budget completes.

At the first snapshot and roughly every five million transitions, a development
preview also evaluates eight resets at seed 7 and renders one trajectory. PPO
waits with its optimizer resident, then continues; this preview does not restart
training. Preview JSON is marked `kind: development_preview` and `passed: null`,
and **cannot promote a stage**. Its video is `<stage>-preview.mp4`; the complete
two-seed gate and final evaluation video remain separate. This makes early
falls and later changes visible without interpreting a short preview as proof.

Checkpoints are saved approximately every million transitions. At the first
checkpoint, approximately every five million transitions afterwards, and the
stage's final checkpoint, `-Publish` exports the actor, critic, observation normalizer,
config and manifest, then commits and pushes only the owned curriculum artifact
paths. Git keeps earlier snapshots; local numbered snapshots are also retained.
Unrelated staged changes block publication. A publication/child failure is
reported and stops the job with its latest valid local checkpoint preserved.
The gate result is published before any next stage begins.

To stop, send SIGTERM to the Linux supervisor PID from `state.json`:

```powershell
wsl -e kill -TERM <supervisor-pid>
```

To recover, use the same launcher with `-WarmStart`. **Brax 0.14.2 does not expose
the optimizer or rollout state in its policy callback.** These new snapshots
restore the actor, critic and normalizer, but initialize Adam, RNG and physical
rounds again. Cumulative stage counters include the restored checkpoint, and
only the remaining budget runs. They are explicitly marked `optimizer_saved:
false`; they are not equivalent to the full-state assisted checkpoints.

For a manual single-stage experiment and deterministic evaluation:

```bash
export PYTHONPATH=. JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
"$HOME/venvs/boxing_curriculum/bin/python" -m boxer.curriculum_train \
  --stage balance --output runs/my_balance --steps 200000000
"$HOME/venvs/boxing_curriculum/bin/python" -m boxer.curriculum_eval \
  runs/my_balance/checkpoint --output runs/my_balance/evaluation.json \
  --video runs/my_balance/evaluation.mp4
"$HOME/venvs/boxing_curriculum/bin/python" -m pytest \
  tests/test_curriculum.py tests/test_curriculum_job.py -q
```

## Initial validation

A 20,480-transition GPU smoke run saved and reloaded the new policy. Its
deterministic standing evaluation failed, as expected at this budget: zero
10-second survivors, with mean survival roughly 0.74/0.89 s at seeds 7/23
(eight reset states per seed). This verifies that the measurement detects
falls; it is not evidence of learned balance. The long run starts separately
from fresh parameters rather than transplanting this smoke policy.

A separate 61,440-transition, 1,024-world balance pilot with the normalization
floor reached about **43,810 transitions/s** in its last training epoch after
compilation, with finite weights and losses. This is a short throughput
measurement, not a sustained-speed guarantee or an evaluation of balance.

The balance-to-walk and walk-to-approach warm starts each completed 40,960
additional smoke transitions with finite weights and losses. Twelve focused
tests passed, including physical leg authority, checksum rejection, failed-gate
handling, publication exclusion and the assisted report regression checks.
The setup script completed and the dependency checker accepted all 101 installed
packages. [Validation records](results/curriculum/validation.json), pilot metrics
and the runtime package list are committed under `docs/results/curriculum/`.

## Upstream attribution

This implementation depends on the Apache-2.0 licensed
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
G1 joystick task at `12ad7f3842791ced39f27e5ac96d066bbe361653` and
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
G1 assets at `1b86ece576591213e2b666ebf59508454200ca97`.
The upstream repository's licenses remain with those dependencies. No humanoid
model or animation was invented for this experiment. The architecture, reward
changes, physical gates, checkpoint manifests and job management here are
project additions.
