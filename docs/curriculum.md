# Adaptive unassisted G1 curriculum

**Current stage (2026-10-22):** all four foundation levels are certified by
[two distinct consecutive complete physical passes](results/curriculum/foundation-certification/README.md).
The foundation finished normally at **474,992,640 consumed main transitions**
under the **475M** ceiling. Full walking passed at 474,562,560; approach then
added 430,080 transitions. The exact actor payload was retained at transition.

**Current operation:** the [opponent-aware striking pilot](striking-pilot.md)
adds trainable arm residuals and independent fist-impact events over the unchanged
certified motor actor. Main boxing PPO remains stopped at **1,048,576 learner
transitions**, without a budget extension or actor adoption. The matched
whole-body authority comparison completed; both controllers had 7/8 paired
survival and zero deterministic impacts. The separate eight-episode saved-Gaussian
checks also found zero impacts. The subsequent
[completed 256-episode holdout and causal contact audit](striking-body-holdout-review.md)
found one reproducible stable two-way exchange, but no robust benefit from body
authority. The [next bounded boxing curriculum pilot](striking-partner-curriculum.md)
uses one learner against a neutral reactive motor partner, retaining both free
bodies and all physical contacts. Its separate allocation is 131,072 new diagnostic
learner transitions, with source/final partner trials and unchanged full self-play
evaluation. The original main actor is copied with all learning state retained;
no failed candidate is adopted. Inspect
`/home/pedro/boxing-runs/striking-partner-1048576/partner.json` and the active child
before any launch. Keep main and foundation stopped; do not repeat completed
reward/body comparisons or holdouts. Diagnostics do not count as main advancement.
No gate is relaxed, stage promoted or candidate composite controller certified.

Full walking resumed at 422.89152M main transitions after a
[supervisor log-read fault](curriculum-log-read-recovery.md). The targeted bounded
retry preserves warning inspection and the latest policy; no training or gate
setting changed during that recovery.

**Earlier recovery:** the original shared allocation stopped at
197.30M without repeated gate passes. The [recovery investigation](curriculum-recovery.md)
selects the preserved 181.58M actor, retains all 197.30M consumed transitions,
enables full randomized resets and training disturbances, and reduces PPO's
learning rate. That recovery launched the earlier **400M total** allocation.
The promotion protocol below remains unchanged; the 200M references describe
the initial/default allocation, not an additional per-level requirement.

The [walking tracking comparison](curriculum-tracking-experiment.md) documents
the subsequent directional plateau and a bounded training-only reward experiment.
The [reviewed result](results/curriculum/tracking-comparison-112066560/analysis.md)
supports easy-walking training with tracking width 0.01. The original main policy
resumed at 319.8976M consumed transitions without a diagnostic actor substitution.
At that point the physical benchmark and then-current 400M ceiling were unchanged.

After the lateral displacement plateau persisted, the
[lateral-practice investigation](curriculum-lateral-practice.md) added targeted
sideways commands. That continuation started at 360.36608M consumed transitions
with the same actor and then-current 400M ceiling. The later reviewed 475M ceiling
above is now active.

## Current protocol

The user approved replacing fixed per-stage budgets with performance-based
promotion and gradually mixed standing/walking practice on 2026-10-21.
**200M transitions is now an initial shared ceiling across the foundation,
including the 66,232,320 transitions already trained. It is not a minimum or a
200M allocation for every stage.** It may finish earlier when skills pass their
gates. A budget is not evidence that a skill has been learned.

One full-body actor controls all 29 joints, including the 12 leg joints. Each
level loads the preceding actor, critic and observation statistics after checking
the same model/action/observation contract. No random policy replaces a trained
one. This is a single policy learning progressively harder tasks, rather than
independent agents relearning each task from scratch.

There are no pelvis springs, root welds, gravity compensation, root position
controllers or gait playback. Position servos produce joint torques at 50 Hz;
ten 2 ms MuJoCo steps resolve each action. The policy uses 103 actor inputs and
the critic 216 privileged inputs. The compact boxing guard and all joint actions
remain available throughout. Normalization has variance epsilon 0.01 so new
nonzero commands cannot explode previously constant input dimensions.

## Mixed practice and increasing difficulty

| Level | Training command distribution | Complete physical gate |
| --- | --- | --- |
| Balance | 70% zero commands; otherwise very gentle forward/backward, lateral and yaw requests (x -0.03 to +0.05 m/s, y +/-0.02 m/s, yaw +/-0.05 rad/s) | At least 95% survive 10 s, linear error <=0.15 m/s and yaw error <=0.25 rad/s; repeat with a horizontal impulse at 4 s |
| Walk easy | 40% zero; 30% pure lateral; 30% mixed commands at 25% of full walking ranges | At least 90% survive every direction, linear error <=0.20 m/s, yaw error <=0.25 rad/s and actual directed displacement >=45% of requested speed x10 s |
| Walk | 40% zero commands; x -0.35 to +0.55 m/s, y +/-0.25 m/s, yaw +/-0.5 rad/s | Same physical criteria at full evaluation commands |
| Approach | Navigate toward a fixed marker initially 1.8-2.2 m ahead, stopping around 0.75 m away | At least 90% survive and finish within 0.95 m, with walking tracking-error limits |

Standing practice is retained when walking starts. The air-time reward is zero
for zero commands, so standing trials do not reward unnecessary hopping. The
balance level begins learning small movement corrections before its standing
gate passes. Reward changes and command ranges are recorded in each config.

Evaluation uses seeds **7 and 23**, with **32 reset states per trial**. Walking
tests standing, +0.40/-0.25 m/s forward/backward, +/-0.20 m/s lateral and
+/-0.40 rad/s yaw; easy walking tests one quarter of those commands. Balance
tests both undisturbed trials and a [0.2, -0.2] m/s horizontal velocity impulse
at four seconds. That impulse is a disturbance, never a stabilizing controller.

Every trial ends at its first fall: pelvis below 0.55 m, torso up component
below 0.60, the upstream leg-contact termination, or non-finite state. Failed
trials never reset into the survival count. Tracking errors exclude the first
second of settling; survival includes it. Distances come from simulated qpos,
and a fallen trial contributes zero to the required directed displacement.
The evaluation gate checks every seed, disturbance and requested direction.

## Adaptive advancement and budget

The learner performs a full gate before optimization and approximately every
**5M new transitions**, rounded to checkpoint boundaries. The optimizer stays
resident while a small evaluation subprocess runs. Failed gates reset the
passing streak and training continues; they do not restart PPO or stop it for
a short plateau. No training reward threshold advances a level.

**Two consecutive complete passes on distinct checkpoints are required.** A
preview, fewer than 32 worlds, a missing direction, wrong stage, stale checkpoint
hash or identical policy cannot certify promotion. The supervisor independently
checks the gate history and latest checkpoint before starting the next level.
Promotion ends the current PPO invocation immediately; it does not wait for a
fixed 200M horizon. The next level fine-tunes the same networks and normalizer
with fresh Adam and RNG, as explicitly recorded in its config.

All levels share `--max-total-steps` (default 200M). Earlier foundation work is
included once; the old assisted experiment is separate and is not added. The
learner rounds its requested remaining ceiling DOWN to complete PPO epochs, so
it cannot overshoot the shared ceiling. A small remainder can be unused.
Without two passing gates, exhausting that allocation ends in `budget_reached`,
with no automatic promotion, endless retry or random reset. An explicit later
extension can use `-MaxTotalSteps <larger-total> -WarmStart`.

This protocol does not assert that any particular level needs 200M steps or
that the foundation will succeed within that initial total.

## Preserved training and recovery semantics

The preceding fixed-budget balance run stopped at a durable checkpoint of
**66,232,320** transitions. An immutable pre-migration snapshot and original
config are in `checkpoints/curriculum/archive-fixed-budget-66232320/`. Its last
small preview is not a promotion certificate. Adaptive-v2 initially warm-started
that policy with gentle mixed commands. Recovery-v3 now restores the preserved
181.58M candidate at the 197.30M consumed offset, as described in the recovery report.

**Brax 0.14.2's policy callback does not expose Adam or simulator/RNG state.**
Snapshots include the actor, critic, normalizer and exact contract, but are
marked `optimizer_saved: false`. This one-time migration therefore initializes
Adam and RNG again; it is not a full-state continuation. The same limitation
applies after shutdown or interruption and at a level change. It does not apply
to periodic evaluations within one live PPO invocation: Adam remains resident.

The earlier assisted boxing run stopped at 116,633,600 observed transitions,
with its last durable checkpoint at 115,712,000. That independent experiment
and its real contact-scoring results remain archived in
`checkpoints/skills_200m/stopped-step-115712000/` and `docs/skills-200m.md`.
Its six-action policy is not used as a replacement for learned leg control.

## Operation and published artifacts

The isolated Linux/WSL GPU runtime is `$HOME/venvs/boxing_curriculum`:

```bash
bash tools/setup_curriculum.sh
```

Setup uses uv (also detected in the existing project venv) or Python 3.12 with
venv/pip. It installs `requirements-curriculum.txt` and fetches only the pinned
G1 asset directory. `BOXER_CURRICULUM_MENAGERIE` chooses another asset cache;
the environment rejects a different revision. The default is 1,024 worlds on
the 6 GB laptop, with `JAX_PLATFORMS=cuda,cpu`, preallocation disabled and EGL
rendering. Existing assisted-training environments are not modified.

From PowerShell in the repository, start a persistent hidden WSL host:

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_curriculum.ps1 -Publish -WarmStart -MaxTotalSteps 475000000
# A fresh output without prior state does not need -WarmStart.
```

`-MaxTotalSteps` sets the shared total ceiling, `-GateEvery` sets the evaluation
interval, and `-Worlds` sets the parallel batch. `BOXER_CURRICULUM_PYTHON`
selects another interpreter. Keep Windows and WSL running. The launcher checks
for a duplicate host; the supervisor holds both a run lock and a publication
namespace lock. Stopping the supervisor terminates the entire learner process
group, including an active evaluator, while retaining completed checkpoints.

- `runs/curriculum/state.json`: supervisor status, active level and total at the last supervisor event.
- `runs/curriculum/<level>/progress.json`: live level and total transition counters and PPO metrics.
- `runs/curriculum/<level>/checkpoint/`: numbered atomic, checksum-verified policies.
- `runs/curriculum/<level>/gates/`: exact historical gate results for each evaluated checkpoint.
- `runs/curriculum/<level>/gate_history.json`: full promotion history.
- `runs/curriculum/<level>/training_result.json`: passing streak and promotion/budget status.
- `checkpoints/curriculum/<level>/`: latest published policy and complete contract/config.
- `docs/results/curriculum/`: published progress, gate history and job state.
- `fights/curriculum/<level>.mp4`: one actual simulated trajectory from the latest full gate.

With `-Publish`, completed gates export artifacts, commit and push only the
owned curriculum paths. Git retains previous policies. Unrelated staged files
block publication. Preview files from the original protocol remain historical
records; their own counters/hashes identify them and they cannot promote a level.
Failures retain the last valid checkpoint and are recorded explicitly.

Stop by sending SIGTERM to the Linux supervisor PID from `state.json`:

```powershell
wsl -e kill -TERM <supervisor-pid>
```

For standalone evaluation:

```bash
export PYTHONPATH=. JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
"$HOME/venvs/boxing_curriculum/bin/python" -m boxer.curriculum_eval \
  runs/curriculum/balance/checkpoint --output runs/manual_gate.json
```

## Scope, validation and upstream attribution

The foundation uses the established feet-only G1 locomotion model. It has real
floor contacts and joint dynamics, but does not contain a physical opponent or
the boxing fist/head/torso collision scene. The navigation marker is not a
support or a boxer. After all levels pass, the supervisor ends honestly in
`foundation_completed`. Transfer to two free boxers remains a separate
integration and contact-validation experiment; this job never silently
re-enables pelvis springs or calls navigation boxing.

Before paired training, check joint order, gains, default pose, action scaling,
observations and physical dynamics against the boxing scene. Then evaluate real
legal fist contacts, falls, footwork and engagement from wider spawns, exchanging
corners and using held-out seeds. The two scenes are not assumed interchangeable.

The [CPU transfer audit](curriculum-transfer-audit.md) now records the measured
gain, pose, timestep, and action-scale differences that the adapter must resolve.

Historical smoke tests, pilot metrics and dependency versions remain under
`docs/results/curriculum/`. Adaptive tests additionally verify early exit before
the budget, repeated complete gates, shared accounting, stale/preview rejection,
command mixtures and physical leg authority. These are engineering validations,
not evidence that locomotion or boxing has already been mastered.

The adaptive change passed **20 focused tests**. A separate GPU diagnostic
warm-started the preserved actor into easy walking, executed 20,480 transitions
and completed two 14-trial gates (32 resets per trial, seeds 7/23). Both gates
failed, so it exhausted its deliberately small diagnostic allocation without
promotion. Parameters and losses remained finite. This diagnostic is separate
from the main counter and is not its initialization source. See the
[adaptive validation record](results/curriculum/adaptive-validation.json).

Dependencies are the Apache-2.0 licensed
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) G1 task
at `12ad7f3842791ced39f27e5ac96d066bbe361653` and
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) assets at
`1b86ece576591213e2b666ebf59508454200ca97`. Upstream licenses remain with those
dependencies; no humanoid model or animation was invented for this experiment.
