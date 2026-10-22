# Balance recovery investigation, 2026-10-22

## What stopped and what was preserved

The adaptive run exhausted its shared allocation at **197,304,320** consumed
transitions. One of 26 complete gates passed (181,575,680), followed by three
failures. There were no two consecutive passes, so advancement was correctly
blocked. At the last frozen gate, survival was 60/64 without the impulse and
49/64 with it. The 95% per-trial criterion is unchanged.

Both original policies and configurations are preserved:

- `checkpoints/curriculum/archive-before-recovery-197304320/`
- `checkpoints/curriculum/recovery-seed-181575680/`

## Investigation and evidence

The new `boxer.curriculum_diagnose` runs first-fall traces for both policies on
seeds 7/23 and diagnostic holdouts 31/47, with 32 resets per seed and condition.
It records first-fall time, torso tilt, pelvis height, leg contacts and finite
state. The latter two seeds are not added to, or substituted for, the promotion
gate. See [raw comparison](results/curriculum/recovery/first-fall-comparison.json).

| Diagnostic sample | 181.58M policy | 197.30M policy |
| --- | --- | --- |
| All eight trials | 246/256 survivors | 216/256 survivors |
| Holdout seeds 31/47, both conditions | 121/128 survivors | 108/128 survivors |
| First-fall low-pelvis flag | 8 | 40 |
| First-fall tilt flag | 2 | 0 |
| Leg-contact / non-finite flags | 0 / 0 | 0 / 0 |

Most observed failures were pelvis collapse rather than a leg-contact
termination or numerical NaN. Several failures happened before the four-second
impulse, so recovery from pushes is not the only weakness. These data favor the
181.58M policy as a recovery seed on this sample; they do not establish a
universally best policy or certify progression.

The diagnostic uses a 256-world batched JIT scan, whereas the frozen gate runs
32 worlds per trial sequentially. Outcomes are not interchangeable: even paired
identical inputs sometimes diverged before the impulse, indicating numerical
execution sensitivity. The diagnostic is explicitly ineligible for promotion.
The original full gate, including commands, seeds, impulses and thresholds,
remains authoritative. No threshold is relaxed to make the results pass.

Source inspection found two training coverage/configuration problems:

1. Training disabled pushes, while promotion required surviving one. Initial
   random root velocity is not equivalent to recovering from a disturbance
   after the robot has settled.
2. Playground's default cached autoreset reuses each world's first data/obs but
   keeps episode info. This repeats the original spawn pool and retains gait,
   action and command history across physical resets. This documented speed
   optimization is a poor fit for the generalization and consistent episode
   state required here.

Continued PPO also lost measured robustness after 181M. The investigation does
not isolate a single causal mechanism for every regression. The changes below
address demonstrated coverage/state issues and use a more conservative update;
they are not a claim that the instability has already been solved.

## Recovery protocol

- Full randomized resets (`full_reset=True`) refresh physical state and all
  controller episode history together, including previous actions and phase.
- Training applies random horizontal velocity disturbances of 0.10-0.30 m/s
  every 3-5 seconds. They are disturbances, not pelvis support. There are still
  no root tracking forces, welds or gravity compensation.
- The evaluator explicitly disables those training disturbances. It applies
  only its original fixed impulse, keeping the benchmark unchanged.
- PPO learning rate is 0.0001, down from 0.0003. The actor, critic and normalizer
  warm-start from 181.58M; Adam and RNG are fresh, as before.
- The first rollback retained **197,304,320** consumed transitions, including
  discarded updates. The capacity incident below adds another 1,003,520; the
  safe restart retains **198,307,840** consumed transitions. A
  rollback never rewinds compute accounting or pretends that the old actor
  received the discarded training.
- The one-time recovery seed SHA is recorded in every new policy. Reusing the
  recovery flag after a shutdown retains subsequent learning instead of
  resetting to that seed again. Physical gates must still pass twice on distinct
  checkpoints before the same policy advances through the existing curriculum.

The authorized continuation extends the **shared total ceiling to 400M**.
This is approximately 202.7M additional available transitions, not 400M per
level. The launcher default remains 200M for unrelated new experiments.

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_curriculum.ps1 `
  -Publish -WarmStart -MaxTotalSteps 400000000 `
  -RestartFrom checkpoints/curriculum/recovery-seed-181575680
```

The current supervisor still advances by performance, never by time spent or
reward alone. Paired boxing integration remains downstream of the foundation.

## Constraint capacity discovered during continuation

The first full-scale continuation emitted six `nefc overflow` messages after
the short smoke test. Full random resets exercise many more sampled states
than cached spawns; the inherited 90-row training constraint allocation was
insufficient for rare samples. The prior adaptive run and frozen evaluation
logs did not contain this warning, so it is not presented as an explanation
for every earlier regression.

The learner was stopped at 198,307,840 consumed transitions. Its 1,003,520 new
updates are rejected and archived under
`checkpoints/curriculum/rejected-overflow-198307840/`. They are not used as the
recovery source. A one-time explicit forced rollback restores the original
181.58M policy while retaining the higher consumed counter. Ordinary later
resumes do not force another rollback.

Training now allocates **256 constraint rows per world**. The frozen evaluator
retains its original 90-row allocation, which showed no overflow in the recorded
gates; its new log check invalidates any gate that ever reports one. The
supervisor also monitors new child output and terminates the entire process
group on physics-allocation overflow instead of silently training on dropped
constraints. Old log warnings are excluded from the new-process scan.

## Validation and limits

Twenty-four focused tests passed, including a real forced episode timeout
that verifies fresh qpos and cleared action/gait history, a frozen evaluator
configuration, and one-time rollback without rewinding consumed steps.

A separate 61,440-transition recovery smoke run restored the chosen seed at the
197.30M accounting offset, exercised full-reset GPU PPO and produced finite
parameters/losses. Its last short epoch measured about 36,393 transitions/s;
this is not a sustained throughput guarantee. It is not included in the main
policy counter and is not evidence of improved recovery. The continuing main
job's unchanged full gates will determine whether robustness improves.

After the capacity fix, an additional isolated **1,003,520-transition** GPU
smoke run completed with finite parameters and **zero overflow messages** at
256 rows. Its offsets are diagnostic accounting, not new main-policy progress.
The supervisor test also confirms that a child emitting an overflow warning is
terminated instead of continuing for its requested runtime. Ordinary recovery
resumes preserve subsequent learning; the tested explicit forced option is used
only to discard the update interval affected by the capacity incident.

Published measurements and the validation record are under
`docs/results/curriculum/recovery/`. Code, documentation, checksums and recovery
artifacts are committed and pushed; local detailed traces remain under
`runs/diagnosis/`.
