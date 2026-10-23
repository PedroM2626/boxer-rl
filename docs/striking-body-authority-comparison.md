# Controlled whole-body residual authority investigation

## Motion-reference comparison result

Both 131,072-step diagnostic arms finished normally. Their learning-state leaves,
losses and checkpoint hashes were finite/verified; Adam count was 18,432 and no
physics overflow or invalid-state warnings were present. Main PPO remains stopped
at **1,048,576**, combined main consumption **476,041,216**. No diagnostic actor
has been adopted and no main budget extension is granted.

| Arm | Pair survivors | Red / blue impacts | Impacts in surviving episodes |
| --- | ---: | ---: | ---: |
| Original reward | 7/8 | 2 / 1 | 0 |
| CMU motion-reference reward | 7/8 | 0 / 0 | 0 |

The baseline reproduced the preceding comparison's checkpoint hash exactly.
These repeated inputs are not independent additional behavioral successes.
Motion shaping produced six training events over 142 completed rounds, including
16 falls; baseline produced five over 139 rounds, including eleven falls.
Neither met the predeclared screening criteria. Reject the motion reward and
adopt neither actor. [Full reports](results/striking-pilot/imitation-comparison-1048576/completed/analysis.json)
and both immutable payloads are archived. The completed reach and imitation
comparisons must not be relaunched.

This result does not prove imitation or the architecture cannot work with more
training. It does fail to support another reward-only variation at this point.
The older unassisted policy could coordinate all 29 joint targets, while the
current combat head directly adjusts arms only. Its frozen motor responds to
physical feedback, but cannot learn new balance responses to punches. Prior
zero-arm and amplitude probes already exposed a contact/stability tradeoff.
Investigate additional learned body authority while holding reward fixed.

## Explicit diagnostic interface migration

`WholeBodyStrikingEnv` preserves the original first 146 observation values and
seventeen outputs. Fifteen additional outputs adjust the twelve leg and three
waist joint targets after the learned motor actor, with the same 0.8/0.2 action
filter. Fifteen previous high-level body outputs are appended to observations,
giving **161 inputs and 32 outputs**. The paired engine records the actual applied
29 joint actions, including these corrections, for motor feedback. There is no
root action, support force, weld, gravity compensation, gait or strike playback.

The certified motor's network weights remain frozen. This is **learned whole-body
residual adaptation**, not end-to-end fine-tuning of that network or restoration
of the old policy. The underlying model, feet, inertias, servos, physics interval,
contacts, free roots and physical impact/fall criteria remain unchanged.

Both diagnostic arms start from the **same explicit expansion** of the unchanged
main 1,048,576 checkpoint, SHA256
`efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0`:

- All old actor/critic weights and normalization features are retained.
- New input weights and output means are zero; added log standard deviations
  start at -1.9. At zero new features, old means/values agree numerically and
  new output means are exactly zero. No random replacement initializes combat.
- Existing Adam moments and its count are retained. Added moments are zero.
  The shared Adam timestep is not reset for old or new parameters.
- New normalization features start with mean zero/variance one while the existing
  global sample count remains unchanged. RNG and all consumed/optimized counters
  are retained. Simulation state is reset as on ordinary resumes.

This is a documented migration for isolated diagnostics, not a forced rollback
or a routine resume of the main job. After migration, ordinary checkpoint/resume
within each interface keeps its exact learning state. Main remains the original
17-output actor and must not silently load this family.

## Matched experiment and budget

Both arms have identical expanded networks, Adam, normalization, RNG, action noise
and input/output dimensions. They differ in only the allowed body correction:

1. **Frozen-body control:** fifteen new outputs have zero joint authority.
2. **Adaptive-body candidate:** each new output has at most **0.1 radian** of
   joint-target authority (about 5.7 degrees), filtered over time.

The control still learns through the original arm and command outputs. Its unused
body outputs/history and extra network dimensions match the candidate; it is not
the old 17-output baseline with a different random-sample shape. At identical
zero added actions, the augmented environment reproduces the base physical
trajectory and its first 146 inputs exactly in the regression assay. Stochastic
training deliberately explores the candidate's small body corrections; zero
initial means do not imply identical sampled physical trajectories.

Each arm receives **131,072 new diagnostic learner transitions** (262,144 total),
sequentially from the common expanded source. Both use the unchanged original
reward, 0.45 m engagement profile, 2 m nominal initial separation, four worlds,
128-control unroll and identical PPO hyperparameters. No proximity or motion
reference reward is enabled. Body authority does not add a reward term.

Both must complete the unchanged eight-episode ten-second physical evaluator,
with seeds 7/23/31/47 and both side arrangements. Compare actual impacts,
survival, complete-episode impacts, scoring seed coverage and finite/overflow
state. Keep the previous screening: at least 7/8 pair survival, no worse than
the matched control, impacts in surviving episodes and scoring on at least two
distinct seeds. Neither is adopted automatically. A short failed pilot cannot
prove the whole-body approach is impossible, but it cannot justify blind main
extensions or importing the less bad actor.

The original foundation certification remains valid for its preserved motor
actor. It **does not certify the new composite controller**, which can change leg
targets. A favorable candidate must receive broader held-out paired evaluation
and balance/locomotion regression checks with body corrections active before
any main adoption. This comparison is not a curriculum promotion or boxing
certification, and no existing gate threshold is weakened.

## Operation

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_body_compare.ps1
```

Durable diagnostic output:
`/home/pedro/boxing-runs/striking-body-compare-1048576/comparison.json`.
Read parent state and the active child's `progress.json`, logs and process before
launching anything else. Parent counters update at arm boundaries; live diagnostic
work is the sum of child `consumed_steps - 1048576`, including discarded attempts.
Lineage counters inside child manifests are not new main work. The hidden WSL
host runs sequential children under the existing single-learner lock. The saved
monitor schedule remains unchanged. Inspect both full evaluations when finished,
archive payloads/results, and make a concrete continuation decision.

## Validation and compute ledger

The preceding completed motion comparison consumed 262,144 additional diagnostic
learner transitions and 7,173 evaluation world controls. Totals before interface
validation were **528,128 diagnostic PPO transitions** and **84,809 formal
striking diagnostic/evaluation world controls**.

The full body smoke used 1,024 transitions per arm, retained Adam count 16,400,
and completed all eight physical episodes per arm with finite payloads and
original impact rules. It added 2,048 diagnostic PPO transitions and 6,321 world
controls. Both survived 6/8; the candidate's three events occurred in a falling
episode. These are implementation checks, not a favorable training result.

A separate default 17-output compatibility smoke consumed 1,024 transitions.
Every actor, critic, optimizer, normalizer and RNG leaf matched the corresponding
pre-refactor smoke **bit for bit**. Trainer generalization therefore preserved
the tested default PPO update; metadata fields alone were added. Its four-control
evaluator check also completed with the original 146/17 interface. Smoke actors
are archived and initialize neither full diagnostic arm.

Totals before the full comparison are **531,200 diagnostic PPO transitions** and
**91,134 formal striking diagnostic/evaluation world controls**. At its full
budget, diagnostic PPO will total **793,344**; append the two actual evaluations
separately. Main and foundation counters remain unchanged throughout.

Fifty focused tests passed across migration, action history/authority, paired
physics, physical scorecard, legacy PPO/profile behavior and curriculum control.
The added zero-correction regression checks exact qpos, reward and base-input
agreement over five real physics controls; the migration checks preserve old
outputs/values within numerical tolerance and every original optimizer element.
The PowerShell launcher passed syntax parsing.

The full comparison runs through hidden Windows host PID 3340, Linux parent 704
and single frozen-body child 845 at launch inspection. It advanced past the
source with finite verified 161/32 learning state and zero body authority in the
control. The main payload hash and complete foundation motor certifications were
rechecked unchanged. The [launch record](results/striking-pilot/body-comparison-1048576/launched.json)
pins code hashes and runtime counters; [smoke validation](results/striking-pilot/body-comparison-1048576/validation.json)
records separate compute and compatibility evidence. No main or foundation
learner was active, and the monitoring schedule is unchanged.

## Completed body comparison and holdouts

Both body-authority arms finished at their reviewed diagnostic budget, each
with 7/8 pair survival and zero deterministic impacts. Neither was adopted.
The [completed 256-episode holdout and contact audit](striking-body-holdout-review.md)
record one stable two-way exchange without a robust body-authority advantage.
Main remains stopped at 1,048,576 and neither actor is adopted. The next reviewed
decision is [bounded single-learner partner practice](striking-partner-curriculum.md),
using the original main checkpoint and separate diagnostic compute. Inspect that
job's state and process before launching anything else.
