# Bounded ordinary continuation of the latest retention actor

**Completed:** both distinct complete generations fail. Final models, full
source/final fresh reports, per-block metrics, `completed-continuation.json` and
`analysis.json` are archived. Native states/losses are finite and error logs empty.
See the [completed review and next bounded task](../../../striking-neutral-retention-curriculum.md).
The launch/recovery notes below retain their original scope.

Live state is `/home/pedro/boxing-runs/striking-retention-1605632/continuation.json`.
`launch-state.json` is a snapshot, not live accounting. The exact source policy and
configuration are archived in
`checkpoints/striking-validation/retention-1572864/ppo_retention/`.
The latest checkpoint hash was verified against its payload after launch.

See [the reviewed allocation](../../../striking-retention-review.md): at most
32,768 new learner credits / 16,384 collection world controls, two distinct full
screen evaluations 16,384 credits apart, source/final fresh seeds 8,000,003–006,
at most 48,000 evaluation world controls. Stop on a screen profile with at most
3/8 survivors; unchanged full promotion criteria still apply. No rollback,
optimizer reset, profile change, main adoption or automatic extension.

Pre-job diagnostic PPO is 1,515,776 and formal diagnostic simulation 611,668 world
controls. Main foundation/boxing counters remain 474,992,640 / 1,048,576. Preserve
all attempted collection and evaluation controls and inspect the active child,
checkpoint, progress, full gate history and native logs before another launch.
Archive both complete gate generations and final independent fresh results when
the bounded job finishes; this launch snapshot is not final training evidence.

The first native start failed before any training physics/update because the
supervisor and child shared `.lock` in this single output directory. The corrected
supervisor owns `.continuation.lock`. `pretraining-lock-failure.txt` preserves the
original error; `recovery-state.json` records verified zero-training recovery with
all completed source evaluation retained. `recovery-validation.json` records a
subsequent finite real PPO checkpoint, correct independent counts and an empty
native error log. No rollback, reset or evaluation repeat occurred. The special
recovery flag is invalid once training has consumed steps.
