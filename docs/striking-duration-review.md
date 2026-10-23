# Striking review at 524,288 transitions

On 2026-10-23 the 0.45 m range continuation stopped normally at its reviewed
524,288 boxing learner-transition ceiling. The immutable checkpoint is
`checkpoints/striking-pilot/step-524288`, SHA256
`c9e6ceeb5164e158daf5d92bd35004bbb9cc6a6b70772395e5f1f8bae00b7205`.
All actor, critic, Adam, normalizer and RNG leaves and recorded losses were finite.
Adam count was 8,192, consistent with all completed updates. There were no
physics overflow/nonfinite warnings or discarded collected transitions. All four
foundation certifications were rechecked against complete gate histories and
latest checkpoint hashes; the foundation remains stopped at 474,992,640.

## Physical results and investigation

The invocation recorded 33 qualifying training events (red 13, blue 20) across
279 completed rounds, including 30 fall rounds. The preceding invocation had
15 events across 238 rounds, including 13 falls. These are invocation counters,
not cumulative policy steps, and are not independent certification results.

The unchanged deterministic evaluation used seeds 7/23/31/47, both side
arrangements and ten-second first-fall episodes. It found **zero events, zero
raw legal impulses and 7/8 surviving pairs**. Seed 47 without side swap fell at
1.30 s. This regressed from the same-policy 262,144 range diagnostic's 8/8
survival and two red events. The earlier 97,280 snapshot also remains archived;
its two blue events do not transfer to the current checkpoint by assumption.

`tools/diagnose_striking_behavior.py` runs read-only interventions using the exact
saved checkpoint, independent RNG and unchanged physics/impact scorecard:

| Behavior | Pair survivors | Qualifying impacts |
| --- | ---: | ---: |
| Deterministic policy | 7/8 | 0 |
| Saved Gaussian exploration scale | 7/8 | 0 |
| Deterministic arms, zero navigation offsets | 7/8 | 0 |
| Deterministic navigation, zero arm residuals | 8/8 | 0 |

Removing navigation offsets did not recover contacts. Removing arm residuals
restored survival on this small sample, with final root separation 0.498-0.511 m,
but did not teach punching. This implicates learned arm/body coordination rather
than locked legs or a numerical crash; it does not isolate every failure's cause.
The diagnostic interventions never replace the training actor. The sampled arm
uses the learned bounded Gaussian scale, not the initial scale or a new optimizer.

## Bounded continuation decision

Allow one additional **524,288 learner transitions**, for a total boxing ceiling
of **1,048,576**. This tests learning duration while retaining the latest actor,
critic, Adam, normalization, RNG, 0.45 m engagement/reward profile and all consumed
work. It introduces no rollback, gate relaxation, assistance or diagnostic actor.
Nominal spawns remain 2 m apart, both roots and legs remain free, and the certified
motor actor remains frozen. Training still produces intermittent actual events;
only 262,144 transitions had been optimized under the corrected range. These
facts justify a finite duration experiment, not an assertion that time solves
the regression. Increased training falls and failed independent contact tests
remain explicit counterevidence.

At the next stop, evaluate the exact latest checkpoint using the unchanged eight
physical episodes and inspect event/fall histories. **If independent contacts or
survival do not recover, do not grant another unchanged duration extension.**
Run a bounded controlled reward/exploration comparison with separate diagnostic
compute instead. Sparse events, survival or shaped reward do not certify boxing.
No curriculum stage promotion is inferred from this pilot.

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking.ps1 -Resume -Ceiling 1048576
```

Ordinary resumes inherit saved profiles and preserve learning state. Simulation
state is not saved, so new episodes start on resume. Use the existing hidden WSL
host and check process/locks before launching. The 30-minute monitoring schedule
is unchanged.

## Evidence and accounting

[Decision and raw reports](results/striking-pilot/review-524288/continuation-decision.json)
include the final evaluation, behavior interventions, full parsed training
history, integrity checks, source configuration and stopped counters. The actual
evaluation video is `fights/striking-pilot/step-524288.mp4`; its first episode
contains no scored impacts. The immutable payload preserves complete learning
state and its source foundation hash.

This review simulated **15,056 diagnostic world controls**, including first-fall
truncations (3,565 ordinary evaluation, 11,491 intervention controls). Formal
striking-interface diagnostics now total **42,656 world controls**. Separately
recorded diagnostic PPO remains **768 learner transitions**; static reach samples,
earlier paired-transfer and foundation comparisons remain in their own ledgers.
No diagnostic compute is imported into main counters. Main consumed work is
475,516,928 combined transitions at this stop and at most 476,041,216 at the new
ceiling, including the unchanged 474,992,640 foundation transitions.

The independent evaluator completed all eight requested episodes, and the new
diagnostic tool completed all 24 interventions with finite observations and
first-fall truncation. The resumed hidden host was verified as Windows PID 25104
and Linux learner 351, with exactly one learner and no foundation job. At the
launch check, 528,384 consumed/optimized steps, Adam count 8,256 and a verified
finite payload demonstrated ordinary continuation past the stop. The 0.45 m
profile was inherited without an override. The launcher successfully archived
the stopped host logs before replacing their live names. No core training,
physics, scoring or promotion implementation changed in this review; validation
therefore focused on actual physical diagnostics, provenance and runtime checks.
See the [launch record](results/striking-pilot/review-524288/launched.json).

Subsequent [closed-fist replay restoration](closed-fist-rendering.md) changes only
the model used to render recorded coordinates. It adds 1,000 replay world controls
to the diagnostic ledger (43,656 total), without changing this training decision,
the active simulation, physical criteria or checkpoint lineage.
