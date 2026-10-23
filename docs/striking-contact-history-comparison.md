# Matched supervised contact-history comparison

This implements the bounded protocol in
[the completed initialization review](striking-demonstration-initialization.md).
Neither previous supervised checkpoint passed the physical phase. Main boxing
remains stopped at 1,048,576 learner transitions; foundation remains certified
at 474,992,640. No PPO or new main allocation is granted here.

## Controlled change

Both arms start from supervised update 2,000, SHA
`fc48cecf46647ba70fb372a657e9b8a889d37ff7b4dfbdddeb2453872a42540f`.
The actor and critic gain two zero input rows, expanding 146 to 148 inputs while
retaining seventeen outputs. New normalizer means are zero and variances one.
Existing weights, Adam moments/count 2,000, normalizer prefix/count, policy key,
critic and all consumed/optimized/world lineage counters are retained. This is
an explicit supervised migration, not a PPO-compatible optimizer.

The extra physical observations report whether each own fist has recorded a
valid event this round. They come directly from finite scorecard timestamps and
reset with the scorecard. No force, joint target, contact pair, scoring condition,
motor action, root, assistance setting or termination changes. Both free bodies
still spawn apart. This history represents the demonstrated first-strike/recovery
phase; it is not a complete repeated-punch controller.

The `zero_history` control trains with the new input columns zero. The
`contact_history` treatment uses event flags from the fifteen exactly verified
500-control trajectories. A flag appears at the first observation after the
event's control finishes, never at an earlier substep timestamp. Evaluations
provide real physical flags to both arms; the control's appended actor weights
and Adam moments are checked to remain exactly zero.

Both arms retain the original 7,500-row dataset, channel weights, and fixed row
weights computed from original PPO SHA
`87eb01ff6392b592395b64fc6f6a29f78e35fd1c1ad2941fafbcc8001146f5d8`.
The original jitted inference path computes those weights. Learning uses the
same batch size 256, Adam 0.0001 and global gradient clipping at one.

The old BC snapshot omitted its separate NumPy sampler. Recovery replays seed
313 through exactly 2,000 uniform batches over 7,500 rows, checked against the
source supervised and Adam counts. Both arms clone that recovered sampler and
must finish with identical sampler states. The expanded source and every new
snapshot store the sampler explicitly. Future snapshots in the original BC tool
also save it; the completed experiment is not rerun.

## Allocation and physical review

Maximum new compute is **2,000 supervised updates per arm**, 4,000 new updates /
1,024,000 example presentations total. Each final actor has a 4,000-update BC
lineage; their shared 2,000-update prefix is counted once in cumulative compute.
There are zero new PPO transitions. Both final actors receive all sixteen neutral
role trials and eight active self-play trials on seeds 7/23/31/47 with both swaps,
ten seconds per trial: 48 complete episodes, at most 24,000 world controls.

Source-level survival remains red 7/8, blue 8/8, active self-play at least 7/8;
surviving own scoring must cover at least two seeds in each role. No criterion is
lowered to match the weaker latest BC checkpoint. Distinct comparison arms are
not consecutive certification passes. A promising candidate needs fresh seeds
and motor regressions before adoption. No adoption or further allocation occurs
automatically, and this comparison alone cannot establish robust boxing.

## Operation and validation

The hidden persistent WSL launcher is
`tools/start_striking_contact_compare.ps1`. It starts the sequential CPU parent
`tools/striking_contact_compare.py` in
`/home/pedro/boxing-runs/striking-contact-compare-bc2000`. Inspect `comparison.json`,
its `diagnostic_output`, child process, per-arm reports and error logs before
another launch. Host PID/start-time checks, a parent file lock and an existing
state refusal prevent duplicate/overwritten jobs. Each evaluator records attempted
controls durably before stepping; after failure, reconcile its progress file
with completed reports rather than ignoring partial physical work or blindly
restarting the experiment. The existing monitor schedule is unchanged.

Before launch, eighteen focused tests passed across actual BC migration,
contact timing/reset, unchanged base physics, zero-control moments, sampler
recovery, PPO resume guards and directed-helper history aliasing. Six focused
contact/noise tests passed after final evaluator/provenance changes. Native
unit trajectories are engineering checks, not phase evidence. All four complete
foundation gate histories match their archives (29/41/13/2 records), still pass
the two-distinct-checkpoint rule, and their current hashes verify. The main
striking hash is unchanged. Prior completed BC evaluator logs contain no native
nonfinite/capacity warnings. Its 19,993 world controls remain included in the
**366,084** formal diagnostic controls before this job; diagnostic PPO remains
**1,188,096**. Main project total remains **476,041,216**.

## Completed result and continuation decision

Both arms completed all learning and 48 full physical episodes. Independent
[integrity, phase-fit and physical analysis](results/striking-pilot/contact-history-bc2000/analysis.json)
agrees with the parent summaries. The full reports/progress/error logs and both
final models are archived; no native numerical or capacity warnings appeared.

| Final actor | Red survivors / own surviving events | Blue survivors / own surviving events | Active self-play survivors / surviving events |
| --- | ---: | ---: | ---: |
| Zero-history control | 6/8 / 1 | 7/8 / 0 | 3/8 / 2 |
| Actual contact history | 6/8 / 1 | 6/8 / 0 | 5/8 / 1 |

The control's two self-play events occur in one surviving two-way episode on seed
23. Treatment has only one surviving self-play event on seed 7. This small sample
does not establish an advantage from feedback. Neither arm passes the complete
screen: both miss role survival and scoring coverage, and self-play remains below
7/8. Do not promote either arm, substitute the earlier 500-update actor, or extend
blind imitation. Preserve both final branches without adoption. There is no
productive approved learning job to keep running after this allocation ends.

The treatment really uses the feature: on recorded post-hit demonstration states,
zeroing only history changes mean absolute actions by about 0.022 in both roles.
The control has exactly zero history sensitivity. However, pre-hit left-arm mean
absolute action error is about 0.049/0.052 by role in both arms, versus about 0.017
after impact. Pre-hit maximum action errors still reach 0.83/0.99. Only 1,526 of
7,500 recorded rows precede the first own impact. These are in-dataset action
fits, not physical evidence or an isolated causal explanation. Observable history
addresses the demonstrated aliasing but has not solved reaching and survival.

Actual additional cost is **19,259 evaluation world controls**, giving **385,343**
formal diagnostic controls. Diagnostic PPO remains **1,188,096**. New supervised
compute is **4,000 updates / 1,024,000 presentations**; cumulative compute is
**6,000 updates / 1,536,000 presentations**, counting the shared prefix once.
All final Adam counts are 4,000 and sampler states match. Critic, normalization,
policy key, exploration parameter and inherited physical/PPO counters are exact.
The main total remains 476,041,216 and every foundation gate is unchanged.

The comparison is completed; inspect
`/home/pedro/boxing-runs/striking-contact-compare-bc2000/comparison.json` and
processes before further work. No comparison/trainer is expected to remain live.
The existing monitor schedule remains active.

The next useful investigation is policy-state coverage and first-fall traces,
rather than more supervised updates or PPO. Prepare a separate read-only audit of
the latest contact-history branch, preserving its 148/17 interface and real event
history. Record pre-action observations, actions, first failure/contact timing,
and per-role normalized distance to the verified demonstrations. Compare motor,
opponent and history input blocks separately so one block cannot hide a mismatch.
Use the existing sixteen neutral-role plus eight active self-play episodes, at
most **12,000 world controls**, zero learning and no action/helper intervention.
This diagnostic repeats a checkpoint for traces and cannot count as another
certification pass. Its bounded cost is justified by the failed complete gates
and the absence of policy-state traces in the existing reports; record every
attempted control and review coverage/failure evidence before any learning
allocation. Implementation and instrumentation checks must precede that launch;
this audit has not yet been started. No extra supervised/PPO allocation is granted.
