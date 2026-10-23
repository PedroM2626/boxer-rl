# Bounded supervised initialization from physical demonstrations

**Current status:** both students and all 48 physical trials are complete.
Neither passes. All jobs are stopped; no PPO is resumed. The next authorized
comparison and its required migration tests are specified at the end.

## Verified evidence and decision

The [directed feasibility experiment](striking-directed-feasibility.md) produced
four reproducible diagnostic trajectories on its initial seeds. Its fixed
head-directed/bounded-advance controller then completed 64 held-out trials on
seeds 2,000,003–2,000,018, both controlled corners and both side arrangements.
**15/64 pairs survived ten seconds with a legal left-fist event**: nine red-role
and six blue-role trials. The other 49 fell. This is useful demonstration
coverage, not a robust controller or a learned boxing policy.

Every observation, action, physical event/time/impulse and complete survival in
all fifteen compressed trajectories was replayed exactly in the normal environment
without the geometric helper. [Full held-out trials, trajectories and replay](results/striking-pilot/directed-holdout-1441792/analysis.json)
retain failures as well as successes. The held-out trial cost was **15,802 world
controls**, plus **7,500** exact replay controls. Formal striking diagnostics now
total **346,091 world controls**; diagnostic PPO remains **1,188,096**, main boxing
**1,048,576**, foundation **474,992,640**, combined main **476,041,216**.

The next experiment initializes the existing residual actor by supervised action
imitation, then tests whether the **network alone** executes stable strikes.
No new PPO allocation, main adoption, motor update, assistance or gate change is
approved. The original four trajectories are excluded from this student's data;
only the fifteen fresh-seed complete trajectories are used. Those seeds now become
training data and cannot serve as this student's generalization benchmark.

## Preregistered learning protocol

Warm-start the exact latest 146-input/17-output actor at lineage 1,441,792, SHA
`87eb01ff6392b592395b64fc6f6a29f78e35fd1c1ad2941fafbcc8001146f5d8`.
Retain its normalizer, source critic and policy RNG unchanged. The certified
29-joint motor remains frozen. Verify every demonstration file against the exact
replay hash and its 500-by-146 observation / 500-by-17 bounded action arrays.
There are **7,500 training rows**; these are previously counted diagnostic
physical observations, not newly consumed PPO transitions.

Train actor means for at most **2,000 supervised updates**, batches of 256 drawn
with fixed NumPy seed 313: at most **512,000 example presentations**. Use fresh
actor-only Adam at 0.0001 with global gradient norm clip 1. The objective is MSE
between bounded `tanh(mean)` and recorded high-level actions, with channel weights
2 for three command offsets, 4 for seven demonstrated left-arm joints and 1 for
seven right-arm joints. A row receives weight 5 if any recorded action differs
from the unchanged source's prediction on that same observation by more than
0.05; otherwise weight 1. All weights are fixed before physical results. This
focuses on demonstrated corrections while retaining ordinary approach/recovery
rows. It does not change an environment reward or independent contact score.

The source Gaussian exploration parameter is unsupervised and unchanged. This
experiment uses **new supervised optimizer state**, distinct from the archived PPO
optimizer. Checkpoints explicitly declare `training_kind=supervised_demonstration`
and `ppo_optimizer_compatible=false`. PPO resumes reject such payloads until a
deliberate tested optimizer/value migration is implemented. Source PPO physical
consumed/optimized counters remain lineage metadata; supervised update counts
and example presentations are recorded separately, never added as PPO steps.

## Independent physical evaluations

Save immutable actors after **500 and 2,000 updates**. At each checkpoint, run
all eight deterministic trials for each neutral-partner learner role and the
original eight deterministic active self-play trials, seeds 7/23/31/47, both side
arrangements, ten seconds, original first-fall and impact criteria. This is
**48 full trials total**, at most **24,000 evaluation world controls**. No teacher,
recorded actions, Jacobian correction, advanced navigation override, sampled
perturbation or support is available in these separate evaluator processes.
The network supplies every high-level action; original learned motor physics
executes it. Close fists remain the existing visual-only rendering feature.

For a promising neutral phase, retain at least **7/8 red and 8/8 blue pair
survival**, matching the latest source, and own legal contacts in surviving
episodes on at least two distinct seeds **for each role**. Source active self-play
survival is 7/8; compare both corners' complete-episode events and seed coverage.
Two different complete consecutive policy checks are required before considering
phase advancement; no automatic promotion/adoption is implemented. The original
research screen has been inspected repeatedly, so any promising student also
needs genuinely fresh physical evaluation and motor regression checks before
claims of generalization or a new main training allocation.

The teacher stops directing after its first hit, whereas the actor's inputs do
not explicitly contain that teacher flag. Complete successful trajectories are
also a selected subset of an unstable controller's states. These are real limits
of imitation: low supervised loss cannot prove that a closed-loop student knows
when to extend/retract or remains balanced outside those trajectories. Review
all final physical results and failures before a further learning decision;
do not extend supervised iterations or resume PPO blindly if imitation fails.

## Operation and checks

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_demonstration_init.ps1
```

Inspect `/home/pedro/boxing-runs/striking-demonstration-init-1441792/initialization.json`,
the active checkpoint/evaluator, logs and process before another launch. The hidden
persistent WSL parent runs CPU learning and sequential independent evaluations;
it refuses to overwrite an existing experiment. Checkpoint folders preserve each
supervised optimizer and full results. Cumulative PPO and main counters remain
unchanged; add actual evaluation controls to 346,091. The monitoring schedule is
unchanged. No routine forced rollback or source substitution is performed.

Thirteen focused tests passed across supervised provenance/log-standard-deviation
gradients, existing PPO/resume contracts and read-only directed actions. The new
tests verify that supervised optimizer payloads cannot silently resume as PPO,
legacy PPO payloads remain accepted, and behavior imitation supplies no gradient
to the exploration parameter. Actual checkpoint and physical checks remain the
evidence for learned striking.

## Completed result

| Actor | Red-role survivors / own surviving events | Blue-role survivors / own surviving events | Active self-play survivors / surviving events |
| --- | ---: | ---: | ---: |
| Original source | 7/8 / 0 | 8/8 / 0 | 7/8 / 0 |
| Supervised update 500 | 6/8 / 1 | 7/8 / 1 | 6/8 / 0 |
| Supervised update 2,000 | 6/8 / 0 | 8/8 / 1 | 4/8 / 0 |

The 500-update network independently lands one surviving own event on seed 7 as
red and seed 31 as blue, without the geometric helper or recorded actions. These
are isolated learned strikes against a neutral partner. It still regresses
survival and scores on only one seed per role. At 2,000 updates red loses its
surviving event, blue scores on seed 23, and active self-play falls to 4/8. Neither
passes a phase or establishes two agents boxing. Do not adopt the favorable
500-update snapshot or extend blind imitation.

[Both students, all complete trials and integrity analysis](results/striking-pilot/demonstration-init-1441792/completed/analysis.json)
are archived. All learning leaves were finite; the source critic, normalizer,
policy key and log-standard-deviation were unchanged. Both supervised payloads
reject PPO resume. Evaluator error logs were empty. Actual cost was **19,993
evaluation world controls**, for **366,084 formal diagnostic controls** total.
Supervised compute was **2,000 updates / 512,000 example presentations**.
Diagnostic PPO remains **1,188,096**; main and foundation are unchanged.

## Next bounded comparison: observable contact history

The teacher changes behavior after its first valid left-fist event, but the legacy
actor's 146 inputs omit scored-hit history. A controlled engineering test holds
qpos/qvel/physical observations fixed and changes only scorecard history: actor
observations remain exactly equal while teacher reaching/recovery actions differ.
This proves missing phase information for representing that teacher, **not** that
it is the sole cause of these student failures. Both directed-helper tests pass;
this history intervention is not a physical trial and consumes no world controls.

The next reviewed work is a matched contact-feedback comparison, contingent on
implementation and migration tests before launch:

- Warm-start **both arms from the latest 2,000-update student**, SHA
  `fc48cecf46647ba70fb372a657e9b8a889d37ff7b4dfbdddeb2453872a42540f`.
  Do not roll back to update 500. Append two inputs indicating whether each own
  fist has recorded a valid event in this round. This is actual physical feedback
  for a first-strike/recovery curriculum, not a changed score or a complete
  repeated-boxing design.
- Expand 146 to 148 inputs identically for both arms: zero new actor/critic kernel
  rows and optimizer moments; normalizer means zero/variances one. Preserve old
  weights/moments, BC Adam count, normalizer prefix and policy key. Test source
  predictions, zero-feature physical agreement and full state preservation.
- Control trains with the two new columns zero; treatment uses actual hit flags
  reconstructed from verified event histories, visible only in the following
  control observation. Use the same fifteen trajectories, original fixed
  row/channel weights and optimizer. The control's new weights/moments must stay
  zero. Evaluation uses real physical flags for both; zero input weights make
  those flags irrelevant to the control. No geometric helper is deployed.
- Preserve BC sampling provenance. Its old snapshot omitted the separate NumPy
  sampler state: reconstruct seed 313 after exactly 2,000 batches of 256 uniform
  draws over 7,500 rows, verify the prefix and save sampler state explicitly in
  every new snapshot. Do not silently reset Adam or sampling on continuation.
- Maximum new allocation: **2,000 supervised updates per arm**, 4,000 total,
  zero PPO. Evaluate each final actor over all sixteen neutral-role and eight
  active-self-play trials: at most **24,000** further world controls. Implement
  and test the 148-input physical observation/evaluator before launching.

Unchanged phase criteria still require source-level survival, sufficient own
scoring seeds in surviving episodes for each role, and two distinct consecutive
complete passes. Two comparison arms are not consecutive passes of one policy.
Any favorable candidate also needs fresh physical seeds and motor regression
checks. No main adoption or further allocation is preapproved. This comparison
is the next authorized work, not an already running job or proof that contact
feedback solves boxing. Inspect completed state and processes first.

## Contact-history implementation launched

The reviewed [matched comparison](striking-contact-history-comparison.md) has
passed migration, feedback timing, unchanged physics and sampler tests and is
running. Inspect its own durable state and active evaluator before another job.
The earlier initialization remains completed; neither old checkpoint is adopted.

## Contact-history comparison completed

The [matched comparison](striking-contact-history-comparison.md) completed both
2,000-update continuations and all 48 physical trials. Neither passed or was
adopted; no further learning is allocated. The review archives all results and
defines the next bounded read-only coverage/first-fall investigation. Formal
world controls now total 385,343; diagnostic PPO and main counters are unchanged.
