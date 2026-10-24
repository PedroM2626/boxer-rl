# Mixed continuation regression and policy-update investigation

**Current outcome:** yaw scaling is rejected after the complete comparison:
active survival regresses in both samples. See the [full result and next bounded
fixed-opponent contract](striking-fixed-opponent-comparison.md). No modified
yaw controller is adopted and the original physical profile remains in use.

**Current decision:** two original complete screens now pass consecutively,
but fresh 20M neutral-0 still fails. The [larger independent reset/opponent
audit](striking-generalization-audit.md) measures remaining generalization
with zero optimization. No main adoption or phase promotion.

## Complete physical outcome

The [ordinary mixed continuation](striking-mixed-partner-comparison.md#completed-comparison-and-one-bounded-continuation-2026-10-23)
completed its 65,536-credit cap normally at **20:41:09 UTC**. Neither distinct
complete generation passes. Intermediate neutral-role profiles each meet their
unchanged criteria, but active pair survival is 6/8. The final regresses in both
neutral roles and on exactly the same fresh 14M seeds as the source.

| Policy / sample | Neutral 0 survival / own events | Neutral 1 survival / own events | Active survival | Surviving active red / blue events | Surviving active two-way episodes |
| --- | --- | --- | --- | --- | --- |
| Source / fresh 14M | 8/8 / 1 | 7/8 / 0 | 6/8 | 2 / 0 | 0 |
| +32,768 / original screen | 7/8 / 2 | 8/8 / 2 | 6/8 | 1 / 1 | 0 |
| +65,536 / original screen | 6/8 / 2 | 4/8 / 1 | 6/8 | 2 / 3 | 1 |
| +65,536 / fresh 14M | 5/8 / 1 | 5/8 / 1 | 5/8 | 0 / 1 | 0 |

The original first-gate vector improves neutral scoring-seed breadth, so the
predeclared stop policy correctly permitted the remaining fixed allocation.
That was not a phase pass or proof of general progress. All **96 complete
episodes / 42,608 evaluation controls**, two exact gate models, 64 finite PPO
blocks and all prefixes are independently verified and archived under
`mixed-partner-continue-1769472` by `tools/review_striking_mixed_continuation.py`.
Foundation complete histories remain 29/41/13/2 and their latest payloads are
finite with correct hashes; the main boxing checkpoint is unchanged. Native
error logs are empty. There is no observed nonfinite/capacity incident to discard.

Intermediate SHA:
`5b8704fa38ab28eaebab1933ec4eba71f29c6acdf7eb70fee73dd0e5de968fb1`.
Latest final SHA:
`8c99c2dfe21cd6c94a90ed31c487620118df5b0d0fdfe430eabe2471745a31c2`.
The final lineage is **1,835,008 consumed/optimized learner credits / 1,212,416
world controls**, actor/critic Adam **11,144 / 34,816**. Mixed mode/corner ledger
is **[[24,338, 24,814], [25,140, 24,012]]**, historical neutral prefix 50,229/48,075
unchanged. Do not substitute the more favorable intermediate actor for this final.

Current diagnostic PPO is **1,810,688**, formal diagnostic simulation **989,355
world controls**. This job added 65,536 collection controls and 262,144 integrated
replay presentations separately. Standalone BC stays 8,000 updates; main boxing
**1,048,576**, combined main **476,041,216**, all four certifications and foundation
ceilings remain unchanged. All previous allocations are exhausted evidence.

## Zero-simulation archived-state audit

`tools/audit_striking_policy_drift.py` checks every saved source/1,024-step block
through the final: **65 finite, provenance-valid checkpoint payloads**. It compares
exact diagonal-Gaussian policy KL on a fixed hashed corpus: 1,024 deterministic
demonstration samples, 1,024 motor anchors and 1,024 archived active/neutral
contact states. Tanh is bijective, so raw/squashed joint Gaussian KL is identical
for fixed inputs. Command and arm contributions are reported separately.
No physics, optimizer, standalone supervised or replay work is performed.

Holding the source normalizer fixed, source-to-final mean KL on archived contact
states is **1.7935 nats**; holding the source actor fixed and changing only the
normalizer gives **0.01705 nats**. Arm changes dominate this fixed-corpus parameter
shift. Mean bounded-action RMS changes are 0.05739 (parameters only) versus
0.00528 (normalizer only). Exploration log standard deviations barely change.
Demonstration MSE actually improves **0.004676 to 0.004423**, and anchor MSE
**0.011114 to 0.010787**, despite worse physical results. This rejects treating
offline imitation error as a boxing gate or sufficient stability signal.

Largest consecutive-block mean KL is 0.05929 on demonstration states, 0.09421
on anchors and 0.04409 on archived contact states. These states are off-policy
probes from earlier controllers. The exact KL is descriptive on those inputs;
it is **not the actual rollout/minibatch KL**, an on-policy error estimate or a
causal explanation of falls. Actor/normalizer decomposition does not replay a
new trajectory. No optimizer reset or normalizer reset is justified by this audit.
`policy-drift.json` retains corpus selection/hashes, all per-block hashes and
measurements, endpoint decomposition and limitations.

## One bounded on-policy update-geometry probe

The larger parameter drift and repeated physical regression justify measuring
actual update geometry before choosing another training profile. Use the latest
final above, ordinary version-6 warm state and unchanged objective, optimizer,
learning rate, eight-world 50/50 task, full learned actions, physical gates,
fists, independent free roots/legs and separated spawns. Do not repeat a source
transition, use a teacher, alter rewards or roll back to the intermediate.

Allocate **exactly one 1,024-learner-credit / 1,024-world-control rollout**, followed
by its existing 16 minibatch updates. No additional evaluation rounds. The optional
`--update-audit-directory` observer is restricted to this one-block mixed path.
Measure exact Gaussian KL, sampled approximate KL and ratio clip fraction on
the same collected 1,024 learner observations before updates, after every update
and after normalizer refresh, separately for 512 neutral/512 active rows. The
observer makes no gradient, optimizer, action, objective, RNG or trust-region
stopping change. A bitwise update-equivalence test checks the observed versus
unobserved parameters, moments and loss. No full training continuation is granted.

Maximum new diagnostic PPO is **1,024 credits**, new collection **1,024 controls**,
integrated replay **4,096 presentations** (3,072 demonstration/1,024 anchor).
At full use diagnostic PPO reaches **1,811,712**, while formal diagnostic
evaluation stays **989,355**. Main and standalone-supervised counters do not move.
The diagnostic child lineage reaches **1,836,032 / 1,213,440 world controls**,
Adam **11,160 / 34,832**. Retain all attempted/discarded work if it fails.
This short child is not a validated boxer or an adopted replacement for the
latest full final; it only supplies update measurements. Review its full metrics
and limitations before choosing a correction or granting any further allocation.

Hidden persistent launcher: `tools/start_striking_update_geometry_probe.ps1`.
Live state:
`/home/pedro/boxing-runs/striking-update-geometry-1835008/probe.json`.
Inspect state, process, child progress, native logs and measurements before any
other launch. The native/global and supervisor locks remain separate. Preserve
the existing monitoring schedule. Twenty-five focused checks pass, covering
the passive observer, Gaussian KL, existing PPO/legacy behavior and mixed native
routing/provenance; Python, shell and PowerShell syntax checks are required.

## Completed actual-rollout measurement

The native probe completed normally: all 18 measurements and the payload are
finite, and native error logs are empty. After 16 updates, exact rollout Gaussian
KL is **0.0379053 nats**, p95 **0.0700719**, sampled approximate KL **0.0625229**,
and **52.34375%** of sampled action ratios are outside the clipping interval.
Neutral/active clip fractions are 50.1953% / 54.4922%. Normalizer refresh changes
mean KL only from 0.037901 to 0.037905. Clipped objectives do not enforce a strict
policy-movement bound; this measurement does not prove the cause of falls.

[Archived probe](results/striking-pilot/update-geometry-1835008/README.md) retains
the exact diagnostic child, all measurements and consumed prefixes. Diagnostic
PPO is now **1,811,712**, formal evaluation remains **989,355**. The child is not
adopted. Subsequent matched branches start from the latest **fully physically
evaluated** final `8c99c2dfe21cd6c94a90ed31c487620118df5b0d0fdfe430eabe2471745a31c2`;
the extra diagnostic block stays in global accounting.

## Bounded matched actor-rate comparison

Compare actor Adam learning rate **0.00001** against unchanged **0.0001** from
that same final. Critic learning rate stays 0.0001; retain all actor/critic Adam
moments and counts, normalizer, policy, RNG seed, joint gradient clipping,
retention objective, gamma 0.999, balanced mixed task and physical gates.
Version 7 adds explicit actor-rate/transition provenance; version 6 control
behavior stays unchanged. Critic trajectories can later differ because policy
changes affect collected data; only its learning rate is held fixed.

Allocate **16,384 learner/world controls per arm**, at most **32,768** total.
The first 1,024 controls in each allocation carry the passive observer and a
finite-state/prefix guard. The reduced-rate arm must reduce both measured KL
and clip fraction relative to the completed probe before continuing. Evaluate
all three unchanged complete profiles at +8,192 and +16,384, using original
four seeds and both swaps. Stop an arm early if any profile has at most 3/8
surviving pairs, retaining all attempted compute. Evaluate source and both finals
on the same fresh seeds 15,000,003 through 15,000,006 with both swaps. Maximum
**168 evaluation episodes / 84,000 world controls**. No adoption, main extension,
phase promotion or further automatic allocation follows this comparison.
Two distinct consecutive complete physical passes remain necessary.

At full use, each branch reaches lineage **1,851,392 learner / 1,228,800 world**,
Adam **11,400 / 35,072**, balanced mixed-mode totals **57,344 / 57,344**, with
historical neutral ledger **50,229 / 48,075** unchanged. Global diagnostic PPO
ceiling becomes **1,844,480** and formal evaluation ceiling **1,073,355**.
New collection is separately capped at 32,768 controls and integrated replay
at 131,072 presentations; standalone BC stays 8,000 updates. Main counters,
foundation certifications and ceilings remain unchanged.

Launcher: `tools/start_striking_actor_rate_compare.ps1`; live state:
`/home/pedro/boxing-runs/striking-actor-rate-1835008/comparison.json`.
Inspect that process and diagnostic output before launching any other job.
Thirty-two focused tests pass, including actor-only step scaling with identical
existing moments, unchanged default/critic updates, passive observer equivalence,
profile/provenance rejection and mixed native routing. Preserve monitoring.

## Native reduced-rate first block verified

The version-7 branch completed its included 1,024-control guard with finite
parameters, normalizer and moments and an empty native error log. All old source
leaves in the initial transition are exactly equal. Its payload SHA is
`b0f0155901f006458b3b5336826c2b06164531179bd548f8d5b82fab1deeadb6`;
lineage 1,836,032 / 1,213,440 world controls and Adam 11,160 / 34,832 match the
retained prefixes. Mixed mode totals are 49,664 each; historical neutral counts
are unchanged. The initial state, exact first payload, measurements and proof
are archived under `actor-rate-1835008` in result/model validation folders.

Actual-rollout mean Gaussian KL is **0.00291492** (previous-rate probe 0.0379053),
p95 **0.00481091**, sampled approximate KL **0.00429074**, ratio clip fraction
**3.125%** (previous 52.34375%). The predeclared integration guard passes and the
bounded comparison continues. This demonstrates smaller initial policy movement,
not improved boxing, survival, a complete physical pass or promotion.

Source fresh 15M complete reports are also archived: neutral survival 5/8 and
4/8, active 7/8 with two two-way surviving episodes; the complete gate fails.
They consumed **10,395** evaluation controls, bringing formal diagnostic compute
to **999,750** before arm screens. These source results cannot predict either
final. Live progress can advance beyond the archived launch snapshot; account
all attempted work from both child progress files before deciding continuation.

## Completed actor-rate comparison and confirmation decision

All three full screen generations and three fresh sets are verified: **144
complete episodes / 62,822 evaluation controls**, 24 finite PPO blocks, exact
checkpoint hashes and optimizer/consumed/mode prefixes. Native error logs are
empty. Complete foundation histories remain 29/41/13/2 with two distinct final
passes each and finite correct payloads; the stopped main boxing hash is intact.
The control's first-block measurements exactly reproduce the prior rate probe.

| Policy / same fresh 15M sample | Neutral 0 survivors | Neutral 1 survivors | Active survivors | Surviving active red / blue events | Active two-way episodes |
| --- | --- | --- | --- | --- | --- |
| Source | 5/8 | 4/8 | 7/8 | 2 / 2 | 2 |
| Reduced rate / +8,192 | 5/8 | 3/8 | 5/8 | 2 / 3 | 2 |
| Original rate / +16,384 | 6/8 | 7/8 | 6/8 | 8 / 9 | 4 |

The reduced-rate branch correctly stopped at its first screen (neutral 0 only
3/8 survivors). Lower initial KL is not a physical recovery. The original-rate
control ends with 19/24 fresh survivors versus 16/24 source and 17 surviving
active events across all four seeds versus four source events across two seeds.
However, active survival regresses 7/8 to 6/8 and neutral-0 scoring breadth
regresses; no complete generation or fresh set passes. This is mixed evidence,
not reliable boxing or a reason for an unrestricted extension.

Archive and independent recheck: `tools/review_striking_actor_rate.py` and
`docs/results/striking-pilot/actor-rate-1835008/analysis.json`. Global diagnostic
PPO is **1,836,288**, formal evaluation **1,052,177**. Added collection 24,576
and integrated replay 98,304 presentations remain separate; BC 8,000 updates,
main boxing 1,048,576 and combined main 476,041,216 are unchanged. Unused prior
allocation is not consumed compute and does not silently fund later jobs.

The broader surviving active scoring and paired overall survival justify one
**16,384-credit confirmation** from the latest complete control final SHA
`c6f7fbf4c5ef941c43b8a71be7f6f6680dd657d38e96a7ed112a0f2f8f79c04b`.
This is an ordinary version-6 resume with actor/critic rate 0.0001, gamma 0.999,
retention 1, balanced mixed worlds, inherited engagement profile and every
normalizer/Adam/RNG/sampler prefix retained. No reduced-rate state, rollback,
teacher or change of body authority is applied.

Full unchanged screens at **+8,192 / +16,384**; stop if any profile has at most
3/8 survivors, or at the first screen if no measured survival/scoring-breadth/
two-way component improves over the source original screen. These measurements
control early stopping only; all complete criteria remain necessary. Source
and final receive identical fresh **16,000,003 through 16,000,006**, both swaps.
Maximum **96 episodes / 48,000 evaluation controls**; no automatic subsequent
allocation. Review all complete trials, including falls, before deciding again.

New global diagnostic ceilings: **1,852,672 PPO / 1,100,177 evaluation controls**.
New collection is at most 16,384 and replay 65,536 presentations. At full use
lineage becomes **1,867,776 learner / 1,245,184 world**, Adam **11,656 / 35,328**,
mixed-mode totals **65,536 / 65,536**, historical neutral prefix unchanged.
Launcher: `tools/start_striking_mixed_confirm_continue.ps1`; live state:
`/home/pedro/boxing-runs/striking-mixed-confirm-1851392/continuation.json`.
Preserve the existing monitoring schedule and inspect this job before any launch.

## Confirmation stopped and passive failure investigation

The 16,384-credit confirmation stopped correctly at **8,192 consumed/optimized
learner/world controls**. Its first complete screen vector changed from
(7,3,7,1,6,3,2) to (7,2,7,1,6,3,1): no early-stop component improves. The failed
complete set is retained, not replaced by fresh results. All **72 complete
episodes / 32,798 evaluation controls**, eight finite PPO blocks, gate/final
payload, empty native error logs and complete foundation/main prefixes are
independently verified by `tools/review_striking_mixed_confirmation.py`.

| Same fresh 16M sample | Neutral 0 survivors / own events | Neutral 1 survivors / own events | Active survivors / red-blue events | Active two-way episodes |
| --- | --- | --- | --- | --- |
| Source | 8/8 / 9 | 6/8 / 2 | 5/8 / 7-11 | 3 |
| +8,192 final | 7/8 / 5 | 5/8 / 1 | 6/8 / 9-8 | 2 |

Survival totals change 19/24 to 18/24; neutral scoring breadth also regresses.
The latest final SHA is
`6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a`,
lineage **1,859,584 learner / 1,236,992 world**, Adam **11,528 / 35,200**,
mixed-mode totals **61,440 / 61,440**, historical neutral 50,229/48,075 intact.
Global diagnostic PPO is **1,844,480**, formal evaluation **1,084,975**;
new collection 8,192 and replay 32,768 presentations are separate. Unused
allocation is not consumed. Main counters and all foundation certifications
stay unchanged. No result establishes that another unchanged extension helps.

Allocate a passive matched replay of the **source and final, all three profiles,
four existing 16M seeds, both swaps**: **48 complete episodes / at most 24,000
world controls**, no training, new optimization, supervised work or adoption.
Every pre-existing trial field must match the original untraced physical result
exactly, and every trace must be finite with a checked SHA. Stop and investigate
if instrumentation changes outcomes. Record qpos/qvel, actions, commands,
residuals, first contact, first fall, corner terminal conditions, actual scoring
times and reward components. Compare contact-before-fall, precontact posture,
and late commands; these are temporal associations, not causal proof. Do not
repeat the previously rejected command mask or body-authority change based only
on correlations. The data will determine the next bounded intervention.

PPO remains **1,844,480**. Formal diagnostic ceiling becomes **1,108,975**
(1,084,975 + 24,000), all attempted simulation retained on failure. There is no
PPO allocation while this diagnostic is active. Hidden launcher:
`tools/start_striking_mixed_failure_trace.ps1`; live output:
`/home/pedro/boxing-runs/striking-mixed-failure-trace-1859584/trace-review.json`.
Inspect this process and its diagnostic output before launching anything else;
preserve the monitoring schedule and all unassisted physical/gate settings.

The passive audit's first full eight-episode profile is verified identical to
its original physical results with finite hash-checked traces (4,000 controls).
Eleven focused checks pass, including a native passive-equivalence test. A
missing source-reference archive initially stopped analysis after that completed
replay; the reviewer now copies source reports. Explicit diagnostic recovery
retains the failed state and all 4,000 consumed controls, reuses the completed
report without rerunning physics and rejects partially attempted reports.
The resumed job continues remaining profiles under the original total cap.
See the trace result README and `failure-before-resume.json` for the evidence.
