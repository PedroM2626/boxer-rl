# GPU recovery and physical validation

Reviewed 2026-10-29. The project has learned valid individual punches and a few
reciprocal examples, but no complete recent punch-quality/stability gate. V17
ended at its allocation; no unchanged CPU PPO extension is justified. Historical
contact-only certifications do not certify complete punches.

The current physics now runs on CUDA with unchanged free roots, actuators,
contacts and 2 ms integration. `paired_warp_physics.py` is an experimental
primitive; `paired_warp_features.py` batches the existing 103 motor / 148 boxing
inputs. Four asymmetric history/frame/profile tests match the native inputs.
The reference adapter copies GPU output to native measurement structures without
stepping native physics; an independent two-impact export check matches direct
GPU forces to native `mj_contactForce` readings. Default production physics and
native gates remain unchanged. Experimental backend/solver reports cannot promote.

## Evidence so far

- The original 3 solver / 5 line-search limits produce solver flags and noticeable
  CPU/GPU trajectory differences. Raising solver iterations alone is insufficient.
- At 30 / 20 on **both** engines, the two saved impact windows agree to about
  6e-6 in qpos and 6.2e-4 in qvel over 0.5 s, without solver flags. The separated
  random reset still diverges, chiefly an ankle, after its first few steps:
  maximum joint difference about 0.039 rad. This is not proven closed-loop parity.
- A 45-control prototype with the actual final v17 actor and certified frozen
  motor measured about 7,260 world controls/s at 128 worlds and 14,368 at 512.
  This includes CPU inference, GPU physics, observations, transfers, every-substep
  overflow inspection and durable accounting. It excludes PPO, actual cycle
  rewards/history, reset-on-fall and certification. Do not advertise these as
  achieved training rates. All original solver flags remain recorded; repetitive
  solver **printing only** is disabled in timing, capacity warnings remain enabled.
- A first dynamics attempt stopped because Windows held a repository JSON file
  during atomic replacement. Preserve its 80 attempted CPU/GPU substeps. Durable
  live diagnostics now write to the WSL filesystem and archive completed results.

Full raw data, source hashes and counters are archived under
`docs/results/striking-pilot/gpu-*`. No learner updates were performed.

## Prospective closed-loop comparison

Run actual final v17 `f311640d...` with its inherited 0.45 m engagement profile,
unmodified motor/actions and original punch criteria. Reuse the exact historical
native 3/5 fresh gate (34002003–34002006, both corner placements), with zero
new debit. Compare three new arms, each with neutral-0, neutral-1 and mirror:

1. Native 30 solver / 20 line-search iterations.
2. CUDA 3 / 5, retaining the original model settings.
3. CUDA 30 / 20, the converged-impact counterfactual.

Maximum **36,000 new physical evaluation world controls** (72 episodes of at
most 500 controls). Formal before **2,271,001**, ceiling **2,307,001**. Debit
attempts before stepping, preserve failures and unused allowance expires. No PPO,
optimizer change, rollback, adoption or automatic extension. Source hashes,
finite state, force capacities, event windows, both actors' survival and complete
cycle conjunctions must be reviewed. A changed solver is an explicit diagnostic
physics intervention, not permission to redefine past certifications.

Live state: `/home/pedro/boxing-runs/gpu-closed-loop-v1/state.json`. Inspect it and
the process before restarting. Use the hidden persistent WSL launcher; no duplicate
diagnostic jobs. Archive complete profiles and summarize all arms, including
failures. Separate GPU physical substeps and warm-ups from world control credits.

The hidden follow-through coordinator is active at
`/home/pedro/boxing-runs/gpu-validation-followthrough/state.json`. It waits without
running physics, then rechecks complete raw traces/hashes for all nine profiles,
archives/publishes the comparison and runs the predeclared legacy diagnostic.
It stops on a failed/stopped comparison rather than duplicating it. Publication
uses only owned result paths and defers if unrelated staged changes exist.
The existing three-hour monitoring schedule is unchanged. Inspect both live states.

To reduce reference-export overhead, `warp_readback.py` stages the upstream
exporter's exact read set in pinned host memory and calls the same exporter after
one synchronization. Two saved impact poses match direct export exactly for
state, sensors, frames, velocities, constraints and contacts. This is a measurement
implementation change only; it leaves dynamics/scoring untouched. Earlier profiles
may use direct export; the report records the readback implementation.

At the interim review, native 30/20 yields pair survival 6/8, 4/8, 3/8; no full
profile set passes. Original native 3/5 was 5/8, 7/8, 3/8. CUDA 3/5 neutral-0
has 5/8 survivors and 17 valid cycles, matching those **aggregate** original
native figures; neutral-1 has 4/8 survivors and 9 valid cycles. These incomplete
comparisons do not establish equivalence, causation, or an improved controller.

After six complete profiles (16,267 new controls; formal 2,287,268), a Windows/WSL
archive overwrite failed while Git was reading an already archived trace. The live
source and all six profile hashes were independently rechecked; no physical child
was interrupted and no profile was repeated. Archiving now skips identical files,
uses atomic replacement with bounded transient-access retries, and cannot stop
productive physics on a per-profile publication failure. The explicit reviewed
resume retains all counters/profiles and starts only CUDA 30/20. The waiting
coordinator was restarted only after that verification finished. Failure/recovery
history remains in live state and `archive-interruption-stderr.txt`.

## Decisions after the comparison

If the reference adapter cannot reproduce adequate physical behavior, inspect
the first divergent contact/reset rather than spending PPO on an unvalidated
backend. If usable, port the verified scorecard, complete-cycle history, rewards
and full randomized resets to the batched path, compare their outputs, and then
benchmark PPO. Native unchanged gates remain authoritative during GPU training.
Also measure retained Warp6/7 policies in their own engine under shared fist-only
travel/impact/return criteria; old force/damage counters are not a matched result.
No incompatible checkpoint transplantation or claim that the new architecture
is inherently superior follows from the present data.

The separate legacy protocol (`tools/evaluate_legacy_punch_cycles.py`) can start
only after the above comparison completes and releases the global locks. It uses
the two verified 200,212,480-step checkpoints, seeds 7/23/31/47, original 0.75 m
and separately identified 2 m spawns: at most 16 episodes / **7,680 new controls**.
Its formal before counter is the completed comparison's actual total; unused
comparison allocation expires rather than rolling over. Retain a separate
legacy counter, every GPU substep, overflow flags, complete event traces and
checkpoint/config hashes. No PPO or historical executable equivalence claim.
Historical source inspection found another compatibility trap: Warp6's 128 inputs
predate damage/stun (introduced in Git `b123d8e` with 131 inputs). Applying current
defaults would unfairly add knockouts and stun to that old policy's evaluation.
The new diagnostic explicitly restores no stun and an unreachable damage cap for
Warp6, while preserving Warp7's recorded cap 3 / stun 10. A focused contract test
checks both cases. This is historical reconstruction, not a gate relaxation.
The old 4 ms step must be supplied explicitly when measuring contact duration;
using the new engine's 2 ms would undercount it by half. Warp7 damage
termination and different self-leg collision geometry limit comparisons of pair
survival. Do not issue current full-gate certification from these results.
The legacy worker independently rechecks saved trace hashes, finite arrays,
all sixteen episode counters and complete-cycle conjunctions before returning.
It writes `completed-review.json` and a compact English README separating valid
cycles before an early stop from reciprocal rounds actually completed without
falling. The coordinator archives and publishes those results together.

Completion still requires stable repeated strikes in both corners, reciprocal
exchanges against active/varied opponents, fresh-condition generalization and
longer rounds, with reproducible videos and independent physical evidence.
