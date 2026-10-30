# Functional completion estimate and remaining evidence

**Update 2026-10-30: about 60 %.** The [sparring motor](spar-motor.md) closes
the stability gap that held this estimate at 40 %. Two unassisted robots now
exchange punches for 20 s in 8/8 bouts (18.6 clean punches per bout, zero
falls). Still outstanding: learned tactics (punch selection is scripted),
defense and blocking, punch types beyond straight punches, clinch breaking,
and longer or multi-knockdown rounds. The 2026-10-29 text below is the
historical record.

| Capability (2026-10-30) | Evidence/status |
| --- | --- |
| Balance while punching and being shoved | `spar` motor: 100 % solo survival, full gate pass |
| Two-sided unassisted exchanges | 8/8 bouts × 20 s, both corners score and win |
| Learned tactics | Outstanding: scripted `RingBrain` |
| Defense, hooks/uppercuts, clinch breaking | Outstanding |

Reviewed 2026-10-29. **Approximately 40%** is a subjective estimate of demonstrated
functional scope, not a measured learning-convergence percentage, fraction of
remaining work hours, or promise that the rest is routine implementation.
The core learned boxing behavior is still unresolved despite substantial
infrastructure and certified locomotion. Do not label the project near-complete
because code, reports and checkpoints exist.

Latest reviewed outcome: the [first scaled GPU block](results/striking-pilot/gpu-learning-v1/README.md)
completed with all six complete native sets failing. Final fresh 27 neutral valid
cycles and mirror 5/1 cycles establish retained isolated striking, while 4/6/3
pair survival and no valid reciprocal surviving seed leave the core stability/
two-sided boxing gap open. Scope remains approximately 40%; training compute is
not functional completion. A separately bounded matched native/CUDA follow-up
and crash-safe accounting optimization precede any new learning allocation.

Latest engineering continuation: scaled GPU integration now exactly replays all
256 updates; a separate [bounded GPU learning job](gpu-bounded-learning.md) uses
its actual final, with a new native baseline and repeated unchanged physical
gates. Collection+PPO measured about 870 credits/s in one block, excluding startup
and evaluations. Live totals/process state supersede the historical idle/budget
notes below. No new functional boxing certification follows from this launch.

Current evidence supersedes the historical notes below: final v17 has 33 fresh
neutral valid cycles, 5 fresh mirror cycles and reciprocal strikes in one
surviving fresh episode, but pair survival is only 5/8, 7/8, 3/8. No complete
recent gate or reliable reciprocal boxing. The bounded
[GPU recovery comparison](gpu-training-recovery.md), batched mechanics and
[one-block GPU PPO integration](results/striking-pilot/gpu-batched-integration-v1/README.md)
are complete. The finite integration checkpoint preserves all learned state and
consumes exactly 1,024 credits; it establishes implementation correctness rather
than improved boxing. Complete collection measured 601.5 world controls/s at 128
worlds, excluding PPO; eight worlds still show no throughput benefit. No PPO job
is active after the mandatory integration review. Collector optimization,
validated larger-batch learning, stable two-sided exchanges and broader/longer
fresh validation remain outstanding. Functional scope is still approximately
40%; an engineering integration does not change the learned-behavior estimate.

| Capability | Evidence/status |
| --- | --- |
| Two free bodies, real contacts, fists, separated spawns | Implemented and physically checked; no support forces |
| Standing, walking, approach | Four foundation stages certified by distinct consecutive full passes |
| Paired training, provenance, native motion evaluation | Implemented, with complete failures and budgets retained |
| Learned complete valid punch | v15 at 1,048,576 new credits: six deterministic cycles against neutral partners, both corners; known seeds only |
| Stable repeated attacks and return in both corners | Outstanding |
| Reciprocal boxing, defense and fresh/longer-round robustness | Outstanding; survival alone is insufficient |

The completed v15 final produces 23 valid neutral-practice cycles on fresh seeds
(13/10 across learner corners), but fresh mirror survival is 2/8 with zero valid
cycles. Original mirror has two valid corner-1 cycles with only 1/8 pair survival.
These results strengthen the isolated learned-strike evidence without closing
the stable reciprocal-fight gap. No complete gate passes. Unchanged training is
stopped at the approved ceiling; a bounded frozen-opponent diagnostic is active.

V16 is now complete: its final produced 42 valid neutral cycles on fresh seeds,
and both actors made valid strikes in one surviving fresh mirror episode. Fresh
pair survival of 3/8, 7/8, 1/8 fails the complete gate. PPO is stopped at its
approved ceiling while a separate arm-residual diagnostic investigates balance.

The first deterministic neutral-practice strikes are now supported by native
motion/contact/recovery measurements. At the subsequent 1,572,864-credit gate,
neutral valid cycles fall from six to two; mirror survival recovers to 2/8 but
still has zero valid cycles. No full pass or consistent retention is established.
The preceding 1,048,576-credit checkpoint's
evaluation fails all eight mirror
episodes on pair survival, with zero scored contacts or valid cycles there.
This is a serious stability regression from 5/8 mirror survivors at the preceding
gate, not completion of reciprocal boxing. No causal explanation is established.
The next concrete gap is preserving balance while both policies act, then
repeating valid strikes against active opponents across fresh conditions.
The [v15 protocol](striking-contact-progress-training.md) addresses the
measured separation between paid preparation and actual target contact, without
weakening full gates or rolling back the actual latest final. It is a bounded
next learning intervention, not a claim that the remaining gap has been closed.

Progress must be reported by milestones: complete learned strikes in both roles,
then repeated stable reciprocal exchanges, then broader fresh-condition/longer-
round validation. Keep two distinct consecutive complete passes and fresh checks
before declaring the corresponding limited milestone. General boxing completion
requires the roadmap's broader evidence, not a successful isolated frame.
