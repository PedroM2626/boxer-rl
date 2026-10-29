# Batched GPU mechanics development protocol

Prospective 2026-10-29 continuation of `gpu-batched-training-next.md`.
No PPO or formal gate allocation is used. At most **200,000 attempted GPU world
substeps** across the bounded development checks/benchmarks, with reset forwards,
native reset forwards, active and discarded-terminal substeps recorded separately.
No change to original 3/5 solver limits, 2 ms physics, action authority, rewards
or physical gates. Main 476,041,216, diagnostic PPO 9,504,000 and formal 2,298,509
remain retained. A failure must be inspected and its compute retained before retry.

Implementation targets: vectorized same-hand impulse/motion and scalar-identical
impact windows; precomputed rows in the unchanged native cycle monitors; all v17
reward terms; independent per-world resets without forwarding live survivors;
first-fall terminal snapshots and explicit timeout handling. CUDA kernels for
still-live worlds may compute unused substeps for already terminated neighbors;
those are never scored or placed in histories and are reported as discarded
terminal compute, never credited as learner data or apparent survival.

Validation uses native API measurements on identical snapshots and original
native control/reward code replaying the actual CUDA snapshots, selective reset invariance,
forced first-fall/timeout boundary states, and the two archived independently
verified positive v15 event windows. A frozen learned-policy collection/benchmark
follows only after these checks pass. Engineering tests cannot promote a policy.
The conditional 1,024-credit PPO integration remains downstream of all required
mechanical/reset/measurement checks and an explicit warm-start transition.

Live output: `/home/pedro/boxing-runs/batched-gpu-mechanics-v1/`. Inspect its
progress/process and the global physics lock before launching another job.

## Completed review

The first independent-rollout comparison stopped after **22 GPU world substeps**
when its observation error reached 1.317e-5 against a 3e-6 development tolerance.
The failure and its compute remain archived in the root progress file. Independent
low-iteration CUDA trajectories are sensitive to numerical differences; this is
not evidence of interchangeable native/CUDA dynamics. No physical gate changed.

Attempt 2 instead compares the original native measurement, controller, event and
reward code on the **same actual CUDA trajectory**, with no extra native physics
steps. It passed all five checks: both placements over 60 controls (maximum
observation error zero), selective reset invariance, first-fall truncation and
round-clock termination, and both archived complete-cycle windows. Resetting one
world preserves surviving physics, warm starts, controller state and history
bitwise. Nine terminal-neighbor substeps in the forced-fall test were discarded
and counted separately. This tests implementation semantics, not independent
CPU/CUDA trajectory equivalence or learned-policy robustness.

The complete frozen-policy benchmark includes both actors, motor, every-substep
contact/kinematics/history/reward, randomized resets, evidence and durable ledger
writes. After eight warm-up controls it measured **85.5 world controls/s at eight
worlds** and **601.5 at 128 worlds**, excluding Gaussian sampling, critic/PPO
updates and native evaluation. The earlier thousands/s bare-path measurement did
not include this work. The current eight-world collector does not recover the
old training throughput. Bulk readback/history overhead and a separately validated
larger-batch optimizer schedule remain necessary before long GPU training.

The 128-world frozen benchmark captured one independently reclassified valid
learner straight-punch cycle against an active frozen opponent. Its raw window,
hash, source and physical measurements are archived with the benchmark. This is
an engineering positive case from the retained policy, not a new training result
or a full boxing gate pass. It also had 36 falls/resets.

The machine-readable `completed-review.json` reconciles **174,941 attempted GPU
world substeps**, including the failed attempt, under the 200,000 development
ceiling. Reset forwards and active/discarded/warm-up substeps remain separate.
Thirty-five focused/regression tests passed. Main, formal and diagnostic PPO
counters remain unchanged at this review. The previously conditional **1,024
credit** warm-start integration is now justified; it stops for independent review
and grants no larger PPO allocation or native promotion.
