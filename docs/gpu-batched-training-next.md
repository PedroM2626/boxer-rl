# Next action: complete the batched GPU training mechanics

Decision after both completed comparisons, 2026-10-29. Do useful implementation
work when no physical job is active; the project is not complete and a stopped
PPO job is not a reason to leave this backlog untouched.

The CUDA 3/5 reference executes measured valid straight punches, but no complete
stability gate passes. Native/CUDA 30/20 does not consistently improve behavior;
do not adopt that solver change. Retained Warp6/7 produce no complete straight
cycles in sixteen reconstructed episodes and fail the common posture test from
2 m in all eight transfer episodes. This does not rule out hooks or prove the
frozen-motor architecture superior. Retain actual latest v17 rather than silently
rolling back to an old or intermediate policy.

## Implementation and validation before learning

1. Extend the fast `PairedWarpPhysics` / `PairedFeatures` path with batched
   per-hand contact impulses and own-torso-relative kinematics at every 2 ms
   substep. Match the audited reference adapter, including geom-role ordering,
   pre-impact velocity, first-fall termination and overflow rejection.
2. Preserve the existing complete-cycle history, target-potential/contact
   progress, v16 partial-payment rebalance and v17 arm-effort reward. Compare
   observations, rewards, events and complete-cycle classifications on identical
   trajectories. No reward change, new teacher or weaker punch definition as part
   of the backend migration.
3. Implement full per-world randomized resets: physics state, warm starts,
   phase/frequency, previous/applied actions, residual filter, scorecard and
   history must reset together. A reset in one world must not alter surviving
   worlds' state/observations. Explicitly test first falls, partial control steps,
   round timeouts, both corner placements and legal contacts.
4. Benchmark the complete collection/reward path and PPO overhead, not bare
   physics. World controls, learner credits, native evaluation, GPU substeps/
   forwards and rendering remain separate. Keep the original native physical
   gates authoritative for any candidate learned on CUDA.

No new formal evaluation allocation is implied by this engineering work. Record
bounded development simulation separately, preserve failed attempts, and inspect
all existing processes/ledgers before starting tests. Current formal total is
**2,298,509**, main **476,041,216**, diagnostic PPO **9,504,000**. All unused
comparison allowances expired.

## Conditional first PPO integration allocation

Only after the above mechanical/measurement/reset checks pass, one ordinary
warm-start integration block may consume at most **1,024 new learner credits**
(global diagnostic PPO ceiling **9,505,024**), using the original eight-world,
128-control profile. This is an implementation integration check, not evidence
of learning failure/success and not a long-training allocation. Retain the source
actor, critic, Adam, normalizers, RNG and frozen opponent/task profile. Record an
explicit backend/configuration transition; never relabel it as an identical native
resume. Stop and review the resulting finite checkpoint, exact source inheritance,
consumed credits, world steps and optimizer updates before larger batches or PPO.

Exact source: `checkpoints/striking-validation/arm-effort-v17/step-2097152`,
SHA-256 `f311640d78941546938412f5802cff872e0a3611c392f49a3b4719eefaa1f26a`.
Verified source lineage consumed/optimized **9,420,800**, world controls
**8,798,208**, actor/critic Adam counts **129,672 / 153,344**. The global diagnostic
counter also retains work outside that lineage; never rewind it to the prefix.

After a successful integration and measured throughput, prospectively document
a justified longer bounded GPU learning allocation with regular unchanged native
gates, two distinct complete consecutive passes and a new fresh holdout. Training
may require many hours; this small integration block is not a convergence test.
Stable repeated reciprocal striking, defense and longer/fresh-round robustness
remain the actual completion goal. No user permission round-trip is needed for
this already authorized continuation.
