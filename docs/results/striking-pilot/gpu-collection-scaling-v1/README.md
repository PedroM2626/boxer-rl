# Pinned collection profile and semantic review

2026-10-29, under the prospective `docs/gpu-collection-scaling.md` protocol.
Full frozen means from the latest GPU prefix, 128 worlds, 64 controls, first
eight excluded from timing. Both cases include motor, contacts, histories, v17
reward, evidence handling, randomized resets and durable per-substep accounting.
They include cProfile overhead and exclude Gaussian sampling/critic/PPO.

Baseline: **620.61 world controls/s**, 11.550 seconds for 56 measured controls.
Persistent pinned readback plus cached pre-substep snapshots: **835.11/s**,
8.583 seconds, a **34.6% rate increase** in this matched workload. These are
bounded measurements, not a sustained training guarantee or historical speed
recovery. Both complete fifteen episodes/resets with no complete cycles in this
short frozen benchmark. No learned-behavior improvement follows.

The profiler identifies 17,698 synchronous allocating `.numpy()` reads in the
baseline and 2.48 seconds in explicit synchronization. The optimized path copies
an explicit read set into persistent pinned buffers and synchronizes once;
contact forces use the actual upstream guarded solver extraction with reusable
storage. Previously copied state is reused until physics changes it. No solver,
dt, reward, collision or action-authority change. Kinematics/history/rewards
remain CPU NumPy; they have not been relabeled GPU kernels. Physics and contact
extraction still run on CUDA. All finite/capacity/applied-force checks remain.

The first scaling semantic test failed after eleven GPU substeps because a
test-harness path variable was overwritten by a reward array. That code defect
was corrected with a distinct path name; its failed progress/compute is retained.
The separately named second attempt passes all five semantic checks, including
the original native control/reward/termination code on identical CUDA snapshots,
bitwise survivor reset invariance, partial first-fall/timeout boundaries and the
two unchanged positive complete-cycle windows. No physical tolerance was relaxed.

Development GPU compute so far is **164,810 world substeps** (82,048 baseline,
82,048 optimized, eleven failed test, 703 successful test), under the separate
300,000 ceiling. The forced boundary adds two explicitly recorded GPU forwards;
reset forwards remain in each ledger. The exact-readback GPU regression uses
three additional forwards, no physics steps. Thirty-six focused/regression tests
pass, including explicit 128-world metadata retention and rejection of sixteen
updates per 16,384 credits. The direct-readback test compares arrays/contact
forces exactly at three poses and verifies snapshot lifetime across refreshes.

The conditional 16,384-credit integration may now proceed only after these
checks pass and the implementation is committed. It preserves the latest GPU
prefix and requires 256 updates per actor/critic, with no formal/main adoption.
No long PPO allocation follows from this collector speed measurement alone.
