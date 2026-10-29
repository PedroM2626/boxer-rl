# Completed one-block batched GPU PPO integration

Reviewed 2026-10-29. This is the conditional **1,024-credit** integration declared
before the mechanics work, not a convergence experiment or a physical gate.
The hidden persistent WSL launcher and both global physics/supervisor locks were
used. Implementation was committed and pushed as `79bd1b5` before launch.

The source is the actual final v17 policy, SHA-256
`f311640d78941546938412f5802cff872e0a3611c392f49a3b4719eefaa1f26a`.
Every source leaf is unchanged in `initial.pkl`, apart from explicitly added GPU
backend metadata. Actor, critic, Adam moments/counters, normalizer, RNG, frozen
opponent and all historical metadata/samplers were retained. No rollback,
optimizer initialization, teacher loss, solver change or motor adaptation occurred.
The full original eight-world/128-control v17 task was used with ordinary complete
randomized resets. Physics and contact-force extraction ran on CUDA;
actor/critic/motor used CPU JAX. Batched kinematics, impact windows and cycle
reward/history calculations currently use CPU NumPy after GPU readback. They
are semantically validated, but are not yet fused GPU reward kernels.

The final checkpoint is
`checkpoints/striking-validation/gpu-batched-integration-v1/step-9421824.pkl`, SHA
`be1b5a8d64d88ac5ae2a266f1b975dc79ac39f90f95df1082d438fc86d5327e9`.
It retains consumed/optimized **9,421,824**, world controls **8,799,232**, and
actor/critic Adam **129,688 / 153,360**: exactly 1,024 new credits/world controls
and sixteen updates to each optimizer. Global diagnostic PPO is **9,505,024**,
at its declared ceiling. Main **476,041,216** and formal **2,298,509** are unchanged.
All unused prior comparison allowances remain expired.

`independent-review.json` rechecks payload hashes, all original source leaves,
unchanged final metadata/frozen opponent, finite rollout/parameters, exact RNG
splits, and every stored Gaussian raw action/log probability by replaying the
sampling stream on CPU. Neutral/active credits are exactly 512/512, with corner
counts retained. One episode terminates and resets during collection. This
read-only review runs no new physics, renderer, formal gate or learner data.

The engine used **10,248 GPU world substeps**: 10,235 active, five discarded after
a first fall in a neighboring world, and eight warm-up. There were 32 GPU reset
world-forwards and 26 native reset forwards. Solver/line-search limit bits
512/1024 remain present (combined flags 1536), with no capacity/float/nonfinite
flag. Peak 138 constraints/world remains below the unchanged 2,048 allocation.
These are the original 3/5 solver limits; successful integration does not prove
convergence or native/CUDA trajectory equivalence.

Collection took **11.588 s** including initial actor/critic sampling JIT; sixteen
PPO updates took **0.735 s**, also including compilation. The full job, including
initialization, took 19.791 s. This small eight-world run does not establish a
production throughput improvement. The separate full frozen collector benchmark
measured 601.5 world controls/s at 128 worlds, excluding PPO; bare-physics numbers
of thousands/s are not complete training rates.

## Decision and next work

The bounded integration succeeded and stopped at its exact ceiling. Preserve its
finite final as the latest experimental GPU prefix for ordinary continuation;
the unchanged v17 final remains the last completely physically evaluated source.
No new complete physical pass, main adoption or boxing completion is claimed.

Next remove measured collection overhead (many small synchronized readbacks and
Python history/contact operations), preserving identical-snapshot semantics and
first-fall/reset behavior. Then explicitly version and test a larger collection
profile. For 128 worlds and 128 controls, retaining 256-row minibatches and four
epochs requires **256 optimizer steps per 16,384 credits**, rather than silently
keeping sixteen updates and changing the update/data ratio. Recheck all count
validators, samplers, normalization and mode/corner coverage against the latest
GPU prefix before allocating learning. Prospectively declare a bounded GPU
learning interval and native gate budget only after that integration/performance
review. Require the original two distinct consecutive complete physical passes
and new fresh checks for promotion. Stable repeated reciprocal exchanges and
broader/longer fresh-round validation remain outstanding.
