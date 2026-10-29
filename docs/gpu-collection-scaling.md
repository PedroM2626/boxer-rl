# Batched collection optimization and explicit scaling protocol

Prospective 2026-10-29 continuation after the reviewed first GPU integration.
Source: `checkpoints/striking-validation/gpu-batched-integration-v1/step-9421824.pkl`,
SHA `be1b5a8d64d88ac5ae2a266f1b975dc79ac39f90f95df1082d438fc86d5327e9`.
Retain main 476,041,216, formal 2,298,509, diagnostic PPO 9,505,024 and every
prior development counter. Prior unused allowances expire; they do not roll over.

## Development allocation and stop rules

At most **300,000 new attempted GPU world substeps** across collection profiling,
optimization and semantic regression checks. Record full/active/discarded/warm-up
substeps, reset/native forwards, failures and code versions separately. No new
formal evaluation or renderer allocation. Inspect processes and global locks
before every physical job. A failed check must be investigated before retrying;
do not silently restart budgets or alter physical acceptance rules.

First profile the complete frozen collector, then reduce its measured transfers
and per-world history overhead. Keep the original CPU implementation as a
same-snapshot oracle. Do not change dt, solver limits, collision geometry, action
authority, assistance, first-fall termination, observation/reward semantics,
random resets or complete-cycle criteria. Verify contact/motion/event/finite-state
results, survivor reset invariance and both immutable positive-cycle windows.
Benchmark the complete path with identical workload definitions and honestly
separate initialization, frozen inference, collection and PPO costs.

## Conditional larger-batch integration

Only after the optimized collector passes the above checks and demonstrates a
measured benefit, one explicitly versioned **16,384-credit** integration may use
128 worlds, 128 controls, four epochs and **256-row minibatches**: 64 minibatches
per epoch, **256 actor and critic updates**. This preserves the previous
updates/credit ratio; it changes the collection batch and normalizer update
frequency and must not be called an identical original-profile resume.

Preserve the latest GPU actor/critic/Adam, normalizer, RNG, frozen opponent and
historical sampler state; repeat the eight-world neutral/active assignment across
128 worlds and account all mode/corner credits. Verify checkpoint/count validators
and sampling before using this conditional allowance. Maximum new diagnostic PPO
16,384, resulting global ceiling **9,521,408**; no full-gate allocation or main
adoption follows. Count simulation separately (at most 163,968 GPU world substeps
including warm-up for this integration). Stop and review its finite checkpoint,
provenance, exact optimizer counts and complete measured throughput before a
prospectively justified longer GPU learning interval with unchanged native gates.
