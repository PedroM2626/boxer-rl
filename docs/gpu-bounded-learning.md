# First bounded scaled-GPU learning continuation

Prospective decision 2026-10-29, after complete packed-readback semantic checks,
a measured collector improvement and independent exact replay of the scaled
integration's Gaussian stream, all 256 PPO updates and sampler/optimizer states.
The optimizer replay uses stored data and adds no simulation or learner credit.

Use the actual latest scaled GPU final, SHA
`6432029087cb00e73e39bee3a974e07f47f598771bae05c519e1bcad616c2103`,
`checkpoints/striking-validation/gpu-batched-scaling-v1/step-9438208.pkl`.
Source consumed/optimized 9,438,208, world 8,815,616, Adam 129,944/153,616.
Global diagnostic PPO before **9,521,408**, main **476,041,216**, formal
**2,298,509**. Prior allowances expire; prior simulation, clone/BC/geometry/render
ledgers remain retained. No motor unfreezing, reward redesign or rollback.

## Allocation and rationale

At most **1,048,576 new diagnostic learner credits**, global ceiling **10,569,984**,
with the validated 128-world/128-control, four-epoch/256-row-minibatch profile.
Preserve all actor/critic/Adam/normalizer/RNG/frozen-opponent/sampler prefixes.
No new migration is needed. Each 16,384-credit rollout performs 256 updates per
optimizer, the original updates/credit ratio. Normalizer update frequency differs
from the historical eight-world profile and remains explicitly versioned.

This bounded interval is justified by functional exact-update integration, about
870 collected-and-updated credits/s in its single measured block (initialization
excluded), and the prior genuine complete strikes without stable reciprocal
behavior. It provides productive learning and repeated physical observations
rather than another unchanged eight-world CPU run. It is not a declaration that
one million credits are sufficient for convergence or that a sparse early result
proves learning impossible. Wall-clock estimates remain provisional.

Native unchanged complete gates: a new baseline on this actual source, then at
262,144, 524,288, 786,432 and 1,048,576 new credits; finish with one new fresh
group. At most **72,000 new native controls**, formal ceiling **2,370,509**.
Known seeds 7/23/31/47 and their original two placements remain fixed; final fresh
seeds 37002003/37002004/37002005/37002006. Retain full finite/action/contact/motion
traces and hashes. No renderer allocation. GPU learning simulation is separate:
at most 10,486,784 attempted world substeps including a bounded 1,024 warm-up
allowance for up to eight inspected ordinary process resumes.

## Stop, review and publication

Stop on nonfinite/capacity/float overflow, unexpected native warnings, checkpoint
or provenance/count failure, unoptimized consumed data on resume, or budget
exhaustion. Solver-limit bits 512/1024 at the unchanged 3/5 solver are recorded;
independent CPU/CUDA trajectories are not certified equivalent. Keep native
criteria authoritative. Stop early after two distinct consecutive complete
passes and run the fresh group; do not promote without both and fresh success.
General boxing still additionally needs sustained reciprocal exchanges, defense,
varied opponents and broader/longer fresh rounds.

Use hidden persistent WSL launchers, exclusive global locks and ordinary warm
starts. Save every rollout checkpoint and attempted transitions durably. Preserve
failed attempts/partial native evaluations and require investigation before any
retry. Publish project-owned checkpoint/results/docs at gates with Windows Git;
publication errors must be recorded without silently deleting or replaying data.
No automatic ceiling extension. The existing monitoring interval stays unchanged.
