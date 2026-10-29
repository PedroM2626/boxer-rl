# Reviewed scaled-GPU PPO integration

2026-10-29. The conditional 16,384-credit integration completed from the exact
latest GPU prefix, SHA `be1b5a8d64d88ac5ae2a266f1b975dc79ac39f90f95df1082d438fc86d5327e9`.
It explicitly versions 128 worlds, 128 controls, four epochs, 256-row minibatches
and 256 updates per optimizer. All learned source leaves except the declared
collection profile remain exact at initialization. Source Adam/RNG/normalizer/
frozen opponent and samplers are retained; the normalizer update frequency change
is explicit. No solver, collision, assistance, dt, authority or reward change.

Final SHA `6432029087cb00e73e39bee3a974e07f47f598771bae05c519e1bcad616c2103`,
consumed/optimized **9,438,208**, world controls **8,815,616**, Adam
**129,944 / 153,616**. Global diagnostic PPO is **9,521,408**, at its conditional
ceiling. Main **476,041,216** and formal **2,298,509** remain unchanged.

The independent review verifies payload hashes, source inheritance, metadata,
all Gaussian raw actions/log probabilities and RNG splits, mode/corner credits,
finite rollout/state and exact optimizer increments. Using stored bootstrap and
rollout arrays, the shared `BatchedPPOUpdate` replays **all 256 updates** exactly:
actor, critic, optimizer moments/counts, normalizer, RNG and auxiliary sampler
states equal the saved final. This replay adds no simulation or new learner data.

GPU simulation: **163,968 world substeps**, including 128 warm-up, 163,717 active
and 123 discarded terminal-neighbor substeps. Reset forwards 3,072 GPU / 408 native
remain separately recorded. Only original solver-limit flags 512/1024 occur;
there is no capacity/nonfinite/float overflow. Peak constraints/world 154.

Collection took **17.299 seconds**, updates **1.537 seconds**, both including
first JIT. That is approximately **870 learner credits/s** across collection and
update, excluding initialization, checkpointing and independent native gates.
The whole one-block job took 26.898 seconds. This is a bounded integration timing,
not a sustained long-training rate or recovery of historical thousands/s.

No new formal gate, learned-boxing success, general convergence conclusion or
main adoption is claimed. The separate [bounded continuation](../../../gpu-bounded-learning.md)
prospectively allows 1,048,576 new credits with original native full gates and a
fresh final group. Ordinary continuation must use this actual final rather than
repeating either completed migration/integration.
