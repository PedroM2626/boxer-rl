# Reviewed exact source / guarded / control replay

All 72 deterministic physical episodes reproduce every original trial field
exactly. The independent reviewer checks all NPZ hashes/finite arrays, original
source and latest endpoint payloads, all four foundation histories (29/41/13/2),
the unchanged main boxing hash and empty native error logs. Complete survivor
and failure traces are archived here; live filenames inside raw reports are
historical, with identical trace bytes retained in each local `traces/` folder.

| Policy | Paired survivors | Low-pelvis failures | Self-leg-contact failures | Before first interbody contact |
| --- | --- | --- | --- | --- |
| Source 97af5783 | 23/24 | 1 | 0 | 0 |
| Guarded 3d7468c9 | 19/24 | 3 | 2 | 1 |
| Control 24e23288 | 18/24 | 5 | 1 | 0 |

There are no tilt or nonfinite terminations. Eleven of twelve failures follow
physical interbody contact, often seconds later; this does not establish that
impact caused the fall. The early guarded failure is the inactive partner's
self-leg contact before either agent scores. Do not weaken the first-fall gate
or discard that episode. All three complete physical sets fail.

Actual replay compute is **33,219 world controls**, with zero new collection,
optimizer or supervised updates. Formal evaluation is **1,666,653** and diagnostic
PPO remains **2,016,512**. Main/foundation counters and certifications are intact.

`terminal-credit.json` is a subsequent **zero-simulation** retrospective audit of
all saved observations, critics, own-fall rewards and fragment boundaries. With
gamma .999 / lambda .95, the direct terminal coefficient is .07319 one second
earlier, versus .57549 at lambda .99 within the same fragment. Critic-mediated
credit is distinct, and lambda .99 may raise variance. These deterministic traces
are off-policy and reset-aligned fragments are not the original training batch.
No causal claim, counterfactual physical improvement or adoption follows.

Concrete next decision: one [bounded GAE comparison](../../../striking-gae-comparison.md)
from latest control 24e23288, retaining learned/optimizer/sampler state, original
rewards/covariance/physics and full gates. Compare lambda .99 versus .95, at most
16,384 credits each and 84,000 evaluation controls. New diagnostic PPO ceiling
2,049,280; formal evaluation ceiling 1,750,653. No older favorable actor, masks,
forced rollback, automatic promotion or further blind allocation.
