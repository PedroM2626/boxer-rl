# Learned straight-punch cycles: prospective training protocol v14

Declared 2026-10-25 before collection, after the arm-contract/saved-state audit.
The last scripted failures do not show that PPO cannot learn boxing. They reject
specific short trajectories; no learner was updated in those experiments.
The current task requires learned purposeful strikes and held-out reciprocal
boxing. Long training may be necessary, but elapsed hours are not a certificate.

## Evidence and deliberate objective transition

The [arm audit](results/striking-pilot/arm-contract-v1/README.md) checks compiled
metadata and 32 saved trajectories: correct joint indexing/baseline scaling,
no native actuator control clipping; residual request clipping still occurs.
Wrist gains 2 versus legacy 20 are an already documented contract difference,
not a demonstrated cause or an authorization to change the certified motor.
No new physics, selected-state forwards, geometry searches or learning in audit.

Warm-start the actual latest fixed-control final 3a34aa85, lineage consumed and
optimized 2,080,768, world controls 1,458,176, actor/critic Adam 14,984/38,656.
Keep actor, critic, Adam moments/counts, RNG, normalizer and all prior lineage.
No rollback to bea8147f, optimizer reset, stronger motors or assistance.
Version 14 retains 148 inputs/17 actions, frozen certified foundation b39b1990,
2 m spawns, .45 m engagement, original command/arm bounds, collisions and filter,
eight mixed worlds (four neutral, four immutable active opponents), gamma .999,
GAE lambda .99, original covariance and split-Adam rates/updates.

Explicitly set the old demonstration-retention loss from 1 to **0**: the complete
retained-demo audit established touches rather than positive full cycles. Keep
its dataset/sampler provenance; existing sampler draws may continue but have
zero gradient weight. This is no new BC allocation. Do not secretly call the old
retention coefficient the current objective. Anchor coefficient remains zero.

Remove both +/-2 legacy contact payments, retaining falls, approach and other
base terms. Pay **+10** for own completed physical straight-punch cycles and
**-10** for the opponent's completed cycles, after the full recovery window.
A moving touch earns no full-cycle payment. Native 2 ms measurement requires
one same hand's legal scored contact, onset own-torso-relative speed >=1 m/s,
forward cosine >=.7, pre-window travel >=.12 m, complete 500 ms post-window,
return >=.06 m and post-contact duration <=.10 s. Either learned hand can punch;
the old scripted trials specified left because only that arm was commanded.
This prospectively declared per-hand symmetry does not mix evidence across hands
or certify every punch type, guard recovery or general boxing.

Add bounded dense preparation at rate .5 per simulated second, using each
native sample's positive own-torso forward fist velocity clipped to 1 m/s,
multiplied by target-directed relative-velocity alignment clipped to [0,1],
maximum across own hands and averaged over the 20 ms control. This encourages
exploration toward a strike; it is not a validation gate or proof of a punch.
Monitor history is capped at 650 native samples; completed events pay once.
Reset clears pending history; stopped/restarted physical episodes censor pending
cycles rather than invent recovery or rewind consumed transitions.

## Bounded learning, time and independent checks

Allocate at most **1,048,576 new diagnostic PPO learner credits**, including the
first 1,024-credit integration block. Global diagnostic PPO ceiling **3,212,544**
from **2,163,968**; lineage ceiling **3,129,344** from **2,080,768**. Count exactly
one learner and one collection world control per mixed-world transition; retain
all discarded collection. New training world controls are recorded separately
from formal evaluation controls. Foundation/main combined **476,041,216** and
main boxing **1,048,576** remain unchanged.

Run complete baseline, then complete deterministic three-role gates after
262,144/524,288/786,432/1,048,576 new credits, plus one fresh complete final set:
maximum **72,000 formal evaluation controls** from **2,073,802** to **2,145,802**.
All gates preserve original survival/contact criteria: neutral corner 0 >=7/8,
corner 1 =8/8, mirror >=7/8, neutral surviving own-scoring breadth >=2 seeds.
Seeds 7/23/31/47; fresh 33000003-33000006. Every full gate additionally records
unmodified native trajectories, full physical state and every complete motion
component with an independent acceptance conjunction, without the training reward
observer. Valid-cycle milestone requires surviving own cycles on >=2 seeds in
each neutral role and surviving reciprocal cycles on >=2 mirror seeds, in
addition to the unchanged original complete gate. Require two distinct complete
consecutive training checkpoint passes and the fresh set before calling this a
limited learned straight-punch milestone; no automatic main adoption or general
boxing completion. Stop earlier only if that milestone warrants its fresh check.

Zero early valid cycles does not alone stop learning or establish impossibility.
The first 1,024 block checks finite updates, native errors, exact Adam counts and
on-policy audit. Numerical/nonfinite/overflow/source errors require immediate
investigation. At the declared ceiling review complete history, physical videos
and motion distributions before any separately justified extension; no unlimited
steps or threshold relaxation. Runtime depends on measured throughput; do not
promise an hour count or assume 200M steps are mandatory for every phase.

No additional cloned (6,349), signed-distance (960), renderer (500), or standalone
BC (8,000) allocation. Use hidden persistent WSL/global/output locks, ordinary
warm starts and complete attempted-step ledgers. Archive/commit/push only owned
results/checkpoints at completed evaluations. Preserve existing monitoring
schedule; it should inspect audit.json and the actual active child before any
launch or recovery. Do not repeat completed initialization on routine resumes.
