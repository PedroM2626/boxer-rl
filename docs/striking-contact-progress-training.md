# Target-coupled physical contact progress: prospective PPO v15

Declared 2026-10-26 before any new training or evaluation. V14 is completed and
reviewed; it must not be resumed under its exhausted allocation. This new
profile does not declare a valid punch or weaken a physical acceptance gate.

## Evidence for the intervention

`tools/audit_motion_exploration.py` verifies source hashes and saved arrays with
no new physics/geometry/learning. Across the learner-controlled hands after 3 s,
original-seed median hand speed rises .415 -> .556 m/s, but mean minimum
fist/target-center distance rises .291 -> .347 m; legal contact substep fraction
falls .002229 -> zero. Replayed forward preparation reward rises 11.68 -> 23.18
across complete evaluated cohorts. Saved Gaussian stds remain roughly .13-.15;
there is no demonstrated covariance collapse. These correlations do not isolate
a unique cause. The v14 formula itself can pay for forward preparation without
requiring arrival at a legal target, and no complete-cycle bonus was encountered.

Keep the actual latest final **be59681d**, all actor/critic/Adam/RNG/norm and
sampler prefixes. Lineage consumed/optimized 3,129,344, world controls 2,506,752,
actor/critic Adam 31,368/55,040. No rollback, reset, stronger motors, scripted
punch teacher, direct root force or new leg authority. Keep 148 inputs/17 actions,
frozen certified foundation, separated 2 m spawns, original physics/collisions,
command/arm/filter bounds, eight mixed worlds, immutable active opponent,
zero old teacher loss, gamma .999 and GAE .99. Version 14 remains the parent
checkpoint contract; version 15 explicitly overrides its preparation term.

## Effective reward and subgoal scope

Remove the v14 forward-velocity preparation payment; do **not** sum both shaping
terms. Add target-coupled potential per corner:
`Phi = sum_h exp(-min_legal_target_center_distance(h) / .3)`.
Each step adds `2 * (.999 * Phi_after - Phi_before)`, with Phi_after zero at
original fall **or timeout**. Distances use actual native-state fist/head/torso
centers, not a newly queried contact distance or a claim that a surface is reached.
Phi is bounded [0,2]; terminal handling produces no positive termination bonus.
The discounted potential terms telescope over a complete terminal episode.

At each **actual legacy scored legal contact**, add a graded per-hand subgoal
payment `2 * clip(speed/1) * clip(forward_cosine/.7) * clip(pretravel/.12)`;
subtract the opposing corner's equivalent. Clip each component to [0,1]. Speed,
direction and pretravel must belong to that same native contact/hand. No contact
means no subgoal payment. A partial contact is explicitly not a valid punch.
This supplies credit before the delayed complete recovery window; it is not a
new gate or a reason to count touches as complete strikes.

Keep the v14 **+10/-10 completed-cycle** payment and independent conjunction
unchanged: actual scored legal fist contact, onset own-torso speed >=1 m/s,
forward cosine >=.7, pretravel >=.12 m, complete 500 ms recovery window with
return >=.06 m and contact duration <=.10 s. Either hand is eligible independently.
Keep fall/approach/base terms. Record potential and physical contact subgoals
separately so increasing shaping without real impacts is visible.

## Bounded learning and unchanged independent evaluation

Allocate at most **2,097,152 new diagnostic learner credits**, including the first
1,024 integration credits. Global diagnostic PPO **3,212,544 -> 5,309,696**;
lineage **3,129,344 -> 5,226,496**. Record new training world controls separately;
retain every attempted/discarded transition. Main combined **476,041,216** and
main boxing **1,048,576** stay unchanged. At roughly 90 credits/s this may take
several hours; throughput and success are not promised.

Reuse the already completely reviewed v14 final original-seed full gate as the
**historical baseline**, with its actual **be59681d** hash and every failed case.
All old saved state fields and physical configuration remain exact under this
reward-only transition. Do not relabel the baseline as new evaluation or change
its source hash. Store its reference/equivalence proof; exclude its progress
files from newly consumed evaluation counters.

Run full original three-role/motion gates after 524,288 / 1,048,576 / 1,572,864 /
2,097,152 new credits, plus one fresh final set: at most **60,000 new evaluation
controls**, formal **2,140,974 -> 2,200,974**. Original seeds 7/23/31/47; fresh
34000003-34000006. All original gates unchanged (neutral 0 >=7/8 survivors,
neutral 1 =8/8, mirror >=7/8, neutral own surviving scoring breadth >=2 seeds).
Complete motion milestone still additionally requires own valid cycles on >=2
surviving seeds in each neutral role, reciprocal surviving valid cycles on >=2
mirror seeds, two distinct consecutive complete training passes and a fresh pass.
No partial subgoal can replace the failed full conjunction or certify boxing.

Finite 1,024-credit integration checks exact Adam counts and on-policy updates.
Use existing hidden WSL launcher, configured Windows Git, global/output locks,
ordinary warm starts and durable attempted counters. No duplicate job or
reinitialization on routine resumes. Stop numerical/overflow/source errors
immediately. Do not call early zero-cycle metrics proof of impossibility; at the
ceiling review all gates and reward components before any justified extension.
No automatic adoption, main promotion or general boxing completion.

No new diagnostic cloned queries (6,349), signed distances (960), standalone BC
(8,000) or renderer forwards (2,065) allocation. Native distance calculations
use the already simulated state as part of collection, not extra forward/search.
Publish owned checkpoints/results at complete gates, retain prior failures and
the existing monitoring schedule. Live job:
`/home/pedro/boxing-runs/striking-contact-progress-v15/audit.json` and
`training/progress.json`. Inspect the actual active process before any intervention.

## Passive event evidence addition, 2026-10-26

After the first cumulative training-cycle signal, add bounded raw event/state
capture in future naturally launched training chunks. Do not interrupt productive
collection to apply an observer. Store the original native event windows, qpos/
qvel, actuator controls, applied-force norm and every learner checkpoint version
in a window, including cross-update windows. This is observation only and does not
change rewards, thresholds, action distributions, source, optimizer, physics or
budgets. Artifacts remain training evidence pending independent policy evaluation.
Earlier counter-only events cannot be reconstructed as videos or promoted as
held-out successes. Existing scheduled gates and monitoring remain authoritative.

## Read-only verified-event visualization allocation, 2026-10-26

The first frozen captured-window audit independently verifies 37 stochastic
training cycles; the scheduled 524,288 deterministic gate has zero full cycles
and fails original survival thresholds. Show the earliest verified neutral and
active-opponent windows, not selected largest scores. Two 66-forward renders
(initial fixed camera, revised side view after occlusion QA) add 132 forwards,
cumulative 2,197. All 330 frames of each output decode. No physical integration,
actor execution, new learner credit, geometry search, global training lock or
live-job interruption. Source windows, policies, scope labels and both camera
versions remain archived. This does not change training/evaluation budgets or
constitute a promotion. Maintain the remaining performance-based full checks.
