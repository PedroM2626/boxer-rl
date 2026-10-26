# Target-coupled contact progress PPO v15

[Prospective protocol](../../../striking-contact-progress-training.md) and
[qualitative completion estimate](../../../completion-status.md) define scope.
Approximately 40% is not a remaining-time forecast. Complete learned valid
punches and reciprocal boxing remain unproven.

## Audit and intervention

proposal-audit.json contains verified source arrays with no new physics or
learning. After 3 s in original-seed learner-controlled observations, median hand
speed increases .415 -> .556 m/s from v14 baseline to final, while mean minimum
fist/target-center distance rises .291 -> .347 m and legal contact fraction falls
.002229 -> zero. Replayed preparation payment rises 11.68 -> 23.18. This suggests
an incentive gap and does not isolate a unique cause or prove learning impossible.
Saved distribution std is about .13-.15; this alone does not measure exploration
coverage. No noise, action authority, physics, motor or acceptance gate is changed.

Version 15 replaces forward-air preparation with bounded target-coupled potential
and explicitly graded same-hand physical contact subgoals. Old complete-cycle
payment and all independent original/motion gates remain unchanged. Partial
subgoal rewards are never counted as valid punches. Actual latest v14 final
be59681d warm-starts without changing any old saved state field or resetting Adam,
RNG, normalization or sampler provenance. Native distance calculations read only
the already simulated state; no extra kinematic/search allocation is used.

## Verified integration; active learning

Thirty focused transition/reward/mixed-routing tests pass. The included
1,024-credit real integration has finite updates, exact actor/critic Adam
31,384/55,056, 18 passive on-policy audit records, zero initial ratio clipping
and empty native stderr. Checkpoint 9da56178 and integration-review.json preserve
the evidence. The integration exercises physical subgoal payment 1.41765, with
zero complete learner cycles; that is a reward-path check, not a learned success.

The historical v14 final original-seed full gate is copied with its actual
be59681d source hash/reference, not rerun or charged again. Initial v15 parameters
remain functionally identical under the reward-only metadata transition.
New evaluation controls so far: zero. Live audit/progress supersede these static
records: `/home/pedro/boxing-runs/striking-contact-progress-v15/`.

The hidden persistent job is training toward the first 524,288-credit full gate,
within 2,097,152 new learner credits. Global diagnostic PPO ceiling 5,309,696;
formal evaluation ceiling 2,200,974 from 2,140,974 (up to 60,000 new controls).
Fresh final seeds are 34000003-34000006. Every complete physical/motion gate,
source hash, failed episode and consumed transition must be retained. Require
two distinct consecutive complete passes plus fresh validation for the limited
motion milestone; no automatic main adoption or boxing completion.

Combined main remains 476,041,216; main boxing 1,048,576; standalone BC 8,000;
cloned queries 6,349; signed distance calls 960; renderer forwards 2,065.
The integration is inside the declared new budget. Subsequent actual learning
requires hours; sparse early full-cycle counts are not an impossibility result.
