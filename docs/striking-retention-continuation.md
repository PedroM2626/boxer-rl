# Saved-policy regression review and bounded demonstration-retention PPO

**Completed:** both fixed arms and all full physical trials finished. Retention
restores strikes, including one fresh deterministic two-way surviving episode,
but regresses role-1 survival. Neither actor passes. See the [final review and
bounded replay/trace audit](striking-retention-review.md). Launch details below
are historical; no training extension is granted yet.

## Completed investigations and decision

The [saved-Gaussian audit](results/striking-pilot/postppo-sampling-bc5000/analysis.json)
completed all 72 unchanged ten-second trials. It consumed **34,237 evaluation
world controls**, with no optimizer steps, finite physical state and empty native
error logs. Neither PPO actor recovers stable strikes through ordinary saved
sampling. No policy has a surviving two-way episode. Each row has eight episodes
per profile on seeds 5,000,003–5,000,006, both swaps.

| Saved sampled policy | Neutral role 0 survival / own stable events | Neutral role 1 survival / own stable events | Active survival / stable events |
| --- | --- | --- | --- |
| BC5000 source | 8/8 / 0 | 8/8 / 2 | 8/8 / 2 |
| Final PPO only | 8/8 / 0 | 7/8 / 0 | 7/8 / 0 |
| Final PPO + approach anchor | 8/8 / 0 | 6/8 / 1 | 7/8 / 0 |

The source's blue neutral events cover two seeds; red never scores. Neither final
actor passes the complete criteria, even descriptively under sampled actions.
This audit does not replace the deterministic physical promotion gates. No
sampling scale/hold, contact threshold, assistance or gate was changed.

The [short-rollout gradient assay](results/striking-pilot/gradient-audit-bc5000/analysis.json)
then consumed exactly **1,536 diagnostic world controls**, 512 per source/final
actor, with **zero optimizer updates or PPO training credits**. Four free paired
worlds per actor ran 128 controls from seeds 6,000,003–6,000,006. Their 1,024-row
batches, actual rewards/log probabilities/values, data indices and hashes are
archived. All batch values and recorded gradients are finite; policies retain
their hashes. Native logs contain no numerical/capacity warnings. A wrapper-path
typo in the first host launch was corrected before any simulation started; it
consumed no steps. The completed assay was not repeated.

PPO actor policy-gradient norms were 7.91 / 8.21 / 7.27 for source / PPO-only /
anchor. Unscaled approach-anchor gradient norms were 0.113 / 1.026 / 0.999.
The rejected 0.02 coefficient therefore supplies only about 0.0023 / 0.0205 /
0.0200 of gradient norm before the joint clip. It also never represents a strike
demonstration. Weighted strike-demonstration norms were 0.173 / 1.881 / 2.365.
Cosines with PPO gradients were near zero (-0.035 / 0.039 / 0.004), not evidence
of a strong systematic opposing gradient. These are one short batch per actor,
using each saved policy's own RNG key and trajectories; they are descriptive,
not a controlled causal estimate, convergence test or physical certification.

The concrete next hypothesis is to keep demonstrated strike actions in the
on-policy objective instead of protecting only approach zeros. Do not increase
the same failed PPO/anchor allocation or restore a favorable older BC snapshot.

## Explicit latest-state continuation

Continue the **latest completed PPO-only branch**, SHA
`e3e01b71de64e3d26a431ba87d484a45d349621df5461c274fb7e394b935ef4e`,
from `checkpoints/striking-validation/ppo-anchor-bc5000/ppo_only/`.
This is a diagnostic branch choice, not main adoption. The parallel anchor branch
is rejected for its measured regression and remains archived.

Both new arms retain the same current actor, critic, log standard deviation,
normalizer, policy key, independent Adam moments/counts **7,048 / 30,720**, and
lineage counters **1,572,864 consumed/optimized / 983,040 world controls**. No
optimizer, sampler prefix or main consumed counter is reset. The version-3
configuration explicitly introduces `demonstration_ppo`, coefficient and verified
dataset provenance; it fails ordinary profile mixing. The new sampler starts
from the retained auxiliary sampler state and is saved after each PPO block.
Both arms draw the same indices, including the weight-zero control. BC and the
previous auxiliary sampler histories remain independently retained.

The coefficient-1 arm adds the existing weighted action MSE to PPO, using
**192 verified demonstration rows + 64 approach-anchor rows per minibatch**.
It retains original channel weights (commands 2, left arm 4, right arm 1) and
original demonstration row weights (5 for material teacher corrections, else 1),
computed from the archived original actor. Anchor labels stay the recorded zero
actions. The fixed 15 trajectories / 7,500 rows and 32 safe approach trajectories /
1,600 rows are checksum-verified. No new teacher, root force, scripted controller
or action clamp is deployed. Current normalization is used for these observations.

Coefficient **1**, versus **0** control, is a bounded hypothesis motivated by the
measured gradient scales. It is not claimed optimal. It supplies a substantive
retention gradient without increasing learning rate or changing reward, score,
geometry, actuation, reset distance, frozen 29-joint motor, residual bounds/filter,
physics timing, free legs or independent survival/striking criteria. Auxiliary
gradients affect only the actor; PPO actor and critic still share one global
gradient clip before their independently counted Adam updates.

## Allocation and review contract

Grant **32,768 new diagnostic learner credits per arm**, **65,536 total**,
at most **32,768 new collection world controls**. This is one quarter of the
previous arm allocation, conditional on verified integration, with no automatic
extension. Both arms first retain one **1,024-credit / 512-world-control** block,
verify finite checkpoint, exact consumption and Adam counts **7,064 / 30,736**,
then resume normally for the remaining **31,744** credits. Final expected counts
are **7,560 / 31,232**. Both arms share this physical reset boundary; native world
state is not serialized, while all policy/optimizer/normalizer/sampler state is.

This adds 512 integrated PPO optimizer updates per arm. Each arm reads 131,072
auxiliary examples, 262,144 across both arms; only the coefficient-1 arm applies
their retention loss (98,304 demo + 32,768 anchor presentations). These are
integrated PPO auxiliary presentations, not separate BC updates. Cumulative
standalone BC stays **8,000 updates / 2,048,000 presentations**.

Evaluate the unchanged source once on **new preregistered seeds 7,000,003–7,000,006**,
all three partner profiles, both swaps, ten seconds (24 episodes). Each final
arm receives the complete original screen and the same fresh set (48 episodes).
Maximum evaluation allocation is **120 episodes / 60,000 world controls**.
The fresh data are not used for checkpoint selection or allocation changes.
Report per-role surviving own multi-seed strikes, complete survival, actual
active contacts by both corners and surviving two-way episodes. Two parallel
arms are not two consecutive passing generations. No stage advances without the
established distinct consecutive complete physical gates and motor checks.

Before this comparison, cumulative diagnostic PPO is **1,450,240** and formal
diagnostic simulation is **547,894 world controls** (including the short gradient
assay). Foundation/main boxing remains **474,992,640 / 1,048,576**, combined
**476,041,216**, with no main ceiling change. Retain every attempted collection
credit and record new collection/evaluation controls separately on errors.

The opt-in trainer uses `--contact-ppo --demonstration-ppo`, strict version-3
configuration and retention coefficient/provenance matching. **21 focused tests
passed**, including legacy update equivalence, exact retained state, weight-zero
update identity, matched samplers, finite count increments, actor-only retention
gradients and configuration/CLI rejection. The final retention suite passed again
after strengthening prefix validation. Python tools compile and the shell wrappers
pass syntax checks. Real first-block validation remains required before claiming
the complete new integration works.

The hidden launcher is `tools/start_striking_retention_compare.ps1`; durable state:
`/home/pedro/boxing-runs/striking-retention-1572864/comparison.json`.
Inspect it and its active child, progress, checkpoint hashes and native logs
before launching anything else. The parent refuses existing state and native
training retains the global unassisted-boxing lock. Preserve the existing monitor.
After completion, archive both unchanged complete final evaluations and make a
concrete decision; no automatic actor adoption or unlimited continuation is allowed.

## Launch verification

The job started at **2026-10-23 13:24 UTC**. Source fresh evaluation completed all
24 trials (11,236 controls), with neutral survival 7/8 and 8/8, active 7/8, and
zero surviving own strikes. The control's retained first block passed at 13:26
UTC with exactly 1,024 credits / 512 controls, counts 7,064 / 30,736, finite
checkpoint and empty native error logs. Its ordinary remaining invocation is
active. The treatment's same real guard remains pending. All four full foundation
histories, current payload hashes/finite values and the unchanged main actor were
reverified. [Launch and integration artifacts](results/striking-pilot/retention-1572864/README.md)
include the complete source baseline and exact transition/first-block models.
