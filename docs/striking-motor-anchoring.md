# Fresh pre-contact samples and matched motor anchoring

## Reviewed collection

The [coverage/channel investigation](striking-policy-coverage-audit.md) found
pre-contact collapses and a residual-channel contribution in selected short
trials. Its bounded fresh-seed comparison is now completed. Both profiles used
the latest contact-history actor SHA
`2da4302cb992d5067a86c8886f72bac2cf54f2def1c41f7f87201d5f76c0bf9a`,
unchanged learned motor and free-body physics, seeds 3,000,003 through 3,000,010,
both swaps, one-second prefixes. No optimizer updates were performed.

| Profile | Red-neutral pair survivors | Blue-neutral pair survivors | Active pair survivors |
| --- | ---: | ---: | ---: |
| Original high-level actor | 14/16 | 16/16 | 13/16 |
| Zero all high-level outputs | 16/16 | 16/16 | 16/16 |

All initial observations match exactly across paired profiles. Positive-force
contacts are recorded at native substeps, and pre/post actual root distances are
stored. The original profile has 14/16, 16/16 and 12/16 eligible pre-contact
prefixes; one otherwise surviving active prefix includes an inter-body load.
All zero-output prefixes meet the strict candidate filter: complete fifty-control
prefix, both standing/finite, separation strictly above 0.9 m throughout and no
positive-force inter-body contact. This selection does not change any physical
gate. One-second survival is not certified safety or a boxing demonstration.

Zero-output trajectories are identical across the three evaluation contexts.
They are explicitly deduplicated by seed/swap/controlled role, with exact array
agreement required for duplicates: **32 unique role trajectories / 1,600 rows**,
800 per role. These are motor-preservation examples, not successful punches.
All failures and exclusions from both profiles are archived in
[the complete data review](results/striking-pilot/precontact-anchors-bc4000/analysis.json).
`dataset.json` and `anchors.npz` carry source/trace hashes and the deduplication
provenance. Diagnosed benchmark seeds 7/23/31/47 are excluded from training data.

Actual cost is **4,716 world controls**, giving **403,150 formal diagnostic
controls**. Diagnostic PPO remains 1,188,096 and cumulative supervised compute
6,000 updates / 1,536,000 presentations. Main boxing remains 1,048,576,
foundation 474,992,640, and main project total 476,041,216. No actor is adopted.

## Bounded learning decision

The repeated intervention benefit on fresh states and verified candidate data
justify a targeted matched supervised correction rather than more unchanged
imitation. Allocate **1,000 new BC updates per arm**, 2,000 total / 512,000 example
presentations, zero PPO. Both arms ordinarily warm-start the latest 4,000-update
contact branch above. Retain actor, critic, existing BC Adam moments/count 4,000,
frozen normalizer/count, exploration parameter, policy key, sampler state and
every inherited consumed/optimized/world counter. No older checkpoint is selected.

Each batch has the same 192 uniform rows from the fifteen original successful
striking demonstrations and 64 uniform rows from the verified fresh anchors.
Keep the original demonstration labels, actual event history, fixed original-PPO
row weights and channel weights. Anchor row weights are one; all channels keep
the same weights. Both arms draw identical indices from cloned source sampler
states. The changed 192/64 sampling distribution is explicit; sampling is not reset.
All snapshots save the resulting sampler. Adam remains 0.0001 with global norm
clipping at one, and normalizer/critic are not optimized.

The sole label intervention is on the 64 anchor rows. `preserve_anchor` targets
the latest source network's bounded mean predictions on those same observations,
so its initial anchor loss is zero. `zero_anchor` targets the actually recorded
zero high-level actions from the verified motor-preserving prefixes. No output
clamp, geometric teacher or new action gating is deployed at evaluation. Learned
predictions must preserve approach on their own while retaining demonstrated
striking. New snapshots have 5,000-update BC lineage and explicitly reject PPO
resume; a supervised optimizer is never silently substituted into PPO.

## Complete independent physical review

Both final actors receive all sixteen neutral-role and eight active self-play
trials on existing screen seeds 7/23/31/47, both swaps, ten seconds. They also
receive the same complete profiles on preregistered fresh seeds **4,000,003 through
4,000,006**. The unmodified latest source receives that fresh profile once as a
matched baseline. These fresh seeds are neither demonstrations nor anchor seeds;
no fresh result is used to select updates, labels or a checkpoint.

This is 120 complete ten-second episodes at most **60,000 additional diagnostic
world controls**: source fresh 24, two actors each screen 24 plus fresh 24. This
explicit finite evaluation allocation is justified by the previous failures and
need to distinguish a targeted correction from new holdout regression. Preserve
all failures and attempted controls. No further allocation is automatic.

Unchanged criteria require at least red 7/8, blue 8/8 and active 7/8 pair survival,
with own valid scoring in surviving episodes covering at least two seeds per
neutral role. Fresh distances or a safe prefix do not replace those criteria.
Even a candidate passing both profiles still supplies only one distinct checkpoint
generation; it cannot promote a stage without a second distinct consecutive
complete pass and necessary motor regression checks. The comparison arms do not
count as consecutive checkpoints. No phase or main actor is adopted automatically.

## Operations and validation

Inspect `/home/pedro/boxing-runs/striking-anchor-contact-bc4000/anchors.json` and
its complete review before learning. The hidden persistent launcher for the new
comparison is `tools/start_striking_anchor_learning.ps1`, durable state
`/home/pedro/boxing-runs/striking-anchor-learning-bc4000/learning.json`.
The sequential CPU parent refuses an existing experiment and holds a file lock;
the Windows host validates PID/start-time. Inspect active checkpoint, evaluator,
progress and logs before any recovery. Partial physical attempts remain in
per-evaluator progress files and must be reconciled before a restart. Keep the
existing monitoring schedule. Main/foundation training remain stopped.

The new tests verify strict prefix exclusions, fresh data identity/deduplication,
actual saved Adam/sampler continuation, zero initial preservation loss, nonzero
correction loss, unchanged exploration/critic/normalizer/key and PPO rejection.
Passive native trace tests verify exact dynamics and root-distance continuity.
No engineering test is counted as a physical phase pass.

## Completed comparison and continuation decision

All 120 full ten-second episodes completed. The independent
[complete review](results/striking-pilot/motor-anchoring-bc4000/analysis.json)
agrees with the parent and verifies finite learning leaves, identical final
sampler states, Adam count 5,000, unchanged critic/normalizer/key/log-standard
deviation and all inherited physical/PPO counters. Both snapshots reject PPO
resume. Eight focused selection, saved-state and passive-trace tests passed.
Native evaluation logs have no numerical or capacity warnings.

| Actor / profile | Red survivors / own surviving events | Blue survivors / own surviving events | Active survivors / surviving events |
| --- | ---: | ---: | ---: |
| Source / fresh | 8/8 / 3 | 8/8 / 2 | 6/8 / 0 |
| Preserve-anchor / screen | 6/8 / 0 | 8/8 / 2 | 6/8 / 0 |
| Preserve-anchor / fresh | 6/8 / 2 | 8/8 / 0 | 6/8 / 0 |
| Zero-anchor / screen | 7/8 / 0 | 8/8 / 1 | 5/8 / 0 |
| Zero-anchor / fresh | 8/8 / 0 | 8/8 / 1 | 7/8 / 1 |

The source has own surviving neutral strikes on two fresh seeds in each role,
but fails active survival and has no surviving active impacts. Anchoring improves
approach output preservation and some survival outcomes, while own scoring
regresses. On recorded anchors, mean absolute commands/arms fall from source
0.581/0.297 to treatment 0.070/0.052. The preserve control remains 0.582/0.298.
These are recorded-state action metrics, not direct safety guarantees. No final
actor passes the complete screen or fresh criteria, and neither has a surviving
two-way exchange in these evaluations. Do not promote either actor, deploy a
zero-output clamp, or continue increasing the same imitation objective blindly.

Actual evaluation cost is **55,319 world controls**, giving **458,469 formal
diagnostic controls**. New supervised compute is **2,000 updates / 512,000
presentations**, cumulative **8,000 updates / 2,048,000 presentations**. Diagnostic
PPO remains **1,188,096**. Main boxing/foundation counters are unchanged.
Both learned branches and complete reports are archived; this job is completed.

The next work is the [explicit BC-to-PPO warm-start contract](striking-ppo-warmstart-plan.md).
Use the latest zero-anchor treatment at 5,000 updates, SHA
`c7d6f1d5458f4767e6d7bedd98350ed8f79231de1030783fe9feb4b779d81e19`.
This retains the learned approach correction without reverting to a favorable
older source. It is a diagnostic continuation source, not an adopted main actor.
Preserve the control and all earlier snapshots. No PPO launch is permitted before
the documented optimizer, interface and accounting validation; current BC payloads
must continue to reject direct PPO resume. Main training and promotion remain stopped.
