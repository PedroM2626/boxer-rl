# Completed BC5000-to-PPO comparison and regression investigation

**Follow-up completed:** saved-Gaussian evaluation and the short gradient assay
confirm no recovered stable striking. See the [result review and distinct bounded
demonstration-retention continuation](striking-retention-continuation.md). The
launch plan below is historical; the PPO/anchor-only objective is not extended.

Both matched arms finished their fixed 131,072 new learner credits and all 120
source/final full physical episodes. Neither final actor passes the unchanged
complete criteria. **No further PPO is allocated**, no actor is adopted and no
curriculum stage advances. All four foundation certifications remain valid;
main foundation/boxing counters remain 474,992,640 / 1,048,576.

## Independent complete physical results

Each entry below is pair survival out of eight ten-second episodes, both swaps.
Neutral-role scoring counts only the learner's actual legal fist events in
surviving pairs. The full reports retain impacts in failed episodes separately.

| Policy / sample | Neutral role 0 | Neutral role 1 | Active self-play | Surviving active events (red / blue) |
| --- | --- | --- | --- | --- |
| Latest BC5000 source / fresh | 7/8 | 8/8 | 6/8 | 2 / 2 |
| PPO only / screen | 8/8 | 8/8 | 8/8 | 0 / 1 |
| PPO only / fresh | 7/8 | 7/8 | 6/8 | 0 / 0 |
| PPO + 0.02 anchor / screen | 5/8 | 6/8 | 5/8 | 0 / 0 |
| PPO + 0.02 anchor / fresh | 5/8 | 5/8 | 6/8 | 0 / 0 |

The source scores once in each neutral role, in only one surviving fresh seed per
role. Both trained actors have **zero surviving own neutral events in all four
role/sample checks**. Neither trained actor has a surviving two-way active
exchange. PPO-only fresh active play has six events in one episode that ends in
a fall; these do not count as stable boxing. The original screen's survival
improvement is not reproduced on fresh seeds. The small approach-anchor loss
provides no measured benefit in this comparison and is rejected for continuation.

## Verified learning state and compute

All 128 PPO block losses per arm are finite, final payloads and optimizer moments
are finite, hashes match, and native training/evaluation error logs are empty.
Both independently counted Adam states reach **7,048 actor / 30,720 critic**.
Both arms consume exactly 131,072 credits / 65,536 world controls from the retained
lineage offset, retain the original BC sampler and finish with the same auxiliary
sampler stream. There is no rollback or duplicated allocation. Complete live
foundation histories still match archived histories (29/41/13/2 records), with
two distinct consecutive passes; their hashes and finite payloads verify. The
stopped main boxing actor's hash is unchanged.

Final actor SHA-256 values:

- PPO only: `e3e01b71de64e3d26a431ba87d484a45d349621df5461c274fb7e394b935ef4e`.
- PPO + anchor: `3a5bae07922ad435ead3ee027263d505b13967119468e3ef13750b2ddcab90fe`.

This job adds **262,144 diagnostic PPO credits**, **131,072 separately recorded
PPO collection world controls**, and **53,652 evaluation world controls**.
Cumulative diagnostic PPO is **1,450,240** and formal evaluation is **512,121**
world controls before the next audit. Cumulative supervised compute stays at
8,000 updates / 2,048,000 example presentations. Historical lineage offsets in
snapshots do not increase main counters.

Final policies/configuration, raw trials, training metrics and reproducible
analysis are archived in
`checkpoints/striking-validation/ppo-anchor-bc5000/{ppo_only,ppo_anchor}/` and
[the result directory](results/striking-pilot/ppo-anchor-bc5000/analysis.json).
Run `tools/review_striking_ppo_anchor.py --input <durable-comparison-directory>`
to repeat integrity/accounting checks and offline measurements without physics
or optimizer updates.

## Offline localization and bounded next action

On the same 7,500 verified demonstration observations, unweighted action MSE grows
from **0.001629** in the source to **0.016180** (PPO only) and **0.021553** (anchor),
approximately 10x and 13x. On 1,600 verified approach-anchor rows, MSE grows from
0.006546 to 0.016522 / 0.022600. These are descriptive fixed-data diagnostics,
not new gates or estimates of performance on unseen states.

Applying the source normalizer to the final actors leaves demonstration MSE at
0.016469 / 0.021496. Applying the final normalizers to the unchanged source actor
gives 0.001774 / 0.001763. This crossed offline check localizes most measured
action drift to actor parameters rather than normalization; it does not prove
the cause of every physical fall. Learned pre-tanh log standard deviations remain
approximately -2.05 to -1.90, so training still samples actions while the promotion
evaluations use the deterministic mean. Training contact totals cannot substitute
for the complete saved-policy evaluations.

Before changing the objective or granting more PPO, run **one bounded saved-Gaussian
diagnostic** of the source and both final policies on seeds 5,000,003–5,000,006,
both swaps, all three partner profiles, ten seconds: **72 episodes, at most 36,000
evaluation world controls**, no optimizer steps. Use the saved standard deviation,
iid perturbations with scale 1 and hold 1, unchanged physical scoring and falls.
The evaluator's matching per-trial random stream is retained across policies.
This distinguishes sampled-policy behavior from deterministic-mean regression;
it cannot certify a phase or justify adoption on its own. These seeds are observed
diagnostic history now, not a new independent holdout. Do not tune noise to hits.

The hidden persistent job started at **2026-10-23 12:15 UTC**, launched by
`tools/start_striking_postppo_sampling_audit.ps1`. Its durable state is
`/home/pedro/boxing-runs/striking-postppo-sampling-bc5000/sampling.json`.
Inspect that process, active evaluation/progress, logs and hashes before another
launch. Preserve the existing monitoring schedule. When it finishes, archive all
three policies' unchanged full sampled results, compare them with these fixed
mean results, and decide whether subsequent work should address policy retention,
credit assignment or demonstrated strike coverage. No budget extension is automatic.
