# Matched PPO continuation from the latest BC5000 actor

**Completed:** final physical reports, actor snapshots, training metrics and
`analysis.json` are now archived. Both arms fail; no actor is adopted or PPO
extension granted. See the [final review](../../../striking-ppo-anchor-review.md).

The sections below retain the original launch and integration evidence.
The comparison started on 2026-10-23 at 11:32 UTC under the
[reviewed continuation contract](../../../striking-ppo-warmstart-plan.md).

The migrated checkpoint retains the latest zero-anchor BC actor, its 5,000-update
Adam history, and the unchanged critic with its recovered 28,672-update history.
Migration consumes no physical transitions or optimizer updates. Hashes and exact
prefix checks are recorded in `migration-integrity.json`; the complete converted
payload is in `checkpoints/striking-validation/ppo-anchor-bc5000/migration/`.
`validation.json` records the 19 passing focused tests and foundation/main audit.

The source is first evaluated in 24 complete trials using new seeds
5,000,003–5,000,006. Each matched active PPO arm then receives 131,072 new learner
credits, including a retained initial 1,024-credit integration block. Auxiliary
approach anchoring weights are 0 and 0.02; physical parameters and scoring gates
are unchanged. Final evaluation comprises the original screen and the same fresh
seeds for each arm: 120 total source/final episodes, at most 60,000 evaluation world
controls. PPO collection is recorded separately (at most 131,072 world controls).

Before launch, diagnostic PPO consumption was 1,188,096 learner credits and formal
diagnostic evaluation consumption was 458,469 world controls. Supervised compute
remains 8,000 cumulative updates, with 5,000 in the selected actor's lineage. Main
foundation/boxing consumption remains 474,992,640 / 1,048,576. Historical lineage
offsets in diagnostic snapshots are not new main consumption.

Live durable state is
`/home/pedro/boxing-runs/striking-ppo-anchor-bc5000/comparison.json`.
Inspect its active child and per-arm `progress.json`, checkpoint hash, initial-block
proof and error logs before another intervention. `launch-state.json` is an initial
snapshot, not a live counter. No actor is adopted and no stage is promoted by this
launch. Review all unchanged complete physical criteria after both arms finish;
survival or an optimizer integration check alone is not successful boxing.

## First physical integration check

The `ppo_only` arm completed its retained 1,024-credit / 512-world-control block
at 11:34 UTC. The finite checkpoint SHA is
`7313b4d4b7fd7355ba898a1d4eea5e515caa85c16e16373480c71002a8125d6b`;
actor/critic Adam counts are exactly 5,016 / 28,688. Both initial-block and resumed
native error logs were empty at review. The remaining 130,048-credit invocation
is running from this checkpoint, without a rollback. Its snapshot is archived
under `checkpoints/striking-validation/ppo-anchor-bc5000/ppo-only-initial-block/`.

The 24 fresh source trials consumed **11,287 evaluation world controls**. Pair
survival was 7/8 for neutral role 0, 8/8 for neutral role 1 and 6/8 for self-play.
Each neutral role scored in only one surviving seed; the complete criteria fail.
The source baseline is a fixed comparison reference, not an advance. At the
verified block boundary, cumulative diagnostic PPO is **1,189,120** and formal
evaluation controls **469,756**, plus **512** separately recorded new PPO collection
world controls. Live training continues to add diagnostic consumption beyond this
archived boundary. No completed arm comparison or final certification exists yet.
