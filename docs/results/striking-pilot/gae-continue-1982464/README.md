# Reviewed ordinary longer-GAE continuation: pass followed by regression

The full 96 episodes / 46,422 evaluation controls and sixteen PPO blocks are
independently reviewed. All source/gate/final reports, both exact gate models,
metrics, inherited configuration and empty native error logs are archived.

| Checkpoint / evaluation | Neutral 0 survival | Neutral 1 survival | Mirror survival | Surviving mirror red / blue impacts | Reciprocal mirror episodes | Complete gate |
| --- | --- | --- | --- | --- | --- | --- |
| Source / new 27M | 8/8 | 8/8 | 6/8 | 7 / 8 | 4 | FAIL |
| +8,192 / original | 8/8 | 8/8 | 8/8 | 9 / 6 | 3 | PASS |
| +16,384 / original | 4/8 | 5/8 | 8/8 | 5 / 5 | 1 | FAIL |
| Latest final / new 27M | 7/8 | 7/8 | 8/8 | 13 / 14 | 4 | FAIL |

The intermediate full pass has neutral own-strike seed breadth three/four and
passes every unchanged criterion. It is followed by a complete failure. The
version-12 original history is **fail/fail/pass/fail**, current pass streak zero.
No two consecutive complete passes, independent fresh pass, promotion or main
adoption. The favorable intermediate is preserved, not restored for training.

Passing SHA-256:
`fa8080110edee9b73f2272f4a8d6f94bebc640a0fd56b04f2e66c8ce44c12548`.
Latest final SHA-256:
`3052e0f69fcadc823be14069ed462ee3057ae75a8028cb2d8aa5bf3615aa48b5`.
Latest final consumed/optimized learner credits 1,998,848, world controls
1,376,256, actor/critic Adam 13,704/37,376. All sixteen payloads retain exact
optimizer, mode/corner and frozen-opponent prefixes with finite learned state.
All fourteen native error logs are empty. Foundation histories remain 29/41/13/2
with valid payload hashes, and the stopped main boxing hash remains unchanged.

Actual new PPO/physical collection is 16,384, with 65,536 integrated retention
presentations separately. Diagnostic PPO now totals **2,065,664**; formal diagnostic
simulation **1,792,372**. Main/foundation and standalone BC counters are unchanged.
No failed episode, update interval or source counter is discarded.

Decision: stop unchanged PPO. Run a [bounded frozen gradient/critic assay](../../../striking-gae-gradient-audit.md)
on source, passing checkpoint and latest final, preserving all saved policy state.
At most 12,288 additional physical controls, zero optimizer/supervised updates.
New formal diagnostic ceiling 1,804,660; no PPO extension, training rollback,
weaker gates, main adoption or automatic follow-on allocation.
