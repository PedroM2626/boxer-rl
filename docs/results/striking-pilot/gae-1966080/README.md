# Reviewed longer-GAE comparison and native integration

**Completed:** `analysis.json` and `completed-comparison.json` now preserve the
full independently reviewed comparison. Integration notes below are historical.

The included first 1,024-credit block is independently verified, with explicit
version-12 configuration and training lambda .99. Checkpoint SHA-256 is
`c6463d90f2864088d678ea4562e2241bea9037ca2775a61263b57e1f05b01985`,
archived in `checkpoints/striking-validation/gae-1966080/integration-1024`.
Consumed/optimized learner credits are 1,967,104 and world controls 1,344,512.
Actor and critic Adam clocks advance exactly sixteen each, to 13,208/36,880.
All learned state is finite; original frozen-opponent fingerprint and source
leaf retention are verified. There is no optimizer reset or earlier actor.

The native block records two red and one blue score while collecting, without
a completed round. This is stochastic training evidence, not a physical gate
or a claim of improvement. Matching the control's initial sampled batch was pending at this snapshot;
the completed review below verifies the match. The source's independent fresh 26M set has survival 8/7/6 and fails
its complete gate. Both arms and their complete evaluations still must finish.

Inspect `/home/pedro/boxing-runs/striking-gae-1966080/comparison.json` and child
processes before intervention. The live job may have advanced beyond the saved
integration snapshot. The [bounded protocol](../../../striking-gae-comparison.md)
retains all criteria and compute ceilings; no adoption, promotion or automatic
extension follows from this integration check.

## Full physical result

| Policy | Original +8,192 survival | Original +16,384 survival | Fresh final survival | Fresh mirror red / blue surviving impacts | Fresh reciprocal mirror episodes |
| --- | --- | --- | --- | --- | --- |
| Source 24e23288 | — | — | 8/7/6 | 6 / 8 | 4 |
| Lambda .99 | 7/7/7 | 8/7/8 | 6/7/8 | 13 / 16 | 6 |
| Lambda .95 control | 6/6/7 | 5/8/6 | 7/6/7 | 13 / 16 | 4 |

Every complete original and fresh gate fails. In particular neutral-1 requires
8/8, not 7/8. The active-play improvement cannot waive it. Fresh aggregate
survival 21/24 matches source, versus 20/24 control; this is a specific exchange
improvement, not proven overall robustness. Both distinct longer-GAE original
checkpoints have six reciprocal mirror episodes; its original mirror survival
improves from 7/8 to 8/8. The latest endpoint is 95dbdff0; the control is 5c8d1fca.
Full hashes and immutable gate payloads are archived alongside their reports.

All 32 payloads retain exact consumed/world, actor/critic Adam, neutral/mixed
credit, finite-state and frozen-opponent prefixes. Both final payloads have
1,982,464 consumed/optimized learner credits, 1,359,872 world controls and Adam
13,448/37,120. All initial source leaves and first sampled observations/actions/
log-probabilities match. GAE targets intentionally differ. All 27 native error
logs are empty; all four foundation histories and main boxing hash are unchanged.

Actual new PPO/physical collection is 32,768; formal evaluation is 79,297 controls
across 168 episodes; integrated retention presentations 131,072 are separate.
Diagnostic PPO is 2,049,280 and formal evaluation 1,745,950. Version 8 has thirteen
unique original records and current streak zero; version 12 has two failures.
No family passes, promotion or main adoption are inherited.

Next: one [bounded ordinary continuation](../../../striking-gae-continuation.md)
of latest 95dbdff0, no transition/reset or earlier actor restoration. At most
16,384 credits, complete screens at +8,192/+16,384, fresh 27M source/final full
sets, at most 48,000 evaluation controls. Stop at the first screen on severe
regression or no source-relative full-screen progress. Diagnostic PPO ceiling
2,065,664, formal ceiling 1,793,950. Preserve every failed gate and unused budget;
no automatic adoption, promotion or further extension.
