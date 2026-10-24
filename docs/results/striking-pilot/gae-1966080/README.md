# Longer GAE native integration, comparison in progress

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
or a claim of improvement. Matching the control's initial sampled batch remains
pending. The source's independent fresh 26M set has survival 8/7/6 and fails
its complete gate. Both arms and their complete evaluations still must finish.

Inspect `/home/pedro/boxing-runs/striking-gae-1966080/comparison.json` and child
processes before intervention. The live job may have advanced beyond the saved
integration snapshot. The [bounded protocol](../../../striking-gae-comparison.md)
retains all criteria and compute ceilings; no adoption, promotion or automatic
extension follows from this integration check.
