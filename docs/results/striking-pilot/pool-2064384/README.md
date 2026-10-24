# Opponent-pool comparison: verified integration, outcome pending

See the [prospective contract](../../../striking-opponent-pool-comparison.md)
for the hypothesis, bounded compute, unchanged physical gates and stop rules.

`integration.json` independently verifies both initial 1,024-credit checkpoints:
lossless learned/Adam/normalizer/sampler source state at conversion, finite first
updates, exact physical and optimizer counts, empty native error logs and initial
pool assignment accounting. Those evaluated integration payloads and configurations
are archived under `checkpoints/striking-validation/pool-2064384/*/integration`.
Forty-one focused tests passed before launch.

The initial four uniform draws all selected the old adversary. Those first blocks
therefore do not establish native coverage of all three policies or improved
boxing. The ordinary pool resume to +8,192 subsequently collected against all
three snapshots with unchanged learning state.
Local consumed ledgers remain authoritative while the job runs.

The [partial monitor review](monitor-20261024T1544.json) verifies the pool final
d93a0008 and both original screens: survival profiles 8/7/7 and 6/8/8, both
complete failures. Fresh survival is 8/6/5 versus source 8/6/7, with one surviving
reciprocal mirror episode versus four at source. This is not a demonstrated
improvement. The fixed-opponent control is still running; the matched comparison
and any subsequent allocation decision remain pending. Native training/evaluation
error logs are empty, frozen foundation/main hashes and certifications are intact,
and both current checkpoints are finite. No training was restarted or extended.

The monitor also records a separate read-only checker environment mistake: an
initial invocation omitted the CPU backend and emitted CUDA allocation failures.
It performed no physical collection or updates and was repeated explicitly on CPU.
The persistent jobs retained their CPU configuration and empty native error logs.
The full comparison reviewer now defaults to CPU and rejects an explicit different
backend before importing JAX, preventing the same accidental GPU reservation.

Inspect `/home/pedro/boxing-runs/striking-pool-2064384/comparison.json`, its active
process and both arms' progress before intervening. Do not launch another job,
rerun the one-time metadata transition, use an earlier actor or adopt a checkpoint
from partial results. On completion, independently review both complete physical
gate histories and fresh outcomes with `tools/review_striking_pool_compare.py`,
then document what worked, what failed and the bounded continuation decision.
Main adoption and general boxing certification remain false.
