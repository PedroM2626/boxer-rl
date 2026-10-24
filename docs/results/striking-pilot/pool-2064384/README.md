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
three snapshots with unchanged learning state; its screen is pending.
Local consumed ledgers remain authoritative while the job runs.

Inspect `/home/pedro/boxing-runs/striking-pool-2064384/comparison.json`, its active
process and both arms' progress before intervening. Do not launch another job,
rerun the one-time metadata transition, use an earlier actor or adopt a checkpoint
from partial results. On completion, independently review both complete physical
gate histories and fresh outcomes with `tools/review_striking_pool_compare.py`,
then document what worked, what failed and the bounded continuation decision.
Main adoption and general boxing certification remain false.
