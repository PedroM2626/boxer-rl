# Opponent-pool comparison: completed, no demonstrated improvement

The [independent review](analysis.json) accounts for both full 16,384-credit arms,
all 168 episodes / 78,563 evaluation controls, every block hash, finite checkpoint,
optimizer count, immutable opponent fingerprint, credit ledger and complete gate.
Native training/evaluation error logs are empty; foundation and main policies are
unchanged. No policy is adopted or promoted.

| Policy | Original +8,192 | Original +16,384 | Fresh neutral-0 / neutral-1 / mirror | Fresh surviving reciprocal mirror episodes |
| --- | --- | --- | --- | --- |
| Source bea8147f | historical source milestone | historical source milestone | 8 / 6 / 7, fail | 4 |
| Pool d93a0008 | 8 / 7 / 7, fail | 6 / 8 / 8, fail | 8 / 6 / 5, fail | 1 |
| Fixed control 3a34aa85 | 7 / 8 / 7, pass | 7 / 8 / 6, fail | 8 / 6 / 6, fail | 2 |

All original/fresh figures use the unchanged full profiles and both spawn swaps.
The pool has one fewer fresh paired survivor than control and two fewer than
source; this run does not demonstrate a diversity benefit. It is not proof that
all opponent pools fail. Exposure was uneven: the newest-source adversary received
839 / 8,192 active credits, all in learner corner 1. Keep that limitation visible.
Version-13 history contains fail/fail, streak zero. The version-12 canonical history
now appends the control's pass/fail to the previous eight records: ten records,
current streak zero. The earlier frozen-source readiness milestone is preserved.

The [continuation decision](decision.json) freezes both actual finals and grants
zero additional PPO or physical controls. The user's visual observation also
exposed a separate [contact-versus-punch objective gap](../../../striking-punch-quality-audit.md).
Measure punch mechanics before another objective/compute allocation. Qualifying
contact counters must not be presented as proof of recognizable boxing punches.

Diagnostic PPO is 2,163,968, formal physical simulation 2,015,379. Main boxing
1,048,576, foundation 474,992,640, combined main 476,041,216 and standalone BC
8,000 remain unchanged. No training job is active and unused evaluation allowance
is not automatically resumed. The following integration/monitoring notes preserve
the earlier trajectory rather than describe a currently running job.

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
