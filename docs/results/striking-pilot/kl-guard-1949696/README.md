# Actor KL-guard integration and ongoing comparison

The bounded comparison is still running. This directory preserves an immutable
integration snapshot, not a completed comparison or a policy adoption.

The first 1,024-credit native block has checkpoint SHA-256
`751ed28b3b320f57078527fc6954422e27dbce8823d97247d5700d6679c4e3c9`,
archived under `checkpoints/striking-validation/kl-guard-1949696/integration-1024`.
Its actor ledger is `[6]`: actor Adam advances from 12,936 to 12,942, while all
sixteen critic updates advance its clock from 36,608 to 36,624. Consumed and
optimized learner counters are both 1,950,720; physical world controls are
1,328,128. Payload validation confirms finite state and the original frozen
opponent fingerprint. The parent verified that every source leaf was retained
at the initial transition. The matching control batch check remains pending.

Exact rollout Gaussian KL is .0257323 before normalization and .0258045 after
normalization. The .02 target stops subsequent actor steps; it does not undo the
crossing step or impose a hard bound. The passive geometry records and native
metrics are retained alongside the validation JSON. These are implementation
checks, not evidence of better boxing.

The source's untouched 25M-seed physical evaluation consumes 11,596 controls and
has paired survival 8/7/8 across neutral-0, neutral-1 and mirror profiles. Its
complete gate fails. The first guarded checkpoint at +8,192 has survival 6/8/6,
four reciprocal surviving mirror episodes, and also fails its complete gate.
That screen consumes 11,071 controls. Every raw report and empty native error
log is retained; no threshold, profile or failed episode is removed.

Inspect `/home/pedro/boxing-runs/striking-kl-guard-1949696/comparison.json` and its
processes before intervention. The live job may have advanced beyond this
historical snapshot. Complete both bounded arms and independently review their
full screens and fresh results before deciding any continuation. Maximum new
diagnostic PPO is 32,768 and evaluation is 84,000 controls; no promotion or main
adoption follows automatically. See [the protocol](../../../striking-kl-guard-comparison.md).
