# Actor KL-guard integration and ongoing comparison

The bounded comparison completed normally and is independently reviewed.
`analysis.json` and `completed-comparison.json` preserve the complete result;
integration snapshots below remain historical evidence. No policy is adopted.

The first 1,024-credit native block has checkpoint SHA-256
`751ed28b3b320f57078527fc6954422e27dbce8823d97247d5700d6679c4e3c9`,
archived under `checkpoints/striking-validation/kl-guard-1949696/integration-1024`.
Its actor ledger is `[6]`: actor Adam advances from 12,936 to 12,942, while all
sixteen critic updates advance its clock from 36,608 to 36,624. Consumed and
optimized learner counters are both 1,950,720; physical world controls are
1,328,128. Payload validation confirms finite state and the original frozen
opponent fingerprint. The parent verified that every source leaf was retained
at the initial transition. At that integration snapshot the matching control batch check was pending;
the completed review below now verifies the exact match.

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

The initial protocol allocated at most 32,768 diagnostic PPO credits and 84,000
evaluation controls. Both arms and their independent fresh gates are now reviewed
below. No promotion or main adoption follows. See
[the protocol](../../../striking-kl-guard-comparison.md).

## Complete result

| Policy | Original +8,192 | Original +16,384 | Fresh final survival | Fresh surviving mirror red / blue events | Fresh reciprocal episodes |
| --- | --- | --- | --- | --- | --- |
| Source | — | — | 8/7/8 | 5 / 16 | 4 |
| KL guard | FAIL 6/8/6 | FAIL 5/8/5 | 6/8/5 | 6 / 4 | 3 |
| Control | PASS 7/8/7 | FAIL 4/5/3 | 7/6/5 | 5 / 7 | 4 |

All fresh complete sets fail. The control's final mirror survival of 3/8 triggers
the predefined severe-regression rule at the end of its allocation. No additional
training is launched. Guarding actor drift is not validated as a remedy, although
its original final survival exceeds control on this one sample.

The initial collected batches match exactly (SHA-256 in `analysis.json`). The
guarded actor applies 104 total Adam steps, versus 256 control; both critics apply
256. Final actor/critic clocks are 13,040/36,864 and 13,192/36,864 respectively.
Every one of 32 block payloads is finite and preserves source/frozen-opponent,
mode/corner and optimizer prefixes. All 27 native error logs are empty. Foundation
histories remain 29/41/13/2 and the main boxing payload hash is unchanged.

Actual new PPO/physical collection is 32,768; formal evaluation is 74,819 controls
across 168 episodes. Integrated retention presentations are 131,072 separately.
Diagnostic PPO is now 2,016,512; formal evaluation 1,633,434. No steps or failed
episodes are discarded. The eleven-record original version-8 history has current
streak zero; version 11 has two failures. No family pass is inherited.

Next: [passively replay these same final resets](../../../striking-kl-failure-trace.md)
to locate the new failures relative to physical contact and scoring before another
training change. At most 36,000 additional evaluation controls, zero optimizer
updates. No rollback, weaker gates, adoption or automatic extension.
