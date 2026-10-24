# Matched static guard versus motor-only result

Completed and independently reviewed on 2026-10-24 under the
[prospective protocol](../../../striking-guard-control-comparison.md).
The frozen motor, free-body physics, approach profile and original arm bounds
are unchanged. The intervention holds the original v1 guard instead of extending
and retracting. All exact archived approach prefixes reproduce.

| Condition | Paired ten-second survivors | Acting-body first falls | Other-body first falls |
| --- | --- | --- | --- |
| Motor only | 8/8 | 0 | 0 |
| Static guard only | 0/8 | 8 | 0 |
| Historical matched jab v1 | 0/8 | 8 | 0 |

Motor-only corner labels duplicate **four unique reset conditions**, not eight
independent baseline samples. Guard-only falls occur at 4.204–6.638 seconds:
four low-pelvis and four self-foot/leg-contact terminations. Its two legal contact
events are not punch certification; six cases fall without any scored contact.
Median guard tracking error is 5.0 cm and 15.6% of active arm-channel action
requests clip. All traces are finite and native stderr is empty.

This matched intervention shows that the static-guard controller is sufficient
to reproduce failure on these conditions where the motor-only control survives.
Extension/retraction is not necessary for these failures. It does not isolate
the separate contribution of final guard geometry, transient entry, arm torques,
motor response or body coordination, and does not prove every learned fall has
the same cause. Two seeds and ten-second scripted episodes do not certify boxing.

## Accounting, evidence and decision

The complete experiment used **6,155 additional physical controls**, cumulative
**2,041,130**, under its declared 2,042,975 ceiling. Unused controls expire at
review. Diagnostic PPO remains 2,163,968, separate cloned geometry 4,339 and main
combined transitions 476,041,216. No optimization, imitation or policy adoption.

[analysis.json](analysis.json) contains every matched case, terminal predicate
and summary. [audit.json](audit.json) retains source hashes and all full/native
trace hashes; `full/` and `native/` archive every episode. Progress preserves
actual control consumption and `native-errors.log` is intentionally empty.
The independent review checks all cases, exact approach-state continuity,
intended action channels, filtered-target agreement, native events, finite state,
duplicate baseline identity, source certification and accounting with zero new
physical integration. Nine focused tests passed before collection.

Reject this static guard as a positive demonstration. Do not resume PPO or
imitation from it, relax gates or repeat the failed cycle unchanged. The concrete
next question is whether entering the **same** guard gradually reduces failure.
A separately declared two-second smooth-entry probe will isolate that change
before changing the final guard geometry or adapting the motor. No claim that
the ramp works follows from this result.

Follow-up: the [completed smooth-entry review](../guard-ramp-v1/README.md) also
fails all eight cases despite improved hand tracking. Preserve this earlier
decision as the reason for that test; entry smoothing is now rejected as a
sufficient repair.

Further source/array review confirms the foundation already carries a compact
guard. [learned-guard-offsets.json](learned-guard-offsets.json) measures the extra
10.3–11.4 cm displacement requested by this script using hash-checked motor-only
native frames; it adds no simulation and does not isolate causality.
The [motor-guard cycle v2](../motor-guard-cycle-v2/README.md) subsequently retains
the learned guard and filtered return: 8/8 survive, but no physical strikes occur.
