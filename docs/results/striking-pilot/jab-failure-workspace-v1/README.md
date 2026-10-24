# Exact jab first-fall replay and static workspace review

Completed and independently reviewed on 2026-10-24 under the
[prospective protocol](../../../striking-jab-failure-workspace.md).
All sixteen failed v1 cases reproduced **every original native trace array**,
elapsed time, scores and phase counts exactly. The new full-body traces retain
before/after state, motor targets, requested/clipped actions and terminal contacts.
All recorded states are finite; native stderr is empty. This is a scripted
diagnostic, not a learned policy, successful punch or curriculum pass.

## Results and interpretation

The acting body triggered the first-fall predicates in all sixteen cases:
eleven low-pelvis terminations, one excessive tilt and four self-foot/leg contact
terminations. The other body triggered none. These are observed termination
predicates, not a causal isolation of the mechanism producing every fall.

Among 17,542 active arm-channel values, 19.6% of action requests were clipped;
5.7% of desired joint targets exceeded the static motor-centered residual box.
Median Cartesian tracking error was 7.0 cm in guard, 14.2 cm during extension,
12.7 cm during retraction and 9.5 cm during rest. The maximum filtered joint-target
request error was 0.828 rad. These measurements show tracking and authority limits;
they do not establish that saturation alone caused the falls.

Cloned static geometry used the recorded roots, legs, motor baseline and original
seven left-arm joint limits, with two bounded local solver starts per query:

| Recorded snapshot | Queried endpoint | Median error | Range |
| --- | --- | --- | --- |
| First guard control | Guard | <0.001 mm | <0.001 mm |
| First guard control | Full strike | 36.5 cm | 34.1–37.6 cm |
| First half-extension control | Guard | <0.001 mm | <0.001 mm |
| First half-extension control | Full strike | 11.0 cm | 9.3–12.0 cm |

The guard endpoint is geometrically matched in these sampled boxes. The full
strike endpoint is not matched by this local search. At an early guard or
half-extension state, that full endpoint is **not the interpolated command being
requested at every instant**. Opponent position and the motor baseline move later.
Two local starts do not prove global impossibility. Static geometry does not
test balance, collision safety, velocity/torque limits or dynamic reachability.
No cloned root/leg constraint was applied to a live physical episode.

## Accounting and provenance

- New physical controls: **4,048**, cumulative formal diagnostic physical
  controls **2,034,975**, exactly the declared ceiling.
- Cloned kinematic evaluations: **4,275** producer + **64** independent review =
  **4,339**, separately recorded under a 20,000 ceiling; no physical integration.
- Diagnostic PPO remains **2,163,968**; main transitions remain **476,041,216**
  combined, including 1,048,576 main boxing transitions. No optimizer/BC updates.
- The certified motor remains SHA-256
  `b39b1990945e8a374802c3a190e5ac5e026ae39e7fd5ef61c5c2b01ad291c6fc`.
  Original controller/profile and checkpoint provenance are retained in
  [audit.json](audit.json); source hashes and full-trace hashes are checked.

[analysis.json](analysis.json) contains the reviewed measurements and all terminal
records. [snapshots.json](snapshots.json) retains the exact source poses/targets;
`full/` contains all sixteen compressed full-body traces. The audit records both
solver attempts, bounds, target positions and per-query compute. Progress and
review-progress separately preserve physical and cloned-query consumption.
`native-errors.log` is intentionally empty. The immutable pre-review completion
record is preserved in [completed-before-review.json](completed-before-review.json).

Eight focused workspace/controller/full-trace tests pass. The independent reviewer
checks exact native reproduction, full-state continuity, filtered-target agreement,
snapshot provenance, finite values, terminal predicates, joint boxes and all
64 best-position results using fresh cloned forward calls. Re-running that
review spends additional geometry compute and requires separate accounting.

## Continuation decision

Reject unchanged v1 and do not use it for imitation or resume PPO from it.
The next controlled question is whether holding this guard already destabilizes
the frozen motor, or whether extension/retraction adds the decisive failure.
Prepare a prospective matched **motor-only versus guard-only** physical comparison
before redesigning the cycle. Reference existing matched v1 cases rather than
spending controls to repeat them again. This review grants no additional physical,
PPO or supervised allocation. No job is active; no gate or policy is promoted.
