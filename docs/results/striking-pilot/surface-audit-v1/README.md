# Signed surfaces: recorded fists meet the guard, not legal targets

Completed and independently reviewed on 2026-10-25 under the
[prospective protocol](../../../striking-surface-audit.md). All 24 recorded
terminal-workspace snapshots and all 24 previously saved local-best poses are
included. Each pose measures the acting left-hand capsule against the ten
unchanged opposing upper-target geoms. All queried pairs exist in the physical
model; no query is capped or nonfinite. No physical trajectory was repeated.

| Pose set | Minimum legal gap: min / median / max | Legal overlap | Blocking overlap |
| --- | --- | --- | --- |
| Recorded | 7.59 / 21.13 / 41.00 cm | 0/24 | 20/24 |
| Saved local-best endpoint poses | -4.02 / 6.43 / 36.22 cm | 6/24 | 21/24 |

For each pose the legal gap is the smaller signed fist-to-head/torso surface
distance. Blocking overlap means at least one shoulder/elbow/wrist/hand gap is
nonpositive. Negative is geometric intersection, not an impact or impulse.
Recorded penetrations are small (deepest nearest blocker 0.89 mm), consistent
with soft-contact geometry; they are not a claim that simulation passes freely
through the guard. Five of the six local-best legal overlaps also intersect a
blocking surface. The remaining pose has a positive blocking gap, but its
dynamic path and stability are unverified.

The nearest overlapping blocker in recorded poses is the opposing right hand
in fifteen cases, right wrist in four and right shoulder in one. The physical
source retains an earlier positive-force left-fist/blocking event by each of
the 24 snapshot times. Those first events are kept beside each query: they do
not establish that the same contact remains active at that exact later time.
Likewise, a negative blocker distance at a hypothetical best pose does not prove
its entire path is blocked. No static overlap is a certified physical strike.

## Evidence and compute

[audit.json](audit.json) records every signed gap and nearest-point segment,
source case/control/pose hash, declared-pair flag, legal/blocking label and
historical physical block events. Independent review reconstructs all source
poses and original arm boxes, reproduces all 480 distances and segments, checks
finite sources/checkpoint provenance and consumes its own explicit query budget.
The analytic sphere regression distinguishes center distance, positive surface
separation and negative overlap, and checks no state mutation and hard limits.
Three focused geometry/workspace tests pass; the subsequent path integration
suite passes sixteen focused tests.

Producer: **48 cloned forwards / 480 signed-distance calls**. Independent review:
**48 / 480** again. Cumulative cloned queries **6,349** (previous 6,253 + 96),
separately **960** signed-distance calls. Both explicit new allocations are
exhausted. Progress is persisted before each query; native stderr is empty in
both runs. Physical diagnostic controls remain **2,051,072**, PPO **2,163,968**,
main combined transitions **476,041,216**, BC **8,000**. The previous 500 visual
render forwards remain separately recorded. No learning or gate changes.

## Continuation decision

Do not enlarge arm authority or repeat the straight dwell unchanged on the
assumption that endpoint distance alone proves insufficient reach. The signed
surfaces expose blocking interactions and some static legal overlaps inside
the original box. Test a small lateral path variation while preserving motor
guard, original endpoints, authority, all contacts and complete punch criteria.
The separately declared [6/12 cm arc comparison](../../../striking-arc-path-comparison.md)
is the next bounded physical experiment. Neither these cloned poses nor that
pending experiment authorizes positive-demo adoption or a training extension.

Follow-up: the [completed arc comparison](../arc-path-v4/README.md) rejects both
amplitudes as positive targets: zero intended-left contact/candidate punches,
survival 7/8 and 8/8. The sole legacy event comes from the other hand. Next
prospectively test the existing lateral movement command through the learned
motor; these arm-only failures do not prove the full action space infeasible.
