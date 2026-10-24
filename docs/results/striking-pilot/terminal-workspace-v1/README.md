# Recorded terminal workspace: endpoint remains unmatched in local searches

Completed and independently reviewed on 2026-10-24 under the
[prospective protocol](../../../striking-terminal-workspace.md). All 24 planned
snapshots are included: first dwell's last control, lowest-error dwell control
and last dwell's last control from each of eight reviewed physical cases.
No physical episode or learner update was added.

| Distance to the commanded full endpoint | Minimum | Median | Maximum |
| --- | --- | --- | --- |
| Actual recorded fist | 22.7 cm | 34.9 cm | 56.6 cm |
| Best of two bounded local static starts | 11.6 cm | 22.9 cm | 52.9 cm |

Each search uses the recorded motor-centered +/- .8 rad left-arm box intersected
with original joint limits. Source qpos, baseline, target, bounds, both solver
attempts and best poses remain in [audit.json](audit.json). Roots, legs and other
joints are fixed only in disposable clones for geometry, never in live physics.
The actor and original physical traces remain untouched.

The sampled endpoint is not matched by these local searches. This is evidence
against solving the current endpoint problem merely by another unchanged dwell,
but **not a global impossibility proof**. A static solution would not establish
collision safety, balance or dynamic reachability either. The command is a point
near the opposing torso center with a height offset; distance to that point is
**not collision-surface separation**. Contact may occur without reaching the
commanded point. Do not interpret this table as proof that every legal physical
contact is unreachable or as permission to enlarge arm authority blindly.

## Verification and compute

Producer: **1,890 cloned kinematic evaluations**. Independent review: **24** new
cloned forwards verifying each best position/error, source snapshot selection,
joint coordinates/bounds and source telemetry. Cumulative cloned evaluation
consumption is **6,253** (prior 4,339 + 1,914), below the separately declared
10,339 ceiling. Unused compute expires at review; it is not an automatic next
allocation. Progress is durable before every producer/review query. Native stderr
is empty; no nonfinite values or unexpected source changes were found.

Physical diagnostic controls remain **2,051,072**, PPO **2,163,968**, main combined
transitions **476,041,216**, BC **8,000**. [analysis.json](analysis.json), progress,
review progress and the pre-review completion record retain all compute evidence.
Re-running independent FK verification costs new queries and requires accounting.

## Concrete continuation decision

Reject unchanged dwell/timing repetition and do not resume imitation or PPO from
these failed punch targets. Next inspect legal fist-to-torso/head **surface** gaps
at the recorded and locally optimized poses, together with blocking/inter-arm
contacts, before selecting a bounded surface-target or authority intervention.
Keep the learned guard, free legs, unassisted physics and complete independent
criteria. No policy promotion, motor unfreezing or new allocation is declared.
