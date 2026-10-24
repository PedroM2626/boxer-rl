# Native punch-motion audit: completed, contact criteria are insufficient

The [independent review](analysis.json) verifies eight selected known episodes,
all 4,000 attempted physical controls, source/reference/trace hashes, finite native
arrays and exact equality of every original archived trial field. Survival,
scores, event timing, impulses, distances and residual statistics are unchanged.
Native child error logs are empty. Fifteen focused tests pass, including exact
native trajectory equivalence and rejection of rigid body translation/rotation
as arm motion. No optimizer updates, main adoption or promotion occurred.

| Selected policy | Legacy contacts | Onset speed below 1 m/s | Forward cosine below .7 | Both reference components met | Hand displacement in preceding 250 ms, min–max |
| --- | --- | --- | --- | --- | --- |
| Published video source bea8147f | 6 | 3 | 6 | 0 | 5.3–9.1 cm |
| Pool final d93a0008 | 2 | 1 | 1 | 0 | 3.2–5.5 cm |
| Fixed control final 3a34aa85 | 3 | 2 | 2 | 0 | 3.4–6.7 cm |
| Earlier long-GAE 95dbdff0 | 8 | 5 | 6 | 0 | 3.1–10.3 cm |

Across these samples, **0/19** legacy contact events simultaneously meet the
old default 1 m/s speed and .7 forward-cone component references at onset.
Eleven are below the speed reference and fifteen below the direction reference.
The two repetitions of the published video have a median 7.7 cm hand excursion,
elbow excursion .10–.22 rad (about 6–12 degrees), and .50–.67 N s accumulated at
the scoring event. Those are measured short motions, consistent with the user's
observation. They do not establish confident straight-punch mechanics.

The references are descriptive comparisons, not a replacement gate or universal
punch classifier. A hook need not advance like a jab; a fist can decelerate;
the pre-window does not cover an entire preparation; and five 500 ms post-event
windows are censored by round end. Return toward a pre-window hand position is
not proof of guard recovery. This selected eight-episode sample is not an unbiased
policy comparison or generalization estimate. The earlier long-GAE model is still
a paired-pilot policy: original `boxer/env.py` video actors and directed teachers
were not replayed in this allocation.

## Decision and next engineering work

Keep all current policies frozen. **No unchanged PPO extension** and no arbitrary
threshold increase. Preserve historical contact gates and their pass/fail records;
none of these measurements retroactively certifies punches or changes those gates.

Implement a separate versioned evaluation of committed strike motion, using
measured speed/direction, arm travel and recovery rather than a contact counter
alone. Audit archived strike demonstrations and applicable older motion criteria
to establish positive as well as passive/leaning/lifting examples. Validate that
the frozen motor and available arm authority can execute the intended movement.
Only then specify a prospective objective comparison with explicit state migration,
independent physical criteria and bounded PPO. The audit supports changing what
is measured and taught, not merely adding adversaries or training longer for taps.

The [original protocol](../../../striking-punch-quality-replays.md) remains the
pre-collection contract. Formal diagnostic consumption is now **2,019,379**, at
this audit's ceiling. Diagnostic PPO remains **2,163,968**; main boxing 1,048,576,
foundation 474,992,640, combined main 476,041,216 and standalone BC 8,000 are
unchanged. No training/replay job is active and no extra allocation is implicit.
