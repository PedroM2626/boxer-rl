# Smooth entry into the same guard: rejected as a sufficient repair

Completed and independently reviewed on 2026-10-24 under the
[prospective protocol](../../../striking-guard-ramp.md). Eight new physical
conditions change only the target entry: a two-second smooth transition from
the current own-torso-relative fist point into the identical original guard.
The final guard, frozen motor, residual authority, filter and free physics remain.

| Matched diagnostic | Paired ten-second survivors |
| --- | --- |
| Previously reviewed motor only | 8/8 (four unique reset conditions) |
| Previously reviewed immediate guard | 0/8 |
| New two-second guard entry | 0/8 |

All eight new first falls are on the acting body: four low-pelvis and four
self-foot/leg-contact terminations. The other body never triggers a first-fall
predicate. Falls occur at 4.448–5.750 seconds. All cases reach the end of the
two-second **command** ramp, followed by 0.668–2.170 seconds commanding the final
guard before falling. The report field `held_final_guard_seconds` measures this
command duration; it does not claim that an exact physical guard pose was held.

Median target tracking error falls from about 5.0 cm in immediate guard to
**0.99 cm** with the ramp. Active arm-channel clipping is 14.6%, versus 15.6%
before. Better hand tracking does not produce paired survival. Five legacy legal
contact events are not certified punches. Entry smoothing alone is rejected as
a sufficient repair on these conditions. These results do not establish which
final posture, arm dynamics or learned motor response causes each fall, and do
not prove all possible smooth transitions would fail.

## Verification, compute and completion error

The run used **1,942 new physical controls**, cumulative **2,043,072**, below
its 2,045,130 ceiling. Unused controls expire. Diagnostic PPO remains 2,163,968,
cloned kinematics 4,339, main combined transitions 476,041,216 and standalone
BC 8,000. No optimization, imitation, external force or policy adoption.

All eight approach prefixes match the archived static-guard cases exactly.
Independent review checks full-state continuity, finite arrays, source/checkpoint
hashes, original action channels/filter, terminal predicates, native score events,
captured entry position, complete smoothstep target trajectory, ramp clock and
actual control accounting. Ten focused tests pass. Native physics stderr is empty.

After all eight episodes had already been saved, a final bookkeeping assertion
incorrectly expected sixteen, inherited from the earlier two-mode comparison.
The job stopped and its failed state/host traceback are preserved in
[failed-before-completion-recovery.json](failed-before-completion-recovery.json)
and `completion-error.log`. The assertion now derives expected count from the
configured modes/seeds/corners/swaps. The narrowly scoped independent review
verified every already-produced episode before marking completion; **zero new
physical controls** or optimizer updates were spent on recovery. No failed
physical trial was discarded or rerun.

[analysis.json](analysis.json) contains all matched outcomes and limitations.
[audit.json](audit.json), progress, `full/` and `native/` preserve all traces,
source hashes and accounting. `native-errors.log` is intentionally empty.

## Concrete continuation decision

Do not repeat v1, immediate guard or this ramp unchanged; do not train imitation
from their failed trajectories or extend PPO with the same target. The original
arm-only guard already fails where the unchanged motor-only control survives,
and slowing entry improves hand tracking without fixing the body. Investigate
guard geometry and its compatibility with the learned whole-body response
before another physical redesign: retain actual motor inputs/leg actions and
compare against the matched stable motor-only traces. A coordination change
needs its own prospective bounded comparison and foundation-retention checks;
earlier body-authority experiments did not demonstrate robust benefit. No new
training allocation or automatic motor unfreezing follows from this result.

Follow-up: source inspection confirms the foundation already has a learned
compact guard. The [motor-guard cycle v2](../motor-guard-cycle-v2/README.md)
preserves it between extension attempts: 8/8 pair survival, but still zero
physical impacts and candidate punches. This is a stable scripted control,
not a trained boxer or a positive punch target.
