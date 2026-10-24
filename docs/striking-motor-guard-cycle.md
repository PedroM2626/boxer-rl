# Preserve the learned guard between explicit extension attempts

**Completed and independently reviewed:** [result and limitations](results/striking-pilot/motor-guard-cycle-v2/README.md).
All eight pairs survive ten seconds, but there are zero legal impacts or candidate
punches. Actual consumption is 4,000 physical controls, cumulative 2,047,072.
No learner updates or positive-demo adoption. The prospective protocol follows.

Declared 2026-10-24 before new collection. The matched static guard and two-second
entry both fail all eight cases where motor-only survives. The foundation already
initializes/trains `GUARD_ARMS_BOXING` (`curriculum_env.py`); it was not trained
with hanging arms. In archived motor-only native traces, the left fist at guard
activation is about `[.26–.27,.14–.17,.09–.11]` m in its torso frame. The failed
script's `[.16,.16,.12]` target asks for an additional 10–11 cm displacement.
That offset is measured, not proof that this displacement alone causes failure.

## New controller and unchanged acceptance scope

Preserve the learned motor guard: zero head actions throughout approach, guard,
retraction and rest. At each first extension control, capture the actual left
fist in its own rotating torso frame. Use that local point as the extension's
starting reference, then the original raised-cosine .18 s extension toward the
unchanged opponent torso target. At retraction, issue zero head residual actions;
the original .8/.2 filter decays the arm override back toward the reactive motor.
No extra Cartesian guard hold or new body authority. Repeat original .35/.18/
.25/.30 s cycle timings, without stopping after a scored contact.

This is a version-2 scripted diagnostic design, not an optimized boxer. Keep
original motor/checkpoints, .45 m approach, .75 m activation, .8 rad arm box,
Jacobian damping/correction cap, physical contacts, free roots/legs and independent
gates. Only seven acting left-arm channels may be nonzero, and only during
extension. No direct leg actions, forces, assistance, actor updates or rewards.

Run seeds 7/23 × both acting corners × both swaps: eight cases, at most 500
controls each. Preserve exact archived approach prefixes. Record complete
full/native traces, actual cycle phase, active extension, entry-reference capture,
contacts, first-fall flags, tracking and clipping. Use the **unchanged** v1
candidate conjunction: acting left legal contact; onset speed >=1 m/s and
forward cosine >=.7; preceding hand excursion >=.12 m; complete post-window
with >=.06 m return toward its pre-window point; contact duration <=.10 s.
A feasible episode must survive ten seconds and contain at least two candidates.
This diagnostic is not a broader learned-boxing or guard-recovery certification.

## Bounded compute and decision

Start cumulative diagnostic physical controls **2,043,072**. New allocation at
most **4,000**, ceiling **2,047,072**, justified by eliminating the separately
failed guard override while testing an actual repeatable cycle. No reuse of
unused ramp compute. PPO 2,163,968, cloned geometry 4,339 and all main/BC counters
unchanged. No additional inference search, gradient steps or data adoption.

Use the hidden persistent launcher and existing locks. Inspect
`/home/pedro/boxing-runs/striking-motor-guard-cycle-v2/audit.json`, progress and
process before recovery. Stop on first fall per episode, native/overflow warning,
nonfinite state, source/profile or prefix mismatch, unexpected stderr or ceiling.
Preserve partial attempted-control accounting and all failed cases. Independently
verify all eight cases, timings, zero non-extension actions, source/native hashes,
unchanged candidate conjunction and actual consumption before interpretation.
If this also fails, do not extend unchanged PPO or call the poses successful
demonstrations. Review extension/whole-body compatibility using the retained
stable motor control before another prospectively bounded change. Even a pass
needs wider holdouts before using it as a positive training target.
