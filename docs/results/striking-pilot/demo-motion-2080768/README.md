# Retained demonstration motion: complete exact replay, no validated strike target

The [independent review](analysis.json) covers every file in the current
fifteen-trajectory retention manifest. All 7,500 pre-action observations and
saved actions replay exactly with the same survival, timing, native events and
impulses. All traces are finite, native error output is empty, and frozen
foundation/main certifications and hashes are retained. The source is the old
146-input geometric feasibility run; no live teacher, PPO or BC updates were used.

All fifteen acting-corner contact events are below **both** inherited 1 m/s speed
and .7 forward-cosine references:

| Acting-hand measurement at the legacy event | Minimum | Median | Maximum |
| --- | --- | --- | --- |
| Own-torso-relative onset speed | .263 m/s | .457 m/s | .629 m/s |
| Hand excursion over preceding 250 ms | 8.2 cm | 9.2 cm | 10.6 cm |
| Elbow excursion over preceding 250 ms | .260 rad | .301 rad | .344 rad |

The new version-1 motion evidence is applied identically to all fifteen targets
and the nineteen earlier learned-policy events. No event in either set meets
both speed/forward references. These are descriptive references, not universal
boxing acceptance criteria or a classifier for every punch type. The dataset
does not provide a validated positive straight-strike cycle for the next target.

Code inspection independently explains the target's intended stopping condition:
`directed_action` stops correcting the arm after the first scored contact. The
retention sampler still presents these same fifteen files, with extra weight on
their left-arm labels. First-impact history inputs also remain set after a score.
This establishes a mismatch between a reach-once target and the desired repeated
strike/recovery task; it does not isolate retention or history features as the
sole cause of every learned failure. No actor/normalizer/optimizer/data was changed.

Twenty-one focused motion/observer/scorecard/evaluator tests pass. The evidence
interface prevents borrowing speed from one hand and direction from another,
marks censored recovery unavailable and never returns a certified punch count.
Historical contact gates and their complete failures remain unchanged.

## Decision

Do not extend unchanged imitation, retention or PPO. Validate a physically
executable positive strike cycle under the existing free-body motor and .8-rad
arm authority before another training objective. The next
[bounded jab-cycle feasibility probe](../../../striking-jab-cycle-feasibility.md)
uses explicit guard/extend/retract/rest phases and does not stop after the first
contact. It is a scripted diagnostic, not a deployed or trained boxer.

This audit ends at **2,026,879 formal diagnostic controls**, with diagnostic PPO
2,163,968, main boxing 1,048,576, foundation 474,992,640, combined main 476,041,216
and standalone BC 8,000 unchanged. The separately declared feasibility allocation
is not an unchanged training extension. No promotion or policy adoption.
