# Fixed straight jab with lateral motor commands: reviewed result

All sixteen prospectively declared cases completed and passed independent
review. See [protocol](../../../striking-lateral-command-comparison.md), audit,
analysis, complete full/native traces and contact-provenance.json.

| Original lateral action | Pair survivors | Acting falls | Candidate punches | Commanded-left target contact substeps |
| --- | --- | --- | --- | --- |
| -1 (requested -.15 m/s) | 8/8 | 0 | 0 | 0 |
| +1 (requested +.15 m/s) | 2/8 | 6 | 0 | 0 |

Five positive-direction failures are low-pelvis and one is self-leg contact;
the other body never falls. The negative direction has one legacy scored event,
from the uncommanded right hand: speed .328 m/s, forward cosine .265,
pre-travel 3.64 cm and return 2.74 cm. It fails intended hand and multiple motion
criteria. It is not a punch. No feasible full episode in either arm.

Across extension/dwell, median target error is 29.20 cm; 24.26% of active left
arm action requests clip. Root-distance medians are .492 m (negative) and
.572 m (positive); ranges .398-.591 m and .486-1.135 m respectively. These are
state observations, not proof that root distance alone explains missing contact.
The old zero-offset straight control had 8/8 survivors and no impacts.

Review checks all sixteen cases, bitwise pre-guard qpos/qvel prefixes against
archived v3 controls, original straight path/phase/filter bounds, scheduled
movement commands before the motor arm baseline, exact recorded command scaling
and clipping, both bodies' first falls, finite full/native arrays, native time
spacing, force absence, source hashes, unchanged full foundation/learner gate
provenance and every original candidate conjunction. Native stderr is empty;
no nonfinite or constraint/contact overflow. Sixteen focused tests pass.

Physical controls: 7,244 new, cumulative **2,066,307**, below **2,067,063**;
unused 756 expire now. PPO 2,163,968, main 476,041,216, cloned evaluations 6,349,
signed-distance queries 960, renderer forwards 500 and standalone BC 8,000 remain
unchanged. No optimizer update, checkpoint adoption, gate relaxation or promotion.
The producer and independent reviewer run on CPU; source policies remain frozen.

## Continuation decision and limitations

Reject both directions as positive demonstration targets. Preserve the stable
negative-direction result diagnostically, without calling it learned footwork
or a certified boxer. Do not repeat the same sweep or extend imitation/PPO from
failed strike targets. The positive direction regresses pair stability relative
to the matched straight control; first isolate that existing command schedule
with arm residual requests disabled in a prospective motor-only comparison.
This separates command-only instability from interaction with extension before
changing the motor or adding authority. No further physical allocation is active.

The two seeds and ten-second scripted episodes do not establish broader
performance. A constant torso-local/pelvis-local direction is not a learned
choice of attack angle. No conclusion here disproves the entire 17-action
interface, every timing choice or all boxing trajectories. A positive future
scripted example will still require holdouts and subsequent learned validation.
