# Lateral bows: no complete punches, one incidental opposite-hand touch

Completed and independently reviewed on 2026-10-25 under the
[prospective protocol](../../../striking-arc-path-comparison.md). Two full arms
use a 6 cm or 12 cm outward bow during the original extension. Endpoints,
learned motor guard, terminal dwell, arm authority, damping/filter and all
physical contacts remain unchanged. No direct leg/root actuation or assistance.

| Arm | Ten-second pair survivors | Legacy scored events | Candidate punches | Feasible episodes |
| --- | --- | --- | --- | --- |
| Historical straight v3 | 8/8 | 0 | 0 | 0/8 |
| 6 cm bow | 7/8 | 0 | 0 | 0/8 |
| 12 cm bow | 8/8 | 1 | 0 | 0/8 |

The 6 cm case at corner 1 / seed 23 / no swap terminates at 9.812 seconds
on the acting body's low-pelvis predicate. Every other case survives. The sole
legacy event is in the 12 cm corner-1 / seed-23 / swapped case, but it is the
**right hand**, not the commanded left hand. Its onset relative speed is
0.294 m/s, forward cosine 0.036, preceding hand excursion 3.95 cm and return
2.55 cm. Its 0.520 N s impulse and 58 ms contact qualify only for the old
contact counter, not the unchanged complete punch conjunction. Do not report
this as a successful left jab or an improved punch count.

Across all sixteen cases, **zero acting-left-fist target contact substeps**
occur. [contact-provenance.json](contact-provenance.json) retains the per-case
native hashes, counts and all events. A curved arm motion is insufficient on
these conditions; no positive demonstration is established. These are scripted
diagnostics against a motor-controlled partner, not learned self-play.

## Verification and accounting

Sixteen focused controller/workspace/trace/signed-distance tests pass. Zero-arc
actions match the original dwell controller; both endpoints are preserved and
the bow rotates with the torso. Independent review checks both complete arms,
source hashes, exact approach/guard prefixes before the first changed extension,
the saved Cartesian path formula, all intended channels and physical limits,
phase/capture clocks, native events and every unchanged candidate requirement.
All recorded values are finite and native stderr is empty.

Actual physical consumption is **7,991 new controls**, cumulative **2,059,063**,
under its explicit 2,059,072 ceiling. The unused nine controls expire. No training
updates, cloned searches or extra distance queries. Diagnostic PPO remains
**2,163,968**, cloned forwards **6,349**, signed distances **960**, main combined
transitions **476,041,216**, BC **8,000**. The pure-array contact summary adds no
physical, optimization or geometry compute. All full/native traces, progress,
pre-review completion and [analysis.json](analysis.json) remain archived.

## Concrete continuation decision

Reject both amplitudes as positive punch targets. Do not repeat these arm-only
sweeps unchanged, relax criteria, enlarge authority or resume PPO from these
failed targets. All studied scripted attempts leave the original three movement
command offsets zero. Their failures therefore do not establish that the full
existing 17-action policy interface cannot solve the task.

The next bounded comparison should keep the arm path fixed and vary the
existing **lateral movement-command offset** through the learned walking motor,
to test angle/range adjustment around the opponent's guard. Preserve free legs,
unassisted roots, all collisions, original command/arm bounds, source provenance
and complete independent criteria; it must not become a direct root controller.
Review prior body-authority failures before implementation and do not blindly
unfreeze the motor. Declare its exact command schedule and new budget before
collection. No new physical, PPO or demo-adoption allocation is granted here.
