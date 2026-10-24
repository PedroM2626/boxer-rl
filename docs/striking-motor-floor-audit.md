# Matched motor-floor diagnostic

The complete [generalization review](striking-generalization-audit.md) finds
all four independent full groups failing. Ten neutral-profile first falls are
of the learner and three of the motor-only partner; some early failures occur
at large separation. Cross-policy matches also include both learner and opponent
falls. This justifies separating reset/motor limitations from head/contact
limitations before adding PPO, changing leg authority or building a larger league.
Earlier action-mask and yaw probes did not generalize and remain rejected.

Retain the exact latest **670bed08** payload, original frozen adversary, actor,
critic, Adam moments/counts, normalizer and all random/sampler histories. No
optimizer, collection, replay or supervised updates are allocated. All original
free-body physics, learned motor actions, initial separation, scoring, fall and
10-second limits remain unchanged. Do not restore an older favorable actor.

1. Evaluate all 16 already audited seeds 21,000,003..21,000,018, both swaps,
   with both heads' three command offsets and fourteen arm residuals zeroed.
   These **32 episodes** leave the inherited automatic approach commands and
   all 29 learned motor joint actions active. This is a diagnostic motor-only
   comparison, not a boxing policy, assistance or a replacement gate.
2. Passively replay the full head on seeds **21,000,006 / 21,000,009 /
   21,000,011 / 21,000,018**, both swaps and all three original profiles:
   **24 episodes**. These were selected before masked outcomes for early/far
   and early near-contact failures. Require every original physical report field
   to match exactly. Save finite native state/control traces, real first interbody
   contacts, terminal pelvis/upright/leg conditions and reward components.

Compare the same reset/side identities, not averaged outcomes on unrelated seeds.
A precontact motor-only failure is evidence for an inherited reset/motor limit;
a failure removed by masking identifies a head-dependent trajectory, without
proving which individual channel caused it. A surviving masked trajectory does
not imply learned boxing and cannot justify permanently disabling head actions.
Contacts without scored punches still count as contacts. Do not label every
unscored fall precontact. Report all outcomes including worsened masked trials.

Total **56 episodes / at most 28,000 attempted evaluation controls**, counted
before each native step. Formal evaluation starts at **1,351,541**, ceiling
**1,379,541**. Diagnostic PPO stays **1,918,208**; main boxing **1,048,576** and
combined main **476,041,216** stay stopped. No new training budget is implied.
Stop on nonfinite state, native error/overflow, hash mismatch or altered passive
replay; retain all attempted controls and investigate before any resume.

Launch with `tools/start_striking_motor_floor_audit.ps1` (hidden persistent WSL).
Live state: `/home/pedro/boxing-runs/striking-motor-floor-1916928/audit.json`.
Inspect that state and process before launching anything else; do not duplicate
or restart completed jobs. On completion independently review/archive all trials,
then choose a concrete bounded intervention from the evidence. No automatic
promotion, main adoption or unlimited PPO; keep the existing monitoring schedule.
