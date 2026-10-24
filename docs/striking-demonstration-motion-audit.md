# Versioned motion evidence and audit of all retained demonstrations

**Completed:** all fifteen demonstrations / 7,500 controls exactly replayed and
are [independently reviewed](results/striking-pilot/demo-motion-2080768/README.md).
All fifteen acting-hand events are below both motion references. Twenty-one
focused tests pass. No PPO or supervised updates; this completed allocation is
not authority to replay again. The subsequently declared jab-cycle prototype is
a separate physical diagnostic with its own consumed accounting and decision.

Declared 2026-10-24 before new physical collection. The
[eight-episode native audit](results/striking-pilot/punch-quality-2080768/README.md)
found short arm motions and no event satisfying both old speed/direction component
references. Before another objective/PPO change, determine what the retained
demonstration targets actually teach. The old geometric generator stops correcting
the left arm after its first scored contact (`tools/striking_directed_feasibility.py`);
the fifteen saved trajectories remain active retention targets. That code path
does not enforce a full strike/retraction cycle. It is a hypothesis about an
inadequate target, not proof that retention causes every policy failure.

## New evaluation interface, not a replacement scorecard

`boxer/strike_motion.py` defines immutable version-1 motion-reference evidence:
per-event own-torso-relative speed >=1 m/s and forward cosine >=.7, with the
existing 250/500 ms descriptive windows, measured arm travel/elbow excursion,
impulse and recovery availability. Components are never borrowed across hands.
Censored return measurements are unavailable rather than false successes/failures.
Even both components plus return motion do not certify a boxing punch or guard.
The interface explicitly returns no certified-punch count. These references are
inherited descriptive anchors, not universal acceptance thresholds for all punches.
Legacy contacts, rewards, physical gates and all archived failures are unchanged.

## Bounded exact physical replay

The saved demonstration files contain only observations and actions; they do not
contain qpos/qvel. New native kinematic measurement therefore requires physical
replay, not an invented offline reconstruction. Allocate **at most 7,500 diagnostic
controls**: all fifteen 500-control demonstrations in the exact existing retention
manifest, with zero optimization, PPO collection or supervised updates.
Formal simulation starts at **2,019,379**, new explicit ceiling **2,026,879**.
Diagnostic PPO stays 2,163,968; main boxing 1,048,576, foundation 474,992,640,
combined main 476,041,216 and standalone BC 8,000 are unchanged.

Source `checkpoints/striking-validation/partner-duration-1441792` has full SHA
87eb01ff6392b592395b64fc6f6a29f78e35fd1c1ad2941fafbcc8001146f5d8.
Use `docs/results/striking-pilot/directed-holdout-1441792/replay.json` for all
fifteen file hashes and `feasibility.json` for the original seed/corner/swap,
elapsed time, scores and event records. Both manifest hashes are recorded before
collection. Original .45 engagement, 146-input StrikingEnv, frozen motor, root/leg
freedom, arm authority/filter and impact scorer remain unchanged. Replay saved
actions verbatim; no live teacher, policy replacement or optimizer is deployed.

Every pre-action learner observation must equal its saved row exactly. Require
the same survival, 10-second duration, scores and every native-timed event/impulse.
The optional passive 2 ms observer measures all hands. Separate learner-corner
events from opposing events; do not label all contact events positive examples.
Retain raw traces and motion evidence for every demonstration, including failures
of reference components and censored recovery windows.

Count attempted physics durably before execution. Stop on an observation/event,
hash/provenance or counter mismatch, nonfinite values, constraint/contact overflow
or unexpected stderr; do not retry/rollback silently. Hold global boxing and
diagnostic/output locks. Start through a hidden persistent WSL host and inspect
`/home/pedro/boxing-runs/striking-demo-motion-2080768/audit.json` and `progress.json`
before recovery. No duplication, new training or promotion is automatic.

Interpret the complete manifest together with the prior selected learned-policy
traces. If there is no measured positive motion target, establish and validate
physical strike feasibility before more imitation or an unchanged retention/PPO
extension. If examples support only some component references, retain that limited
scope; do not declare them competent boxing. This audit does not recreate every
historical policy/video and cannot by itself certify generalization or all punch types.
