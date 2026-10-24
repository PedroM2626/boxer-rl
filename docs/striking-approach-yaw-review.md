# Passive failure outcome and bounded automatic-yaw probe

**Current outcome:** yaw scaling is rejected after the complete comparison:
active survival regresses in both samples. See the [full result and next bounded
fixed-opponent contract](striking-fixed-opponent-comparison.md). No modified
yaw controller is adopted and the original physical profile remains in use.

## Completed exact replay

All **48 complete episodes / 21,499 world controls** reproduce every original
source/final physical trial field exactly. All trajectory arrays are finite and
hash checked; native logs are empty. Archived original/replayed trials, raw
qpos/qvel/actions/commands/reward terms, terminal conditions and timing are in
`docs/results/striking-pilot/mixed-failure-trace-1859584`. Independent reviewer:
`tools/review_striking_mixed_failure_trace.py`. Complete foundation histories
29/41/13/2 and finite payload hashes are reverified; main boxing hash is intact.

Across both policies and three profiles, **11 falls** occur: **10 low-pelvis,
1 torso-tilt, 0 self-leg contact**. Two happen before any interbody contact. Both
are neutral-role-1 episodes where the red partner receives zero learned boxing
action; that motor still follows automatic approach commands. Its yaw command
is +0.5 rad/s while strafing; the source falls at 1.672 s, final at 1.460 s.
Nine other falls follow contact, often without any qualifying scored hit. These
associations do not establish that turning, contact or leg authority causes the
failures. In particular, lowering an optimizer rate did not resolve stability,
and previous whole-body/command-mask experiments did not establish robust benefit.

Diagnostic PPO remains **1,844,480**; formal evaluation is **1,106,474**. The
trace recovery reused the first completed 4,000 controls; they are counted once
in 21,499, not discarded or added again. Main boxing stays 1,048,576 and combined
main 476,041,216; all four foundation certifications and standalone BC 8,000
updates are retained. No unchanged PPO allocation or phase promotion is granted.

## Single causal controller-path probe

Test only the automatic approach yaw, scaled **0.5 versus original 1.0**, on
the latest fully evaluated final
`6bf3273ab5cd83e42b588403a7361140e4ee42afe0fbec20732e0a010bb34e7a`.
The evaluation-only environment scales base yaw before the original learned
offset addition and final clipping. It preserves all 17 learned actions,
translational command formulas, weights/normalizer, arm authority/filter,
engagement distances, physical roots/legs, impact scoring and rewards. Motor
observations reflect the actual changed base command. Closed-loop observations
and effective saturation can consequently change; no outcome is guaranteed.
This is a hypothesis test about motor-command compatibility, not support or a
scripted punch, a change to learned action dimensions, or a deployed correction.
The ordinary evaluator and training path keep their original environment.

Reuse the verified latest-final control on the existing **16M** sample. Evaluate
modified yaw on all three same profiles (four seeds, both swaps), then original
and modified controllers on **fresh 17,000,003..17,000,006**, all three profiles,
both swaps. **72 new complete episodes / maximum 36,000 evaluation controls**,
zero PPO, collection or supervised compute. Retain all attempted controls on
failure. No cherry-picked-only continuation: evaluate both fresh arms fully.
Report survival, falls, own-event breadth, per-corner surviving impacts and
two-way episodes under unchanged numerical thresholds. Only evidence consistent
across both samples could justify a subsequent explicitly configured matched
training proposal; do not adopt this diagnostic controller or promote a stage.
If results are mixed or worse, reject this hypothesis as a practical correction.
No automatic follow-on allocation is granted.

The modified report carries `diagnostic_base_yaw_scale=0.5`; the ordinary full
promotion reviewer explicitly rejects it. Thus diagnostic criterion matching
cannot be confused with two complete ordinary consecutive passes. Default scale
1 uses the original exact command path; no base command is altered in training.
The numeric physical criteria themselves are unchanged.

Diagnostic PPO remains **1,844,480**, formal evaluation ceiling **1,142,474**
(1,106,474 + 36,000). Hidden persistent launcher:
`tools/start_striking_base_yaw_compare.ps1`; live output:
`/home/pedro/boxing-runs/striking-base-yaw-1859584/comparison.json`.
Inspect this output, process, native logs and complete reports before any other
launch. Preserve the existing monitoring schedule. Archive all results and
limitations, and decide using full physical evidence before any PPO allocation.
