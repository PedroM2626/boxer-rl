# Positive lateral command: isolate scripted arm interaction

**Completed and independently reviewed:** [all results](results/striking-pilot/lateral-isolation-v6/README.md). Motor-only survival 8/8 versus 2/8 in the archived arm-plus-command control; all six failures disappear. Zero scored impacts or candidate punches. Consumed 4,000 controls, cumulative 2,070,307. The original prospective protocol below remains unchanged.

Declared 2026-10-25 before collection, after the complete lateral-command-v5
review. Its positive direction has six acting-body failures in eight cases,
versus zero in the matched straight no-lateral control. Its negative direction
survives all eight but has no commanded-left target contact. Neither arm is a
positive punch target. Investigate this regression before changing the motor.

Reuse all eight positive-direction cases as the fixed historical control.
Run eight new motor-only cases: seeds 7/23, both acting corners, both swaps.
Remove all 14 arm residual requests; retain the original frozen motor (including
its learned arm actions), positive lateral action +1 during guard/extension/dwell,
zero otherwise, X/yaw and partner head zero, identical .75 m activation and
350/180/120/250/300 ms clock. No motion is prescribed to a root or leg.
The command channel adds .15 m/s in the acting pelvis-local lateral direction,
with unchanged original clipping. Recompute automatic approach from live state.
Keep 2 m spawns, .45 m approach stop, contacts, physics, source hashes and bounds.
Require bitwise full-state prefixes through the first removed arm request.

At most 4,000 new physical controls (eight ten-second episodes), from 2,066,307
to **2,070,307**. The earlier 756 unused controls expired and are not reused.
No new PPO, BC, cloned geometry, distance or rendering allocation. Retain main
476,041,216, diagnostic PPO 2,163,968, BC 8,000, cloned 6,349, signed-distance
960 and renderer 500. Record attempted controls before stepping and all partial
failures. Existing hidden persistent WSL launcher/global/output locks prevent
duplicates; stop each episode on first original fall or timeout, and the job on
nonfinite/overflow, unexpected native stderr, source/prefix mismatch or ceiling.

Review every case, frozen source/checkpoint/full gate histories, commands,
zero arm residual requests, native traces and physical contacts. Preserve the
unchanged complete jab-candidate conjunction even though no jab is requested:
acting-left legal event, onset relative speed >=1 m/s, forward cosine >=.7,
pre-travel >=.12 m, complete recovery window and return >=.06 m, duration <=.10 s;
at least two candidates plus ten-second pair survival for feasibility.
No incidental touch, scripted controller or survival-only result certifies boxing.

If motor-only still falls, the command alone is sufficient on the affected
conditions; do not attribute all failures to arms. If motor-only removes falls,
retain evidence of an arm-command interaction in this matched sample, not a
universal causal mechanism. Mixed outcomes require case-wise interpretation.
No automatic PPO/imitation extension, motor unfreezing, checkpoint adoption or
gate relaxation. Declare a separate bounded next intervention only after review.
