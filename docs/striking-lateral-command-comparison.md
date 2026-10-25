# Lateral movement-command comparison

**Completed and independently reviewed:** [full results](results/striking-pilot/lateral-command-v5/README.md). Both arms have zero candidate punches; survival 8/8 versus 2/8. Consumed 7,244 controls, cumulative 2,066,307. Unused 756 expire. The prospective protocol below remains unchanged.

Declared 2026-10-25 before collection. The reviewed arc-path study has no
commanded-left contact in sixteen cases; signed surface evidence shows frequent
blocking overlaps. All previous scripted cycles kept movement offsets zero.
Test angle adjustment through the existing frozen walking motor; the failed
adaptive body holdout does not justify unfreezing it.

Keep the original straight terminal-dwell-v3 arm path, phase clock, targets,
filter, correction/residual bounds, 2 m spawns, approach stop distance .45 m,
contacts and frozen foundation checkpoint. Compare lateral action -1 versus +1
(own pelvis coordinates, requested additions -.15/+ .15 m/s, original command
clipping) during guard and extension including dwell; zero during approach,
retraction and rest. X/yaw offsets and partner heads remain zero. Compute the
arm baseline with the same movement command actually sent to the motor.
No direct root/leg control, assistance, policy updates or changed physics.

Two directions x seeds 7/23 x both corners x both swaps: sixteen episodes,
maximum 500 controls each. Reuse all eight archived v3 straight controls and
require bitwise state prefixes before the first guard movement command.
Preserve full/native traces, command baselines, exact phases and all failures.

Allocate at most 8,000 physical controls from 2,059,063 to **2,067,063**.
Unused prior controls expired. PPO remains 2,163,968, main 476,041,216,
cloned queries 6,349, signed-distance queries 960, render forwards 500,
standalone BC updates 8,000. No new optimizer, geometry or renderer allocation.
Use existing global/output locks and hidden persistent WSL. Persist attempts
before stepping; stop on first fall per episode, nonfinite/overflow, unexpected
native stderr, source mismatch or ceiling; preserve partial failure records.

The unchanged complete candidate requires acting-left legal contact, onset
relative speed >=1 m/s, forward cosine >=.7, pre-travel >=.12 m, complete
post-window and return >=.06 m, duration <=.10 s. A feasible episode requires
at least two candidates and ten-second pair survival. Review both full arms,
not selected frames. Zero-candidate arms are rejected as positive targets;
no automatic imitation/PPO extension. Even a positive scripted result requires
broader holdouts before learning adoption and does not certify learned boxing.
