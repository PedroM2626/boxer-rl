# Bounded terminal-extension dwell comparison

**Completed and independently reviewed:** see [results and decision](results/striking-pilot/terminal-dwell-v3/README.md).
All eight pairs survive, but zero impacts or candidate punches occur. Actual
consumption is 4,000 controls, cumulative 2,051,072. No new learner updates.
The original prospective protocol follows.

Declared 2026-10-24 before collecting new physical episodes. The reviewed
[motor-guard v2 control](results/striking-pilot/motor-guard-cycle-v2/README.md)
survives all eight matched ten-second episodes but records zero impacts and zero
acting-left contact substeps. Median tracking error during its 180 ms extension
is 14.7 cm. Test whether releasing the arm override immediately at that endpoint
prevents contact, preserving the already-learned guard and original outbound motion.

## Single changed factor and fixed physical criteria

Version 3 adds **120 ms at full extension target** after the unchanged 180 ms
raised-cosine extension. During that dwell the same bounded original Jacobian
controller tracks the opponent torso endpoint. Retain 350 ms guard, 250 ms
motor/filter retraction and 300 ms rest; cycle length becomes 1.20 s. Capture
the learned guard at first extension of each actual cycle, not at the old 1.08 s
clock. All non-extension/dwell head actions remain zero. No gain, authority,
root/leg control, contact, assistance, reward or source checkpoint changes.

Run the same eight cases: seeds 7/23 × both acting corners × both swaps.
Use all previously reviewed v2 cases as the fixed historical matched control;
do not spend new controls repeating it. Require exact qpos/qvel agreement through
the first outbound extension, immediately before the first new dwell control.
Retain all full/native traces and original action-channel/filter telemetry.

The original complete candidate conjunction is unchanged: legal acting-left
contact, onset speed >=1 m/s, forward cosine >=.7, pre-window hand excursion
>=.12 m, complete post-window with >=.06 m return toward its pre-window point,
and contact duration <=.10 s. A feasible episode survives ten seconds and has
at least two such candidates. A longer contact or low-speed touch cannot become
a punch merely because it occurs during the new dwell. No promotion, learned
boxing certification or positive-demo adoption from a selected clip.

## Budget, stop and review

Formal diagnostic physical controls start at **2,047,072**. Allocate at most
**4,000** new controls, new ceiling **2,051,072**, specifically to distinguish
premature return from lack of physical contact/reach. PPO remains **2,163,968**,
cloned kinematics **4,339**, combined main transitions **476,041,216**, BC **8,000**.
No new optimization, imitation or automatic extension. Eight episodes at most
500 controls each; first-fall termination and all native overflow/nonfinite
checks remain. Stop on unexpected stderr, source/profile/prefix mismatch or ceiling.
Preserve partial attempted-control accounting; no blind retry or forced rollback.

Use the hidden persistent WSL launcher and existing global/per-output locks.
Inspect `/home/pedro/boxing-runs/striking-terminal-dwell-v3/audit.json`, progress
and process before launching or recovery. Independently check all conditions,
full candidate conjunction, clock/capture changes, unchanged outbound prefix,
source hashes, finite traces, zero external forces and actual consumed controls.

If contact remains absent, reject dwell as a sufficient repair and inspect
terminal reach/physical geometry using the archived evidence before another
bounded design. If contact appears but stability or punch criteria fail, retain
all failures and diagnose that tradeoff rather than extend PPO or lower gates.
Even a positive cycle requires broader holdouts before training-target adoption.
