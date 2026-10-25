# Matched lateral-bow path comparison

**Completed and independently reviewed:** see [full results and decision](results/striking-pilot/arc-path-v4/README.md).
Actual consumption 7,991 controls, cumulative 2,059,063. Neither complete arm
produces a candidate punch. The only legacy impact is an incidental right-hand
touch, not the commanded left hand. Unused compute expires at review; the
prospective protocol below remains unchanged.

Declared 2026-10-25 after independent signed-surface review and before collection.
Recorded poses have no legal surface overlap, but 20/24 intersect a blocking
surface. Six of the old static-best poses intersect a legal target, five also
intersecting a blocking surface. This motivates a path comparison, not authority
increase or a claim that every actual path is blocked.

Compare **6 cm and 12 cm outward bows** against the already-reviewed zero-bow
terminal-dwell control. During outbound motion, add
`torso_left_axis * amplitude * sin(pi*fraction)` to the original Cartesian
interpolation. Both endpoints and the full-target dwell remain identical. The
bow is zero at either endpoint, maximal halfway and rotates with the acting torso.
Preserve learned motor guard, 350/180/120/250/300 ms phase clock, .8 rad residual
box, .16 rad Jacobian correction cap, damping, filter, physical contact pairs,
approach distance and source checkpoints. No root/leg action authority, forces,
assistance, modified collisions, policy/optimizer updates or new geometry search.

Run both amplitudes on seeds 7/23, both acting corners and both swaps: sixteen
new episodes, at most 500 controls each. Use the complete archived zero-bow cohort
as the fixed historical control, without repeating it. Require exact full-state
approach/guard prefix through the first changed outbound control. Preserve all
full/native traces, path targets, phase clocks, contacts and failed conditions.

The complete original candidate conjunction remains unchanged: actual legal
acting-left contact, onset speed >=1 m/s, forward cosine >=.7, pre-window hand
excursion >=.12 m, complete post-window with >=.06 m return toward its pre-window
point, contact duration <=.10 s. At least two candidates and ten-second survival
define a feasible diagnostic episode. A lateral motion or guard touch does not
qualify by itself; do not redefine criteria if a curved attempt fails them.

Declare at most **8,000 new physical controls** from **2,051,072**, explicit
ceiling **2,059,072**, justified by testing a path change identified by actual
blocking evidence. No reuse of expired allocations. PPO remains **2,163,968**,
cloned queries **6,349**, signed surface queries **960**, combined main transitions
**476,041,216**, BC **8,000**. Normal per-control motor/Jacobian calculations
remain part of the physical trial, not a new search or learner update.

Use hidden persistent WSL and existing global/output locks. Inspect
`/home/pedro/boxing-runs/striking-arc-path-v4/audit.json`, progress and process
before another launch/recovery. Stop on first fall per episode, native overflow
or nonfinite values, unexpected stderr, source/prefix/profile mismatch or ceiling.
Persist attempted controls before stepping; preserve all partial records and
do not retry blindly. Independently review both complete arms and unchanged
candidate criteria, source hashes, path formula, limits and consumed counts.

If neither arm yields a surviving full cycle, reject these amplitudes without
threshold relaxation or unchanged PPO extension. If one produces candidates,
retain both arms' failures and require broader holdouts before demo adoption.
No scripted result certifies a learned boxer or authorizes main promotion.
