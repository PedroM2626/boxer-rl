# First independently reproduced impact events, 2026-10-23 UTC

The main boxing learner's first nonzero event counts appeared at consumed
counters 90,112 (blue +1) and 91,136 (red +2), using the unchanged scorecard.
These are logged checkpoint intervals, not exact per-action occurrence counters.
The complete [increment records](training-event-increments.json) are preserved.

An immutable **97,280-step** policy snapshot was taken without stopping training:
`checkpoints/striking-pilot/first-observed-impacts/`. Its checksum is
`196d6aeee9b499bf3ea9daa94a8da063d821768049aa0ba097cdcbd05afffa5b`.
The original frozen motor hash remains
`b39b1990945e8a374802c3a190e5ac5e026ae39e7fd5ef61c5c2b01ad291c6fc`.
All actor/critic/Adam/normalizer/RNG leaves were finite, all four foundation
certifications remained intact, and training source checksums match the original
launch record. No impact, physics, reward, or assistance setting changed.

The [independent deterministic evaluation](deterministic-evaluation.json) uses
seeds 7/23/31/47, both side arrangements and ten-second first-fall episodes.
**8/8 paired episodes survived; two actual impact events were credited**, both
by blue's left hand on holdout seed 31, once per arrangement. No exploration
noise was added. The other six episodes scored zero. Thus this is **2/8 episodes
with a valid event**, not consistent two-way boxing or a new certification.
The 32,768-step baseline scored zero in every corresponding episode.

The same two inputs were [replayed](confirmed-replay.json) to log exact timing:

| Side arrangement | Hand | Score time | Contact-window impulse at end of control step |
| --- | --- | --- | --- |
| Original | Blue left | 4.046 s | 1.0931 N s |
| Swapped | Blue left | 4.104 s | 0.9476 N s |

Both robots were upright at both events and survived the round. These impulses
are the hand's accumulated legal-contact window as inspected after the 20 ms
control step, not a claim about peak force or an exact threshold-crossing
impulse. Raw round-long legal impulses were 7.096/3.309 N s for blue, but only
one qualifying event per episode was credited; sustained contact did not turn
into repeated hits. Red had raw impulse without a credited active strike.
The replay reproduces the original two successes; it is not counted as two new
independent successes.

The actual simulated video is
`fights/striking-pilot/first-confirmed-impacts.mp4`. It shows the original seed-31
arrangement, with score/time text added to the unchanged captured frames. The
frame clock includes the 20 ms first action, so the first displayed scored frame
is 4.06 simulated seconds, immediately after the 4.046 s event. It was visually
inspected. Arm motion is still modest and attacks are rare/asymmetric; an impact
counter is not evidence of polished or robust sparring.

The evaluation used 4,000 diagnostic world-control transitions and replay added
1,000, with zero optimization updates. Formal striking-interface diagnostics now
total 17,100 world transitions, plus the separately recorded 640 diagnostic PPO
learner transitions and earlier foundation/transfer experiments. Main boxing
consumption and optimizer state continue independently in the durable ledger.

**Decision:** keep the single productive learner running unchanged within the
existing **262,144 boxing-step ceiling**. No budget extension or stage promotion
is made by this review. Continue to the planned final evaluation, inspect
survival, actual contacts and both-corner asymmetry, and archive the final policy
before deciding any later allocation. Retain misses and falls in all denominators.
The existing 30-minute monitor schedule is unchanged.
