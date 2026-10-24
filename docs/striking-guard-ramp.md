# Two-second entry into the unchanged guard

**Completed and independently reviewed:** the
[full result](results/striking-pilot/guard-ramp-v1/README.md) rejects smoothing as
a sufficient repair (0/8 pair survival despite improved hand tracking).
Actual consumption is 1,942 physical controls, cumulative 2,043,072. A post-run
episode-count bookkeeping error was preserved and recovered with zero additional
physical controls. The prospective protocol below remains unchanged.

Declared 2026-10-24 after the independently reviewed
[motor/static-guard comparison](results/striking-pilot/guard-control-v1/README.md),
before collecting new ramp trajectories. Static guard fails 0/8 survival versus
8/8 motor-only. A complete jab cycle is unnecessary to reproduce those failures.
Test whether changing entry alone improves stability, without claiming the
final posture or all body coordination is already correct.

## Single changed factor

At the unchanged .75 m guard activation, capture the acting left fist in its
own rotating torso frame. Interpolate that local point to the **same** original
guard `[.16,.16,.12]` m over two seconds using `3t²−2t³`, then hold the exact
original target until the ten-second episode ends or first fall. Keep the same
Jacobian, .16 rad correction cap, .05 damping, .8 rad residual box, .8/.2 filter,
frozen motor, .45 m approach profile and unassisted free roots/legs. No artificial
forces, direct leg actions, root constraints, reward or termination changes.

Run eight cases: seeds 7/23, both acting corners and both swaps. Compare against
all already-reviewed matched motor/static-guard/v1 cases without repeating them.
Require identical full-state approach prefixes. Retain full/native traces,
entry/target telemetry, first falls, phase timing and guard tracking/clipping.
Report ten-second pair survival, not a new boxing/punch promotion gate. Check
that a survivor actually reaches the end of the two-second ramp before calling
the final-guard hold stable. No new PPO, imitation or positive-demo adoption.

## Bounded allocation and review

Start cumulative diagnostic physical controls **2,041,130**. Allocate at most
**4,000** new controls, explicit ceiling **2,045,130**. This justified measurement
distinguishes guard-entry transients from the unchanged final target. Diagnostic
PPO remains 2,163,968; cloned geometry remains 4,339; all main/BC counters unchanged.
Old unused allocations expire and are not silently reused.

Use hidden persistent WSL launching and existing global/per-output locks. Inspect
`/home/pedro/boxing-runs/striking-guard-ramp-v1/audit.json`, progress and process
before recovery. Stop on native/overflow warning, nonfinite state, source/profile
or prefix mismatch, unexpected stderr, first fall per episode or budget ceiling.
Preserve attempted-control accounting and partial results; no blind retry.
Independently verify every condition, source hashes, intended channels, ramp
timing, terminal flags, finite traces, native event counts and actual consumption.

If all ramp cases still fail, reject entry smoothing as a sufficient repair and
investigate a guard compatible with the learned motor before adding punching.
If stability improves, report matched cases and held-guard duration before a new
positive-cycle test. No extrapolation to unseen seeds, learned boxing or full
certification and no automatic training extension.
