# Preserve the learned guard: stable sample, no physical strikes yet

Completed and independently reviewed on 2026-10-24 under the
[prospective protocol](../../../striking-motor-guard-cycle.md). This scripted
version 2 leaves the learned motor guard intact between extensions and decays
the residual through the original filter during retraction/rest. It does not
impose the earlier arbitrary Cartesian guard point or update any policy.

| Matched condition | Ten-second pair survivors | Feasible punch-cycle episodes |
| --- | --- | --- |
| Motor-only control | 8/8 (four unique reset conditions) | Not a punch attempt |
| Immediate scripted guard | 0/8 | 0/8 |
| Two-second scripted guard entry | 0/8 | 0/8 |
| Learned motor guard between extensions | **8/8** | **0/8** |

No first-fall predicates occur in the new eight cases. However, there are
**zero legal scored impacts, zero acting-left-fist contact substeps and zero
candidate punches**. It improves stability in this matched sample, not boxing
completion or a positive punch-demonstration qualification. Both neutral
partners remain controlled by the same motor; this is not trained self-play.

The foundation already uses `GUARD_ARMS_BOXING`; it did not need a new arbitrary
guard to acquire raised hands. The separately verified
[archived offset measurement](../guard-control-v1/learned-guard-offsets.json)
finds that the failed Cartesian guard requests an extra **10.3–11.4 cm** of hand
displacement from that learned posture at activation. This does not isolate
every contribution to earlier falls. Version 2 also changes recovery to the
original motor/filter; do not attribute its survival improvement solely to one
Cartesian coordinate.

## Remaining motion problem and continuation decision

Median extension tracking error remains **14.7 cm**; 14.7% of active arm-channel
requests clip. [extension-motion.json](extension-motion.json) records every
extension pulse directly from the original native arrays. Peak own-torso-relative
speed during extension is not impact-onset speed; even a large peak does not
qualify a punch without real contact and the unchanged full candidate conjunction.
Center distances include either own hand and are not left-fist surface gaps.

Retain this as a stable scripted control for the next comparison. It is not
approved for imitation or PPO extension: the positive strike-cycle target is
still missing. Next inspect extension timing, target lag and closest physical
approach from these traces, then prospectively compare a bounded terminal-target
dwell against the original immediate return. Preserve the learned guard and all
independent contact/motion/survival criteria; do not lower punch thresholds to
declare these non-contact motions successful. Wider holdouts are required even
if a later positive cycle is found.

## Evidence and accounting

All eight cases and exact archived approach prefixes are independently checked.
Zero non-extension head actions, seven-channel authority, cycle phases, learned
entry capture, original filter targets, complete candidate criteria, native events,
full-state continuity, finite arrays, source/checkpoint hashes and terminal flags
are verified. Twelve focused controller/workspace/trace tests pass; native stderr
is empty. No source actor, Adam, normalizer, gate or assistance changes.

The run consumed **4,000 physical controls**, cumulative **2,047,072**, at its
explicit ceiling. Diagnostic PPO remains **2,163,968**, separate cloned geometry
**4,339**, main combined transitions **476,041,216**, standalone BC **8,000**.
The two descriptive array measurements add zero physics, cloned queries or
optimizer updates. No further allocation follows automatically.

[analysis.json](analysis.json) retains all matched outcomes and failures of the
candidate criterion. [audit.json](audit.json), progress, `full/` and `native/`
retain every trajectory and hash; `native-errors.log` is intentionally empty.
The protocol/source implementations and all prior rejected results remain.
