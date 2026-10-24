# Passive punch-quality replay protocol

**Completed:** all eight episodes / 4,000 controls are
[independently reviewed](results/striking-pilot/punch-quality-2080768/README.md),
with exact original trial equality. The observer suite now has fifteen passing
checks, including rotation/translation compensation. Diagnostic physical compute
is 2,019,379; PPO remains 2,163,968. No replay or learner is active and this
completed allocation cannot be restarted automatically.

Declared 2026-10-24 before collection, following the user's video observation and
the [static objective audit](striking-punch-quality-audit.md). The completed pool
comparison grants no PPO extension. The new question is what the legacy contact
events actually look like in native fist motion, not whether more unchanged PPO
can improve a permissive counter.

Allocate **at most 4,000 diagnostic physical controls**, zero optimizer updates,
zero new PPO collection: eight selected episodes of at most 500 controls each.
Formal cumulative consumption starts at 2,015,379; this audit's new explicit
ceiling is **2,019,379**. The unused allowance from the completed comparison is
not silently reused. Diagnostic PPO remains 2,163,968; all main/BC counters stay
unchanged. Count each attempted replay control before execution, including errors.

Replay both swaps of: published video source bea8147f at seed 29,000,003; actual
pool d93a0008 and control 3a34aa85 finals at 32,000,003; earlier long-GAE 95dbdff0
at 26,000,003. Complete source hashes, checkpoint/reference paths and reference
report hashes are pinned by `tools/striking_punch_quality_audit.py`. These are
selected known episodes, with different source conditions; their descriptive
statistics are not an unbiased policy comparison or new holdout certification.
The earlier long-GAE actor still uses the new paired interface. It is not a
recreation of every older assisted/unassisted `boxer/env.py` video.

An optional passive observer records the pre-solver fist position/velocity in
its own moving/rotating torso frame, elbow angle, nearest-target advancement,
actual contact impulse, scorecard contact/onset state and scoring time at every
2 ms native substep. Per scored event, describe the preceding 250 ms of speed,
hand travel and elbow excursion, the impulse accumulation, and 500 ms of later
contact/retraction toward the pre-window hand position. That reference pose is
not assumed to be a competent guard. Mark truncated post-event windows as censored.
The 250/500 ms windows are descriptive choices, not new punch gates.

Keep the original physics, policy/normalizer, motor, scorecard, reward, resets,
free roots/legs and impacts unchanged. Compare every original physical trial
field against its archived exact replay, including survival, time, raw impulse,
scores/events, residual activity and distance. Any mismatch, nonfinite/overflow,
unexpected child stderr or source/hash/accounting failure stops the audit. No
retry, restoration or training allocation is automatic. Nine focused tests pass,
including exact native qpos/qvel/ctrl/score equivalence with and without the
observer and correct onset/censored-window reporting.

The audit holds the established global boxing lock and a diagnostic supervisor
lock. Use the hidden persistent WSL launcher and inspect
`/home/pedro/boxing-runs/striking-punch-quality-2080768/audit.json` and its active
process before recovery. Each child has durable attempted-control progress and
empty expected native stderr. Completed traces and reports are archived under
`docs/results/striking-pilot/punch-quality-2080768/` for independent interpretation.

Counts below old default speed (1 m/s) and forward-cone (.7 cosine) components
are reference comparisons only: they do not reproduce the entire old commitment
function or establish universal boxing thresholds. Do not call a fast collision,
an arbitrary extension length or a returned hand a certified punch. Interpret
the motion/impulse distributions and recovery before specifying a versioned
objective and prospective acceptance criteria. No main adoption or promotion.
