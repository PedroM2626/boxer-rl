# Reject proximity shaping and test the preserved motion reference

## Completed reward comparison

Both 131,072-transition arms completed with finite actor, critic, Adam, normalizer,
RNG and losses, verified payload hashes, inherited Adam count 18,432 and no
physics overflow/invalid-state warnings. The full eight-episode physical
evaluation remained unchanged.

| Arm | Pair survivors | Red / blue impacts | Impacts in surviving episodes | Scoring seeds |
| --- | ---: | ---: | ---: | --- |
| Original reward | 7/8 | 2 / 1 | 0 | 47 |
| Additional hand proximity | 6/8 | 3 / 1 | 0 | 7 |

The proximity arm generated 22 training events over 143 rounds, including 23 fall
rounds; baseline generated five over 139 rounds, including eleven falls. More
training events did not transfer to safer evaluation. Neither met the predeclared
survival, complete-episode impacts and multiple-seed screening criteria. Reject
the additional proximity reward and adopt neither diagnostic policy. Keep main
PPO stopped at **1,048,576** and combined main consumption at **476,041,216**.
All foundation certifications remain valid; no foundation training is restarted.

[Full analysis and both arms](results/striking-pilot/reward-comparison-1048576/analysis.json)
preserve configuration, diagnostic profiles, counters, full evaluations and
parsed training histories. Immutable payloads are under
`checkpoints/striking-validation/reward-comparison-1048576/`. Their counters
include the copied 1,048,576 lineage offset: only the new 131,072 per arm belongs
to the diagnostic compute ledger. These policies are not main progress.

## Action/body interface investigation

A read-only factorial probe held the exact main actor and normalization fixed
while changing only arm residual amplitude and requested engagement range. Each
profile used all eight original first-fall episodes and the fixed impact rules.
No profile was trained or adopted.

| Arm residual bound | Engagement range | Pair survivors | Impacts | Impacts in surviving episodes |
| --- | --- | ---: | ---: | ---: |
| 0.8 rad | 0.45 m (production) | 6/8 | 3 | 0 |
| 0.8 rad | 0.35 m | 5/8 | 4 | 0 |
| 0.4 rad | 0.45 m | 7/8 | 0 | 0 |
| 0.4 rad | 0.35 m | 7/8 | 0 | 0 |

The production replay exactly matched the archived source trial fields. A smaller
arm amplitude reduced falls but removed contacts; a closer range with full
amplitude increased falls. Thus another range/amplitude adjustment is not a
validated remedy. This sample does not prove the frozen motor architecture is
incapable of boxing; it exposes a coordination/strike-discovery tradeoff that
proximity shaping did not resolve. [Raw probe results](results/striking-pilot/interface-probe-1048576/probe.json)
record 13,487 new diagnostic world controls and zero optimization.

## Older unassisted system audit and next hypothesis

The preserved `runs/warp6/warp_config.json` and `runs/warp7/warp_config.json`
configure **29 direct joint actions**, 2,048 worlds, 0.75 m spawns and a CMU jab
reference with imitation weight 0.3. The current system has seventeen high-level
outputs, a frozen motor, nominal 2 m spawns and no motion reference. These are
material differences, not proof that any one of them explains the entire gap.
Old and new impact instruments, dynamics and reset conditions also differ, so
their event counts cannot be ranked directly as the same benchmark.

`WarpBoxing`'s old imitation term rewards similarity to **any** strike-window
pose, gated by fist motion. It does not command a timed punch cycle. Restore this
specific missing learning signal in an isolated experiment before replacing the
whole motor/control interface. The existing CMU-derived thirteen-pose left-jab
reference is preserved at `data/reference/cmu-13-17-jab.json`, with provenance and
acknowledgement in `LICENSES.md` and the adjacent README. No old boxing weights,
assisted controller, root force or scripted strike is imported.

## Bounded second comparison

Clone the **same unchanged main 1,048,576 checkpoint** into two fresh diagnostic
arms, each with **131,072 new transitions**:

1. Original reward, as the common-source control.
2. Add an untimed motion bonus: `dt * max_pose(exp(-sum((q_arm-q_pose)^2)/0.6^2))
   * clip(relative_extension_speed/1 m/s, 0, 1)` while both robots stand.

The bonus is bounded to one reward unit per simulated second. It uses the left
shoulder pitch/roll/yaw and elbow, matching the recorded arm, independently for
both corners. Own torso translation and rotation are subtracted from fist
velocity before projection toward the nearest legal target. A stationary guard,
whole-body translation or fallen round earns no motion bonus. There is no clock,
required punch phase, pose playback, forced root motion or action substitution.
Actual joint servos and the existing 0.8-rad residual/filter execute policy outputs.
The 0.45 m engagement profile remains unchanged. This is learning guidance, not
an independent impact score or evidence of boxing completion.

The old implementation is reused as a hypothesis, not copied as an identical
reward scale or a claim that the old policy transfers. The stricter relative
motion gate avoids paying whole-body movement. The body's frozen adaptation
limit remains a possible blocker; if this comparison fails, investigate a
controlled whole-body adaptation interface rather than another reward-only loop.

Evaluation runs in separate fresh processes using the unmodified production
environment, all eight ten-second episodes and the existing physical scorecard.
The motion reward wrapper is absent. Keep the preceding screening criteria:
at least 7/8 pair survival, no worse than baseline, actual impacts in complete
surviving episodes and scoring across at least two seeds. No candidate is adopted
automatically; a favorable small sample still needs broader validation. Reject
both if neither qualifies, and do not extend main work or import diagnostic actors.

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_reward_compare.ps1 -Candidate imitation
```

Active durable output is
`/home/pedro/boxing-runs/striking-imitation-compare-1048576/comparison.json`.
Inspect that state, active child progress/logs and processes before any launch.
Arms run sequentially under the existing single-learner lock in a persistent
hidden WSL host. Child counters include their source offset; sum only
`consumed_steps - 1048576` from each active/completed arm. The preceding completed
reach comparison must not be relaunched. The original monitor schedule is unchanged.

Before the motion smoke, total diagnostic PPO consumption is **264,960 learner
transitions** and formal striking evaluation/diagnostic simulation is **74,456
world controls** (60,969 after the first comparison plus 13,487 interface probe).
Append the motion smoke and this comparison's actual deltas/evaluations when
available. Main counters and ceilings remain unchanged throughout.

## Validated start and accounting update

Sixteen focused reward/training tests passed, including motion similarity,
bounded reward and zero payment for static/retracting/fallen cases. The launcher
passed PowerShell parsing. An isolated imitation smoke completed one full
1,024-transition PPO block with finite checksum-verified learning state and
retained Adam count 16,400. Its full physical evaluator completed eight requested
episodes (6/8 survivors, no qualifying impacts). This tiny update validates the
implementation, not the motion-reference hypothesis. The smoke actor is archived
and is not used to initialize either full arm.

Smoke consumption adds 1,024 diagnostic learner transitions and 3,180 evaluation
world controls. Totals before this full comparison are **265,984 diagnostic PPO
learner transitions** and **77,636 formal striking diagnostic/evaluation world
controls**. At its full two-arm budget, diagnostic PPO will total **528,128**;
append the actual full evaluations separately. The comparison repeats the same
baseline as a control for this reward implementation; repeated inputs are not
new independent behavioral successes.

The hidden Windows host was PID 13636, Linux parent 443 and single baseline child
494 at inspection. The baseline progressed beyond the source with finite saved
learning state and no physics warnings. The main payload hash was unchanged.
[Launch evidence](results/striking-pilot/imitation-comparison-1048576/launched.json),
[smoke evidence](results/striking-pilot/imitation-comparison-1048576/validation.json)
and historical Warp6/Warp7 configurations are archived alongside the report.
No main or foundation learner was active, and the existing monitor is unchanged.

## Completed motion-reference comparison and current interface diagnostic

The motion-reference arm finished with 7/8 pair survival and zero evaluated
impacts; the repeated baseline had 7/8 and three impacts in a falling episode.
Neither qualified or was adopted. Main remains stopped at 1,048,576. The
[matched whole-body authority investigation](striking-body-authority-comparison.md)
records the archived result and the active next comparison. Inspect its durable
body-comparison output before any launch; do not repeat the finished reward
experiments. No main budget extension or foundation promotion is granted.
