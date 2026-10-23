# Target-directed arm feasibility diagnostic

**Current status:** all 64 initial trials and four exact physical replays are
complete. Four trials contained a directed left-fist event and full ten-second
survival; overall controller survival was poor, so it is not adopted. The fixed
variant's sixteen-seed holdout also completed: fifteen successful trajectories
were exactly replayed. The next [bounded supervised actor initialization](striking-demonstration-initialization.md)
uses those verified demonstrations, with independent network-only physical checks.
PPO and main remain stopped. The completed protocols/results remain below.

## Purpose and limits

The [duration and exploration reviews](striking-exploration-review.md) found no
stable learned striking. Main and diagnostic PPO remain stopped. This experiment
asks whether a simple closed-loop geometric correction can execute a legal left
fist contact and remain standing through the complete episode using the **existing
0.8-radian residual/filter interface**. It is a diagnostic controller, **not a
trained boxer, a curriculum pass, a scripted product demonstration or a main actor**.
No optimization transitions or training budget are allocated.

Use the latest fixed 1,441,792-lineage policy, SHA
`87eb01ff6392b592395b64fc6f6a29f78e35fd1c1ad2941fafbcc8001146f5d8`,
its normalizer and the certified frozen motor. The saved policy supplies the
controlled robot's navigation and other arm commands. The partner receives zero
high-level residuals, retaining its reactive learned motor. Both robots start
nominally two metres apart with original randomization; every root, leg, floor,
fist, head, torso and blocking contact remains free and active. No position is
teleported during a rollout, no model/inertia or constraint is changed, and no
root force, weld, gravity compensation or balance support is introduced.

## Exact intervention

Once actual root distance is below 0.75 m, compute the left fist's translational
MuJoCo Jacobian with respect to the seven left arm joints. Solve a damped
least-squares step toward the other robot's actual head or torso centre, damping
0.05 and limiting each angular correction to 0.06 rad per control. Predict the
same frozen motor evaluation used by the next real physics step, then convert
the proposed joint target into the existing arm residual coordinates. Invert
the existing filter and clip the resulting high-level action to [-1,1]. Saturation
is allowed to prevent an infeasible request from bypassing the interface. The
real environment still applies the original filter, residual bound, joint servos,
joint limits and ten native physics substeps per 20 ms control.

Jacobian calculation and motor prediction are read-only; all changes occur through
`StrikingEnv.step` with bounded 17-dimensional actions. There is no direct servo
override. After the left fist earns one legal event, disable the geometric
correction and return that arm to the saved policy for the rest of the ten-second
episode. There is no repeating timed punch cycle. This intervention supplies
target direction; it is not evidence that the learned policy discovered it.

Compare the saved navigation against one separately labeled diagnostic approach
variant: bounded forward high-level command `clip(2*(distance-0.5),0,1)` for the
controlled robot only. The partner remains neutral. This is an action within the
existing command-offset box, not the rejected global gain-2 controller or a
training-profile adoption. Its results remain separate. The source navigation
is not silently changed for the original variant.

Head/torso target, original/bounded-advance navigation, both controlled corners,
seeds 7/23/31/47 and both side arrangements give **64 full ten-second first-fall
trials**, at most **32,000 attempted world controls**. Use the independent original
scorecard, including impulse, relative closing/own-fist extension, both-standing,
release and cooldown conditions. An event on either legal surface can score;
intended target direction does not relabel the actual contact. Record hand-specific
event times/impulses, falls, active correction controls, closest distances and
complete survival. Falls before correction activates are distinguishable from
failures after the intervention. Do not reset a fallen trial to finish its clock.

Save observation/action trajectories only when the directed left fist scores and
both robots finish standing. These are diagnostic demonstrations with exact source
and control provenance, not learner rollouts or proof of generalization. If such
trajectories exist, review their forces and replay first, then use fresh seeds
before proposing distillation or another finite learner. If none exist, investigate
reach saturation and balance/control compatibility; do not reward static overlap,
weaken physical gates or blindly restart PPO. No policy adoption or automatic
curriculum advancement is implemented.

## Operation, accounting and validation

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_directed_feasibility.ps1
```

Inspect `/home/pedro/boxing-runs/striking-directed-feasibility-1441792/feasibility.json`,
`progress.json`, hidden-host logs and live process before another launch. The
parent refuses to overwrite an existing run, verifies source hash/finite learning
state and motor provenance, and writes a small attempted-control ledger before
each real physical advance. Full trial records update the parent state; inspect
progress for the active trial's cost. The persistent hidden WSL host uses CPU JAX.
Main, foundation and diagnostic learners remain stopped. The existing thirty-minute
monitoring schedule is unchanged.

Starting formal striking diagnostic/evaluation consumption is **307,839 world
controls**; append actual attempts. Diagnostic PPO remains **1,188,096**. Main
remains **1,048,576 boxing transitions**, foundation **474,992,640**, combined main
**476,041,216**. Source lineage prefixes are not new work.

Three focused tests passed across the directed helper and exploration sampling:
the geometric calculation returns bounded finite actions while preserving partner
commands, qpos, qvel, servo controls and zero external generalized/body forces;
the paired model has no equality support or gravity compensation. The helper's
unit setup places bodies closer solely to exercise its Jacobian math; runtime
trials always use ordinary separated resets. The evaluator/probe modules compile
and the hidden launcher parses. Physics results still require all full trials.

## Initial result and replay integrity

| Intended direction / navigation | Pair survivors | Left-fist scoring episodes | Left-fist scoring episodes completing ten seconds |
| --- | ---: | ---: | ---: |
| Head / saved navigation | 0/16 | 0 | 0 |
| Head / bounded advance | 3/16 | 3 | 3 |
| Torso / saved navigation | 0/16 | 0 | 0 |
| Torso / bounded advance | 1/16 | 2 | 1 |

The four complete trajectories cover three seed/side/role arrangements on only
two seeds (7 and 31); the torso success repeats a seed-7 arrangement. They are
not four independent generalization successes. Intended head direction does not
prove head contact; the original scorecard accepts either legal surface. Recorded
left-fist event windows in the surviving trials were **1.57758, 1.19877, 0.65160
and 0.84925 Ns**. Many failures occurred after geometric correction, and the
controller's poor aggregate survival precludes adoption or a boxing claim.

All four saved observation/action trajectories were replayed from their original
separated resets **without the geometric helper**, applying only the recorded
bounded high-level actions through the normal environment and frozen motor.
Every one of the 500 observations per trajectory, event time/impulse, final score
and ten-second survival matched exactly. The
[labeled replay](../fights/striking-pilot/directed-feasibility-diagnostic.mp4)
was visually checked at 2.02 and 9.98 seconds; its caption explicitly states
"Not a trained boxer". These are physically reproducible diagnostic demonstrations,
not the current learned actor's independent behavior.

The first replay converted saved float32 actions to float64 and correctly failed
the exact observation comparison after two controls. Reproducing that mismatch
used two more controls. Preserving the saved dtype fixed the residual-filter
rounding difference; exact comparisons were kept, rather than loosened. Both
rejected attempts and their source are archived, and their four controls remain
counted. [Full trials, four compressed trajectories, exact replay and analysis](results/striking-pilot/directed-feasibility-1441792/analysis.json)
also retain the original diagnostic source before adding held-out CLI options.

Actual initial cost was **12,946 world controls**, plus **2,000** successful replay
controls and **4** rejected/reproduction controls. Formal striking diagnostics
therefore total **322,789 world controls** before the held-out run. All 64 initial
trials were finite. Diagnostic PPO remains **1,188,096** and main remains unchanged.

## Fixed held-out demonstration collection

The head-directed/bounded-advance variant had three complete successes in the
initial set, versus one for the torso variant. Freeze that configuration and all
source weights; change only reset seeds to **2,000,003 through 2,000,018**. Both
controlled roles and both side arrangements give **64 complete trials** across
sixteen new seed clusters, at most **32,000 attempted world controls**, zero PPO.
These seeds are distinct from the earlier duration/coverage/teacher screens and
the one-million-range body holdouts. Do not tune parameters from partial results.
The trial controller and physical rules are identical; the new CLI selects a
subset of the same configuration space and records its full seed/profile list.

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_directed_holdout.ps1
```

Inspect its `feasibility.json`, `progress.json`, host logs and process before any
launch. Source checking, per-attempt accounting and complete-success-only trajectory
exports are unchanged. The maximum formal count after this allocation is 354,789;
append actual consumed controls instead. Main, foundation and PPO stay stopped.
Review the whole set, successful action provenance, contact events and failures
before proposing supervised skill initialization or a new finite training job.
Any distilled actor must be evaluated as a learned controller without the helper;
teacher existence, hand direction and video playback cannot pass its physical gates.

The held-out run completed at **15,802 world controls**: red-role 9/32 complete
left-fist scoring trials, blue-role 6/32, total **15/64**, with 49 first-fall
failures. All 64 trials were finite. Exact replay of all fifteen complete
trajectories used **7,500** further controls and matched every observation and
physical event. [Held-out results, compressed actions and replay](results/striking-pilot/directed-holdout-1441792/analysis.json)
bring formal diagnostics to **346,091 world controls**. No source policy or
geometric controller is adopted. The next supervised experiment must establish
independent learned behavior and cannot count these teacher successes as its own.
