# Unassisted opponent-aware striking pilot

## Physical source and new policy

The foundation remains certified and stopped at **474,992,640 consumed main
transitions**, within its 475M ceiling. Its final payload SHA-256 is
`b39b1990945e8a374802c3a190e5ac5e026ae39e7fd5ef61c5c2b01ad291c6fc`.
Both corners retain that frozen actor, normalizer, 29 joint actions, free legs,
original feet/inertias, 2 ms physics and 20 ms control. No springs, root forces,
welds, gravity compensation or scripted punch/gait cycles are added.

`StrikingEnv` adds a separate shared self-play policy with **146 inputs**:
103 original proprioceptive values, relative opponent root/head/torso and both
corners' fists, relative root velocity, upright values and previous high-level
action. Its **17 outputs** are three bounded local command offsets and fourteen
arm residuals. Commands ask the learned motor actor to move; they do not apply
root forces. Residuals affect only shoulder/elbow/wrist targets, with a 0.8-rad
maximum and a 0.8/0.2 temporal filter. The motor actor still controls every leg
joint using current physical feedback. Zero residuals preserve its outputs
exactly. Nominal mutual approach now stops around 0.58 m, deliberately entering
potential striking range; initial spawns remain nominally 2 m apart.

The new actor has two 128-unit tanh layers and an exactly zero initial mean head.
Exploration starts with pre-tanh standard deviation exp(-1.9), clipped during
training to exp(-3) through exp(-0.5). There is no pose/clip imitation or scripted
extension/retraction prior. Both corners learn the same residual policy.

## Independent physical impact events

`ImpactScorecard` credits only an actual fist-to-opponent-head/torso contact.
It integrates normal constraint force into impulse, requiring at least **0.5 N s**
in the first **80 ms** of contact, relative closing speed at least **0.3 m/s**,
and fist extension toward the target at least **0.2 m/s relative to its own
moving/rotating torso**. Velocities are sampled before the collision solver.
Walking a motionless guard into the opponent therefore does not qualify as a
punch. Both robots must still be upright. A hand scores once per event, rearms
after 100 ms without legal contact and has a 200 ms scoring cooldown.
Head/body contact slots cannot double-count the same hand event. Forearm, leg,
pelvis, same-corner and blocking contacts do not independently score.

Tests reject static catches, whole-body leaning, fallen rounds, weak sustained
force and contact chatter. A native MuJoCo assay placed legal surfaces in
contact and gave only the red arm an incoming velocity: it produced a red event
and zero blue events even though both corners had raw contact impulse. This is
an engineered scorecard test, not evidence of a learned punch.

Training pays +2 for a qualifying event, -2 for one received and -5 for falling.
Small shaping terms reward facing, engagement and reduction in hand-to-target
distance; arm residual magnitude has a cost. Shaping is disabled after a fall.
Shaping/reward does not certify boxing: independent evaluation reports actual
impact events, falls, distance and finite state. First fall ends both agents'
episode; the ten-second round clock also cuts PPO bootstrapping.

## Validation and bounded allocation

The [independent initial evaluation](results/striking-pilot/interface-baseline.json)
ran seeds 7/23/31/47 with both side arrangements: eight neutral and eight bounded
exploration episodes, ten seconds each. All **16/16 paired episodes survived**
with finite observations and **zero scored impacts**. This tests only one reset
sample per seed/side; it does not erase the earlier paired-transfer fall or
standing drift documented in the previous report.

An isolated PPO smoke used 512 learner transitions and completed finite updates.
A local-filesystem smoke used 64 more, then ordinary resume added 64 while
retaining actor, critic, normalizer, Adam/RNG and counter. Those **640 diagnostic
learner transitions** (320 world-control transitions) are separately archived;
neither smoke head initializes the main pilot. The interface evaluation adds
8,000 diagnostic world transitions. Unit tests and controlled contact assays
are engineering validation. The [launch decision](results/striking-pilot/launch-decision.json)
records the allocation and evidence.

The new pilot ceiling is **32,768 boxing learner transitions total**, or at most
16,384 world-control actions, across four paired worlds. It is a bounded new
boxing pilot, not a foundation restart or an automatic large-budget extension.
There are 32 PPO iterations of 128 controls x8 learner rows, four shuffled
minibatches and four epochs per iteration, Adam 1e-4, gradient norm limit 1,
PPO ratio clip 0.2, gamma 0.99 and GAE lambda 0.95. Normalization stays fixed
through collection and all updates, then updates from collected data. This
pilot is too small to imply learned boxing. Review final policies and full
physical results before a justified later allocation.

## Persistent operation, accounting and resume

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking.ps1
# After a stopped run, keep the reviewed ceiling and matching profile:
powershell -ExecutionPolicy Bypass -File tools/start_striking.ps1 -Resume
```

A hidden Windows WSL host runs native MuJoCo and JAX CPU. Durable output is
`/home/pedro/boxing-runs/striking-pilot-v1`; `runs/striking-pilot` mirrors its
latest complete checkpoint/config/state after every PPO iteration.
`runs/striking_host.json` and `runs/striking_host*.log` identify the actual host.
Read the local Linux `state.json`/`progress.json` when the mirror lags.
Per-action fsync on the mounted Windows drive reduced the smoke to about 12
learner steps/s; local Linux durable writes measured about 39-56 in short smoke
sessions. These are startup-scale CPU measurements, not GPU training throughput.
GPU batching remains future performance work after action/reward validation.

Both output and a global boxing namespace lock reject duplicate jobs. Every
attempted paired control action is charged as two consumed learner transitions
in the durable local ledger, even if a later world/update fails. Checkpoints
store actor/critic/normalizer, optimizer, RNG, source hash and optimized/consumed
counts; filenames and manifests carry SHA-256. Partial collected work can be
discarded on recovery but its consumption is retained. Simulation state is not
saved: resumes reset physical worlds and explicitly retain learning state.
The global consumed project figure in pilot progress is foundation plus this
pilot; earlier smoke/diagnostic compute remains separate. Never overwrite
foundation counters or import an easier diagnostic gate as boxing certification.

At a budget stop, run:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu MUJOCO_GL=egl \
  "$HOME/venvs/boxing_curriculum/bin/python" tools/evaluate_striking.py \
  --checkpoint runs/striking-pilot/checkpoint \
  --output runs/striking-pilot/final-evaluation.json \
  --video runs/striking-pilot/final-evaluation.mp4
```

Compare actual events and falls against the initial neutral/exploration baseline,
inspect complete traces and checkpoint provenance, archive the final actor and
results, then make a concrete bounded continuation decision. No training reward
or pilot step count establishes boxing completion. Keep the existing 30-minute
monitoring schedule and inspect the boxing host/output rather than restarting
the already completed foundation job.

## Launch inspection

The final combined CPU suite passed **58 tests**, including original curriculum
certifications, exact neutral output, arm-only residual authority, true native
active-fist scoring, passive-contact rejection, episodic bootstrap boundaries,
numerical warning rejection and duplicate-job exclusion. PowerShell launcher
syntax passed. A separate one-second replay of the archived smoke checkpoint
validated checksum loading, saved observation normalization and finite deterministic
inference in both corners (100 more diagnostic world transitions, zero updates).

The hidden Windows WSL host launched as PID 15016; its Linux learner was PID 498
at inspection. Exactly one new boxing learner was active and no foundation
learner remained. The source hash and 32,768 ceiling match the reviewed decision.
At 8,192 optimized/consumed pilot transitions it had completed eight rounds,
with no falls, finite losses and no scored impacts. Short sustained throughput
was about 164 learner transitions/s on local Linux storage. This is active
learning plumbing evidence, not learned punching. Source code SHA-256 values are
saved with the launch decision; the monitor should read the latest live state.

## First budget review and bounded continuation

The 32,768-transition run ended normally at its exact ceiling after **32 PPO
updates and 32 completed training rounds, with zero falls and zero scored hits**.
All policy/critic/Adam/normalizer/RNG leaves are finite and the payload matches
its manifest and evaluated hash. The complete snapshot is archived under
`checkpoints/striking-pilot/step-32768/`.

The [independent final evaluation](results/striking-pilot/evaluation-32768.json)
used all four seeds and swapped sides, ten seconds each: **8/8 paired episodes
survived**, with **zero actual legal-target contacts and zero scored impacts**.
Mean absolute filtered arm residuals were about 0.029-0.038 (before the 0.8-rad
scale), so the new head is active but movements remain small. The closest
hand-to-target centres were about 0.301-0.374 m, still short of legal surfaces.
The video `fights/striking-pilot/step-32768.mp4` was visually inspected and
shows physical footwork and guard, not learned punches. Its caption
identifies it as an unassisted striking pilot evaluation.

**Decision:** extend this same boxing pilot to a **262,144 total learner-step
ceiling**, adding at most **229,376** transitions. The first short run establishes
that PPO and bounded residual exploration preserve physical standing; only 32
updates have occurred, so a longer bounded same-profile segment is reasonable
before attributing the missing contacts to a learning plateau. This is not
boxing certification or evidence that more steps alone will solve it.
Preserve the exact learned head, frozen motor, Adam, RNG and observation stats;
use ordinary resume. Physics, reward and independent impact thresholds remain
unchanged. The [decision record](results/striking-pilot/continuation-32768.json)
records the limits and provenance. The source foundation remains stopped at
474,992,640, and diagnostic learning remains separately counted at 640 steps.

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking.ps1 -Resume -Ceiling 262144
```

At the next budget review, missing actual contacts or stationary residuals
require a concrete arm-reach/exploration investigation before another extension.
Inspect held-out physical behavior, not just shaped reward or optimization loss.
Round/fall/hit statistics in live progress are per invocation; consumed and
optimized transition counters are cumulative across resumes. Do not interpret
a session's reset statistics as a rewind of consumed training.

Formal new interface/evaluation diagnostics total **12,100 world-control
transitions** (8,000 neutral/exploration baseline, 100 checkpoint replay, 4,000
post-pilot evaluation), in addition to the separately preserved earlier paired
transfer and tracking investigations. No evaluation transition enters PPO.

The resumed hidden host was verified as Windows PID 34812 and Linux learner
438 at launch. Exactly one boxing learner and no foundation learner were active.
Its consumed/optimized counters advanced beyond 32,768 with finite saved learning
state and a verified checkpoint hash under the new 262,144 ceiling. The source
foundation payload and inherited profile remained unchanged; the existing
30-minute monitor schedule was preserved. Runtime identities are recorded in
`results/striking-pilot/continuation-launched.json`, not reusable restart targets.

## First physical impacts reproduced during continuation

The [97,280-step snapshot review](results/striking-pilot/first-observed-impacts/README.md)
confirmed the first independent events: all eight deterministic paired episodes
survived, and blue landed one qualifying left-hand event in each of the two
holdout-seed-31 side arrangements. The remaining six episodes scored zero.
Training had recorded events from both corners, but deterministic evaluation
has so far validated blue only. Events were replayed at 4.046/4.104 s with
unchanged thresholds; actual video and exact timing are archived. The evaluator
now logs the scoring hand, timestamp, contact-window impulse and first-fall flags
for each event, without changing simulation/actions or score classification.

Continue the unchanged learner under the existing **262,144** ceiling. This is
progress from zero contacts, not boxing completion or a promotion. The snapshot
was isolated while training kept its resident optimizer; no restart, rollback,
new budget or duplicate learner was introduced. No new policy source was adopted.

## Range correction after the 262,144 stop

The [engagement-range investigation](striking-range-recovery.md) records the
final regression, measured workspace and unchanged-policy physical comparison.
The reviewed continuation retains the latest 262,144 actor/Adam/counters and
changes the training approach/reward range to **0.45 m** under a **524,288 total
boxing learner-step ceiling**. Initial spawns remain nominally 2 m and physical
impact criteria remain unchanged. Ordinary resumes inherit this saved profile.
Read that report and its decision record before a new intervention; historical
32,768/262,144 allocations above are not the current ceiling.
