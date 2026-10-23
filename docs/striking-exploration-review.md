# Partner duration stop and physical exploration coverage

**Current status:** the coverage probe completed all 64 trials. Every profile
failed to demonstrate stable striking; none is adopted. The next active work is
[bounded target-directed physical feasibility](striking-directed-feasibility.md),
with all learners stopped and no additional PPO allocation.

## Completed duration review

The [duration experiment](striking-partner-duration-review.md) stopped normally
at its predeclared **two stagnant checks**, after **262,144 new learner
transitions**. The remaining 262,144 of its maximum allocation are unused, not
an entitlement to resume. No phase passed, no actor was adopted, and no physical
criterion changed. Main and foundation remain stopped.

| Lineage checkpoint | Neutral red survivors | Neutral blue survivors | Own impacts in surviving partner episodes | Active self-play survivors / events |
| --- | ---: | ---: | ---: | ---: |
| Source 1,179,648 | 6/8 | 8/8 | 0 | 6/8 / 0 |
| Check 1,310,720 | 6/8 | 8/8 | 0 | 5/8 / 2 |
| Check 1,441,792 | 7/8 | 8/8 | 0 | 7/8 / 0 |

The first checkpoint's two self-play events and two learner-red partner events
occurred in episodes ending in a fall. The second checkpoint had no events in
any of the 24 complete trials. Its better survival is not boxing mastery.
Mean role-wise median closest hand distance worsened from **0.27558 m** at source
to **0.30749 m** and then **0.30910 m**. Training event totals also cannot rescue
the failed independent checks: block one recorded 63 events over 322 rounds with
78 fall endings; block two recorded 24 over 280 rounds with 28 fall endings.
Those totals combine learner and neutral partner, without assigning stable-round
credit to the learner.

Both immutable payloads, all **48 full physical trials**, reviews, complete
512-block training history and runtime counters are archived under
[completed duration results](results/striking-pilot/partner-duration-1179648/completed/analysis.json).
The checkpoints are `checkpoints/striking-validation/partner-duration-1310720/`
and `partner-duration-1441792/`. The final SHA is
`87eb01ff6392b592395b64fc6f6a29f78e35fd1c1ad2941fafbcc8001146f5d8`.
Every saved actor, critic, Adam, normalizer and RNG leaf and every logged loss
was finite. No overflow/invalid-state warning was logged. The full current
foundation histories and checkpoint hashes remain valid.

Diagnostic PPO now totals **1,188,096 learner transitions**. The duration checks
used **21,303 evaluation world controls**, bringing formal striking diagnostics
to **283,004** before the coverage probe below. Main retains **1,048,576 boxing
transitions**, foundation **474,992,640**, combined main **476,041,216**. The generic
live combined counter includes diagnostic lineage and is not main advancement.

## Why inspect exploration before another training allocation

The saved final Gaussian has arm standard deviations around 0.13–0.15 before
`tanh`. The environment then applies a 0.8-radian residual bound and a filter
`r[t] = 0.8*r[t-1] + 0.2*a[t]` every 20 ms. For independent zero-mean noise in
the locally linear approximation, this filter reduces standard deviation by
`sqrt(0.2/1.8) = 1/3`. Thus per-tick exploration can become a small arm
perturbation rather than a large, temporally coherent excursion. This calculation
is a hypothesis about coverage, not proof of the sole cause of failure; actual
means, `tanh`, feedback and contact dynamics matter.

Do not blindly extend the unchanged learner, raise navigation gain, replace the
actor with a favorable earlier seed, or relax impact/fall criteria. The next
authorized work is **read-only coverage evaluation**, using the exact latest
1,441,792 checkpoint, its saved normalizer and full native physics.

## Predeclared factorial probe

Compare four sampled action profiles: arm-noise scale **1 or 3**, crossed with
noise refresh every **1 or 10 controls** (20 or 200 ms). Policy means still react
to observations every control. Holding a perturbation is not holding an action,
joint pose, leg controller or root. Only fourteen arm-noise channels change;
command noise and any body channels retain their original independent draws.
The physical residual bound/filter, frozen 29-joint motor, free roots/legs,
nominal two-metre spawn, 0.45 m engagement profile and all contacts stay intact.
No punch cycle, motion playback, physical assistance or optimizer is used.

Draw the full Gaussian noise array every control in every profile, then reuse
the selected arm draw for the declared interval. This preserves common command
noise streams across profiles. Each profile evaluates both learner roles against
the same neutral learned motor partner, seeds 7/23/31/47 and both side arrangements:
**16 full ten-second first-fall episodes per profile, 64 total**, at most **32,000
new world controls**, zero PPO. Seeds/sides are paired and reused, so this is not
64 independent reset seeds or a fresh generalization benchmark.

Compare actual own events, own events in complete surviving episodes, scoring
seed coverage, pair survival, closest hand distance and residual amplitude. Inspect
all four profiles together before a decision. None can replace the original
deterministic screen or promote a phase. A promising coverage result would need
contacts in surviving episodes across multiple seeds and both roles without
survival loss against the saved-Gaussian control; a few falling impacts do not
justify an exploration change. If no profile finds stable striking, investigate
target-directed executable motion feasibility before allocating another PPO run.

Temporal correlation changes the stochastic process. It must **not** be inserted
into existing per-tick PPO while retaining independent-Gaussian log probabilities.
Any later learner would need a correct decision/conditional-action model and
physical-step accounting, such as an explicitly validated action-duration design.
This probe evaluates coverage only and preapproves no such training migration.

## Operation and validation

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_exploration_probe.ps1
```

Inspect `/home/pedro/boxing-runs/striking-exploration-probe-1441792/probe.json`,
the active evaluator's log/errors and live process before any launch. The hidden
persistent WSL host runs the eight role/profile jobs sequentially, validates source
hashes and complete finite seed/side trials, and refuses to overwrite an existing
probe. Its control ledger updates at complete-job boundaries; the active log
contains completed trial rows. Main, foundation and diagnostic PPO stay stopped.
The existing monitoring schedule is unchanged.

Six focused tests passed: the two new noise tests prove bit-identical default
Gaussian draws and preservation of command/body draws under held/scaled arm noise;
the four existing duration tests retain their regression, stagnation and complete
phase-pass behavior. The evaluator/probe compile and the hidden launcher parses.
New flags require explicit sampled diagnostics and record scale/hold in the report.

## Completed coverage result

| Arm noise refresh / scale | Pair survivors | Own events | Own events in surviving episodes |
| --- | ---: | ---: | ---: |
| Every control / 1x | 15/16 | 0 | 0 |
| Every ten controls / 1x | 15/16 | 0 | 0 |
| Every control / 3x | 15/16 | 0 | 0 |
| Every ten controls / 3x | 0/16 | 3 | 0 |

Holding or scaling alone did not find a strike. Their combination produced three
own events while every pair fell. This sample rejects all profiles as remedies;
it does not prove every possible exploration strategy fails. Do not import the
held/scaled process into PPO or consume the stopped duration job's unused steps.
[All raw profiles and the completed analysis](results/striking-pilot/exploration-probe-1441792/analysis.json)
retain the exact fixed source hash and physical results. All 64 trials were finite
and all evaluator error logs were empty. Actual cost was **24,835 world controls**,
bringing formal diagnostics to **307,839**, with diagnostic PPO still **1,188,096**
and main/foundation unchanged. The next test investigates executable directed
motion through the existing action interface, rather than another noise variant.
