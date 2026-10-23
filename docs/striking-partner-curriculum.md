# Single-learner striking practice with a neutral learned partner

**Current status:** this initial pilot completed its 131,072-step allocation and
failed the physical phase screen. The [completed review and bounded duration
continuation](striking-partner-duration-review.md) supersede its launch status.
Inspect the new `striking-partner-duration-1179648/duration.json` before any launch.
The protocol below remains the original pilot specification.

## Decision and bounded allocation

The [completed 256-episode comparison and contact audit](striking-body-holdout-review.md)
find one reproducible stable two-way exchange, but no repeatable advantage from
body authority. Do not extend the failed main symmetric self-play unchanged or
adopt an actor selected from this rare favorable seed. Instead test a simpler
boxing curriculum task: one learner practices striking a neutral, physically
free partner before both corners actively learn to attack simultaneously.

This is an isolated **131,072-new-learner-transition** diagnostic, not an additional
main budget. It starts by copying the exact original **146-input/17-output**
main checkpoint at 1,048,576, SHA
`efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0`.
All actor, critic, Adam, normalizer and RNG leaves are retained without migration.
It imports neither body-comparison actor. Learning rate, original reward, 0.45 m
engagement profile, timestep, scorecard and ten-second first-fall rounds are
unchanged. There is no motion-reference reward, scripted punch or root assistance.

The neutral partner receives zero high-level arm residuals and command offsets.
Its frozen certified motor still observes the actual simulated body, balances,
walks toward the other robot and reacts through all 29 joints. It is not anchored
or held upright; either corner can fall and terminate the round. Both free roots,
twelve leg joints, floor, fist, head, torso and blocking contacts remain active.
The robots still spawn nominally two metres apart with the original randomization.
Neutral motor motion may itself produce a legal event; the scorecard records both
corners independently. This stage teaches attacking a simpler partner, not defense
against a trained attacker.

Only one corner contributes PPO observations, actions and rewards per world.
Seed parity selects the learner corner and alternates it at subsequent resets;
side swaps remain independent. The wrapper returns that corner's unchanged
observation and reward. The other corner's data are not mislabeled as learner
transitions. Four worlds and 128-control unrolls therefore form **512-row** PPO
blocks, with the original four epochs/four minibatches. The smaller learner batch
is an explicit task change, so results cannot isolate opponent difficulty from
batch-size effects. This is a feasibility curriculum pilot, not a matched claim
of algorithmic superiority.

## Physical review before continuation

Before training, evaluate the source against the neutral partner with each learner
corner separately, seeds 7/23/31/47 and both side arrangements: **16 full trials**.
After training, repeat those same sixteen deterministic trials, sixteen explicit
saved-Gaussian diagnostic trials, and the original **eight full deterministic
self-play trials with both corners active**. Total evaluation allocation is at
most **28,000 world controls**; first falls reduce actual consumption. Saved
checkpoints are finite/hash/provenance checked by the evaluator. Neutral-partner
reports explicitly identify the learner corner and cannot replace the original
self-play screen. No contact intervention from the causal audit is enabled.

For a promising neutral-partner phase result, require at least **7/8 surviving
pairs for each learner role**, no survival regression against the source in either
role, and own legal impacts in full surviving episodes on at least **two distinct
seeds for each role**. One lucky seeded exchange or sampled-only contacts do not
pass. A phase promotion would require two distinct consecutive complete checkpoint
passes; this launcher never promotes automatically. The original self-play screen
and downstream balance/locomotion regressions still govern any main adoption.
Success at this easier task does not establish defense or complete the boxing
project. Failure triggers policy/contact/exploration review before any new finite
allocation; it does not silently grant another 131,072 steps.

If this task earns two complete passes, the next proposed task is best-response
training against a frozen attacking checkpoint that passed it, followed by an
opponent pool and active two-sided sparring. Those stages are not implemented or
preapproved budget extensions here. Each needs explicit physical evaluation and
its own reviewed finite allocation; all retain free bodies and separated spawns.

## Operation, provenance and accounting

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_partner.ps1
```

Inspect `/home/pedro/boxing-runs/striking-partner-1048576/partner.json`, its
`active_evaluation`, `learner/state.json`, `learner/progress.json`, all logs and
live process before launching anything else. The hidden persistent WSL parent
runs source checks, one learner child under the existing single-learner lock, and
the full final evaluations sequentially. The parent refuses to overwrite an
existing experiment. Inspect any stopped/failed job before recovery; the child
supports ordinary warm resumes with its saved config and the same explicit
ceiling. Never recopy the original source over a partially trained child.

The source prefix is **1,048,576 learner transitions / 524,288 world controls**.
Subtract that prefix from this diagnostic's lineage counters. Every new attempted
paired control consumes one learner transition and one world control, rather than
the old two learners per world. Explicit physical counters persist in progress,
payloads and manifests, retain attempted/discarded work, and preserve old implicit
two-learner prefixes. At the full allocation the lineage is 1,179,648 learner
transitions and 655,360 physical controls; neither is a new main counter.
Parent training totals update at child completion; inspect live child deltas while
training. Evaluation compute is separately accumulated at complete-job boundaries.

Main remains stopped at **1,048,576 boxing transitions**, foundation at
**474,992,640**, combined main **476,041,216**. Before this new job, diagnostic PPO
totals **794,880**, including the 512-row partner and 1,024-row legacy smoke tests.
At its full budget it will total **925,952**, without further smoke/diagnostic work.
Formal striking diagnostics/evaluations start this job at **225,344 world controls**;
add actual source/final evaluation controls separately from PPO collection.
The existing thirty-minute monitoring schedule is preserved.

## Validation

**53 focused tests passed**, including exact physical/reward agreement between
the role wrapper and the original two-body environment for both learner corners,
zero partner high-level commands, contact-force removal, scorecard and complete
curriculum promotion behavior. The wrapper changes data collection, not physical
constraints. A real 512-transition partner PPO smoke completed with finite learning
state and exactly 512 new physical controls. The default two-learner 1,024-transition
smoke remained bit-identical in every actor, critic, optimizer, normalizer and RNG
leaf to its archived pre-change counterpart. Only physical-counter metadata was
added. Three two-control evaluator checks covered default and both neutral roles.
The hidden launcher passed PowerShell parsing. Smoke actors are archived under
`checkpoints/striking-validation/partner-practice-smoke-1048576/` and initialize
neither main nor the full diagnostic.

## Verified launch

The full job completed all sixteen source trials: learner-red trials survived
6/8, learner-blue trials 8/8, with no impacts in surviving episodes in either
role. They used **7,226 evaluation world controls**, so formal diagnostic/evaluation
consumption is **232,570** before the final checks. This is the measured starting
point, not a phase pass. The
[launch snapshot and source reports](results/striking-pilot/partner-practice-1048576/launched.json)
pin the checkpoint, code hashes, counters and live finite-state inspection.

Training then started in hidden Windows WSL host 20676, Linux parent 499 and its
single learner child 1105. The first complete 512-row update had a verified finite
payload and exactly 512 additional physical controls. No other main, foundation or
diagnostic PPO process was active and no physics warnings were found. Parent
training totals remain zero until child completion; its live progress ledger is
authoritative for current diagnostic work. These are observations at launch, not
permanent process identities or a completed training result.
