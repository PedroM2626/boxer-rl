# Whole-body comparison result and broader fixed-policy assessment

**Current decision (2026-10-23):** all 256 holdout episodes and the contact
intervention audit are complete. Neither controller is adopted. The next bounded
experiment is [single-learner partner practice](striking-partner-curriculum.md),
starting from the preserved original main actor. Main and foundation stay stopped.

## Completed holdouts and contact audit

| Controller / action mode | Both standing at ten seconds | Physical events | Events in surviving episodes |
| --- | ---: | ---: | ---: |
| Zero body / mean | 57/64 | 3 | 0 |
| Adaptive body / mean | 53/64 | 2 | 0 |
| Zero body / sampled | 55/64 | 6 | 0 |
| Adaptive body / sampled | 53/64 | 6 | 2 |

The two surviving events belong to **one episode on one seed**, 1,000,008 without
the side swap. Red scores at 1.640 s and blue at 1.694 s; both remain standing
through ten seconds. Their contact-window impulses are 0.56018 and 1.04431 Ns.
The [actual closed-fist replay](../fights/striking-pilot/stable-two-way-exchange-1179648.mp4)
was checked visually at 2.02 and 9.98 seconds. Its two trial records reproduce
the original held-out rows exactly. This is executable stable two-way striking,
but **1/64 sampled episodes on one seed is not reliable boxing**. The adaptive
controller also has worse survival than its matched control in both action modes.
The original deterministic screening remains failed; no controller is adopted.

The audit replays **all seven scoring episodes**, not just the favorable video.
Every unmodified baseline reproduces its original events, impulses, survival and
elapsed time exactly. In the six falling episodes, remove only the four explicit
opponent-fist-to-victim-head/torso pairs and replay the recorded high-level actions.
The frozen motor remains reactive; roots, legs, inertias and other contacts remain
unchanged. Three counterfactuals avoid the original fall within its observed
horizon, while three still fall. These interventions support a role for incoming
contacts in some failures; they do not make every fall a proven knockout or an
unforced balance failure. Their horizons are censored at the original first fall,
not extended to certify ten-second survival. No modified-physics trial is eligible
for evaluation, training, promotion or boxing evidence.

Separately, zeroing the adaptive body authority while replaying its high-level
actions preserves the stable episode and its one event per corner. Two other
adaptive cases still fall and one avoids the original fall over its short horizon.
Thus the single stable exchange does not establish that extra body authority is
necessary or generally helpful. Full closed-loop removal could differ from this
fixed high-level action intervention.

An earlier attempt to suppress forces using contact `gap` failed its independent
solver-force assay. Its entire interpretation is rejected and archived, together
with the exact attempted tool source. Its **2,970** executed controls remain in
the ledger. The corrected method omits the four explicit pairs at model
compilation; the assay confirms positive baseline force, zero removed-pair force,
unchanged masses/inertias and no assistance. It uses **2,959** controls. Neither
attempt trained a policy. Never use the rejected counterfactuals as causal evidence.

[Raw holdouts, corrected/rejected audits, replay and integrity analysis](results/striking-pilot/body-holdout-1179648/analysis.json)
archive all four result sets and reconcile **113,067** holdout world controls,
**1,000** replay controls and both audit costs. Before the new partner job,
diagnostic PPO totals **794,880 learner transitions**, including 1,536 new smoke
transitions; formal striking evaluation/diagnostic controls total **225,344**,
including twelve smoke evaluator controls. PPO collection controls remain recorded
separately in each diagnostic's physical counter. Main totals are unchanged.

The completed-protocol description below is retained as the pre-result plan.

## Completed learning comparison

The matched zero/0.1-rad body-authority arms completed 131,072 new diagnostic
transitions each. All saved learning leaves/losses were finite, hashes verified,
Adam count was 18,432, and neither log contained physics overflow/invalid-state
warnings. Foundation motor certifications remain intact; they do not certify
the composite controllers. Main remains at **1,048,576 boxing transitions** and
combined main consumption **476,041,216**, with no budget extension or adoption.

| Arm | Training events | Completed training rounds / fall rounds | Deterministic pair survivors | Deterministic impacts |
| --- | ---: | ---: | ---: | ---: |
| Zero body authority | 16 | 148 / 23 | 7/8 | 0 |
| 0.1-rad body authority | 40 | 145 / 21 | 7/8 | 0 |

Neither met the predeclared physical screening. The larger training event count
and similar fall fraction are a reason to investigate, not evidence of stable
boxing: aggregate training logs do not establish which events belonged to rounds
that later survived. Immutable 1,179,648-lineage checkpoints are archived under
`checkpoints/striking-validation/body-comparison-1048576/`. Their 1,048,576 copied
source offset is not additional diagnostic or main consumption.

As a separate check, both saved Gaussian policies were sampled for the original
eight episodes with fixed common random streams. Both survived 7/8 with zero
impacts, just as their deterministic policies did. This does not rescue their
failed deterministic screening. The evaluator's default remains deterministic;
`--sampled` is explicit and requires a checkpoint. Impact/fall rules, reset physics
and ten-second episode duration are identical in both action modes.

[Full comparison, integrity and sampled reports](results/striking-pilot/body-comparison-1048576/completed/analysis.json)
preserve the evidence. The source payloads are:

- Zero body authority: `e95cf8a6ab17ed40ed231035a9e995bdcc9f5299444f7fcce02d2ad05391bb4f`.
- Adaptive body: `3d3e21617adab8c146579cdf66ebf2835894746be33899bd9745884dc74b359f`.

## Coverage investigation rather than another training variation

Eight reset/side episodes are a small sample of a rare-event behavior, particularly
when a few unstable exchanges can generate several training events. Before adding
steps or changing the controller again, evaluate both fixed policies more broadly:

- Seeds **1,000,003 through 1,000,034**, thirty-two new reset seeds, chosen outside
  the current boxing learner's reset-seed ranges and the prior diagnostic sets.
- Both side arrangements for each seed: **64 episodes per job**.
- Both controllers and both deterministic/saved-Gaussian modes: **four jobs,
  256 paired episodes**, up to **128,000 new world controls**, zero optimization.
- Original full ten-second, first-fall trials and unchanged physical scorecard.
  No survivor reset or threshold substitution is allowed.

The same seed/side combinations and sampled noise streams are paired across
controllers. Side arrangements and action modes reuse seed clusters; the 256
episodes are not 256 independent random reset draws. This is a supplemental
diagnostic, not a replacement gate or a new certification. Neither a larger
sample nor sampled actions erase the original failed deterministic screen.

Review all four completed jobs together, including actual impacts by corner,
survival, impacts in complete surviving episodes, scoring seed coverage and
impact-to-fall timing. A fall after an incoming legal impact is **contact-associated**,
not automatically an unforced balance failure; timing alone also does not prove
a causal knockdown. Retain the unchanged survival reading and investigate causal
replays separately if such exchanges dominate. Stable contact and legitimate
knockdowns are distinct physical questions in a boxing project.

If the larger set still contains no stable striking evidence, do not grant another
blind main extension or reward/interface variation. Investigate executable
standing-strike feasibility and exploration before further PPO. If adaptive body
authority yields repeated contacts in surviving episodes across new seeds without
a survival regression against control, review a bounded diagnostic continuation
and broader balance/locomotion checks. That possibility is not a preapproval:
no automatic actor adoption, extra steps or gate promotion is implemented.

## Operation and ledger

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_holdout.ps1
```

Durable read-only output:
`/home/pedro/boxing-runs/striking-body-holdout-1179648/holdout.json`.
Inspect parent state, active job log/error file and live evaluator process before
launching anything. Jobs run sequentially in a persistent hidden WSL host.
Parent world-control totals update at job boundaries; the active log contains
completed trial rows. Archive all four complete JSON reports and reconcile exact
world controls when done. The tool refuses to overwrite an existing holdout and
validates source hashes, complete seed/side sets, finite trials and trial counts.
Main and foundation learners remain stopped; no duplicate policy learner is needed.
The existing 30-minute monitoring schedule remains unchanged.

The completed body comparison brings diagnostic PPO to **793,344 learner
transitions** and formal striking diagnostics to **98,246 world controls**.
The two eight-episode sampled checks add **7,090 world controls**, for **105,336**
before the larger holdouts. No new PPO steps or migration compute were performed
in this review. Main remains 1,048,576 and foundation remains 474,992,640.

The two eight-episode sampled checks completed with finite observations and
verified saved learning states. Both evaluator/job modules passed compilation,
and the hidden launcher passed PowerShell syntax parsing. At launch verification,
the first complete 64-episode control job had passed source/seed/side/finite-state
validation; the adaptive deterministic job was active. Partial results are not
used to select a controller before all four jobs finish.
The hidden Windows host was PID 30132, Linux parent 351 and active evaluator 1298
at inspection. No main, foundation or diagnostic PPO learner was active. Source
learning states were finite, main stopped counters unchanged and no physics
warnings were found. The [launch snapshot](results/striking-pilot/body-holdout-1179648/launched.json)
records hashes, parent progress and the distinction between completed-job counts
and an active job's pending controls. Runtime identities are observations only.
