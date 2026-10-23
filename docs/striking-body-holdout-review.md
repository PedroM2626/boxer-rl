# Whole-body comparison result and broader fixed-policy assessment

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
