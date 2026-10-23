# Neutral-partner result and bounded duration review

**Current status:** the duration test completed after 262,144 new transitions,
stopping on two stagnant checks. The remaining allocation is not being consumed.
The [complete result and exploration coverage audit](striking-exploration-review.md)
supersede the launch status below. No phase passed or main actor was adopted.

## Completed pilot

The first [single-learner partner pilot](striking-partner-curriculum.md) finished
its 131,072 new learner transitions normally. Its original main source, foundation
certifications and counters are intact. All 256 PPO blocks had finite losses;
the final actor, critic, Adam, normalizer and RNG were finite, with verified SHA
`38cdfbee805babc6a40f164c9064ca99e1f3cb70e19d436fd3fd37f79e954a48`.
No physics overflow or invalid-state warning was logged. Full gate histories and
their current foundation payloads were rechecked. The new actor is archived under
`checkpoints/striking-validation/partner-practice-1179648/`, not adopted by main.

| Full deterministic evaluation | Source survivors | Final survivors | Final learner impacts |
| --- | ---: | ---: | ---: |
| Learner red / neutral blue | 6/8 | 6/8 | 0 |
| Learner blue / neutral red | 8/8 | 8/8 | 0 |
| Both corners active | 6/8 | 6/8 | 0 in either corner |

The separate saved-Gaussian partner checks also had zero learner impacts, with
the same 6/8 and 8/8 survival. One neutral partner impact occurred in a falling
episode; it is not learner progress. The phase fails its unchanged physical
criteria, and has zero consecutive passes. The training log's sixteen events
over 285 completed rounds (29 ending in a fall) combine both corners and do not
establish learned stable striking. The source's three learner-red evaluation
events occurred in a falling episode and disappeared after training.

There is a smaller measurable learning signal: among surviving deterministic
episodes, median closest own-hand/target centre distance fell from **0.32551 to
0.27758 m** for red and **0.31106 to 0.27358 m** for blue. These are centre distances,
not impact evidence. Survival counts did not improve, and red's seed-23/original
fall became earlier (3.696 to 1.114 s). The short pilot neither passes the task
nor proves additional duration will solve it.

[Complete source/final reports, training history and integrity review](results/striking-pilot/partner-practice-1048576/completed/analysis.json)
preserve all seven evaluations. They used **24,531 world controls**, including
source checks already counted at launch. Diagnostic PPO now totals **925,952**;
formal striking diagnostic/evaluation controls total **249,875** before the probes
below. Main remains **1,048,576 boxing transitions**, foundation **474,992,640**,
combined main **476,041,216**. The pilot's copied lineage prefix is not new work.

## Rejected navigation explanations

The surviving robots usually finish around 0.60 m apart despite the 0.45 m
approach stop request. Test the fixed final actor without changing physical
contacts, roots, impact rules or first-fall thresholds:

| Diagnostic profile, both learner roles | Pair survivors | Events | Events in surviving episodes | World controls |
| --- | ---: | ---: | ---: | ---: |
| Original (already evaluated) | 14/16 | 0 | 0 | Included above |
| Zero high-level command offsets | 13/16 | 0 | 0 | 6,686 |
| Approach proportional gain 2.0 instead of 0.8 | 7/16 | 15 | 0 | 5,140 |

Zeroing offsets did not remove the approximately 0.60 m separation and added a
fall; the learned offsets are not a demonstrated sole cause. Raising approach
gain brings surviving pairs nearer (roughly 0.51–0.55 m), but greatly increases
first falls. Contact-associated falls are not automatically knockouts, as the
earlier causal audit explains. Reject **both** changes; do not train or adopt
either profile. The evaluator's explicit diagnostic flags record these overrides;
its default navigation, all physical criteria and the training task are unchanged.
[Raw probe reports](results/striking-pilot/partner-practice-1048576/completed/navigation-probe-analysis.json)
record zero optimization and no warning findings. Formal controls now total
**261,701**. The faster-gain results cannot be substituted for the original screen.

## Reviewed finite duration experiment

Keep the original neutral-partner task, reward, 0.8 approach gain, 0.45 m stop,
learned command offsets, free bodies, full contacts and nominal two-metre spawns.
Warm-start the exact completed pilot actor **with Adam, normalizer, RNG and all
consumed counters**. This is a diagnostic continuation, not adoption into main or
a forced rollback. No architecture, reward, exploration scale or physical rule
changes. The reason to test duration is the approximately four-centimetre reach
improvement without aggregate survival loss after only 131,072 transitions in
this new task. The lack of legal strikes and earlier red fall limit that evidence;
they require intermediate independent checks instead of a large unchecked run.

The maximum extension is **524,288 new learner transitions**, four blocks of
**131,072**. After every block, stop the learner normally and run all sixteen
deterministic neutral-partner trials plus the original eight active self-play
trials, with the same seeds, swaps, ten-second durations and physical scorecard.
These are complete physical checks, not training-reward previews. Warm starts
between blocks retain learning state but reset unsaved physical episodes, as
ordinary native PPO resumes already do. Reuse the archived exact-source evaluation
instead of needlessly simulating the same baseline again.

The parent stops automatically for review on:

- Any neutral-role survival regression below its source **6/8 red or 8/8 blue**.
- Two consecutive checks without a new own-scoring surviving role/seed or at
  least **0.01 m** improvement in the average of the two roles' median closest
  hand distances relative to the last qualifying progress checkpoint.
- Two distinct consecutive complete neutral-phase passes: each role must have
  at least 7/8 survival, no regression against source, and own legal impacts in
  surviving episodes on at least two distinct seeds. Intermediate improvement
  alone cannot pass a phase. No automatic main adoption is implemented.
- The four-block budget, even if closer hands continue improving without impacts.

Distance progress is solely a duration-review diagnostic; it does not replace
contact gates or establish boxing. Active self-play reports remain mandatory,
and any eventual main candidate still needs the original screen, broader held-out
seeds and appropriate balance/locomotion regression checks. Sample reuse across
intermediate decisions requires fresh holdouts before claiming generalization.
No further extension is preapproved at this experiment's stop.

## Operation and accounting

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_partner_duration.ps1
```

Inspect **`/home/pedro/boxing-runs/striking-partner-duration-1179648/duration.json`**,
the active child `learner/progress.json`, logs, checkpoint and process before any
launch. The old `striking-partner-1048576/partner.json` is completed historical
state. The persistent hidden WSL parent runs at most one learner under the existing
single-learner lock. Complete check directories retain full results and checkpoint
hashes; the learner directory retains every immutable step payload. A stopped or
failed parent must be inspected before manual recovery; do not recopy its source
over existing learning state. The existing monitoring schedule is unchanged.

The copied source prefix is **1,179,648 learner transitions / 655,360 physical
world controls**, including the previous diagnostic's 131,072 transitions. Subtract
that prefix to count this duration experiment, without adding old main or previous
diagnostic steps again. Each new control is one learner transition and one paired
physical advance. Parent training counters update at block boundaries; inspect
live child deltas for in-progress consumed/optimized work. Discarded attempts stay
in the durable consumed ledger. The live generic `combined_project_consumed_steps`
includes diagnostic lineage and must not be reported as an increased main total.

At the full maximum, this diagnostic's lineage ceiling is **1,703,936**, world
counter **1,179,648**, and cumulative diagnostic PPO **1,450,240**. Actual early
stops may be lower. Full intermediate evaluations add at most **48,000** world
controls, separate from PPO collection, starting from formal count **261,701**.
Main and foundation remain stopped at their counters above.

Sixteen focused tests passed, including the four new duration-decision cases,
role routing and existing PPO/resume contracts. They verify that closer hands
alone cannot promote, survival regression overrides event improvement, two
stagnant checks stop, and two complete consecutive passes stop for review.
The diagnostic evaluator/job modules compile and the hidden launcher parses.

The [verified launch snapshot](results/striking-pilot/partner-duration-1179648/launched.json)
records hidden WSL host 30832, Linux parent 352 and its single learner child 460.
Learning advanced beyond the retained source with finite verified payloads and
matching learner/physical deltas, without physics warnings or duplicate jobs.
The first launcher attempt stopped before Python because its new Bash file had
Windows CRLF endings. Normalizing it to LF fixed startup; that failed attempt
consumed zero learner transitions or world controls and is separately recorded.
Process identities and live counts in the snapshot are observations, not final
results. The parent ledger updates training consumption at block boundaries;
inspect the active child's counters between them.
