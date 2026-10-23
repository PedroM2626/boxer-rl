# Boxing stop at 1,048,576 and controlled reward comparison

The unchanged duration experiment stopped normally on 2026-10-23 at exactly
1,048,576 consumed/optimized boxing learner transitions. The source payload is
`checkpoints/striking-pilot/step-1048576`, SHA256
`efad0ccd1051ee5305d4ccf8ebbd7d0e402211f12ecd0cdf175b473b24a7b5b0`.
The foundation remains certified and stopped at 474,992,640; combined main
consumption is 476,041,216. The main boxing ceiling has **not** been extended.

## Final physical review

The invocation produced 39 training events (red 19, blue 20) over 571 completed
rounds, with 72 fall rounds. These statistics reset per invocation; main learner
counters remain cumulative. Adam count 16,384, all learning-state leaves and
recorded losses were finite. No constraint overflow or invalid-physics warnings
were found. All four complete foundation gate histories and latest hashes still
certify their stages; no curriculum retraining or promotion is requested.

The unchanged eight-episode first-fall evaluation (seeds 7/23/31/47, both side
arrangements, ten seconds) found **6/8 surviving pairs**, versus 7/8 at 524,288
and 8/8 in the earlier 262,144 range diagnostic. Seed 23 without side swap
produced red one and blue two actual qualifying events while both were upright,
then red fell at 2.388 s. Seed 47 without swap fell at 0.808 s. The other six
episodes survived with zero events. Thus both corners can produce valid impacts,
but no complete surviving episode in this evaluation included a scored event.
This does not demonstrate stable boxing or physical superiority to old policies.

The final source is preserved, not rolled back. The prior review required a
controlled comparison if contacts or survival failed to recover. Although some
contacts returned, survival worsened. Another unchanged main duration extension
is therefore rejected in favor of the bounded diagnostic below.

[Raw final results](results/striking-pilot/review-1048576/evaluation.json),
[summary](results/striking-pilot/review-1048576/summary.json),
[integrity](results/striking-pilot/review-1048576/integrity.json) and parsed training
history preserve the evidence. The first-seed video at
`fights/striking-pilot/step-1048576.mp4` contains no impacts; do not present it as
the seed-23 exchange.

## Predeclared diagnostic

Both arms clone the exact same final actor, critic, Adam, normalizer and RNG.
They inherit the same four worlds, 128-control unroll, episode seed derivation,
0.45 m engagement profile and all PPO hyperparameters. Each gets **131,072 new
diagnostic learner transitions**; the combined ceiling is **262,144**, excluded
from main work. Arms execute sequentially under the existing single-learner lock.

1. **Baseline:** unchanged training reward.
2. **Reach:** add `0.25 * dt * exp(-(d / 0.2)^2)` per corner while both stand,
   where `d` is the minimum fist-to-opponent-head/torso centre distance in metres.
   The bonus is bounded to 0.25 reward per simulated second. It changes no action,
   collision geometry, impact threshold, assistance or fall rule.

The hypothesis is that a sustained, hand-specific proximity signal counters the
observed non-contact arm posture better than the existing once-only approach
progress and root-distance reward. This is an experiment: it may reward a static
guard, pressure or increased falls. Only the unchanged independent evaluator can
show executable impacts, and proximity itself never becomes a scored punch.

Each arm is evaluated in a fresh process with all eight original physical
episodes, using production `StrikingEnv`, its saved normalization and fixed
`ImpactScorecard`. The diagnostic reward wrapper is absent from evaluation.
Compare survival, actual impacts, complete-episode impacts, seed coverage and
finite/overflow state, not shaped reward or loss. A useful candidate must reach
at least 7/8 pair survival, match or exceed baseline survival, produce impacts in
complete surviving episodes, and spread impacts across at least two distinct
seeds. These are diagnostic screening criteria, not new curriculum gates or a
boxing certification. Failure means neither arm is eligible on this sample;
do not automatically adopt the less bad policy or extend main steps.

Even a favorable small comparison needs review and broader physical validation.
No automatic policy adoption is implemented. Main actor and counters remain
unchanged, and no diagnostic actor is silently imported. Inspect both complete
arms before making a concrete subsequent decision. If neither improves, examine
the action/body coordination interface and the preserved older unassisted system
rather than repeating this reward experiment without evidence.

## Operation and accounting

The persistent hidden WSL launcher is:

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking_reward_compare.ps1
```

Durable diagnostic output:
`/home/pedro/boxing-runs/striking-reward-compare-1048576`.
Read `comparison.json`, then the active arm's `state.json`, `progress.json`,
`training.log` and `errors.log`. During an arm, parent aggregate counters update
at boundaries: live new diagnostic consumption is the sum of each arm's
`progress.consumed_steps - 1048576`, including attempted/discarded controls.
Child checkpoint counters include their copied lineage offset and must **never**
be added wholesale to main consumption. Each `diagnostic-profile.json` records
the source and tool hash. On completion, archive both manifests/payloads,
configuration, counters and full evaluation reports, then review them.

The tool refuses to replace an existing comparison. Do not launch a duplicate
or resume main while it is active. The original 30-minute monitor is preserved.
The main job remains at `budget_reached`; this is an intentional investigation,
not a learner crash or a claim that productive main PPO is still running.

Final source evaluation adds 3,161 world controls, bringing formal striking
diagnostics to 46,817 before the comparison smoke. Existing diagnostic PPO was
768 learner transitions. Smoke and actual comparison deltas and evaluation
world controls must be appended separately when their results are available.

## Validated launch

Fifteen focused training/reward tests passed; the hidden launcher passed
PowerShell syntax parsing. An isolated two-arm smoke completed 1,024 PPO
transitions per arm from the same source, retained Adam (count 16,400), produced
finite checksum-verified learning states, and ran the full eight-episode evaluator
for each arm. Both survived 7/8; these tiny updates are plumbing validation, not
evidence selecting a reward. The smoke checkpoints and reports are archived under
`checkpoints/striking-validation/reward-comparison-smoke/` and
`results/striking-pilot/review-1048576/smoke/`. Neither initializes the actual arms.

Smoke compute adds 2,048 diagnostic learner transitions and 7,084 evaluation world
controls. Totals **before the actual comparison** are therefore 2,816 diagnostic
PPO learner transitions and 53,901 formal striking evaluation/diagnostic world
controls. At the actual comparison's full budget, diagnostic PPO will total
264,960 transitions; reconcile attempted controls even if an arm fails. Its two
evaluations are not yet included in the world-control total above.

The actual comparison runs through hidden Windows host PID 34284, parent Linux
PID 2034 and the single active baseline learner PID 2079 at launch inspection.
It advanced beyond the copied source with finite verified learning state and no
physics warnings. The main source hash and its 1,048,576 stopped counters were
unchanged. Parent state and active counters are recorded in the
[launch snapshot](results/striking-pilot/review-1048576/comparison-launched.json).
Process identities are observations, not reusable restart targets. No main or
foundation learner was active, and the monitoring schedule remains unchanged.

## Completed proximity comparison and next investigation

Both proximity-comparison arms finished; neither met the predeclared physical
screening criteria, and neither actor was adopted. Main steps remain stopped
at 1,048,576. The [result and current CMU-reference comparison](striking-motion-reference-comparison.md)
record the full evaluations, finite-state checks, factorial amplitude/range
probe, audit of older unassisted configs and separately bounded next diagnostic.
Inspect the current imitation comparison's durable state before any launch;
do not rerun the completed reach comparison.
