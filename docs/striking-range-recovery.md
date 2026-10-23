# Striking engagement-range recovery

## Budget stop and actual regression

The boxing run stopped normally at **262,144 consumed/optimized learner
transitions**. Foundation consumption remains **474,992,640** and all four
foundation certifications are unchanged. The source policy SHA-256 is
`191dc2b107cd84210bc5bb0a95c6b916713718a32b08893c42b4d81ae28d71bc`.
The complete latest policy/config/Adam/normalizer/RNG/counters are preserved at
`checkpoints/striking-pilot/step-262144/`.

The resumed training session completed 238 rounds, including 13 first-fall
endings, and recorded 10 red plus 5 blue qualifying events. The last event
increment was logged at 172,032; roughly 90k further transitions scored none.
The [final independent evaluation](results/striking-pilot/review-262144/final-evaluation.json)
had **8/8 surviving pairs but zero raw legal contacts and zero qualifying hits**.
This regressed from the 97,280 actor's two events; neither low PPO loss nor
survival makes it learned boxing. The final policy/optimizer leaves are finite,
payload checksum matches, and no overflow/nonfinite/traceback was logged.

## Concrete reach and control investigation

`tools/diagnose_striking_reach.py` first settles both free bodies through their
learned motor actors, then performs separately labeled static workspace tests
and physical constant-command trials. Static candidates call forward kinematics;
roots are repositioned only in that diagnostic data, never held by a runtime
controller or imported into a training rollout. Geometric overlap is not a punch.

The [initial 4,000-sample-per-arm-set test](results/striking-pilot/review-262144/reach-initial.json)
found no legal surface overlaps in either small filtered initial-noise poses or
the full bounded residual box at the old settled separation. The closest centre
was about 0.231 m even in the full box. A [common-sample separation sweep](results/striking-pilot/review-262144/reach-sweep.json)
then held poses/sample vectors constant and measured:

| Root separation | Legal surface overlaps / 1,000 candidate arm poses |
| --- | --- |
| 0.60 m | 0 |
| 0.55 m | 1 |
| 0.50 m | 17 |
| 0.45 m | 68 |
| 0.40 m | 167 |

This finite random sample does not prove absolute reach limits. It does show
that the old engagement reward peak at **0.60 m**, combined with a **0.58 m**
approach stop, favored a sampled non-contact region. Initial filtered noise
explored an even narrower arm workspace.

The frozen motor opposed about 15% of a sustained 0.4-rad elbow request along
that request's direction, but the observed elbow still moved about 0.35 rad.
A full 0.8-rad constant elbow request produced raw contact but fell after 140
controls; the broad best kinematic vector fell after 58. Static reach therefore
does not establish executable balance. These data do not support blaming a
complete motor cancellation or overriding the learned leg controller. The motor,
residual scale/filter, action/observation shapes and physical assistance settings
remain unchanged. Constant-command probes are diagnostics, not scripted punch
cycles or a training prior.

## Same-policy physical comparison

The [0.45 m diagnostic](results/striking-pilot/review-262144/range-045-evaluation.json)
loads the **exact same 262,144 policy, frozen motor and normalizer**, with no
optimization. It changes only the approach/engagement goal. It uses the same
seeds 7/23/31/47, both corner arrangements and ten-second first-fall episodes.
**8/8 pairs survived**, with two qualifying red impacts in seed 7/original
arrangement; the other seven episodes scored zero. End distances ranged from
0.306 to 0.563 m, so collision/command transients can overshoot the request.
This small comparison is evidence for a bounded training-profile correction,
not consistent boxing, a new certification, or demonstrated improvement on every
seed. All impact criteria remained unchanged.

## Reviewed continuation

Adopt **0.45 m** for both the requested approach stop and the training engagement
reward peak. This aligns practice with the measured reachable region rather
than paying the old stationary range. Keep nominal **2 m initial spawns**, free
roots/legs, original physical model, frozen motor, 0.8-rad residual bound/filter,
first-fall rules, scorecard impulse/closing/extension thresholds and cooldowns.
The latest main actor/critic/Adam/normalizer/RNG is retained, with **no rollback
and no diagnostic actor substitution**. Retain all **262,144** boxing transitions.

The new bounded boxing ceiling is **524,288 total learner transitions**, allowing
at most **262,144 additional learner transitions**. Review the final independent physical
results and both corners before another extension. Persistent missing contacts
or survival regression require investigation, not blind extra steps.
The [decision record](results/striking-pilot/review-262144/continuation-decision.json)
records the policy hash, source counts, old/new profiles and evidence.

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_striking.ps1 `
  -Resume -Ceiling 524288 -EngagementDistance 0.45
```

The new parameter is an explicit reviewed profile adoption. Ordinary future
resumes omit it and inherit both recorded distances; legacy checkpoints without
these fields retain exactly 0.58/0.60. Explicit adoption cannot bypass unrelated
configuration mismatches. The launcher also inherits the saved ceiling on a
plain `-Resume`, instead of silently reverting to its new-run default.

## Validation, accounting and limits

**64 focused CPU tests passed**, including legacy/new profile inheritance,
invalid values, explicit-only adoption, unrelated-setting rejection, the existing
physical scorecard and all foundation promotion checks. PowerShell parsing
passed. A separate diagnostic warm start advanced its saved head from 128 to
192 with the explicit new profile; a plain resume then advanced to 256 and
retained 0.45/0.45, normalizer and Adam/RNG. Its Adam count reached 64 versus the
main source's 4,096, confirming the separate small diagnostic lineage. All saved
leaves/checksums were verified. That head does not initialize main training.

This review used **10,500 diagnostic world-control transitions**: 4,000 old-profile
evaluation, 4,000 candidate evaluation, and 1,248 each in the two physical probe
runs, plus four controls in a checkpoint-profile reader smoke. Static forward-kinematic candidates are separately counted at **15,000**.
Formal striking-interface diagnostics now total **27,600 world controls**, and
all isolated PPO smoke runs total **768 diagnostic learner transitions** (384
world controls). Earlier foundation/transfer investigations retain their own
accounting. None enters main PPO or rewinds consumed counters. No simulation
state is restored on warm start; policy/learning state is retained as documented.

The persistent hidden WSL launcher and existing 30-minute monitor are preserved.
Native state, logs and snapshots remain in the documented Linux durable output
and `runs/striking-pilot` mirror. The completed foundation must stay certified
and stopped. Locomotion and isolated contacts still do not finish the boxing goal.


## Verified restart

The persistent hidden host resumed as Windows PID 14968 and Linux learner 411
at inspection. One learner was active, with no foundation learner or duplicate
training job. Counters advanced beyond 262,144 without discarded work; payload
hashes and all learning-state leaves remained finite. The stored Adam count
continued above its original 4,096 and matches the completed PPO update count.
The [launch record](results/striking-pilot/review-262144/launched.json) preserves
source/current hashes, code fingerprints and inherited settings. The evaluator
reads range metadata from the manifest/config and rejects disagreements;
a four-control replay verified inherited 0.45 without an override. Legacy
metadata still resolves to the historical profile.

The launcher now archives the stopped host's stdout/stderr under
`runs/striking-host-logs/` before replacing the live log names on future resumes.
Prior event increments and final counters are preserved in this report; complete
old stdout cannot be reconstructed after an already completed overwrite.
This logging change does not alter learning or physics. The existing automation
schedule is unchanged. Video comparisons are archived at
`fights/striking-pilot/step-262144.mp4` and
`fights/striking-pilot/range-045-comparison.mp4`.

## Subsequent 524,288-step review

The [duration review](striking-duration-review.md) supersedes the preceding budget.
Independent evaluation regressed to 7/8 pair survival and zero impacts. Read-only
exploration/navigation/arm interventions and finite-state checks are archived.
One bounded unchanged duration experiment now has a **1,048,576 total boxing
learner-transition ceiling**, retaining the latest actor and all learning state.
If physical contacts or survival do not recover, another unchanged extension
requires a controlled reward/exploration investigation instead. No physical
criterion, source foundation policy or initial separation changed.
