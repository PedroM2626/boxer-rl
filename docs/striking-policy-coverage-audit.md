# Passive policy-state coverage and first-fall audit

The [completed contact-history comparison](striking-contact-history-comparison.md)
rejected both final actors. History affected the treatment network's predictions,
but pre-impact fit remained worse than recovery fit and both roles lacked robust
own scoring. These observations justify the previously bounded read-only trace
audit, not another supervised/PPO extension.

## Frozen protocol

Source is the latest contact-history BC branch at 4,000 updates, SHA
`2da4302cb992d5067a86c8886f72bac2cf54f2def1c41f7f87201d5f76c0bf9a`.
Do not substitute the control, original PPO actor or earlier favorable BC actor.
Keep its exact 148-input/17-output interface, normalization and engagement profile.
The frozen motor remains SHA
`b39b1990945e8a374802c3a190e5ac5e026ae39e7fd5ef61c5c2b01ad291c6fc`.

Collect the existing complete sixteen neutral-role and eight active self-play
episodes on seeds 7/23/31/47 with both swaps, ten seconds each. Maximum allocation
is **12,000 new diagnostic world controls**, zero learning. Before stepping, each
attempted control is written durably. Do not treat repeated checkpoints as new
consecutive certification passes. No score, gate, termination, root force, joint
authority, spawn separation or assistance setting changes.

Passive traces save both corners' pre-action and next observations, actual
bounded actions, commands, applied motor actions, filtered arm residuals, native
qpos/qvel, pelvis height/upright, scores and falls. An opt-in native recorder notes
the first positive normal contact force for each geometry pair at 2 ms substep
resolution. It only calls the read-only contact-force accessor. Merely allocated
zero-force contact records do not count. Impact events retain the independent
scorecard's original timing and criteria; positive normal force alone is not a
legal punch. Terminal self-contact records disambiguate the existing self-leg
contact rule from low-pelvis and tilt thresholds.

The optional recorder is absent during default use. Instrumentation tests compare
traced and original qpos/qvel, observations, rewards and scores exactly under the
same action sequence. The audit also requires every original complete trial's
reported physical fields to match exactly. A disagreement stops the audit for
investigation, while retaining all attempted controls. Traces do not affect actions.

## Coverage analysis and limits

Use the same fifteen verified demonstrations, with reconstructed true history,
and the source actor's frozen normalizer. Compare each controlled corner only to
demonstrations in that same role. For self-play, analyze both corners; a neutral
corner is not a policy-controlled sample. Exact nearest-neighbor RMS distances
are reported separately for motor inputs [0,103), opponent inputs [103,129),
previous actions [129,146), and contact history [146,148). Different blocks may
select different neighbors; these are marginal block distances, not joint-state
coverage or certified safety metrics.

Calibrate descriptive q50/q95/q99 distances by leaving an entire demonstration
trajectory out of the same-role reference pool. This avoids self-neighbor zeros
and adjacent-state leakage. Report pre-hit/post-hit and survival/failure groups;
q99 exceedance describes departure from these successful examples and is never a
new physical gate. Demonstrations were selected for success and are not a random
sample of the reachable state space. A distance association cannot establish
causality or prescribe safe corrections by itself.

The continuation decision must consider first contact, first legal score and
first fall, not only average distances. A falling episode with no own valid hit
cannot be presented as learned boxing. New training requires an evidence-based
bounded plan after review; this audit allocates no optimizer updates.

## Operation and accounting

`tools/start_striking_coverage_audit.ps1` uses the persistent hidden WSL host and
`tools/run_striking_coverage_audit.sh`. The sequential parent writes
`/home/pedro/boxing-runs/striking-coverage-contact-bc4000/coverage.json`.
Inspect it, the active evaluator/progress and logs before any launch. Existing
state, parent flock and PID/start-time guards prevent duplicate jobs. At failure,
reconcile partial evaluator progress before any recovery; no automatic rerun.
The monitoring schedule is unchanged.

Before this audit, formal diagnostic controls total **385,343**; diagnostic PPO
is **1,188,096** and cumulative supervised compute **6,000 updates / 1,536,000
presentations**. Main boxing remains **1,048,576**, foundation **474,992,640**,
and main project total **476,041,216**. Four complete foundation histories
(29/41/13/2 records), their finite payloads and the unchanged main actor hash were
checked before instrumentation. Both previous comparison arms are completed;
their evaluator error logs are empty. No productive learning job is interrupted.

## Completed traces and bounded continuation

All 24 episodes completed with **9,698** world controls, bringing formal
diagnostic controls to **395,041**. Every original physical trial field matches
exactly. Ten paired/trace tests passed; the earlier six contact/trace checks also
passed. No learning, main counter or gate changed. Full traces, native logs,
checksums, leave-trajectory-out calibration and role-level analysis are archived
under `docs/results/striking-pilot/policy-coverage-contact-bc4000/`.

Seven pair failures occurred across the neutral-role and active trials. Six
terminate on low pelvis and one on the unchanged self-leg contact rule; none
terminate on the upright threshold. The red learner at seed 47 without a swap
collapses at 0.908 s against a neutral partner and 0.902 s in self-play, before
any positive-force inter-body contact. Those two failures cannot be attributed
to a received physical punch. Other failures follow hand-to-hand contact, but
temporal order alone does not prove the contacts caused them.

Coverage departs from successful demonstrations in some failures: for seed 47
without a swap, all red pre-hit opponent observations and about 74% of motor
observations exceed their respective demonstration q99 distances. The red seed
23 failures also show substantial departure before any own legal hit. However,
some surviving episodes have early exceedances and the blue seed 7 failure has
little pre-hit distance departure. Do not turn these distances into a fall
detector or claim that coverage alone explains every failure.

These traces justify a **separate maximum 3,600-control channel intervention**,
zero learning: use the same latest actor on seeds 23/47, both swaps, red-neutral
and active self-play, three seconds per trial. Compare zero high-level command
offsets, zero fourteen arm residual outputs, and zero both (3 variants x 8 trials
x 150 controls). The original three-second baselines come from the completed
traces and consume no extra controls. Both bodies/legs, automatic learned motor
navigation, contact model and all fall/scoring criteria remain unchanged.

This tests the channel contribution to early collapse, not a new deployed policy.
Actions are changed only for explicitly labeled diagnostic interventions, with
actual action history and score flags retained. Removing a channel also changes
subsequent observations and closed-loop actions; do not claim a single joint or
contact cause from it. Three-second survival cannot substitute for full gates or
establish boxing. Record every attempted control, inspect all variants and swapped
controls, and review before any optimizer allocation. This extension is bounded
because a specific pre-contact failure was verified; no supervised/PPO learning
is allocated. Its parent is `tools/striking_channel_audit.py`, durable state
`/home/pedro/boxing-runs/striking-channel-contact-bc4000/channels.json`.

## Completed channel interventions

All 24 three-second intervention trials completed. The independent
[channel review](results/striking-pilot/action-channels-contact-bc4000/analysis.json)
checks actual stored actions against the original network followed by exactly the
declared channel removal. Both swapped controls are included. No native numerical
or capacity warnings occurred. Five trace/action/noise tests passed after adding
the optional arm-channel intervention.

| Controller during the short diagnostic | Red-neutral pair survivors | Active pair survivors |
| --- | ---: | ---: |
| Original, truncated existing traces | 2/4 | 2/4 |
| Zero command offsets | 3/4 | 3/4 |
| Zero arm residual outputs | 4/4 | 4/4 |
| Zero both | 4/4 | 4/4 |

Zero commands prevent the seed 23 early fall but retain seed 47's pre-contact
collapse. Removing arm residual outputs prevents both selected original early
failures in each context. The learned foundation still controls all 29 joint
actions; neither roots nor legs gain support, locks or artificial forces.
This is evidence for the residual channel's contribution under these closed-loop
interventions, not proof about one joint, a universally safe replacement, robust
ten-second survival or boxing. Do not deploy the zero-output diagnostic as an
adopted policy and do not weaken full gates.

Additional channel cost is **3,393** world controls, so the two completed audits
cost **13,091** and formal diagnostic controls total **398,434**. Diagnostic PPO
remains 1,188,096, cumulative supervised compute remains 6,000 updates, and main
counters are unchanged. Both audit parents are completed. No training or audit
process is expected to remain running.

## Next bounded work: fresh pre-contact anchors

The evidence favors preserving the already learned motor during approach before
adding more punch training. Validate that idea on fresh initial states rather
than fitting the diagnosed benchmark seeds. The next separate comparison has
two fixed profiles: the latest unmodified BC actor and that same actor with all
seventeen high-level outputs zero, retaining the autonomous learned motor's
approach commands and ordinary free-body physics. Use fresh seeds
**3,000,003 through 3,000,010**, both swaps, both neutral roles and active self-play,
one second per trial. That is 96 trials at most **4,800 world controls**, zero
supervised or PPO updates. Keep both profiles' complete failures and all consumed
controls. No benchmark-seed trace may enter training.

Record actual pre-action root separation in addition to the existing traces.
Potential motor-preservation examples must complete the entire prefix with both
bodies standing, finite native state, no inter-body positive-force contact, and
root separation above 0.9 m throughout. Selection is an explicitly restricted
pre-contact data filter, not a changed physical pass criterion. Archive exclusions
and both profiles' same-seed outcomes; short survival alone does not certify a
demonstration or a boxing phase. Compare actual outputs before any first failure
and retain exact hashes/provenance. This collection is not yet started.

Only after that review should a bounded supervised anchoring experiment be
designed from the latest 4,000-update contact branch, retaining its Adam/sampler,
normalizer, critic, policy key and all lineage counters. Preserve the fifteen
striking demonstrations and their event history. Do not train directly on seeds
7/23/31/47, adopt an output clamp, revert to a favorable checkpoint, silently
replace the optimizer, or allocate another long PPO run. Fresh independent full
physical checks remain necessary after any learning; the current protocol grants
only the 4,800-control data comparison, not optimizer compute or promotion.
