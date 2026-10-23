# Demonstration retention restores strikes, with incomplete stability

Both matched arms completed their fixed 32,768 new learner credits, real
first-block guards and all 120 source/final ten-second physical episodes. The
retention arm restores real learned mean-policy strikes, including one fresh
episode where both robots score and remain standing. Neither final actor passes
the unchanged complete physical criteria. No actor is adopted or phase promoted.

## Full independent results

Each survival entry is out of eight episodes, both swaps. Neutral-role events
are the learner's own legal events in surviving pairs. Active events below exclude
episodes ending in a fall; those remain separately recorded in the raw reports.

| Policy / sample | Neutral 0 survival / events | Neutral 1 survival / events | Active survival | Surviving active red / blue events | Surviving two-way episodes |
| --- | --- | --- | --- | --- | --- |
| Source / fresh | 7/8 / 0 | 8/8 / 0 | 7/8 | 0 / 0 | 0 |
| PPO only / screen | 7/8 / 0 | 8/8 / 0 | 8/8 | 0 / 0 | 0 |
| PPO only / fresh | 7/8 / 0 | 8/8 / 0 | 7/8 | 0 / 0 | 0 |
| PPO + retention / screen | 7/8 / 1 | 8/8 / 1 | 7/8 | 2 / 0 | 0 |
| PPO + retention / fresh | 8/8 / 1 | 4/8 / 0 | 6/8 | 3 / 3 | 1 |

The fresh two-way episode uses seed **7,000,003, swap true**, deterministic mean
actions, ten simulated seconds, with **three red and two blue qualifying impacts**.
The other fresh surviving event is blue-only on seed 7,000,005, swap false. An
additional two-way episode on seed 7,000,006 ends in a fall and is not counted as
stable. This is one selected success out of eight fresh active trials, not robust
boxing or two consecutive complete passing generations. Neither neutral role
has surviving own events on two screen/fresh seeds, and role-1 fresh survival
regresses from control 8/8 to 4/8. The tradeoff must be investigated before more
steps are justified.

## Integrity, learned action retention and accounting

All 32 PPO block losses per arm, parameters and optimizer moments are finite.
Native error logs are empty, hashes match, and both retention/auxiliary sampler
streams match across arms. Exact consumed/optimized lineage is **1,605,632** with
**999,424 world controls** per final snapshot, including the retained old prefixes.
Adam counts reach **7,560 actor / 31,232 critic**, without rollback or reset. The
current complete foundation histories (29/41/13/2 records), payload hashes and
finite values were reverified, as was the unchanged stopped main actor.

- PPO-only final SHA: `7c17c4500ef0873b8a7c575e24ead949c8742d31abca566075c92328616d8c6a`.
- Retention final SHA: `f10d817527e521b0f95253c1d8cfdf2db4b3a03195958e15a7638222743fd38c`.

On the fixed 7,500 verified demonstration observations, action MSE is 0.016180
for the source, 0.020311 for PPO-only and **0.005991** for retention. On the 1,600
approach observations, MSE is 0.016522 / 0.020768 / **0.012572** respectively. These
offline measurements are consistent with retaining demonstrated actions and
the observed restoration of legal strikes; they are not independent physical
gates or proof of generalization. The evaluation deploys only the learned actor,
without teacher commands, clamps, assistance or modified impacts.

This job consumed **65,536 new diagnostic learner credits**, **32,768 separately
recorded collection world controls**, and **55,683 evaluation world controls**.
Cumulative diagnostic PPO is **1,515,776**, formal diagnostic simulation
**603,577 world controls** before the replay/trace audit. Standalone BC remains
8,000 updates / 2,048,000 presentations; the retention arm applies an additional
131,072 integrated auxiliary presentations (98,304 demo / 32,768 anchor), without
separate BC optimizer steps. The zero-weight control reads the same number of
examples. Main foundation/boxing remains **474,992,640 / 1,048,576**, combined
**476,041,216**, with no main ceiling extension.

Complete physical reports, per-block training metrics, sampler/count verification,
offline measurements and final models/configuration are archived under
`docs/results/striking-pilot/retention-1572864/` and
`checkpoints/striking-validation/retention-1572864/{ppo_only,ppo_retention}/`.
`tools/review_striking_retention.py` repeats the completed comparison's integrity,
full-criteria and offline checks without physics or optimizer updates.

## Bounded verification before another continuation

Do not extend either training arm yet. Reproduce both swap arrangements of the
selected fresh seed under passive traces and require every original physical
result field to match exactly. Render the recorded successful coordinates with
the original closed-fist meshes; rendering adds no simulation steps and changes
no training geometry. Verify the exact learned three/two impact sequence and
both bodies standing throughout the full ten seconds.

Then trace **all eight fresh neutral-role-1 episodes for each final actor**. This
is a matched control/treatment stability audit, not an action intervention. Record
first contact timing, pelvis/upright/self-leg terminal causes and approach/arm
outputs. Compare each replay with its original complete trial before interpreting
the trace; passive timing alone does not establish collision causality.

The fixed allocation is **18 episodes, at most 9,000 evaluation world controls**,
zero optimizer updates. It does not reuse or extend the prior training budget.
The persistent hidden launcher is `tools/start_striking_retention_trace.ps1`,
durable state `/home/pedro/boxing-runs/striking-retention-trace-1605632/trace-review.json`.
Inspect that state, active child/progress and native logs before new work. On
completion, archive exact-replay proof, raw traces, video hash and failure analysis,
then make a bounded evidence-based decision from the latest retained treatment.
No checkpoint rollback, main adoption, gate weakening or automatic extension is
authorized by the isolated two-way episode. Preserve the existing monitor.

## Completed exact replay and stability trace review

All **18 trials reproduce every original physical field exactly**, with finite
traces and matching hashes. Actual replay/trace cost is **8,091 world controls**,
bringing formal diagnostic simulation to **611,668**; diagnostic PPO remains
1,515,776. The replay video is
`fights/striking-pilot/learned-mean-two-way-retention-1605632.mp4` (640x480, 25 fps,
10 seconds), SHA `a9c3372936884faff0db83faccb92707be632aef23300182d86b51ec29a45651`.
The closed-fist visual model only renders recorded coordinates. A frame was
visually inspected; both complete bodies and feet are visible. The five exact
impact windows are red 8.742 s / 0.688 Ns, blue 8.928 s / 1.695 Ns, red 9.042 s /
1.301 Ns, blue 9.242 s / 1.156 Ns and red 9.828 s / 0.723 Ns. Both agents survive
the complete episode; these are learned deterministic actions, not a teacher or
the earlier selected Gaussian replay.

The matched neutral-role-1 control survives 8/8. Treatment's four failures all
affect blue: seeds 7,000,003 / 004 / 005 with swap true and 006 with swap false.
Two terminate on low pelvis and two on the unchanged self-leg-contact condition;
none terminates on the upright threshold. First positive interbody contacts occur
at 2.050 / 2.258 / 1.812 / 1.892 s, before the respective falls at 5.654 / 2.564 /
4.326 / 9.228 s. These are post-contact finite failures rather than numerical
divergence or isolated early approach falls. Timing alone does not establish that
contact caused the loss of stability. The high-level policy still controls only
navigation commands and arm residuals over the frozen motor; the free legs do
not gain direct new training authority in this continuation.

All native evaluator error logs are empty. The parent emitted Python's warning
about `os.fork` with multithreaded JAX during render/subprocess use, but all
evaluators and rendering completed; there was no numerical/constraint warning,
deadlock or repeat of consumed physics. Raw matching trials, passive traces,
video hash and detailed terminal causes are archived in
`docs/results/striking-pilot/retention-trace-1605632/`.

## Justified single bounded ordinary continuation

The matched strike gain, reduced demonstration error and exact deterministic
two-way replay justify **one further 32,768-credit diagnostic continuation** of
the latest treatment, with unchanged version-3 profile/coefficient 1. This is
not a main extension, restoration of BC or permission to train indefinitely.
Actor/critic, log standard deviation, normalization, policy key, both Adam states
(7,560 / 31,232), original samplers and all consumed counters are retained exactly.
The source lineage is 1,605,632 learner credits / 999,424 world controls, SHA `f10d817527e521b0f95253c1d8cfdf2db4b3a03195958e15a7638222743fd38c`.

The allocation is at most **32,768 new learner credits / 16,384 new collection
world controls**. Evaluate the entire original three-profile screen after **16,384**
and **32,768** new credits, preserving both distinct checkpoint generations and
all full reports. If any screen profile has **at most 3/8 surviving pairs**, stop
training immediately after that full check and still evaluate the last checkpoint
on the registered fresh set. This is a conservative training stop, not a lowered
promotion threshold. At the allocation end, stop for review regardless of scores.
Neither a partial improvement nor two parallel comparison arms is a stage pass.

Preregister fresh seeds **8,000,003–8,000,006**, both swaps, all three profiles,
ten seconds. Evaluate the source once on this set and the final checkpoint once;
do not use it for mid-run allocation or checkpoint selection. Together with the
two complete screen checks this is **96 episodes / at most 48,000 evaluation world
controls**. Retain all attempts and record collection separately. No action,
physics, reward, score or independent full gate changes. Full role survival and
multi-seed own scoring remain required; main adoption and promotion are disabled
pending the completed independent review and established consecutive gates.

If all 32,768 credits complete, lineage reaches **1,638,400** with **1,015,808 world
controls**, Adam **8,072 / 31,744**, cumulative diagnostic PPO **1,548,544**. The
512 integrated updates read/apply another 131,072 auxiliary examples (98,304 demo /
32,768 anchor). Standalone BC and all main counters remain unchanged. No further
extension is automatic, including on a successful selected episode.

The hidden launcher is `tools/start_striking_retention_continue.ps1`; live state
is `/home/pedro/boxing-runs/striking-retention-1605632/continuation.json`. This is an
ordinary warm resume from byte-identical policy/configuration, with no new
optimizer migration or first-block rollback. The same native global lock and
per-step finite/consumption ledger remain active. New orchestration tools compile,
shell syntax checks pass, and the completed comparison's real guards, finite
32-block histories and all foundation/main integrity checks were verified.
Inspect its active process, progress, full gate history, native logs and hashes
before intervening. Preserve the existing monitoring schedule.

The first continuation launch reached the completed fresh source evaluation but
the native trainer refused its first lock before any training state/update or
collection. The single-directory supervisor had held `.lock`, the same mutex its
child legitimately acquires. The supervisor now holds `.continuation.lock`; the
native child retains both its own `.lock` and the global unassisted lock.
The narrowly guarded `-RecoverPretrainingLock` path verified failed status, zero
new training consumption, the original exact checkpoint/configuration, absent
native state and all completed source reports before resuming. The original lock
error is archived. No source evaluation, optimizer prefix or physics was repeated.
The recovered child has since saved finite verified native PPO updates with
expected independent counts and an empty native error log. Routine resumes must
not use this special zero-training recovery path.
