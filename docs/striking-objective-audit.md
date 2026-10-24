# Read-only training-distribution objective audit

The completed [posture-cost comparison](striking-posture-cost-comparison.md)
rejects the extra posture cost: fresh source/control/cost paired survival is
20/24 / 20/24 / 15/24 and all four new original full screens fail. No reward
coefficient increase or identical PPO extension is justified. The next source
is the latest completed control **c47edf0b**, not a restoration of 670bed08.
Both branches, failed generations and all consumed compute remain archived.

Source inspection identifies a concrete objective difference to measure:
the independent physical gate requires pair survival, whereas the original
learner reward applies `-5 * own_fallen` and no corresponding partner-only fall
cost. First fall still ends the shared round. This is not proof that a shared
fall penalty will help or that all partner falls were caused by the learner.
Gate criteria must not be relaxed to hide this difference.

Run one frozen-policy native rollout using the exact latest control checkpoint,
normalizer, saved Gaussian and saved RNG as initial state. Use the original
eight-world task, four active worlds with immutable 6bf3273a mean opponent and
four zero-head neutral partners, both learner roles and the original reset/
swap schedule. The current source is 1,933,312 learner / 1,310,720 world, Adam
12,680 / 36,352. All 29 motor actions, all 17 head channels, free-body physics,
separated spawns, scoring, original rewards and termination remain unchanged.

The diagnostic draws 2,048 sampled actions per world, **16,384 attempted native
controls** total. Start episode indices at consumed/8 and use the original
`53 + 1009 * world + episode` reset seeds. Continue fresh resets on completion.
Retain actor/critic/normalizer/opponent state without updates. This measures the
training task's sampled distribution at a fixed policy; it is not an exact
replay of historical adaptive PPO or an independent deterministic full gate.
Sampling advances a local copied key only; the original checkpoint is never
written. Record partial final rounds rather than pretending they completed.

Record every completed round: role/mode, seed/swap, first-fallen corner(s),
timeout, both valid scores, original terminal/round learner reward, terminal
learner action and the **counterfactual** additional pair-fall cost: five only
when the partner falls and the learner does not. Both-fallen trials already
have an own-fall cost and must not be double charged. Report own-only,
partner-only, simultaneous and timeout counts by mode/learner corner, including
whether partner-only falls occur without valid strikes. Do not apply this
counterfactual to actions, rewards, state, updates or the independent evaluator.
It cannot identify whether each partner fall was a knockdown, incidental
collision or inherited motor failure without further physical evidence.

Maximum **16,384 additional evaluation controls**, zero optimizer/normalizer,
training collection, replay or supervised updates. Formal evaluation starts at
**1,454,558**, ceiling **1,470,942**. Diagnostic PPO stays **1,950,976**; main
boxing 1,048,576 and combined main 476,041,216 remain unchanged. Count before
every physics attempt, stop on invalid/nonfinite physics, constraint/contact
overflow, hash drift or an unexpected process error; retain all attempted cost.

Launcher `tools/start_striking_objective_audit.ps1`, hidden persistent WSL.
Live `/home/pedro/boxing-runs/striking-objective-1933312/audit.json` and its process
must be inspected before any new job. Parent and global native locks prevent
duplicate physics jobs. No automatic retry, PPO allocation, reward change,
promotion or main adoption. On completion verify/archive the whole audit, then
make a concrete bounded continuation decision from measured terminal frequency
and reward/score context. Preserve the existing monitoring schedule.

## Complete sampled audit and matched deterministic follow-up

The sampled audit completed all **16,384** controls, **33 complete rounds** and
eight explicitly partial final rounds. Independent review reconciles every
completed/partial control and reward, mode/corner counters, unique round identity,
counterfactual cost and unchanged source hash. No training or normalizer update.
The native host log contains only the known WSL path-translation message, with
no native exception or physical warning; all physical observations/rewards are
finite. Formal total is **1,470,942**, PPO still **1,950,976**.

Neutral worlds completed 16/16 rounds without a fall. Active frozen-opponent
worlds completed 17 rounds: 12 timeouts, four own-only falls and one partner-only
fall; no simultaneous falls. The partner-only fall has original terminal reward
zero versus counterfactual -5, after both sides scored earlier. It confirms an
objective difference but does not explain most measured failures or justify
an immediate shared-fall penalty. No reward change or PPO extension follows.

The next concrete comparison is **one mean-action rollout**, same c47edf0b
actor/normalizer, frozen opponent, eight worlds, first reset seeds/swaps, reset
schedule and 2,048 controls per world. Only the learner uses tanh(mean) instead
of a saved-Gaussian draw; all action channels remain active. The copied key
still advances locally, and no saved state is written. This is a diagnostic
comparison of action mode, not another independent certification.

Match complete episodes by world/seed/swap/learner role and report unmatched or
partial tails explicitly: first-fall timing changes reset counts, so aggregate
round totals alone are not a fair paired comparison. Separate neutral and
active results, actual surviving scores, own/partner first falls and the original
reward/counterfactual cost. These samples cannot replace the unchanged full gate.

Allocate **at most 16,384 more evaluation controls**, zero optimizer/normalizer,
replay or collection updates. Formal ceiling **1,487,326**, all main/BC/PPO totals
retained. Stop/review rules above remain in force. Launch
`tools/start_striking_objective_mean.ps1`; live state
`/home/pedro/boxing-runs/striking-objective-mean-1933312/audit.json`. Inspect it and
the process before another job. No automatic retry, reward change or further
budget; archive and independently compare both action modes before deciding.
