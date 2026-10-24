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
