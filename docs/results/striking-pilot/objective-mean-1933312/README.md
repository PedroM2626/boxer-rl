# Matched mean-action rollout launch

The sampled audit is independently reviewed in `objective-1933312`: 33 complete
rounds plus eight partial tails, four own-only falls and one partner-only fall;
zero updates. It does not justify changing the shared-fall reward immediately.

This bounded follow-up retains the exact c47edf0b policy, normalizer, frozen
opponent, initial world/reset schedule and all original physics/rewards. Only
the learner action mode changes to tanh(mean). Maximum 16,384 additional
evaluation controls, zero training; formal ceiling 1,487,326 and diagnostic PPO
unchanged at 1,950,976. `launch-state.json` is a historical snapshot.

Read live `/home/pedro/boxing-runs/striking-objective-mean-1933312/audit.json`
and its process before any job. Independently match completed rounds by world,
seed, swap and learner role with the sampled audit; report unmatched/censored
tails and mode/role outcomes. Do not substitute aggregate diagnostic counts for
the unchanged complete physical gates. No automatic training, reward change,
adoption, promotion or further allocation; preserve monitoring.

## Complete independent action-mode review

`completed-audit.json` retains all mean-mode rounds and partial tails;
`analysis.json` records the independently matched comparison: 32 common round
identities, one completed round unique to each mode and eight partial tails per
mode. Neutral sampled/mean pair survival is 16/16 versus 12/16; active survival
11/16 versus 14/16. Surviving active two-way episodes are seven versus three.
Neither action mode dominates, and these diagnostics do not certify boxing.

Formal total 1,487,326, diagnostic PPO still 1,950,976, no new training. Complete
foundation histories/hashes, seven-record original boxing history (current
streak zero) and main boxing hash are rechecked. Native host logs contain only
the known WSL path-translation message, with no physics/overflow exception.

The next concrete operation is the bounded on-policy half-standard-deviation
versus unchanged training experiment in `docs/striking-sampling-comparison.md`.
It retains latest c47edf0b learned state and all original physics/rewards/gates.
No sampled-gate replacement, actor rollback or automatic promotion/adoption.
