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
