# Completed short-rollout gradient assay

`gradients.json` records the actual process, hashes, consumed controls and gradient
measurements. Each compressed batch contains the exact collected rewards,
observations, actions before tanh, log probabilities, values, bootstrap and fixed
training-data indices. Hashes and finite values were rechecked before archiving.
There are exactly 512 attempted world controls per variant, 1,536 total, no Adam
updates and no PPO training credits. Saved policies are unchanged.

`analysis.json` records the reviewed total of **547,894 formal diagnostic world
controls**, including this assay, and the next bounded continuation decision.
These short batches cannot establish episode survival or certify striking. Each
variant uses its own saved policy RNG; cross-policy differences are descriptive.

See [the full review and continuation contract](../../../striking-retention-continuation.md)
for limitations, gradient scales, the completed 72-episode sampled audit and the
explicit 32,768-credit-per-arm demonstration-retention experiment. Source/final
checkpoints remain under `checkpoints/striking-validation/ppo-anchor-bc5000/`.
