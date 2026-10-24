# Reviewed frozen source / passing / final gradient and critic assay

All **12,288 diagnostic physical controls**, twelve exact saved batches and
3,072 diagnostic retention rows are reviewed. There are zero optimizer, parameter,
normalizer, saved-sampler, supervised or PPO training updates. All three saved
checkpoint hashes remain unchanged. Complete batches and raw measurement state
are archived here with checksums in `completed-gradients.json`.

The reviewer recomputes normalized observations, policy log probabilities, critic
values, GAE .999/.99 advantages/returns, PPO/retention/critic gradients and mode
decomposition from the batches. Retention labels/weights match across variants.
All arrays and gradients are finite; native execution completed. Host stderr
contains only the known WSL path-translation notice; native warning guards remain.

| Frozen checkpoint | Mean PPO gradient norm | Mean retention gradient norm | Mean critic / bootstrapped-return MSE |
| --- | --- | --- | --- |
| Source 95dbdff0 | 7.993 | .328 | .903 |
| Passing fa808011 | 7.680 | .303 | .825 |
| Latest final 3052e0f6 | 6.935 | .298 | .624 |

These data do not establish retention-gradient dominance or larger final critic
error as the cause of regression. Nor do they exclude long-run/minibatch/momentum
effects. Neutral/active gradient alignment varies; a negative cosine alone does
not justify an optimizer change. Critic errors use bootstrapped GAE targets, not
true returns. Frozen policies/normalizers and full-batch pre-update gradients
differ from production's changing states and sixteen minibatch updates. Common
exogenous seeds do not imply identical trajectories. Sampled round/fall counts
are descriptive, not physical gates.

Formal diagnostic simulation now totals **1,804,660**, diagnostic PPO stays
**2,065,664**. Main/foundation and BC counters are unchanged. Collection is
included once in the simulation ledger, not as PPO; diagnostic label evaluations
remain separate from training retention presentations.

Decision: one [bounded unchanged consolidation](../../../striking-gae-consolidation.md)
of latest final 3052e0f6, no restored intermediate. Up to 65,536 learner/world
credits, full gates every 16,384, new 29M source/final sets and at most 72,000
evaluation controls. Stop on severe regression, three failed screens without a
new frontier component, or two distinct consecutive complete passes (then freeze
latest and review fresh evidence). This tests consolidation, not a claimed cure.
New PPO ceiling 2,131,200, simulation ceiling 1,876,660. No weaker gates, main
adoption, automatic promotion or further extension.
