# Matched motor anchoring, completed

[Protocol and continuation decision](../../../striking-motor-anchoring.md).
`learning.json` preserves the complete parent state and sampler streams.
`analysis.json` independently verifies both actors, all 120 complete physical
episodes, original/fresh scoring criteria, finite optimizer leaves and unchanged
critic/normalizer/key/exploration/inherited counters. Raw reports, progress and
native logs are archived. Both 5,000-update models with Adam/sampler states are
in `checkpoints/striking-validation/motor-anchoring-bc4000/`.

Neither actor passes both roles and active boxing. Motor anchoring improves some
survival outcomes but reduces own scoring. No actor or stage is adopted. New
compute is 2,000 supervised updates and 55,319 world controls; cumulative BC is
8,000 updates and formal evaluation controls 458,469. No PPO ran. Further blind
imitation is rejected. The next contract requires explicit validated BC-to-PPO
optimizer migration; these BC payloads still reject direct PPO resume.
