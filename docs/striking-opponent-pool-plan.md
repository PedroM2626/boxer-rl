# Next engineering work: explicit opponent-pool training integration

**Implementation update (2026-10-24):** the following is the preserved engineering
plan before implementation. Version 13 now passes 41 focused checks and the
[explicit bounded comparison](striking-opponent-pool-comparison.md) is running.
That later contract declares the new allocation and live output; the historical
no-allocation statements below describe the state when this plan was written.

The [broader frozen audit](results/striking-pilot/gae-generalization-2064384/README.md)
passes every new neutral-role profile but fails three of four complete groups
because active mirror survival is insufficient (23/32). One archived-adversary
corner also fails at 5/8. The existing fixed-opponent learner has useful stable
own strikes, but active interaction does not generalize reliably. Opponent
diversity is the next training hypothesis, not a proven remedy.

No pool implementation, new PPO allocation or running training job is claimed
by this plan. Continue authorized engineering automatically; do not resume the
old single-opponent job or restore an earlier actor while integration is pending.

Implement a distinct, versioned warm-start profile from latest frozen candidate
bea8147f, preserving all actor/critic parameters, Adam moments/counts, normalizer,
samplers, original lambda .99/gamma .999 and consumed prefixes. Preserve the
learned motor, free-body physics, action channels, original reward and full gates.

The pool must contain explicitly hashed archived actor/normalizer snapshots,
including the original adversary and prior competent policies. Never normalize
an opponent using the learner's statistics. Keep neutral practice and balanced
learner corners; assign opponents independently and record per-opponent/corner
physical credits. Their sum must equal active-mode credits, with historical
neutral/mixed prefixes retained. Persist assignment/sampler provenance and reject
ambiguous legacy checkpoints; apply any metadata transition once only. Ordinary
resumes retain the new profile rather than recreating it.

Test snapshot immutability, routing, independent normalization, exact credit
conservation, changed-hash rejection and warm resume before native integration.
Then declare one bounded matched pool-versus-fixed comparison from the same
latest source, with unchanged physical gates, independent fresh conditions,
numerical/overflow guards and finite stop rules before launching PPO. The budget
must be explicit; no ceiling increase is authorized by this document alone.

Diagnostic PPO remains **2,131,200**, reviewed formal simulation **1,936,816**.
Main boxing 1,048,576, foundation 474,992,640, combined main 476,041,216 and
standalone BC 8,000 are unchanged. The original readiness milestone remains
limited; main adoption and general boxing certification remain false.

After active-opponent consolidation, sustained reciprocal exchanges, longer
rounds and unseen spawn/orientation conditions still need prospective validation.
Do not silently change the ten-second environment clock or call a CLI duration
override a longer-round certification without validating actual episode duration.
