# Explicit supervised-to-PPO continuation contract

## Evidence and decision

The [completed anchoring comparison](striking-motor-anchoring.md) improved approach
output preservation and fresh active survival to 7/8, but reduced neutral scoring
and produced no surviving two-way exchanges. More identical imitation is rejected.
The next hypothesis is that physical on-policy credit, after preserving the
learned motor during approach, can improve contact-seeking behavior under an
active opponent. This is a bounded diagnostic, not a claim that longer PPO alone
will solve boxing or an extension of the stopped main run.

Warm-start the latest zero-anchor 5,000-update actor, SHA
`c7d6f1d5458f4767e6d7bedd98350ed8f79231de1030783fe9feb4b779d81e19`,
from `checkpoints/striking-validation/motor-anchoring-bc4000/zero_anchor/`.
Do not substitute the older 4,000-update actor because it scores better on some
trials. Preserve all completed branches. This source is not certified or adopted.

## Migration must precede any rollout

The actor-only BC optimizer is incompatible with the legacy joint actor/critic
PPO optimizer. The current guard must keep rejecting direct resume. Implement a
separate versioned migration and a trainer path that explicitly understands it;
changing `training_kind` alone or silently initializing a new optimizer is invalid.

Preserve the complete actor, critic, frozen-start normalizer/count, policy key,
BC sampler state and every consumed/optimized/world lineage counter. Preserve
the actor Adam moments and count **5,000** exactly. Its log-standard-deviation
must be identical before the first actual PPO update. The current critic is the
unchanged original PPO critic expanded by two zero input rows. Verify that exact
identity against original PPO SHA
`87eb01ff6392b592395b64fc6f6a29f78e35fd1c1ad2941fafbcc8001146f5d8`.
Recover that critic's original Adam moments/count, padding only the two new input
rows with zero moments. Never restore its old actor or rewind any consumed steps.

Use independently counted actor/critic Adam states. Apply the existing joint
global gradient clip once before the two Adam updates. Record both optimizer
source hashes, both counts, state format version and conversion integrity in a
new checkpoint directory; do not overwrite a BC payload or a historical PPO run.
Running normalization may subsequently follow ordinary validated PPO updates;
its initial state must not reset. Preserve the BC sampler as provenance and save
any continuing auxiliary-data sampler explicitly.

Tests must verify exact old parameter/moment prefixes, zero new critic moment
rows, both counts, policy/value prediction agreement, key/counter retention,
finite state and serialization round trip. Show one split update agrees with the
legacy joint Adam update when initial counts/moments are equivalent. Verify
actor and critic counts advance independently and no clipping is applied twice.
The unchanged legacy PPO path must retain its existing behavior and rejection
guards must fail closed on missing/mismatched optimizer formats.

The 148-input/17-output contact-history environment needs an explicit training
integration with configuration/checkpoint agreement checks. Preserve 0.45 m
engagement/reward profiles, original residual scale/filter, 2 ms physics / 20 ms
control, original 29-output learned motor, free roots/legs, all contacts, separated
spawns and independent impact/fall rules. This diagnostic explicitly trains two
active corners, two learner credits per world control, rather than the inherited
neutral-partner one-learner collection mode. That profile transition must be
versioned in the migration/config, never smuggled through an ordinary resume.
Subsequent resumes inherit that exact profile. No curriculum stage is promoted.

## Contingent bounded physical learning comparison

Only after migration/interface/accounting tests pass, allocate **131,072 new
diagnostic PPO learner transitions per arm**, 262,144 total, over the same frozen
motor and actual native fist score. Both arms start from the same verified migrated
state and initial policy key. Keep all attempted credits, including failures and
discarded partial collections; PPO world controls remain separately recorded.
Two learners per world imply at most 131,072 new collection world controls across
both arms under ordinary complete collection. Preserve inherited historical
world offsets rather than recomputing old one-learner data as two-learner data.

Compare ordinary on-policy PPO with the same PPO plus a small motor-preservation
auxiliary loss: coefficient **0.02** times the existing weighted mean action loss
on 64 uniformly sampled verified pre-contact anchor observations and their actual
zero labels per optimizer update. Both arms draw/save the same auxiliary index
stream, including the weight-zero control. The auxiliary gradient affects only
the actor; it is not an action clamp, teacher controller, force or scoring change.
All other PPO hyperparameters/reward terms match the inherited native profile.
The low coefficient tests preserving approach without repeating the dominant
imitation objective that reduced striking; its benefit is a hypothesis to measure.

Before learning, evaluate the migrated source once on preregistered fresh seeds
**5,000,003 through 5,000,006**. After each final arm, evaluate the complete sixteen
neutral-role plus eight active trials on screen seeds 7/23/31/47 and the new fresh
seeds, both swaps, ten seconds. This is at most **60,000 new evaluation world
controls**, separate from PPO collection: source fresh 24 episodes, each final
arm screen 24 plus fresh 24. The previous 4,000,003-series is now observed history,
not a new independent holdout. No checkpoint selection or tuning uses the new
fresh results during the fixed allocation.

Keep source-level survival and own surviving multi-seed scoring criteria unchanged.
Report actual active own contacts for each corner and surviving two-way episodes;
survival alone cannot establish boxing. A final comparison arm is not a second
consecutive checkpoint generation. No main adoption or phase advancement occurs
without the established distinct consecutive complete physical passes and motor
regression checks. No budget extension is automatic after the final review.

## Current status and baseline counters

Migration and PPO comparison are **not implemented or launched yet**. Implement
and validate the contract before creating a new hidden persistent WSL job. Inspect
the completed anchoring `learning.json`, logs, payloads and processes first; no
routine forced rollback or duplicate job is authorized. Preserve the monitor schedule.

Formal diagnostic evaluation controls currently total **458,469**, diagnostic PPO
**1,188,096**, cumulative BC **8,000 updates / 2,048,000 presentations**. The latest
BC actor's own lineage is 5,000 updates. Main foundation **474,992,640**, boxing
**1,048,576**, combined **476,041,216** remain unchanged. The conditional comparison
would add at most 262,144 diagnostic PPO credits and 60,000 evaluation controls;
it does not increase the main or foundation ceiling. Document actual separate
collection/evaluation consumption and review finite/overflow diagnostics before
any continuation.
