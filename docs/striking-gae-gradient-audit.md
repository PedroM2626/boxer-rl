# Frozen gradient and critic assay after longer-GAE regression

The [reviewed continuation](results/striking-pilot/gae-continue-1982464/README.md)
passes all three original full profiles at +8,192 (24/24 paired survivors), then
regresses at +16,384 to 4/5/8. Fresh final survival is 7/7/8, with neutral-1 still
failing. The version-12 original history is **fail/fail/pass/fail**, current streak
zero. No checkpoint is adopted, no old favorable actor is restored and no further
unchanged PPO allocation is justified by this result.

Before changing the objective or optimizer, compare the **source 95dbdff0,
passing fa808011 and latest final 3052e0f6** in a read-only diagnostic. Full SHA-256
manifests and configurations are archived. The intermediate is diagnostic evidence,
not a training rollback. All have version 12 / lambda .99, gamma .999, the original
frozen adversary, original action channels, motor, rewards and unassisted physics.

For each saved policy collect eight mixed worlds, four sequential 128-control
blocks (4,096 physical controls each; **12,288 total**). Use common reset seeds
28,000,003..28,000,010, noise seed 314159 and independent retention sampler seed
9091. Policies and their own normalizers stay frozen, so this differs from updating
PPO and trajectories will diverge between policies despite common exogenous seeds.
These diagnostic seeds cannot later be presented as untouched holdouts.

Save every actual batch: raw observations, sampled pre-tanh actions/log
probabilities, rewards, terminal masks, critic values, bootstrap, normalized GAE
targets, mode labels and exact retention observations/labels/weights. Preserve
finite-state checks, batch checksums and source policy hashes. Continue native
first-fall resets and physics warning checks; count every attempted control before
stepping, including failure. Never silently repeat a partial assay.

Measure PPO, entropy, demonstration-retention and critic loss/gradient norms;
retention/PPO gradient alignment overall and by neutral/active mode; mode-wise
critic/return error; and the hypothetical original joint gradient-clip scale.
Maintain global advantage normalization when decomposing modes. No Adam update,
parameter update, normalizer update, saved sampler advance, standalone supervised
update or PPO training credit occurs. Twelve diagnostic retention draws expose
3,072 rows for gradient measurement; these are separate from training retention
presentations. Gradient measurements are full-batch pre-update quantities, not
the sixteen production minibatch updates. They do not establish a causal remedy
or certify survival; critic errors are bootstrapped GAE errors, not true returns.

Diagnostic PPO stays **2,065,664**. Formal diagnostic simulation begins at
**1,792,372**, new bounded ceiling **1,804,660**. The 12,288 controls are diagnostic
collection, included once in that simulation ledger and never added as PPO/main
credits. Main boxing 1,048,576, foundation 474,992,640, combined main 476,041,216
and BC 8,000 remain unchanged. No weakening gates, masks, reward/physics changes,
policy adoption, promotion or automatic further allocation.

Use hidden persistent `tools/start_striking_gae_gradient_audit.ps1`. Inspect
`/home/pedro/boxing-runs/striking-gae-gradient-1998848/gradients.json`, process and
logs before any launch. Hold both local and global boxing locks. Stop on native
errors, nonfinite values, constraint/contact overflow, changed source payload or
invalid accounting. Archive all three variants and their batches, then decide
from the measurements whether an intervention is justified. Preserve monitoring.

## Checks before launch

Twenty focused checks pass, including a new equal-mode gradient decomposition
check: averaging the two mode gradients recovers the full-batch gradient while
retaining globally normalized advantages. Existing GAE provenance, retention,
mixed routing and continuation-stop checks pass. Python/Bash syntax and
PowerShell parsing pass. Native batch/finite/source-hash checks must still pass
through the assay; software checks alone do not establish a physical outcome.

## Completed review and bounded decision

The [complete result](results/striking-pilot/gae-gradient-1998848/README.md)
verifies 12,288 controls, twelve exact finite batches, unchanged policy hashes and
3,072 diagnostic label rows. Recomputed gradients and bootstrapped return errors
do not establish retention dominance or larger final critic error as the cause
of regression. No survival certification or causal cure follows.

Keep configuration unchanged for one [bounded consolidation](striking-gae-consolidation.md)
from latest 3052e0f6. Repeatable active skills and intermittent complete passes
justify testing patient consolidation, not another parameter search or rollback.
At most 65,536 new credits, full gates every 16,384, new 29M source/final full sets,
at most 72,000 evaluation controls. Stop on severe regression, three failed
screens without a new frontier component, two distinct consecutive full passes,
or the fixed cap. New PPO ceiling 2,131,200; formal diagnostic ceiling 1,876,660.
Fresh review is mandatory; no automatic adoption, promotion or further extension.
