# Bounded actor KL-guard comparison

## Evidence and question

The complete [sampling comparison](striking-sampling-comparison.md) finished
all 168 episodes / 71,289 evaluation controls and 32,768 PPO credits without
native errors. Final fresh half-noise/control survival is **17/24 each**, versus
16/24 source; all complete final sets fail. The isolated version-10 original
pass is followed by failure. Its initial mean KL / ratio clipping is about
.0866 / .635 versus .0291 / .426 control. Narrower sampling is not adopted.

Both profiles still allow sixteen actor updates on a rollout even after its
distribution has moved substantially. Test an explicit **actor-only full-rollout
Gaussian KL stop**, against unchanged training, from the latest completed
ordinary-covariance control. This is distinct from the previously rejected fixed
actor-rate reduction: rates and critic schedule stay unchanged, and the actual
number of actor updates is limited adaptively. A smaller measured KL alone will
not certify physical improvement.

## Exact source, algorithm and provenance

Source **97af578363d9a8e43c2331546d07a85437e44a02a4673b50d7c1035352f09721**,
`checkpoints/striking-validation/sampling-1933312/control/gate-16384`, is the
latest completed control: **1,949,696 learner / 1,327,104 world**, Adam actor
**12,936** / critic **36,608**. Retain all original actor/critic/moments,
normalizer, RNG, sampler, mode/corner and consumed prefixes. Do not restore the
earlier source or favorable intermediate actor. The explicit version-11
transition adds only guard metadata and an initially empty actor-step ledger.

After collecting each original 1,024-credit rollout, preserve the old Gaussian
on all its normalized observations. Before every minibatch, measure the mean
exact joint Gaussian KL(old || current) across that same full rollout. Once it
**exceeds .02**, disable further actor updates for the remainder of that rollout.
Reset the guard only after collecting the next fresh rollout. A crossing update
is retained: **no actor rollback**, line search or discarded native controls.
This is a target/stop rule, not a guaranteed hard KL bound.

All **sixteen critic Adam steps** continue. Preserve the original full joint
gradient clipping calculation, entropy/PPO/retention loss and all shuffled
minibatch/replay draws. Disabled actor steps must preserve actor parameters,
Adam count and both moment trees exactly, including nonzero momentum; zeroing
the gradient while still stepping Adam is not equivalent. Report actual actor
steps per rollout (1..16), keep their cumulative ordered ledger in checkpoints,
and validate the exact actor count as 12,936 + ledger sum. The original critic
prefix remains 36,608 + 16 per optimized 1,024-credit block. Keep all consumed
transitions even when fewer actor steps are applied.

The normalizer still updates after each rollout. Record KL before and after
that update: the guard does not bound normalization-induced changes. All action
channels, original Gaussian covariance, gamma .999, rates .0001, retention 1,
eight worlds / 128 controls and four active/four neutral worlds are inherited.
Original 6bf3273a frozen adversary, learned 29-joint feedback motor, 148/17
interface, rewards, free roots/legs, spawns, scoring and physics stay unchanged.
No support, gravity compensation, root forces, reward shaping or action masks.

## Allocation and unchanged independent decision

At most **16,384 learner/world credits per arm / 32,768 total**. Include a
1,024-credit native integration block and passive update geometry per arm.
Its collected batch hash must match across arms, since only the subsequent
optimizer protocol differs. Both starts must preserve every original source
leaf. Use ordinary warm resumes between blocks; never reapply the transition.

Run original complete screens at **+8,192 / +16,384** and source/final full
fresh sets on untouched **25,000,003..25,000,006**, both swaps/all three roles.
Maximum **168 episodes / 84,000 evaluation controls**. Independent deterministic
physical gates, first-fall rule, ten-second duration, real fist contact thresholds
and all original survival/strike-breadth criteria stay unchanged. Version 11
cannot inherit any previous-family pass. Two distinct consecutive complete
passes and independent generalization evidence remain prerequisites.

Stop an arm if any original complete profile has at most three paired survivors;
stop all on nonfinite/invalid physics, constraint/contact overflow, native error,
batch/provenance mismatch or invalid actor/critic/count ledger. Retain every
attempted world control. Finish the bounded control and archive all failed
generations; no intermediate-actor selection to hide later regression.

Judge stability, surviving own strikes and reciprocal impacts versus source and
control, not KL, loss or fewer updates alone. No automatic adoption, promotion,
budget extension or broader opponent rollout follows a favorable screen.

Diagnostic PPO starts **1,983,744**, ceiling **2,016,512**. Formal evaluation
starts **1,558,615**, ceiling **1,642,615**. New physical collection at most 32,768
world controls and 131,072 retention presentations are separate; these include
minibatches whose actor step is suppressed. BC 8,000, foundation 474,992,640,
main boxing 1,048,576 and combined main 476,041,216 remain unchanged.

Launcher `tools/start_striking_kl_guard_compare.ps1`, hidden persistent WSL.
Inspect `/home/pedro/boxing-runs/striking-kl-guard-1949696/comparison.json`, both
child outputs and processes before any new job. No duplicate jobs or blind
retries. Independently verify/archive complete gates, exact checkpoint hashes,
all per-rollout actor ledgers, critic/consumed prefixes, update geometry and fresh
finals before the next concrete continuation decision. Preserve monitoring.

## Implementation checks

Forty-eight focused tests pass. Real nonzero Adam momentum cannot move a disabled
actor or increment its optimizer clock; the full critic-only loss update retains
the actual source actor/moments while advancing critic parameters/count. Lossless
transition and variable actor-step ledgers are validated, including rejection of
missing/wrong ledgers and mislabeled legacy counters. Existing split-Adam,
mixed/neutral/retention/horizon/sampling and independent evaluator paths pass.
Python/Bash syntax and PowerShell parsing pass.

The passive update observer now records a canonical collected-batch hash without
changing gradients or state. Its note distinguishes passive observation from the
explicit training guard. The native comparison still must verify its included
integration blocks, matching first batches, exact per-rollout ledgers and unchanged
complete physical gates; software checks alone do not establish better boxing.

## Initial native evidence

The [archived integration](results/striking-pilot/kl-guard-1949696/README.md)
verifies six actor updates and all sixteen critic updates in the first included
1,024-credit block, with finite state, exact retained optimizer prefixes and the
original frozen opponent. KL after normalization is .0258045: the crossing
update is retained, as specified, so .02 is a stop target rather than a hard
bound. Matching the control's collected batch remains pending.

The first guarded +8,192 complete screen fails with paired survival 6/8/6 and
four reciprocal mirror episodes. The source's untouched fresh set also fails
(8/7/8). These intermediate observations establish no improvement. Finish the
bounded comparison and review both final fresh sets before a continuation decision.
