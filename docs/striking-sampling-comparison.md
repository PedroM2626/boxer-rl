# Bounded on-policy sampling-profile comparison

## Evidence and hypothesis

The completed [matched action-mode audit](striking-objective-audit.md) retains
32 common round identities. Neutral sampled/mean pair survival is **16/16 vs
12/16**; active frozen-opponent survival is **11/16 vs 14/16**. Sampling has more
surviving active events/two-way rounds, so neither mode dominates all objectives.
The direct mean is not a guaranteed safe replacement for its sampled trajectories.
These are fixed-policy diagnostics in the eight-world training task, not full
independent certification, a broad league or proof about other resets.

Test one bounded hypothesis: training with a narrower action distribution can
improve the deterministic learned mean's robustness by concentrating valid PPO
experience nearer it. Immediate frozen-policy noise changes are not adopted as
a solution; benefits must appear in trained policies under unchanged physical
gates. Earlier arm-only exploration probes used a different older policy and
found no remedy; held/expanded noise, posture cost, shared-fall cost, yaw changes
and additional body authority remain unadopted. No identical PPO extension.

## Exact on-policy change and provenance

Both arms start at latest control **c47edf0b**, lineage **1,933,312 learner /
1,310,720 world**, Adam **12,680 / 36,352**. The half-noise arm adds explicit
version-10 sampling metadata: `sampling_log_std_offset = log(0.5)` after the
existing learned log-std clip. This halves the **pre-tanh Gaussian standard
deviation in all 17 channels**, quartering its variance; it does not rescale the
mean, actions, physical joint authority or existing command bounds. Initially
effective standard deviations become about .063-.074, from .126-.149. Learned
log-std parameters and Adam moments are preserved and remain trainable.

The same Actor distribution is used for action collection, stored raw-action
log probability, PPO likelihood ratio and entropy calculation, including passive
update geometry diagnostics. Do not collect deterministic means and label them
as on-policy Gaussian samples. Deterministic evaluation retains exactly the
same tanh(mean); sampled checkpoint evaluation honors the saved sampling offset.
The unchanged control retains the original version-8 distribution.

At transition retain every original actor/critic/Adam/normalizer/RNG/sampler leaf,
all consumed/optimized/world and mode/corner prefixes, original fixed 6bf3273a
adversary, learned 29-joint feedback motor, 148/17 interface and gamma .999.
Actor/critic rate .0001, retention 1, eight worlds / 128 controls, four active /
four neutral worlds, rewards and reset schedule remain inherited. No optimizer
reset, actor rollback or forced transition on routine resumes. Strict matching
flags/profile are required to resume the new task. The source actor is not
replaced in main training.

## Allocation, physical checks and stop/review rule

At most **16,384 learner/world credits per arm / 32,768 total**. Include one
1,024-credit native integration block per arm, retaining all its updates and
geometry audit; then original complete screens at +8,192 / +16,384. Verify finite
learned state, unchanged opponent fingerprint, exact Adam/consumed/world/mode
prefixes after each block. Stop an arm if any full original profile has at most
three paired survivors; stop all on nonfinite/invalid physics, overflow, unexpected
training error or provenance/accounting failure. Partial attempts stay counted.

Before training evaluate the source on untouched **24,000,003..24,000,006**,
both swaps/all three original profiles; evaluate each latest arm final on the
same full set. Maximum **168 episodes / 84,000 formal evaluation controls**:
source 24, four original full screens 96, two final fresh full sets 48. The
original deterministic evaluator, physical fist events, first-fall rule,
ten-second duration, initial separation, independent thresholds and all contact
physics remain unchanged. No support, welds, root forces or gravity compensation.

Judge full physical passes, paired survival, neutral own-strike breadth and
surviving reciprocal impacts against both the latest source and matched control.
Lower loss, KL or sampling variance alone is insufficient, as is survival bought
by suppressing strikes. Record every failed generation; version 10 cannot inherit
version-8 historical passes. No automatic promotion, main adoption, extra budget
or opponent-pool expansion follows a favorable screen.

Diagnostic PPO starts **1,950,976**, ceiling **1,983,744**. Formal evaluation
starts **1,487,326**, ceiling **1,571,326**. New collection at most 32,768 world
controls and retention at most 131,072 presentations are separate. BC 8,000,
foundation 474,992,640, main boxing 1,048,576 and combined main 476,041,216 remain
unchanged. Unused evaluation/training reservations remain unused.

Launcher `tools/start_striking_sampling_compare.ps1`, hidden persistent WSL.
Inspect `/home/pedro/boxing-runs/striking-sampling-1933312/comparison.json`, both
arm outputs and processes before launching anything else. Ordinary resumes only;
no duplicate jobs or blind retries. Independently archive/review all gates,
checkpoints, metrics, geometry and fresh finals, then make a concrete evidence-based
continuation decision. Preserve the existing monitoring schedule.

## Integration checks before launch

Twenty-nine focused tests pass. They verify lossless source-leaf preservation,
strict profile/resume metadata, bit-identical means under the offset, correct
half-standard-deviation and Gaussian density conversion, unchanged-parameter
PPO ratio one with the new density, and disagreement when the old density is
incorrectly substituted. Float32 standard-deviation checks use four machine
epsilons; no physical criterion changes. Original fixed-opponent, posture,
contact trace, action-mask and independent cross-policy paths also pass.

The evaluator checks agreement between checkpoint manifest and saved sampling
metadata, preserves the original deterministic path, and honors the saved
distribution for sampled diagnostics. Eleven focused checks pass again after
those provenance checks. Python/Bash syntax and PowerShell parsing pass.
The native comparison must still complete its included integration blocks and
unchanged full physical checks before any claim about improved boxing.

## Included native block verified

The first half-noise 1,024-credit block is finite with empty native training
logs, exact Adam 12,696 / 36,368, retained source/profile prefixes and immutable
opponent. Its checkpoint, metric and all 18 passive geometry records are archived
under `sampling-1933312`. Initial after-normalizer Gaussian KL is about .0866
and ratio clip fraction .635, so reduced sampling variance is not assumed to
make PPO updates small or physically safe. The control geometry and all full
physical outcomes remain required; no extra budget or adoption follows.

All initial 24M source profiles completed: 9,085 controls, pair survival 5/8 /
6/8 / 5/8, no complete pass. Formal total after source evaluation is 1,496,411;
PPO after the first verified block is 1,952,000. These are historical snapshots;
live comparison and child progress counters take precedence during training.

## First complete original screen, comparison still in progress

The +8,192 half-noise checkpoint
**c61f6e46c0419699b3850d7176f7d520e66ca0bdf66ea30fdc4fd1990d697e53**
passes all original full criteria: neutral-0 **7/8** pair survivors with own
impacts across two seeds, neutral-1 **8/8** across four scoring seeds, mirror
**7/8** with seven red / eight blue surviving valid impacts and two two-way
episodes. Total pair survival **22/24**. This is the first version-10 full pass;
it does not inherit earlier version-8 successes.

An independent snapshot review checks exact hash and sampling profile, finite
actor/critic/Adam/normalizer/key, immutable adversary, all eight included finite
PPO blocks, exact Adam 12,808 / 36,480 and counters 1,941,504 learner / 1,318,912
world. All 24 complete original trials have correct seed/swap/role/controller
markers and unchanged duration/thresholds; native logs are empty. Actual screen
cost 11,129 controls is already included in live accounting, not charged twice.
Raw reports, model, metrics and a separate one-record version-10 original
history are archived under `sampling-1933312/half_noise/gate-8192`.

The original pass streak is **one**, required two distinct consecutive complete
passes. The +16,384 generation, fresh final and matched control remain pending;
complete all bounded checks before an adoption or extension decision. A favorable
intermediate checkpoint cannot replace later failures or fresh-set evidence.
No stage promotion, main adoption or reliable-boxing claim is made.

## Later distinct generation fails; finish matched control

The following +16,384 half-noise generation
**185e93e0d32a9543360908611a9a948c9d4f20e9a55e50499b9d719b5642ff32**
fails the unchanged original screen: neutral-0 / neutral-1 / mirror survival
**7/8 / 7/8 / 4/8**, total 18/24 and no surviving two-way mirror episode. Fresh
24M final survival is **6/8 / 6/8 / 5/8**, total **17/24**, two surviving two-way
mirror episodes; the complete fresh set also fails (source was 16/24).

Independent verification checks the exact final hash, sampling metadata,
immutable adversary, finite learned state and all 16 finite training metrics,
Adam 12,936 / 36,608 and lineage 1,949,696 learner / 1,327,104 world. Original
screen cost 10,318 and fresh cost 9,175 controls are already included in live
accounting. Both raw full sets, final model, metrics and validation are archived.
The version-10 original history now retains **pass/fail**, current streak zero.
The earlier first-pass validation above is a historical snapshot, not current
certification or permission to restore that intermediate actor.

No further half-noise credits, actor adoption, rollback or phase promotion are
allocated. Finish the already bounded matched control and independently review
both complete arms before any continuation decision. Relative benefit versus
control remains unresolved; software/probability correctness does not establish
physically robust boxing.
