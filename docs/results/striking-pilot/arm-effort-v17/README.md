
## Half-arm comparison and prospective v17 arm-effort PPO, 2026-10-28

The half-arm diagnostic completed and independently verified all matched
fresh-seed traces: **7/8 pair survivors and zero valid cycles**, versus **8/8
and zero** with arm residuals zeroed, and **1/8 with four valid cycles** at
unmodified full authority. Both actor outputs were scaled; physics and navigation
were unchanged. This is strong evidence of a stability/strike tradeoff on these
four seeds, not proof that a particular joint is responsible or that half output
is a suitable permanent controller. Neither scaled condition can promote.

The half-arm test used **3,770** new formal controls; global formal **2,238,008**.
The unused 230 expire. Both diagnostic arms and original unmasked full gate,
traces, finite values, provenance and source hashes are archived. V16 has reached
its original PPO ceiling; no unchanged extension or rollback is performed.

Prospective v17 tests an explicit **training-only smooth effort cost** on the
learner's 14 filtered arm residuals: `2.0 * dt * mean(residual**2)` per control.
At dt 0.02 it is bounded between zero and 0.04 per step. It does not clip,
scale, mask, constrain or otherwise change any action or force; full arm and leg
authority remains available. Retain v16's 0.25 graded partial-contact reward,
target potential, 10-per-complete-cycle bonus, base fall penalty, frozen motor,
physical self contacts, active/neutral task and all independent gate criteria.
This tests whether learning lower effort while preserving high-value punches can
improve stability, not a claim that a penalty will work. The old posture-margin
penalty failed in another profile and is not repeated. The zero/half arm probes
were diagnostic modifications and never serve as evaluation gates.

Start from exact actual v16 final **5bbf6a38**, including both Adam states,
actor/critic, normalization, RNG, samplers and frozen adversary. Preserve all
consumed main and diagnostic counters. New bounded allocation: at most
**2,097,152 PPO credits**, diagnostic PPO ceiling **9,504,000** from 7,406,848;
at most **60,000** formal controls, formal ceiling **2,298,008** from 2,238,008.
V16 lineage starts at **7,323,648 consumed/optimized**, **6,701,056** collection
world controls; main **476,041,216**, BC **8,000**, cloned queries **6,349**,
signed distance calls **960**, renderer forwards **2,947** remain separately
retained. Prior diagnostic caps expire rather than roll over.

Run a 1,024-credit finite integration, then unchanged original full gates at
524,288-credit intervals and a fresh final on four new seeds. Reuse the v16
original full gate as a zero-debit baseline; two distinct consecutive new complete
passes are required before any stage advancement, followed by fresh checks. Review
both strike retention and survival, across corners and swaps. On failure, retain
all attempted counters and inspect before ordinary resume; no forced rollback,
automatic adoption, duplicate jobs or unbounded budget extension.

Thirty focused reward, trace and gate-protection tests passed. Source actor,
critic, both Adam states, normalization, RNG, sampler states and frozen adversary
were compared exactly after transition. Use the persistent hidden WSL launcher;
job state: `/home/pedro/boxing-runs/striking-arm-effort-v17/audit.json`.

### V17 finite integration completed

The 1,024-credit integration finished at lineage **7,324,672** consumed and
optimized, **6,702,080** collection controls. Checkpoint SHA-256
`c8709013df0e558687d852e30efcce4efb06785bd5799d998ae6cc9c2bd6a340` has finite
numeric leaves and actor/critic Adam counts **96,920 / 120,592**, exactly the
source plus 16 updates. All 18 update-audit records exist, with first-rollout
clipped fraction **0.0**. The measured training-only arm cost is positive
(4.606 over this integration), native error log is empty, and all source actor,
critic, optimizer, RNG, normalizer, samplers and frozen opponent were preserved.
This confirms implementation integration, not improved boxing. The hidden
supervisor has started the next chunk toward the first 524,288-credit full gate;
all physical and promotion criteria remain unchanged.

## First v17 full gate: early regression, 2026-10-28

At **524,288 new learner credits**, checkpoint SHA-256
`f0efe32821b70ecf7bf2d44f8f7c1c294a911b3990f3ae91ebdee2830fc9909c`
was independently reviewed against all 24 archived native/full trajectories.
Trace and checkpoint hashes, finite data, native impulse, original complete-cycle
criteria and all foundation histories verify; no numerical/overflow errors.

Pair survival is **3/8 neutral 0, 6/8 neutral 1, 1/8 self-play**, versus
**5/8, 6/8, 3/8** on the inherited v16 final baseline. Neutral valid cycles
are 5 and 13; self-play has one cycle in corner 1, none in corner 0, and zero
surviving reciprocal seeds (versus three on the inherited baseline). Six of seven
mirror failures occur before any scored event. The original physical gate and
self-play motion check both fail; full-pass streak zero. This is an early
regression relative to the frozen predecessor on these known seeds, not proof
that the reward change cannot learn or evidence of generalization.

The full gate consumed **6,127 formal controls**, global formal **2,244,135**;
global diagnostic PPO at the checkpoint **7,931,136**. Lineage consumed and
optimized **7,847,936**, world controls **7,225,344**. Main **476,041,216**,
standalone BC **8,000**, cloned queries **6,349**, signed distances **960**,
renderer forwards **2,947** remain separate. The independently reviewed
checkpoint and trajectories remain preserved; live next-chunk counters advance.

Decision: continue the already approved second chunk to 1,048,576 new credits,
without lowering thresholds or restoring a policy. The single early gate does
not isolate a cause; compare strike retention, early falls and both corners at
the next complete gate, then reassess before the bounded final. Exactly one
supervisor and one learner are active, native error logs empty, and latest live
checkpoint hash/finite state verified. No main adoption or promotion. Evidence:
`step-524288-independent-review.json`.

## Current paired model GPU preflight, 2026-10-28

The user correctly noted that previous versions achieved thousands of steps per
second on the GPU. Current native paired PPO explicitly forces CPU and iterates
native MuJoCo worlds; this was an implementation choice, not a GPU hardware
failure. The existing GPU Warp boxer is a different scene/observation/reward
contract; its historical throughput cannot be assumed for the current model.
The actual WSL runtime exposes the RTX 3060 and installed `mujoco_warp`.

A bounded one-world read-only preflight loaded the actual current paired model
(two free roots, 72 qpos, 70 velocities, 58 actuators, no equality constraints
or gravity compensation) to CUDA and performed one Warp forward. At reset seed
34002003, CPU/GPU max absolute differences were **2.94e-8 qpos**, **1.01e-8
qvel**, **2.34e-7 body positions**. The paired native object performed two resets;
no physics steps, learner updates or formal evaluation controls were consumed.
Source and executable: `paired-warp-preflight.json` and
`tools/probe_paired_warp_forward.py`.

Warp emitted **solver iterations limit reached at 3 iterations** during the
forward. The matching positions are a useful compatibility signal, not proof
of solver convergence or force parity. Contact/constraint capacity, actual
contact impulse forces and native substep punch trace, recurrent PPO observations,
reward/termination behavior, randomized resets, rollout trajectory drift and
buffer-overflow handling must be compared before using a GPU backend for training.
Benchmark the validated batched implementation on this GPU; do not promise the
legacy environment's throughput or silently transplant its incompatible policy.
The live CPU learner was not interrupted by this small preflight.

## V17 completed and GPU contact preflight, 2026-10-28

The v17 arm-effort block completed **2,097,152 new PPO credits** at its declared
ceiling, global diagnostic PPO **9,504,000**. Final policy SHA-256:
`f311640d78941546938412f5802cff872e0a3611c392f49a3b4719eefaa1f26a`.
All five new original/fresh full gates failed. The last original gate has pair
survival **4/8, 3/8, 2/8**, neutral valid cycles 11/10, and no mirror valid
cycles. Its fresh gate has survival **5/8, 7/8, 3/8**, neutral valid cycles
17/16 and mirror cycles 4/1, reciprocal on one surviving seed. The original
v16 source baseline had 5/8, 6/8, 3/8 with reciprocal cycles on three known
seeds. Sequential checkpoints and different fresh seeds cannot establish a
causal comparison; this is not a complete pass, fresh certification, or a
basis for unbounded continuation. Checkpoint/native/full trace checksums, finite
arrays and all complete gate criteria were independently verified; logs are
empty of native errors. No active training process remains.

V17 formal evaluation used **32,993** controls, global **2,271,001**; unused
27,007 expire. Lineage consumed/optimized **9,420,800**, world collection
**8,798,208**. Main **476,041,216**, BC **8,000**, cloned queries **6,349**,
signed distances **960** and render forwards **2,947** remain separately retained.
No rollback, main adoption, altered gate or forced resume.

The user requested a fair comparison with the previous unassisted GPU approach
and a path back to GPU throughput. Local historical Warp checkpoints exist:
`runs/warp6/checkpoint/step-200212480.pkl` SHA-256
`ed0c56e760bb74a63aaa35d8b5d5ed445fa745e66519e5008536548a17368937`, and
`runs/warp7/checkpoint/step-200212480.pkl` SHA-256
`b5a7e7bc837ff3938fcf0dbfd96fcdd74b0c32f711a1da05f7cb919a3be200fd`. They use
different model/action/observation/termination contracts; neither may be loaded
as the current actor or scored under current criteria without an explicit adapter
and independently recorded measurement. Their old video/score alone is not a
matched comparison. Preserve the original checkpoints and build a truthful
shared-measurement comparison rather than assuming the new approach is superior.

A GPU contact preflight used two previously saved v15 impact states, neutral and
active. The initial exploratory `paired-warp-contact-preflight.json` produced a
float-cast overflow and a three-iteration solver warning; its numbers are
**superseded, not parity evidence**. A strict check localized the overflow to
`efm_fluid`, an uninitialized MuJoCo binding scratch array with fluid density and
viscosity both zero; repeated property reads do not share memory. Attempting to
clear the property did not fix it. Two strict retries rejected the float overflow
rather than silently proceeding. This is a transfer-path issue, not evidence that
training data are nonfinite; no original physics state or policy was changed.

The revised diagnostic uses fresh Warp data and copies only finite saved qpos,
qvel, controls and time, with zeroed independent solver state. In both selected
cold-start cases CPU and GPU found identical contact geom pairs (13/13 and 17/17)
and constraints remained below reserved capacity (110/2048 and 126/2048). For
the recorded fist-to-torso pairs, CPU/GPU normal forces were **51.62004/51.62005 N**
and **66.62798/66.62788 N**. Body position maximum differences were below 1.5e-7 m.
Results and limitations: `docs/results/striking-pilot/paired-warp-contact-forwarded.json`.
The separate earlier solver-iteration warning remains unresolved. Two snapshots,
cold-start forces and chosen capacity do not prove long-trajectory, solver,
strike-cycle, reset, buffer-overflow or policy equivalence. There were six
experimental GPU forwards in the contact work (two exploratory, one per rejected
strict retry, two revised), plus the separate one-forward preflight; no GPU
physical stepping, PPO updates or formal evaluation controls were spent.

Decision: do not spend another long CPU PPO block on an unproven change while
GPU compatibility and legacy comparison are outstanding. Continue the GPU port
and equivalence work: quantify solver convergence, first dynamics steps, contact
impulse and event timing, then implement batched current-model observations,
rewards and recorded motion windows; benchmark on this card before adopting a
backend. Keep native CPU full gates authoritative. This is not a GPU training
launch, completed port or declaration that legacy policies pass current gates.
