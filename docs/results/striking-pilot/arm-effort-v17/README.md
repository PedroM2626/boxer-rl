
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
