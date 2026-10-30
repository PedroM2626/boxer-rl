# Post-block throughput and transfer follow-up

Prospective 2026-10-29 local / 2026-10-30 UTC review after all 1,048,576 scaled
GPU credits. Full audit: `results/striking-pilot/gpu-learning-v1/completed-review.json`.
Actual final SHA `639ccec99328b034d7d6c9f1ba08a7305a0edc3e67025aa01cedcc06cf979052`,
consumed/optimized 10,486,784, world 9,864,192, actor/critic Adam 146,328/170,000.
Global diagnostic PPO **10,569,984**, formal **2,340,326**, main **476,041,216**.
All older unused allowances expire, including 30,183 native controls from the
completed job. No unchanged PPO continuation or restoration of an intermediate.

## Reviewed evidence and decision

All 65 checkpoints, 144 native/full trace pairs and 2,900 captured training
events verify. Known neutral valid cycles rise from 22 to 36; pair survival changes
from 5/4/3 to 5/5/3 out of eight. One intermediate has valid reciprocal strikes on
one surviving seed; it is not restored. Final fresh neutral cycles total 27 and
mirror cycles 5/1, but survival 4/6/3 fails every complete profile. No full pass.
Several falls occur during the initial approach, before recognizable exchanges.
This interval is not convergence evidence or proof that further learning cannot
help. It also does not justify blindly extending an exhausted ceiling.

The logged collection+PPO rate declines from 510 to 371 credits/s by quarter.
Complete-job elapsed time includes native gates, evidence compression/copying,
checkpointing and Git publication; the earlier 870 one-block figure was not a
sustained rate. The full audit/history was reserialized and native ledgers
rescanned before every substep, despite no native evaluation running concurrently.
Keep a separate small durable attempted-step frontier, flush the full audit at
rollout/phase/publication boundaries and reconcile both on resume. Reject any
unoptimized attempted frontier instead of rewinding. Test crash/identity cases
and benchmark accounting on stored data, without new physics or learner credit.

## Matched diagnostic before another learning allocation

Run only original **CUDA 3/5** with the established slow audited reference adapter,
unchanged native observation/contact/cycle measurement code, actual source and
actual final, seeds 7/23/31/47 and both placements in all three roles. Reuse the
already verified native baseline and final-known traces byte-for-byte; no new
native control. Do not repeat the old nine-profile solver or legacy comparisons.

New explicit allowance: at most **24,000 CUDA evaluation controls** (48 episodes,
500 controls each), formal before 2,340,326 and ceiling **2,364,326**. Maximum
240,006 GPU world substeps, including six separately counted one-step warm-ups.
Record every attempted substep/reset forward, native reset forward, first fall,
finite/capacity/float diagnostics and source/trace hashes. No PPO or renderer.
No unused control allowance rolls forward. Stop/investigate any partial failure
before retry. Use exclusive global locks and a hidden persistent WSL launcher.

Compare survival, first-fall times/types, contacts and complete cycles on matched
conditions to distinguish persistent early-control fragility from a measured
backend-dependent transfer gap. These diagnostics cannot issue native promotion
or establish exact trajectory equivalence. After the completed comparison,
choose and prospectively document a bounded learning/transfer intervention from
actual results. Preserve the latest final and unchanged full gates; do not adopt
new solver settings, action masks, assistance or a changed punch definition.
