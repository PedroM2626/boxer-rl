# Bounded archived-opponent pool versus fixed-adversary comparison

Declared on 2026-10-24 before new physical collection. This implements the
[pool integration plan](striking-opponent-pool-plan.md), following the reviewed
[broader frozen audit](striking-gae-generalization.md): neutral profiles pass,
but active mirror survival is 23/32 and one archived-adversary corner is 5/8.
Opponent diversity is a testable next hypothesis, not an established remedy.
An unchanged long single-opponent extension is not justified by those results.

## Source and exact intervention

Both arms start at latest bea8147f6a13dd2d473a558f2e2ac68d3492e3136d2899e2d9daf139364bb870,
`checkpoints/striking-validation/gae-consolidate-1998848/gate-65536`.
Retain all actor/critic leaves, Adam moments and counts (14,728 / 38,400),
normalizer, action RNG, auxiliary and retention samplers, consumed/optimized
2,064,384 and physical 1,441,792 prefixes. No actor restoration or reset.

The control retains version 12 and the original 6bf3273a adversary. Version 13
changes only the active training opponent to an immutable three-snapshot pool:
6bf3273a (original adversary), 95dbdff0 (earlier long-GAE final), and bea8147f
(latest source). Full source hashes, actor/normalizer fingerprints, paths and
the explicit one-time transition are pinned in `boxer/striking_opponent_pool.py`.
Each opponent uses its own archived normalization, with the original .01
variance regularizer and clipping; it never shares learner statistics.

Keep four active and four neutral worlds. Draw an opponent uniformly at each
active episode reset, using a separate persistent PCG64 sampler seeded 32,000,053.
Initial assignments, later assignments, sampler state and per-opponent/corner
attempted controls are durable. Neutral assignments are -1 and their opposing
high-level actions stay zero. Historical active corner credits [85,379, 78,461]
are retained; new pool column sums must equal the corresponding active-column
increments. Count an attempted physical step before execution, including failures.

Routine warm resumes preserve the new profile, sampler and all learning state.
Physics is reset as in the inherited trainer; active worlds receive fresh draws
after a routine restart, while the initial transition's recorded draws are used
on the first block. A partial failed collection retains the greater progress
frontier and its pool credits/sampler; it cannot silently rewind consumed compute.
Every checkpoint rechecks all three immutable actor/normalizer fingerprints.

Free roots/legs, learned frozen motor, no support/root forces, separated spawns,
physical fist contacts, impact criteria, rewards, learning rates, original
Gaussian covariance, gamma .999, lambda .99, retention and batch sizes are unchanged.
The retained `fixed_opponent_ppo` label documents the historical lineage and old
immutable adversary; explicit `pool_opponent_ppo` and `mixed_pool_mean` declare
the new task. Ambiguous modes, removed labels and changed snapshots are rejected.

## Allocation, checks and stop rules

Authorize at most **32,768 new diagnostic PPO credits**, **16,384 per arm**.
This raises the diagnostic PPO ceiling from 2,131,200 to **2,163,968**. The first
1,024-credit integration block is included in each arm, followed by ordinary
warm resumes to +8,192 and +16,384. Check initial source leaf equality, finite
losses/state, exact optimizer progression and all mode/pool credits before continuing.
The initial update audits must have an unclipped ratio of one. Pool and control
physical batches intentionally differ; matching batch hashes are not required.

At most **84,000 formal evaluation controls**, raising the reviewed cumulative
1,936,816 to a ceiling of **2,020,816**. This covers 168 full ten-second episodes:
source fresh (24), each arm's +8,192/+16,384 original screens (96), and each final
fresh group (48). Every seed has both spawn swaps. Fresh seeds are untouched
32,000,003 through 32,000,006, shared prospectively across source and arms.
Actual consumption, including incomplete evaluation attempts, remains separate
from the 32,768 new PPO collection world controls. Unused budgets expire at review.

Every evaluation uses the unchanged independent full neutral-0, neutral-1 and
current-policy mirror gates: paired survival at least 7/8, 8/8 and 7/8 respectively,
and neutral own surviving strikes on at least two seeds in each role. The training
pool never replaces the evaluated mirror policy. Stop an arm if any full profile
has at most three survivors; retain that model and evaluate its actual final.
Stop the job on numerical/constraint/contact overflow, unexpected stderr, checksum,
optimizer or accounting mismatch. No automatic retry, fallback actor or unlimited
extension. Neither arm is adopted automatically, even if its complete screens pass.
The new version-13 family's complete pass streak starts at zero; earlier version-12
passes cannot promote it. Later promotion still requires two distinct consecutive
complete passes plus appropriate generalization evidence.

## Operation and verification

`tools/start_striking_pool_compare.ps1` starts the persistent hidden WSL launcher.
Inspect `/home/pedro/boxing-runs/striking-pool-2064384/comparison.json`, its
`diagnostic_output`, host PID/start time and live child before any recovery.
Each arm has durable `progress.json`, `state.json`, per-block logs and checkpoints.
The parent locks its output and `runs/striking_diagnostic_supervisor.lock`;
learners retain `runs/unassisted_boxing.lock`. Existing output is never overwritten.
The existing heartbeat schedule is unchanged. Once the comparison is completed,
run `tools/review_striking_pool_compare.py --input
/home/pedro/boxing-runs/striking-pool-2064384` with the configured WSL Python.
It independently checks all block payloads, source leaves, both complete gate
histories, fresh reports, physics/error logs, foundation certifications and exact
consumption before publishing the evaluated models and evidence. It does not
choose or adopt a policy; document the concrete continuation decision afterward.

Before launch, **41 focused tests passed**, covering snapshot mutation rejection,
lossless learning/optimizer/sampler prefixes, own-normalizer action equivalence,
exact credits, deterministic sampler restoration and physically identical action
routing without assistance, plus historical fixed-opponent/GAE/PPO/evaluator gates.
The first test pass exposed a legacy split-Adam mode whitelist; explicit version-13
support was added and all checks rerun successfully. No physics was collected by
that failed metadata test.

Both native 1,024-credit blocks have now completed: the pool checkpoint is
538e6a3fb166cf62193cf3d571bf904e3e8906590a71242f7d3e6bc5b7d36d25;
the control is b77898bc92271ca212ccff112bd3d3aa93997f83b346f0455985202a882652ad.
Each has finite state/loss, actor/critic counts 14,744 / 38,416, exactly 1,024 new
physical controls and empty native stderr. The first pool assignment happened
to draw the original adversary in all four active worlds; its 512 active credits
are [256,256] for that opponent and zero for the other two. This validates the
native path and accounting, not native coverage of all opponents yet. Later
episode/resume draws remain prospectively uniform; do not reseed to force favorable
coverage. Source fresh evaluations and subsequent ordinary warm resumes are
running; no physical improvement is claimed from these integration blocks.

The next ordinary pool resume reached +8,192 with actor/critic counts
14,856 / 38,528, unchanged sampler/optimizer provenance and empty native stderr.
Its 4,096 new active credits cover all three archived opponents, by corner:
[[756,875], [1461,877], [0,127]]. This verifies native diversity and warm-resume
accounting; uneven coverage in this small allocation is retained rather than
reseeded. Initial evidence
is [archived here](results/striking-pilot/pool-2064384/README.md).

The 2026-10-24 15:44 UTC [partial review](results/striking-pilot/pool-2064384/monitor-20261024T1544.json)
checks both completed pool screens (8/7/7, then 6/8/8) and fresh 8/6/5 survival.
Every complete set fails; no promotion or adoption. The matched control remains
active within its original ceiling. Keep it running, then review both actual
finals before any new allocation; do not restore the source or earlier pool actor.
The read-only monitor consumed zero new physics/optimization. Its initial missing
CPU backend produced GPU allocation diagnostics; explicit CPU revalidation passed,
and native job logs remain empty. The completion reviewer now selects CPU by
default and rejects other explicit backends before loading JAX.

Main boxing 1,048,576, foundation 474,992,640, combined main 476,041,216 and
standalone BC 8,000 remain unchanged. This experiment cannot certify sustained
boxing, longer rounds or unseen spawn/orientation robustness by itself.
