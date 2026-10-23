# Matched mixed-opponent coverage experiment

The full [command-offset intervention review](striking-reward-credit-review.md#completed-full-holdout-and-next-coverage-decision-2026-10-23)
does not justify adopting a masked controller. Original-screen survival improves,
but new-seed neutral own strikes disappear and active survival changes 8/8 to
7/8. Two surviving two-way fresh episodes are useful evidence, not a complete
pass. All 72 new full trials, actual consumption and limitations are retained.

The previous active-only and neutral-only continuations each lost performance
in the other task. The latest gamma-0.999 continuation improves neutral blue
striking but regresses active play on identical seeds. This motivates a fixed
coverage comparison, rather than another unreviewed neutral-only extension or
blanket removal of learned commands. No main ceiling, gate, action/observation
dimension, reward, assistance or physical authority changes.

## Common source and one explicit task transition

Both arms start from latest final SHA
`55976fc15b74b9997b4cb56cf13683f1f87315ac6d67c5cc966ab3c644dd2d73`,
lineage **1,736,704 learner credits / 1,114,112 world controls**, actor/critic
Adam counts **9,608 / 33,280**. Preserve every actor/critic/Adam leaf, normalizer,
RNG, BC/auxiliary/replay sampler and consumed prefix. No rollback or optimizer
reset. Gamma 0.999, lambda 0.95, replay coefficient 1, eight worlds, 128-step
unrolls, 17 outputs/148 inputs and original full learned command/arm authority
remain identical. The zero-command diagnostic is not incorporated.

The control retains ordinary version-5 neutral-only training. The treatment is
explicit version 6 (`--mixed-partner-ppo`): four worlds retain the neutral motor
opponent; four receive the current learner's deterministic mean high-level
opponent actions. Opponent weights/normalizer are fixed during each rollout and
refreshed with the next learner update. Actions follow each opponent's actual
148-element observation; no teacher or scripted punch. World indices 1/2/5/6
are active, 0/3/4/7 neutral, balancing both learner roles at the initial spawn.
The learner corner continues to vary with episode seed parity after resets.
The opponent mode is not added as a new observation feature.

Every world still records **one stochastic learner row per physics control**.
The active counterpart is an opponent, not an extra PPO sample. Both free bodies
receive the original frozen motor feedback and real contacts. There are no
pelvis springs, welds, gravity compensation or applied root forces. First falls
and the 500-control clock retain their original episode semantics. Native
single-learner rewards are exactly sliced from the original paired environment.
Evaluation remains the independent unmodified paired environment, both swaps,
both learned mean actors in active play and the original neutral tests.

The treatment preserves the historical neutral-only role ledger **50,229 /
48,075** as a frozen prefix. New `mixed_credits_by_mode_and_corner` is a 2x2
integer ledger (rows neutral/active, columns red/blue), initially zero, counting
every attempted learner/world control before physics. Its sum equals all new
consumed credits. Keep partial/discarded work on recovery. Optimizer counts
must match optimized blocks; all old source transitions and samplers remain
validated. Ordinary version-5 resumes retain their original behavior/ledger.

## Fixed allocation and stop conditions

Allocate at most **32,768 new learner credits / world controls per arm**, at most
**65,536 total diagnostic PPO credits**. The ordinary neutral arm is an explicit
matched control, not a deployment extension. Each arm's retained first **1,024**
credits must pass native finite/hash/optimizer/ledger guards before further
optimization. The new mixed path runs first so its real integration guard is
inspected early. No guard credits are discarded or counted twice.

At **16,384 and 32,768** new credits preserve distinct checkpoint payloads and
run all three full eight-episode screen profiles (seeds 7/23/31/47, both swaps).
If any full profile has at most **3/8 surviving pairs**, stop that arm early and
run its final fresh check. Keep all failed generations. The shared source and
both final policies are evaluated on preregistered seeds **13,000,003..13,000,006**,
both swaps, all profiles. Total evaluation cap is **168 complete episodes /
84,000 world controls**. No shortened rounds, altered impacts or weakened
survival/multiseed requirements. Two distinct consecutive complete passes remain
necessary; this parent never automatically adopts an actor or promotes a phase.
Review both arms, including any unequal early-stop consumption, before another
allocation. Do not extend either blindly because a replay or return looks better.

At full per-arm allocation lineage reaches **1,769,472 / 1,146,880 world controls**,
Adam **10,120 / 33,792**. Treatment new mode-row totals are 16,384/16,384;
control neutral ledger totals 131,072 including the inherited prefix. Diagnostic
PPO would total **1,745,152**; formal diagnostic simulation starts at **870,055**
and is capped at **954,055** after this job. Record new 65,536 collection controls
and 262,144 integrated replay presentations separately. Standalone BC stays
8,000 updates; main boxing **1,048,576**, combined main **476,041,216**, foundation
**474,992,640 under its 475M ceiling** are unchanged.

## Operation and validation

Hidden persistent launcher: `tools/start_striking_mixed_partner_compare.ps1`;
LF wrapper: `tools/run_striking_mixed_partner_compare.sh`.
Durable state:
`/home/pedro/boxing-runs/striking-mixed-partner-1736704/comparison.json`.
Inspect that state, active child, progress, complete gates, native logs and
checkpoint hashes before any other launch. Supervisor/native child locks are
separate; the learner also retains the global boxing lock. Preserve the existing
monitoring schedule. Do not replay historical forced-rollback recovery flags.

Focused checks cover exact native action/observation/reward routing for both
learner corners and opponent modes, unassisted physics, initial mode/role balance,
exact learned-state preservation, malformed transitions/ledgers/optimizer-count
rejection and unchanged legacy PPO paths. A real retained mixed integration
block remains mandatory; unit tests cannot certify physical training outcomes.


**32 focused tests passed**, including a completed-update plus discarded-attempt
case that retains all world/mode/corner credits without inventing Adam updates.
Python compilation, LF shell syntax and PowerShell launcher parsing pass. The
hidden parent and its native source-evaluation child were verified live without
a duplicate trainer. Native mixed integration proof and full physical outcomes
must still be checked from live state; unit tests are not a phase pass.


## Retained native integration evidence

The live mixed arm completed and verified its first **1,024 learner credits /
world controls**, included in the 32,768 allocation. New mode/corner ledger is
**[[256, 256], [256, 256]]**, historical neutral prefix stays 50,229/48,075,
Adam counts are **9,624 / 33,296**. The finite checkpoint hash is
`a06eb7aead942e022350cb855f16edd225aaa981979f172b5deb6a0624afc87f`.
Every old source field in the initial transition (except explicit opponent-mode
metadata) matches exactly, including actor/critic/moments, normalizer, keys,
all samplers and consumed prefixes. Native logs are empty. The ordinary remaining
invocation is active; control integration and final independent outcomes remain
pending. This guard is integration evidence, not a physical curriculum pass.

The source's complete fresh check used **11,255 evaluation controls**. At this
specific integration boundary cumulative diagnostic PPO is **1,680,640**,
formal diagnostic simulation **881,310**, plus 1,024 new collection controls
recorded separately. Live training can advance beyond this snapshot; inspect
actual child progress before intervention. Source reports, proof and validation
are archived under `docs/results/striking-pilot/mixed-partner-1736704/`; exact
transition and first native model under
`checkpoints/striking-validation/mixed-partner-1736704/`. Main counters and all
four certified foundation histories remain unchanged.
