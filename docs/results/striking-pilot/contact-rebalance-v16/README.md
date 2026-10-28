
## Navigation ablation review and contact-rebalance v16, 2026-10-27

The saved navigation ablation is independently reviewed: **2,178 new formal
controls**, global **2,190,984**, zero PPO. All eight native/full traces, finite
states, unchanged source checkpoint and original punch criteria verify. The
masked pair survives 3/8 with zero valid cycles; the unmasked same-seed mirror
survives 2/8 with zero cycles. Five masked episodes terminate (two low-pelvis,
three self-leg contacts). Masking both actors is diagnostic, not a full gate,
and this small difference neither isolates each actor nor solves instability.
Frozen-opponent comparison similarly found only 3/8 and 4/8 survivors. Neither
result warrants weakening action or physics criteria.

V15 produced 23 fresh neutral cycles but failed active-pair stability. Its graded
partial-contact term pays up to **2** on each qualifying contact, while the fall
penalty is **5**; the last session recorded **27,033** in own partial-contact
payments, creating a plausible incentive to retain close contact at the expense
of balance and full recovery. This is a hypothesis, not isolated proof. An older
posture-cost comparison worsened fresh survival and is not repeated blindly.

Prospective v16 tests one reward change: scale the net graded partial-contact
payment from **2 to 0.25** per event; retain its target-distance potential,
**10** per complete valid cycle, base fall penalty, long-GAE, mixed task,
actor/critic optimizer and all existing action/physical/evaluation profiles.
The arithmetic changes only the graded contact component and applies symmetrically
to both corners. No direct support, pose lock, reduced action authority, new
teacher loss or threshold change. Exact actual v15 final SHA **9132ab4c** is the
source; preserve actor, critic, Adam, normalization, RNG, frozen opponent and
prior consumed transitions exactly, with a new explicit transition metadata
record. This is a bounded new experiment, not a claim of success or causality.

Allocation: at most **2,097,152 new learner credits**, global diagnostic PPO
ceiling **7,406,848** from 5,309,696; formal evaluation at most **60,000 new
controls**, global ceiling **2,250,984** from 2,190,984. Prior diagnostic
allowance expires; there is no budget rollover. Use 1,024-credit integration,
then complete original physical gates at 524,288-credit intervals and a fresh
four-seed final. Reuse v15 final original gate as zero-debit baseline; never count
it as a new pass. Early exit only after two distinct complete consecutive full
passes, then fresh gate. Review failures and ceiling before further allocation.

Source lineage starts at **5,226,496 consumed/optimized**, **4,603,904** physical
collection controls; retain main **476,041,216**, BC **8,000**, cloned queries
**6,349**, signed distances **960**, renderer forwards **2,697**. Use the hidden
persistent WSL launcher and normal warm starts. All progress and evaluation
attempts are counted; do not repeat an interrupted partial gate blindly. Output:
`/home/pedro/boxing-runs/striking-contact-rebalance-v16/audit.json`.
Independent gates remain mandatory; no automatic adoption or boxing completion.

### V16 integration completed

The ordinary 1,024-credit integration completed with **5,227,520** lineage
consumed/optimized and **4,604,928** collection controls. Checkpoint SHA-256
`3c0be5dd77eea17080a717c4402cfff685c98b1b84dae95cbc69d2a9f3139b8a`;
actor/critic Adam counters **64,152 / 87,824** match the retained source plus 16
updates. The payload and all numeric leaves are finite; the 18 update-audit
records have first-rollout clipped fraction **0.0**. No native error output.
The initial transition retained actor, critic, both Adam states, normalization,
RNG, samplers and fixed-opponent payload exactly. Focused v16/v15 cycle and
reward checks: **22 passed**. This is a finite integration check, not evidence
of improved behavior. Supervisor has started the authorized next chunk toward
the first full gate at 524,288 new credits. Integration payload and update audit
are archived, without resetting the live job.

## First full gate: reciprocal motion but incomplete survival, 2026-10-28

At **524,288 new v16 credits**, checkpoint SHA-256
`2d618717c1a1e3ccded77311d3bf45cd9a5737b30653836034337be019df20e3`
produced **10 complete deterministic self-play cycles** (five per corner).
Surviving episodes on known seeds **7 and 31** each include valid cycles by both
actors, so all three motion subchecks pass. The independently reviewed native/full
trace and checkpoint hashes, finite arrays, scored native impulse and full-cycle
conjunction confirm these measurements. This is a first observed limited
reciprocal motion milestone, not an overall gate pass or fresh-seed certification.

Pair survival **5/8 neutral 0, 7/8 neutral 1, 4/8 self-play** misses the unchanged
7/8, 8/8, 7/8 original thresholds. Relative to the inherited original baseline
5/8, 5/8, 1/8, self-play survival and reciprocal cycles improve, but the two
checkpoints are successive training snapshots, not a randomized causal comparison.
Across the same four self-play seeds, all four **swapped** episodes survive and
all four **unswapped** episodes fail, each before a scored contact. This striking
asymmetry needs validation at the next full gate and fresh seeds; it does not by
itself identify the cause. Avoid claiming reliable boxing or promoting the model.

The gate consumed **8,907** formal controls; global formal **2,199,891**. At the
completed gate, global diagnostic PPO **5,833,984**, lineage
**5,750,784 consumed/optimized** and **5,128,192** collection controls. Main
**476,041,216**, BC **8,000**, cloned **6,349**, signed distances **960** remain
unchanged. First physical gate failed, streak zero. Continue the already approved
next chunk to **1,048,576 new v16 credits**, without reset or change of gates;
use its outcomes to test retention, survival and swap asymmetry. The live learner
counter advances independently of this completed-gate snapshot.

For user-visible evidence, select the **seed 31, swapped** ten-second self-play
bout with three corner-0 and one corner-1 valid cycles. Render exactly 250 stored
states at 25 fps, preserving the full bout and marking counts; this is a selected
example from known seeds, not a success-rate estimate or fresh certification.
Allocate **250** read-only renderer forwards, cumulative **2,947** from 2,697;
no actor execution, physics integration, optimizer updates or formal controls.
The separate saved-state renderer does not lock or interrupt the live learner.
Save trace checksums, preview, video hash and decode all frames before publication.

The selected reciprocal full-bout video completed: **250/250 frames decoded**,
25 fps, 10 seconds; preview visually inspected. Source and video SHA-256 values,
per-hand event descriptors, and the selected seed/swap are recorded in
`reciprocal-seed31-video.json`. The known FFmpeg/JAX fork warning appeared, but
the encoder exited and the resulting video fully decoded. The clip is
`fights/striking-pilot/v16-reciprocal-seed31-swapped.mp4`. Renderer total is
**2,947**; no added physical controls or learner updates. It illustrates a selected
surviving episode while the complete survival gate remains failed.

## Second and third full gates: reciprocal consistency remains open, 2026-10-28

Independent saved-trace review of both checkpoints verifies all native/full
trace hashes, finite arrays, native impulse, unchanged complete-cycle rules,
checkpoint provenance and original physical criteria. Neither is a full pass;
no consecutive pass or promotion. Results, each out of eight pair episodes:

| New v16 credits | Neutral 0 / neutral 1 / self-play survivors | Valid self-play cycles by corner | Surviving reciprocal seeds |
| --- | --- | --- | --- |
| 524,288 | 5 / 7 / 4 | 5 / 5 | 7, 31 |
| 1,048,576 | 5 / 5 / 3 | 1 / 1 | 31 |
| 1,572,864 | 4 / 5 / 5 | 2 / 3 | 47 |

Only one surviving seed in each later gate has valid cycles by both agents, below
the unchanged two-seed motion criterion. Self-play survival improved from 3/8 to
5/8 between these checkpoints but still misses 7/8, while neutral survival is
4/8 and 5/8. At the latest gate all four swapped episodes survive, but only one
of four unswapped episodes survives; all three unswapped failures occur before
scored contact. This continues the observed swap asymmetry without identifying
its cause. Results on four known seeds are limited and do not establish fresh
robustness or a causal effect of the reward change.

The second and third gates consumed **7,409 and 8,130** formal controls,
respectively. At the third completed gate: v16 new formal **24,446**, global
formal **2,215,430**; v16 learner/collection credits **1,572,864**, global
diagnostic PPO **6,882,560**, lineage consumed/optimized **6,799,360** and
lineage world controls **6,176,768**. Main **476,041,216**, standalone BC
**8,000**, cloned queries **6,349**, signed distances **960**, render forwards
**2,947** remain separately retained. Live final-chunk counters advance beyond
this snapshot. No nonfinite checkpoint data, overflow warnings or native errors
were found; exactly one supervisor and one learner continue.

Decision: complete the already authorized final chunk at 2,097,152 new credits,
then run both original and fresh evaluations within the existing 60,000-control
cap. Review stability, reciprocal cycles and swap asymmetry across all checkpoints
before deciding a bounded next intervention. Do not reset Adam, roll back, promote
or weaken gates. Reviews: `step-1048576-independent-review.json` and
`step-1572864-independent-review.json`.
