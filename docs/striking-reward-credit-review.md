# Reward-credit review and completed horizon comparison

**Current status:** the full 72-episode zero-command holdout is completed and
reviewed. No complete set passes, so the masked controller is not adopted.
The next bounded training comparison tests 50/50 neutral/current-mean-active
opponent coverage against ordinary neutral-only control, from the unchanged
latest final. Read the [current matched contract](striking-mixed-partner-comparison.md).

## Completed neutral curriculum review

The version-4 neutral-partner task exhausted its fixed **32,768-credit** allocation.
Both distinct complete screen generations fail. No actor is adopted, phase
promoted or unchanged neutral-retention extension granted.

| Checkpoint / sample | Neutral 0 survival / own stable events | Neutral 1 survival / own stable events | Active survival / stable events |
| --- | --- | --- | --- |
| Source / fresh 9M series | 6/8 / 0 | 7/8 / 0 | 7/8 / 2 |
| +16,384 / screen | 6/8 / 0 | 8/8 / 1 (one seed) | 5/8 / 2 |
| +32,768 / screen | 7/8 / 0 | 8/8 / 0 | 7/8 / 0 |
| +32,768 / fresh 9M series | 7/8 / 0 | 8/8 / 0 | 7/8 / 0 |

Final survival meets those profile thresholds, but the independent multi-seed
own-striking requirements fail completely. Survival is not boxing completion.
The previously verified three/two exchange remains historical evidence of one
earlier learned policy, not a claim about this final actor or generalization.

Both complete generations, source/final fresh reports, all 32 finite training
block losses, final role counters and exact checkpoint hashes are archived under
`docs/results/striking-pilot/neutral-retention-1638400/` and
`checkpoints/striking-validation/neutral-retention-1638400/{gate-16384,gate-32768}/`.
Final actor SHA:
`63c06464fa0ce0ec238d349e4fac559c454a323f235fc8630a5bf599753ce8e9`.
Its retained lineage is **1,671,168 consumed/optimized learner credits /
1,048,576 world controls**, Adam **8,584 / 32,256**. New learner credits split
**17,330 red / 15,438 blue**, summing exactly to 32,768 physical world controls;
partner actions were not credited as training samples. All state leaves/moments
are finite, native error logs empty, current full foundation certification
histories/hashes verified and the stopped main actor unchanged.

This allocation adds **32,768 diagnostic PPO credits**, **32,768 separately
recorded collection world controls**, and **43,761 evaluation world controls**.
Cumulative diagnostic PPO is **1,581,312** and formal diagnostic simulation
**699,791 world controls** before the next audit. Main foundation/boxing remains
**474,992,640 / 1,048,576**, combined **476,041,216**, with no ceiling change.
Standalone BC remains 8,000 updates / 2,048,000 presentations; this run applies
another 131,072 integrated replay examples within PPO, not separate BC updates.

## Concrete next action: measure credit before changing it

Do not repeat the same failed allocation. Measure the exact existing reward under
passive full replays of the latest mean actor's three fresh profiles and the
archived successful exchange. The existing reward gives +2 for own legal impact,
-2 for one received, -5 for own fall, plus the original facing/engagement/progress
terms and arm cost while both stand. The joint impact terms cancel by definition;
that algebra alone does not establish a policy optimum or prove reward hacking.

The current discount is gamma **0.99 per 20 ms**, an exponential timescale of
about **1.99 simulated seconds**. The known exact impacts of the archived exchange
arrive at 8.742–9.828 s. Their discounted contribution and the dense reward earned
earlier should be measured explicitly before a different training objective is
justified. Reweighting a fixed recorded trajectory under other gammas is an
offline calculation, not evidence that a newly trained policy would improve.

The historical [proximity reward comparison](striking-reward-comparison.md)
already failed, and its subsequent motion-reference investigation is archived.
Do not rerun those bonuses or substitute a favorable old checkpoint as the new
learner. The older actor is used only as a labeled reference in this audit; the
latest neutral-retention actor and all counters remain preserved for continuation.

The bounded audit has **26 ten-second episodes, at most 13,000 new evaluation
world controls**, and **zero optimizer updates or PPO training credits**:

- Replay both swaps of seed 7,000,003 using the archived learned three/two actor.
- Replay all eight latest 9M-series trials in each of the two neutral profiles
  and active self-play. Every original physical trial field must match exactly.

`StrikingTrace` enables optional read-only reward terms. The original reward
expression, its order and returned float32 values remain unchanged. Components
are computed separately only for traces, checked against the actual reward and
stored with observations/actions/qpos/qvel. No term enters observations, actions,
PPO updates or independent impact scores. Record both undiscounted and gamma
0.99 / 0.995 / 0.999 observed returns by component and corner. These descriptive
comparisons use different selected policies/seeds and cannot isolate causality.

**26 focused tests passed**, including exact native trajectory/reward agreement
with tracing enabled, term reconstruction, impact zero-sum algebra, fall gating,
legacy update equivalence, retained optimizer state and both neutral role routes.
The audit tool compiles and its LF shell wrapper passes syntax checks. Its hidden
launcher is `tools/start_striking_reward_credit_audit.ps1`; durable state:
`/home/pedro/boxing-runs/striking-reward-credit-1671168/reward-credit.json`.
Inspect that process, active evaluation/progress, hashes and native logs before
another launch. Keep the existing monitoring schedule. Archive exact-replay
proof, raw components and actual consumption, then make a concrete decision on
credit horizon/objective or state coverage. No reward, discount, physical gate,
main adoption, rollback or extra training is authorized merely by launching this
measurement.

## Completed measurement and bounded horizon comparison

All **26 physical trials reproduce every original result field exactly** and
their separately measured components reconstruct the unchanged actual rewards.
All values/traces are finite, hashes verify and native evaluator logs are empty.
Actual cost is **12,222 world controls**, zero optimizer updates. Formal
diagnostic simulation becomes **712,013**, diagnostic PPO stays **1,581,312**.
Full rewards, observed discounted components, raw trajectories and exact-replay
proof are archived in `docs/results/striking-pilot/reward-credit-1671168/`.

In the archived selected three/two exchange, red's net impact return is
**0.018561** under gamma 0.99 versus **0.671583** from progress/facing/engagement/
arm shaping. On the identical fixed trajectory, gamma 0.999 gives impact return
**1.247973**, shaping **2.047639**. Blue receives the opposite net impact term;
these five contacts cancel jointly, as intended by the original zero-sum event
reward. A surviving latest neutral-role-0 example (seed 9,000,003, swap true)
earns red shaping **0.639770** at gamma 0.99 with no impact at all. These different
policies/seeds are descriptive examples, not proof that waiting is globally
optimal or that a longer horizon will produce boxing.

The measured temporal attenuation justifies **one matched bounded discount-horizon
comparison**. It does not justify another reach/motion bonus, changing independent
scores, or restoring the older successful actor. Both arms copy the **latest
final neutral actor**, SHA `63c06464fa0ce0ec238d349e4fac559c454a323f235fc8630a5bf599753ce8e9`,
with every actor/critic/normalizer/key/sampler/Adam leaf and consumed prefix exact.
Compare **gamma 0.99 control** with **gamma 0.999** (about 1.99 versus 19.99 s
discount timescale) while retaining GAE lambda 0.95, 128-control unroll, eight
one-learner worlds, coefficient-1 verified replay, learning rate and all native
reward terms. The critic starts identically from its old 0.99 calibration in
both arms; adaptation to a different value scale is a limitation of this warm
comparison, not a hidden critic reset. GAE truncation/bootstrapping still applies
at the real first fall/round end; a longer discount alone does not make the
finite rollout observe an entire ten-second future.

Use explicit **version 5** configuration and `--credit-horizon-ppo
--training-gamma <0.99|0.999>` together with the existing neutral/contact/replay
flags. The checkpoint marker records old/new gamma, source hash, source lineage
**1,671,168 / 1,048,576 world controls** and Adam **8,584 / 32,256**, without reset.
Implicit/mismatched changes fail closed. The existing native GAE now receives
the validated configured gamma/lambda; all old profiles remain exactly 0.99/0.95.
No gamma feature enters physical observations or actions.

Allocate **32,768 new diagnostic learner credits per arm**, **65,536 total**, with
**65,536 collection world controls** because each world has one learner. Both
first retain a **1,024-credit / 1,024-world-control** block, verify finite states,
role accounting and Adam **8,600 / 32,272**, then ordinary warm resume the remaining
31,744 credits. Both arms share this reset boundary, initial model/key and replay
sampler streams; world state is not serialized. This integration block is part
of each allocation, not a discarded smoke or rollback. Final expected counts
are **9,096 / 32,768**, lineage **1,703,936 / 1,081,344 world controls** per arm.

Preregister **new seeds 10,000,003–10,000,006**, both swaps, ten seconds, all three
profiles. Source fresh evaluation is 24 episodes; each final arm receives the
complete original screen plus fresh set (48 episodes). Maximum is **120 episodes /
60,000 evaluation world controls**. No tuning/selection or budget change uses
these fresh results. Report full survival, own surviving multi-seed strikes,
active impacts by corner and surviving two-way episodes. Parallel comparison
arms cannot supply consecutive passing generations. No stage promotion, main
adoption or further allocation is automatic, regardless of shaped return.

At full allocation cumulative diagnostic PPO reaches **1,646,848**. Another
262,144 integrated replay presentations occur across the arms (131,072 each);
standalone BC stays 8,000 updates. All main counters and ceilings remain unchanged.
Keep all attempted/discarded credits and record collection/evaluation separately.

**28 focused tests passed**, including exact learned-state preservation for both
gamma profiles, mismatch/implicit-change rejection, correct discounted credit
and done truncation, read-only reward tracing, native partner routing, optimizer
prefix guards and legacy update equivalence. Parent tools compile and shell
syntax checks pass; real native first-block validation remains mandatory.
The hidden launcher is `tools/start_striking_credit_horizon_compare.ps1`; durable
state `/home/pedro/boxing-runs/striking-credit-horizon-1671168/comparison.json`.
Inspect that job, active child/progress, complete reports, checkpoint hashes and
native logs before any other launch. The measurement audit is completed; preserve
the existing monitoring schedule. At completion archive both complete arms and
make an evidence-based decision instead of extending either blindly.

## Native launch and control integration proof

The comparison started on 2026-10-23 at **16:21:49 UTC**. Its full 24-episode 10M
source baseline consumed **12,000 evaluation controls**; all three profiles
survive 8/8, neutral roles score no surviving own events and active play has one
surviving event on one seed. The complete striking criteria still fail.

The gamma-0.99 control retained and verified its first **1,024 learner credits /
1,024 world controls**, adding 512 credits per corner to the inherited role
ledgers. Adam counts are **8,600 / 32,272**, checkpoint leaves finite, hash
`402a50b3e5f1444bdcdfc5c1a8b4718bab4b06345a6874ecf1a641f26590778e`, and native
logs empty. Its ordinary remaining invocation is active. The longer-horizon
arm's own native integration check remains pending; both tests and this control
guard cannot certify its eventual physical outcome.

Transition/source baseline, configuration and exact control integration model
are archived under `docs/results/striking-pilot/credit-horizon-1671168/` and
`checkpoints/striking-validation/credit-horizon-1671168/`. At the archived boundary
diagnostic PPO is **1,582,336**, formal diagnostic simulation **724,013 world
controls**, plus **1,024** separately recorded new collection controls. Later live
progress continues beyond this snapshot within the fixed matched allocation.


## Completed horizon comparison and bounded continuation (2026-10-23)

Both fixed arms completed normally at **16:42:41 UTC**. The source and all
120 full physical episodes are retained. All checkpoint hashes, finite leaves,
32 update metrics per arm, retained Adam counts, role credit ledgers, matched
auxiliary sampler streams and complete foundation histories were independently
verified with `tools/review_striking_credit_horizon.py`. Native error logs are
empty. Neither final actor passes the unchanged complete criteria.

| Final policy / sample | Neutral 0 survival / own events | Neutral 1 survival / own events | Active survival | Active surviving red / blue events | Active scoring seeds | Surviving two-way episodes |
| --- | --- | --- | --- | --- | --- | --- |
| Gamma 0.99 / screen | 8/8 / 0 | 7/8 / 0 | 8/8 | 0 / 1 | 1 | 0 |
| Gamma 0.99 / fresh | 8/8 / 0 | 8/8 / 0 | 8/8 | 0 / 0 | 0 | 0 |
| Gamma 0.999 / screen | 6/8 / 2 | 8/8 / 0 | 6/8 | 1 / 3 | 2 | 1 |
| Gamma 0.999 / fresh | 8/8 / 0 | 7/8 / 0 | 7/8 | 2 / 4 | 4 | 1 |

Fresh seeds are 10,000,003 through 10,000,006, both swaps. The treatment's
six surviving active events span all four seeds, unlike the control's zero.
One fresh mean-policy episode (10,000,003, swap true) lasts ten seconds with
one red and two blue valid events: blue 9.318 s / 0.635 Ns, red 9.742 s /
1.090 Ns, blue 9.776 s / 0.827 Ns. This is broader measured active striking,
but only one two-way episode out of eight; screen stability regresses, one
fresh active pair falls, and neither neutral role has fresh surviving own
strikes. The longer discount is a promising diagnostic, not certified boxing.
The inherited critic's old discount calibration remains a limitation.

The final gamma-0.99 hash is
`7a5d8b80b6504341e2f026836dbdc17ad31f80f7b29e5cedc30883869940ca16`;
gamma-0.999 is
`22f4908af40b7c5269952d40a98f35630c2300d50d2bf8ba54c125f36cb6dd55`.
Each lineage reaches **1,703,936 learner credits / 1,081,344 world controls**;
actor/critic Adam counts are **9,096 / 32,768**. Final role credit ledgers are
33,112/32,424 (control), 33,273/32,263 (treatment); they include the source
32,768-credit prefix. Final models/configurations, complete reports, per-block
metrics and review are archived under the existing `credit-horizon-1671168`
result and validation directories. No old actor is substituted.

Actual new diagnostic compute is **65,536 PPO credits / 65,536 collection world
controls**, **58,073 evaluation world controls**, and 262,144 integrated replay
presentations. Cumulative diagnostic PPO is **1,646,848**; formal diagnostic
simulation is **770,086 world controls**. Standalone BC remains 8,000 updates;
main boxing **1,048,576**, combined main **476,041,216**, and the foundation's
475M ceiling remain unchanged.

### Next single allocation

The wider fresh active scoring and independently verified warm state justify
**one 32,768-credit diagnostic continuation** from the latest gamma-0.999 final.
This tests whether the new signal survives further ordinary optimization;
it is not permission to continue indefinitely if neutral strikes remain absent.
Inherit version 5 exactly: gamma 0.999, lambda 0.95, replay coefficient 1,
eight one-learner neutral-partner worlds, contact history, normalization,
all optimizer moments/counts, samplers and consumed/role prefixes. Do not reset
the critic or roll back to the selected earlier two-way actor. Physics,
rewards, independent gates, fist geometry and unassisted free legs are unchanged.

The lineage ceiling is **1,736,704 learner credits / 1,114,112 world controls**.
At full allocation actor/critic counts become **9,608 / 33,280**, cumulative
diagnostic PPO **1,679,616**, with 32,768 additional collection controls and
131,072 integrated replay presentations. Keep attempted/discarded work counted.
Check all three full eight-episode screen profiles at **16,384 and 32,768**
new credits, preserving distinct exact checkpoint payloads. If any profile
has at most **3/8 surviving pairs**, stop training early and run the final fresh
check. Source and final fresh seeds are **11,000,003..11,000,006**, both swaps;
these plus the two full screens are at most **96 episodes / 48,000 evaluation
controls**. Retain failed generations. No gate changes, main actor adoption,
phase promotion or additional allocation is automatic; review the complete
result before deciding another step. Two distinct consecutive complete passes
remain required, including role survival and own strikes on multiple seeds.

Hidden persistent launcher: `tools/start_striking_credit_horizon_continue.ps1`;
live state:
`/home/pedro/boxing-runs/striking-credit-horizon-continue-1703936/continuation.json`.
Inspect that state and process before launching anything else. Keep the existing
monitoring schedule. The supervisor and learner use separate locks.


The completed review and continuation guards pass **14 focused tests**, including
rejection of incomplete swaps, stale evaluated hashes and nonfinite trials, and
ensuring a successful selected active exchange cannot mask failed neutral gates.
The existing horizon/state-preservation, optimizer-prefix and neutral-routing
checks also pass. Python compilation, LF shell syntax, PowerShell launcher
parsing and `git diff --check` succeed. The actual hidden process was verified
running its preregistered fresh source evaluation before any new optimization.


## Completed longer-horizon continuation and transfer review (2026-10-23)

The single ordinary gamma-0.999 continuation completed normally at **17:29:17
UTC**, consuming its entire **32,768-credit** allocation. Neither of the two
distinct complete screen generations passes. The complete source/final holdout
and both screens comprise **96 episodes / 42,964 evaluation world controls**.
Checkpoint hashes, finite learned/optimizer leaves, preserved counters, 32 finite
update records and all four complete foundation histories were reverified by
`tools/review_striking_horizon_continuation.py`. Native error logs are empty.

| Policy / sample | Neutral 0 survival / own events | Neutral 1 survival / own events | Active survival | Surviving active red / blue events | Surviving two-way episodes |
| --- | --- | --- | --- | --- | --- |
| Source / fresh 11M | 7/8 / 3 | 7/8 / 0 | 7/8 | 0 / 5 | 0 |
| +16,384 / screen | 5/8 / 1 | 7/8 / 0 | 6/8 | 6 / 7 | 3 |
| +32,768 / screen | 6/8 / 1 | 8/8 / 2 | 5/8 | 1 / 5 | 1 |
| +32,768 / fresh 11M | 7/8 / 1 | 7/8 / 5 | 6/8 | 0 / 1 | 0 |

The screen's three intermediate two-way episodes do not override its failed
survival/neutral criteria. The final neutral-role-1 screen alone meets its
profile threshold: 8/8 pair survival and own surviving strikes on two seeds.
Its fresh own strikes also span two seeds, but fresh survival is 7/8 rather
than the required 8/8. Neutral role 0 still scores on only one seed. On exactly
the same fresh active seeds (11,000,003..11,000,006, both swaps), pair survival
decreases 7/8 to 6/8 and surviving events decrease five to one. This is mixed
transfer with a measured active regression, not a complete phase pass or a
reason to keep extending unchanged neutral-only optimization. Neither selected
intermediate episodes nor older four-seed results replace the current final.

Both exact models/configurations and all reports/metrics are archived under
`credit-horizon-continue-1703936` in the result and checkpoint-validation trees.
Intermediate hash:
`f4d8daf3260280b6046dc5dddedc44b1edb153a0cc7c44c8058223801c9320cc`.
Final hash:
`55976fc15b74b9997b4cb56cf13683f1f87315ac6d67c5cc966ab3c644dd2d73`.
The final lineage is **1,736,704 learner credits / 1,114,112 world controls**,
Adam **9,608 / 33,280**, role credits **50,229 / 48,075**, including the inherited
neutral prefix. Nothing is rolled back or promoted. Current diagnostic PPO is
**1,679,616**, formal diagnostic simulation **813,050 world controls**; this job
also adds 32,768 collection controls and 131,072 integrated replay presentations.
Standalone BC stays 8,000 updates; main boxing and combined main remain
**1,048,576 / 476,041,216**. Foundation certifications and ceilings are unchanged.

### Bounded passive investigation, no training extension

Before choosing a training change, replay **all eight fresh active episodes and
all eight neutral-role-1 episodes for both source and final**. This fixed
32-episode / at most **16,000 world-control** audit preserves every failed case,
same seeds/swaps, original deterministic mean policies and inherited physical
profiles. It uses no optimizer updates, teacher, action intervention, assistance
or gate change. Check all original physical fields for exact equality, hashes,
finite trace arrays and native numerical/capacity logs. Trace actual qpos/qvel,
actions, commands, root/hand distances, pelvis/upright state, legal impacts,
first positive-force interbody contacts and terminal flags. Compare neutral
improvement with active failure onset; contact-before-fall is a temporal
association and does not establish causality. This can inform a subsequent
coverage or control hypothesis, but does not authorize a new training budget.

Persistent launcher: `tools/start_striking_horizon_regression_trace.ps1`.
Live state:
`/home/pedro/boxing-runs/striking-horizon-regression-trace-1736704/trace-review.json`.
Inspect it, its child, progress and all complete replay checks before any further
launch. Record all attempted simulation, including on failure. At the maximum
allocation formal diagnostic simulation would reach **829,050**, while diagnostic
PPO stays **1,679,616**. Preserve the latest final, all consumed work, unassisted
free legs and existing monitoring schedule. The eventual full trace review,
not an isolated frame or reward number, determines the next concrete decision.


Validation for this review/audit: **12 focused tests passed** (existing passive
trace/reward invariants, complete gate evidence guards and five new contact-timing
summary checks). Python compilation, LF shell syntax, PowerShell launcher parsing
and `git diff --check` pass. The hidden audit supervisor and its native child were
verified live; the main checkpoint hash remains unchanged. Startup evidence is
under `docs/results/striking-pilot/horizon-regression-trace-1736704/`.
The two source profiles (16 episodes) have already reproduced every original
physical field exactly, with finite hashed trace arrays and empty native error
logs. Final-policy replay checks remain pending at this snapshot. Both source
falls follow interbody contact: early red low-pelvis termination in active play,
and late blue self-leg contact against a neutral partner. This chronology alone
is not evidence that contact caused either fall; compare the complete final
traces before proposing a correction.


## Completed transfer traces and controller ablation (2026-10-23)

All **32 passive matched episodes reproduced every original physical field
exactly**. Trace hashes and all stored arrays were checked, with no native
numerical/capacity errors. Actual audit simulation is **15,050 world controls**,
zero optimization. Formal diagnostic simulation is now **828,100 controls**;
diagnostic PPO stays **1,679,616**. Main counters, latest final checkpoint,
foundation histories and consumed prefixes remain unchanged. Complete evaluations,
raw traces, exact replay state and `analysis.json` are archived under
`docs/results/striking-pilot/horizon-regression-trace-1736704/`.

All five observed failures occur after interbody contact. In active play, the
source's one and final's two red failures terminate at pelvis height below
0.55 m, without tilt/self-leg flags. The shared early case (seed 11,000,003,
swap true) falls at 1.580 s (source) / 1.810 s (final), after first contacts at
1.460 / 1.498 s. The additional final active failure (11,000,006, swap false)
falls at 8.406 s, first contact 1.568 s. Against the neutral partner, both
source and final blue fail through self-leg contact on seed 11,000,006, swap
true (9.338 / 9.826 s), without low-pelvis/tilt flags at the terminal instant.
The latest final still has more neutral blue strikes and worse active play.
These temporal observations do not identify whether high-level movement,
arm loading, incoming contact or coupled feedback caused the failures.

### One fixed closed-loop channel experiment

Use only the latest final hash
`55976fc15b74b9997b4cb56cf13683f1f87315ac6d67c5cc966ab3c644dd2d73`,
without optimizer updates or policy rollback. Run two separate evaluator
interventions already supported and tested: zero the three high-level command
offsets, or zero the fourteen high-level arm residuals. All other action
channels retain learned authority; the frozen motor still controls free legs,
roots and arms through native contacts. No root forces, constraints, teacher,
reward, scorecard or gate changes. The unchanged baseline is the exact archived
latest-final physical evaluation, not a newly selected favorable policy.

For each intervention run active seeds **11,000,003 and 11,000,006**, both swaps,
and neutral-role-1 seed **11,000,006**, both swaps: **12 episodes / at most 6,000
world controls total**, zero learning. Preserve every selected success/failure,
trace actual state/actions and terminal flags, verify hashes/finite state and
native logs, and compare each seed/swap to its original final-policy baseline.
These seeds were chosen to include every latest-final failure in the traced
profiles, with their opposite swaps retained; this is a targeted mechanism
assay, not an unbiased estimate of general boxing performance. Interventions
alter later feedback/state trajectories and are not gate-eligible policies.
Even a standing zero-arm controller cannot be adopted as a trained boxer.
Review both interventions before selecting a training change; no additional
optimizer allocation follows automatically.

Hidden persistent launcher: `tools/start_striking_horizon_control_ablation.ps1`.
Live state:
`/home/pedro/boxing-runs/striking-horizon-control-ablation-1736704/ablation.json`.
Check that job/process and attempted progress before any other launch. At the
full cap formal diagnostic simulation would be **834,100**, diagnostic PPO
remains **1,679,616**, main boxing **1,048,576**, combined main **476,041,216**;
all standalone and integrated supervised compute is unchanged. Preserve the
existing monitor. Existing channel-isolation and read-only physics tests are
included in the 12 passing checks; Python/shell/PowerShell syntax checks pass.


## Completed controller ablations and full holdout (2026-10-23)

Both preregistered interventions completed all selected cases: **12 episodes /
6,000 world controls**, finite hashed trace arrays, empty native error logs,
zero optimizer updates. All selected failures are rescued: **6/6 surviving
pairs per intervention versus 3/6 in the unchanged selected baseline**. Removing
command offsets retains one active red event and one neutral blue event;
removing arm residuals produces no events. The command intervention rescues the
early and late active red low-pelvis cases and the neutral blue self-leg case,
while leaving high-level arm residuals learned and active. This is a controlled
closed-loop effect on these selected cases, not a unique causal explanation or
proof of reliable boxing. The neutral seed's four baseline surviving blue
strikes become one; stability and scoring must both be assessed on full sets.

Raw reports, all twelve traces, per-case baseline comparisons, completed state
and review are archived under
`docs/results/striking-pilot/horizon-control-ablation-1736704/`.
Formal diagnostic simulation is **834,100 world controls**; diagnostic PPO is
still **1,679,616**, with no new learned state, replay presentations or consumed
training credits. The latest final and all main/foundation accounting remain
unchanged. No actor is adopted and no phase passes.

### Full diagnostic contract, zero optimization

The selected-set result justifies a single broader read-only assay using only
latest final SHA
`55976fc15b74b9997b4cb56cf13683f1f87315ac6d67c5cc966ab3c644dd2d73`.
Evaluate the **unmodified final on new seeds 12,000,003..12,000,006**, both swaps,
all three full profiles (neutral 0/1 and active). Then evaluate the **zero-command
offset intervention** on original screen seeds 7/23/31/47 and on those same new
12M seeds, all profiles/swaps. The unmodified original screen is already fully
archived for this exact final hash. This is **72 complete episodes / at most
36,000 world controls**, no training. Preserve all full reports and attempted
counts, reject native numerical/capacity errors and incomplete/incorrect-hash
reports, and apply the existing complete survival/multiseed-strike criteria
without changing thresholds. Compare against the matched unmodified fresh
baseline, not against hand-picked old clips or different seed pools.

This assay is explicitly ineligible for promotion or actor adoption. A masked
controller is an intervention; two seed sets on one unchanged checkpoint are
not two distinct passing training generations. Any future training integration
requires a separately reviewed explicit profile change preserving checkpoint,
optimizer and consumed-step provenance. No such allocation is granted now.
Unassisted roots/legs, learned motor feedback, fist geometry, reward, scorecard
and independent physical criteria are unchanged. Inspect both complete outcomes
before deciding whether to implement a correction or reject the intervention.

Hidden persistent launcher: `tools/start_striking_zero_command_holdout.ps1`.
Live state:
`/home/pedro/boxing-runs/striking-zero-command-holdout-1736704/holdout.json`.
Read its state/process and attempted progress before another launch. At the
full cap formal diagnostic simulation would reach **870,100**, while diagnostic
PPO remains **1,679,616**, main boxing **1,048,576**, combined main **476,041,216**.
Preserve the existing monitor, latest final and all consumed work. The runner
reuses the tested complete-evidence guards; Python/shell/PowerShell syntax checks
pass. The 12 focused checks cover channel isolation and unchanged native physics.


## Completed full holdout and next coverage decision (2026-10-23)

The full assay completed normally at **18:26:01 UTC**, using **35,955 evaluation
world controls** for all 72 new episodes. Complete original-screen baseline
reports already exist for the same latest-final hash. All seed/swap sets, hashes,
finite states, parent full-criteria summaries and native logs were checked;
there are no numerical/capacity errors. Current final learned state/optimizer
prefixes and complete foundation histories (29/41/13/2 records) remain valid.

| Same final policy / set | Neutral 0 survival / own events | Neutral 1 survival / own events | Active survival | Surviving active red / blue events | Surviving two-way episodes |
| --- | --- | --- | --- | --- | --- |
| Unmodified / original screen | 6/8 / 1 | 8/8 / 2 | 5/8 | 1 / 5 | 1 |
| Zero command offsets / original screen | 8/8 / 1 | 8/8 / 2 | 8/8 | 1 / 0 | 0 |
| Unmodified / fresh 12M | 8/8 / 1 | 8/8 / 1 | 8/8 | 2 / 3 | 0 |
| Zero command offsets / fresh 12M | 8/8 / 0 | 8/8 / 0 | 7/8 | 4 / 3 | 2 |

Fresh seeds 12,000,003..12,000,006 include both swaps and the complete ten-second
profiles. Zero commands improves original-screen survival, but its fresh neutral
roles lose all own events and fresh active survival regresses versus the matched
unmodified policy. Two surviving fresh two-way episodes do not override those
failures. **Neither controller passes a complete set. Reject blanket adoption of
the zero-command intervention**; it remains a descriptive diagnostic. No trained
actor, physical authority, score, gate or main ceiling changes.

All new raw reports, progress, completed state and analysis are archived under
`docs/results/striking-pilot/zero-command-holdout-1736704/`. Actual formal diagnostic
simulation is **870,055 world controls**, PPO still **1,679,616**. No optimizer,
standalone supervised or integrated replay updates occurred. Main boxing
1,048,576 and combined main 476,041,216 retain their prior limits.

The earlier active-only and neutral-only runs each lost task performance. The
longer-horizon final improves neutral blue striking but regresses active play;
removing commands alone is not sufficient. This motivates one **matched bounded
coverage experiment**, preserving full learned authority: 50/50 neutral/current-
mean-active opponent worlds against neutral-only control. The explicit version-6
transition retains all learned/optimizer/sampler/consumed state from the latest
final. See the [complete comparison contract](striking-mixed-partner-comparison.md)
for 32,768 credits per arm, retained native integration blocks, independent full
checks, early-stop rules, exact accounting and no automatic adoption/extension.
**32 focused tests pass**, including new task routing, mode/role balance and
provenance guards plus unchanged horizon/neutral/legacy PPO behavior. Python,
LF shell and PowerShell syntax checks pass. Real mixed-path integration is
required before further new-profile optimization and is included in its budget.
