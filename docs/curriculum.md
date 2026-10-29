# Adaptive unassisted G1 curriculum

Maintain the [trajectory and decision record](experiment-history.md) at every
reviewed intervention, including failed results, accounting and pending evidence.

**Current work (2026-10-29):** [GPU recovery](gpu-training-recovery.md) has
implemented a current-model CUDA dynamics primitive, matching batched policy
feature ordering, a measurement reference adapter and bounded diagnostics.
No new PPO allocation. Closed-loop comparison is active at
`/home/pedro/boxing-runs/gpu-closed-loop-v1/state.json`: native 30/20, Warp 3/5,
Warp 30/20, same final v17 and original fresh seeds/roles. Maximum 36,000 new
formal world controls; before 2,271,001, ceiling 2,307,001. Main 476,041,216 and
diagnostic PPO 9,504,000 retained. No experimental physics promotion.
Inspect actual process/child progress and all completed profiles before another
job. After completion, independently review all arms, then run the separately
declared retained Warp6/7 own-engine punch-cycle diagnostic (maximum 7,680 new
controls, formal before equals the actual completed comparison total). Preserve
all limits, failures and expired allowances. Detailed protocol and limitations
are in the GPU recovery document; do not resume an old reward experiment.
The persistent coordinator at
`/home/pedro/boxing-runs/gpu-validation-followthrough/state.json` already owns
that follow-through sequence; inspect its PID/child before launching the legacy
diagnostic independently. It publishes completed owned evidence and stops on
failures. No second physical job should overlap it.

**Latest v17 completed review (2026-10-28):** all 2,097,152 new PPO credits
finished; no complete gate passed. Final fresh pair survival is 5/8, 7/8, 3/8;
both actors struck in one surviving mirror episode. Formal total 2,271,001 and
diagnostic PPO 9,504,000; no active training, adoption or unchanged extension.
The user requested a fair legacy comparison and GPU throughput recovery. One
current-model Warp forward showed a solver iteration warning. Two cold-start
contact poses using fresh GPU data have matching CPU/GPU contact pairs and
similar normal forces; these limited checks do not establish trajectory,
solver or policy parity. An initial transfer test overflowed on uninitialized
native scratch and is retained as superseded evidence. See the
[completed review and GPU plan](results/striking-pilot/arm-effort-v17/README.md).

**Earlier v17 decision (2026-10-28):** matched fresh-seed arm diagnostics yield
8/8 pair survivors with both learned arm outputs zeroed, 7/8 at half output,
and 1/8 unmodified; zero/half conditions contain no valid strikes and cannot
promote. The [v17 protocol](results/striking-pilot/arm-effort-v17/README.md)
tests a smooth training-only arm-effort cost while retaining complete-cycle
incentives and full action/physics authority, starting from the exact v16 final.
At most 2,097,152 new PPO credits (global ceiling 9,504,000) and 60,000 formal
controls (global ceiling 2,298,008), no budget rollover. One integration block,
original gates every 524,288 credits and a new fresh final; require two distinct
complete consecutive full passes. Inspect live state and process before retry:
`/home/pedro/boxing-runs/striking-arm-effort-v17/audit.json`.

**First v17 gate:** independent review finds pair survival 3/8, 6/8, 1/8
versus 5/8, 6/8, 3/8 on the inherited baseline. Self-play has one valid cycle
in one corner, none in the other, and no surviving reciprocal seed. The full
gate fails. This early regression does not isolate causation; continue the
already approved second chunk to 1,048,576 new credits and review its full gate.
Completed-gate totals: formal 2,244,135 and diagnostic PPO 7,931,136.

**Latest v16 completed review (2026-10-28):** final checkpoint 5bbf6a38 has
original pair survival 5/8, 6/8, 3/8 and fresh 3/8, 7/8, 1/8. Fresh neutral
cycles total 42; the one surviving fresh mirror bout has valid cycles for both
agents. Full gates fail and the approved PPO allocation has ended. All seven
fresh mirror failures occur before scoring, three before any inter-body contact;
terminal predicates include low pelvis and self-leg contacts. These observations
do not establish causation. A separately bounded 4,000-control diagnostic masks
only learned arm residuals in both actors, on the same fresh resets, with full
physics and no PPO. It cannot promote a checkpoint. Global formal ceiling is
2,234,238 from 2,230,238; diagnostic PPO remains 7,406,848. Inspect
`/home/pedro/boxing-runs/striking-v16-arm-ablation/state.json` and the
[protocol](results/striking-pilot/v16-arm-ablation/README.md) before any new job.

The arm ablation is completed and independently reviewed: disabling both
learned arm residuals yields 8/8 pair survivors on the same fresh seeds, versus
1/8 unmasked, but zero valid strikes. The earlier count of pre-contact failures
is corrected to three, not four. This identifies a strong interaction without
isolating individual actors or joints. A separately bounded 4,000-control
half-arm diagnostic is now active; current formal total 2,234,238, diagnostic
ceiling 2,238,238. It preserves physical dynamics but cannot promote a policy.
Inspect `/home/pedro/boxing-runs/striking-v16-half-arm/state.json` and the
[protocol](results/striking-pilot/v16-half-arm/README.md) before any new job.

**Earlier v16 decision (2026-10-27):** the completed navigation ablation has
3/8 masked survivors and no valid mirror cycles versus 2/8 unmasked; unchanged
physics and unmodified original gates remain authoritative. Based on v15's
repeated partial-contact incentives, prospectively test one training reward
change: reduce graded partial-contact payment from 2 to 0.25 while retaining
the full-cycle reward, target potential, terminal fall cost, action interface,
and actual v15 final actor/critic/Adam/RNG. Allocate at most 2,097,152 new PPO
credits (global 7,406,848) and 60,000 new physical evaluation controls (global
2,250,984); no rollover. One integration block, performance gates at 524,288
intervals and fresh final; two distinct consecutive complete passes required.
See [the v16 protocol](results/striking-pilot/contact-rebalance-v16/README.md).
Live output: `/home/pedro/boxing-runs/striking-contact-rebalance-v16/audit.json`.
Inspect process/attempt ledger before any retry or change; no automatic adoption.

**First v16 gate:** independent review verifies ten self-play valid cycles,
five per corner, with reciprocal cycles on surviving known seeds 7 and 31.
The motion subchecks pass, but original survival is 5/8, 7/8, 4/8; the full
gate fails and no promotion occurs. All four swapped self-play episodes survive
and all four unswapped fail before scoring; cause and generalization are unknown.
The predeclared next chunk to 1,048,576 new credits remains active. Completed
gate formal total 2,199,891; diagnostic PPO 5,833,984. A selected known-seed
reciprocal video uses 250 additional read-only render forwards, cumulative 2,947.
See [first-gate analysis](results/striking-pilot/contact-rebalance-v16/README.md).

**Second/third v16 gates:** independently verified self-play survivors are 3/8
then 5/8; each gate has reciprocal valid cycles on only one surviving known seed.
Latest neutral survival is 4/8 and 5/8, and three of four unswapped self-play
episodes fail before scoring. Both complete gates fail. Continue the already
approved final chunk and original/fresh checks; no rollback or changed criteria.
At the third completed gate formal total is 2,215,430, diagnostic PPO 6,882,560.

**Latest completed review (2026-10-26):** v15 exhausted its 2,097,152-credit
allocation with exact final 9132ab4c. Original/fresh neutral valid cycles are
13/23; mirror cycles 2/0. Original survival 5/5/1 and fresh 6/5/2 out of eight
per profile fail full gates. No training remains active, no promotion or rollback.
Global diagnostic PPO 5,309,696 and formal controls 2,183,338. Retain the actual
final optimizer; do not extend unchanged PPO. A separately declared diagnostic
of at most 8,000 formal controls compares final v15 against its exact frozen
training opponent in both roles on the existing fresh seeds, reusing the saved
mirror comparison. Formal ceiling 2,191,338; no PPO, reward or physics changes.
Read [the diagnostic protocol](results/striking-pilot/v15-opponent-gap/README.md)
and inspect its live state before launching anything else:
`/home/pedro/boxing-runs/striking-v15-opponent-gap/state.json`.

That diagnostic is now completed/reviewed: 3/8 and 4/8 pair survivors, with
zero/one learner valid cycles; the learner also falls against its actual frozen
training opponent. Formal total 2,188,806. Opponent identity alone does not resolve
the deficit. The next bounded action-interface diagnostic masks only learned
navigation offsets in both current actors, preserving motor/arms/physics, on
the same fresh set. At most 4,000 new formal controls; ceiling 2,192,806, zero PPO.
Masked diagnostics are never promotable. Active output:
`/home/pedro/boxing-runs/striking-v15-navigation-ablation/state.json`.
Read [the prospective protocol](results/striking-pilot/v15-navigation-ablation/README.md)
before intervening or launching another job.

**Earlier v15 decision (2026-10-26):** the completed v14 block produced no valid
punch cycles. The [saved reward/exploration audit](results/striking-pilot/contact-progress-v15/proposal-audit.json)
finds higher paid preparation/hand speed but larger target-center distances and
fewer legal contacts; no demonstrated saved-covariance collapse. This justifies
[prospective target-coupled contact-progress PPO v15](striking-contact-progress-training.md),
not a success claim or weakened gates. Preserve actual latest be59681d and all
Adam/RNG/norm; replace air-preparation payment with bounded target potential and
explicit graded same-hand physical contact subgoals. Full-cycle criteria unchanged.
New ceiling: at most 2,097,152 learner credits (global PPO 5,309,696) and 60,000
new evaluation controls (global formal 2,200,974); historical v14 baseline reused
with its actual source hash and zero new evaluation debit. No main adoption or
new geometry/BC/render allocation. Live job:
`/home/pedro/boxing-runs/striking-contact-progress-v15/audit.json` and
`training/progress.json`; inspect actual processes before launch or recovery.
At 404,480 new v15 credits the training learner complete-cycle counter first
changes 0 -> 1 (corner 0). This is a counter-only training signal, with no retained
raw event window or held-out confirmation; no promotion or success claim. Preserve
it and adjacent policies in first-training-counter.json. Passive evidence capture
will apply at the next ordinary training-process handoff, without restarting the
productive job or changing rewards/physics. The first full evaluation at 524,288 is now complete: 7/8, 7/8 and 5/8
survivors, 240 legacy contacts but zero full valid cycles. Both gates fail.
Independently checked raw stochastic training windows now establish 37 cycles
meeting the existing references, in both corners (four active-opponent cases);
these are not deterministic held-out passes. See first-captured-event-review.json
and the explicitly scoped normal/slow saved-state video. Keep the productive
approved continuation and all criteria unchanged. Formal evaluation is 2,150,902;
renderer forwards 2,197 after two read-only 66-frame renders. No new training or
physical evaluation allocation, interruption, source rollback or promotion.

**Next full gate, 1,048,576 new v15 credits:** independent saved-data review of
b2a2c04c verifies six deterministic valid cycles (neutral 0: four; neutral 1: two),
on surviving seeds 7 and 23 in both corners. These neutral motion subchecks pass,
but original survival remains 7/8 and 7/8, and mirror survival regresses to 0/8
with no scored contact before termination. No full pass, streak or promotion.
Formal evaluation now totals 2,159,537 (18,563 new in v15); main remains
476,041,216. Continue the already approved next chunk to 1,572,864 new credits,
with no intervention, budget extension or numerical error. See
[the independent review](results/striking-pilot/contact-progress-v15/step-1048576-independent-review.json).
This is limited deterministic striking evidence, not stable reciprocal boxing.

**Third gate, 1,572,864 new credits:** saved-data review verifies checkpoint
39f9e3d3. Neutral complete cycles fall from six to two (one per role, each on
only one surviving seed); mirror has zero. Pair survival is 6/8, 7/8, 2/8.
More scored contacts/speed/pretravel coexist with fewer qualifying returns.
No complete pass or promotion. Continue only the predeclared final chunk and
full/fresh checks at 2,097,152; no rollback, gate change or budget extension.
Formal controls total 2,168,560; diagnostic PPO at the completed gate 4,785,408.
See [third-gate review](results/striking-pilot/contact-progress-v15/step-1572864-independent-review.json).


**Earlier reviewed outcome (2026-10-24):** the
[broader frozen audit](striking-gae-generalization.md) completed 130 episodes /
61,934 controls with no optimization. All new neutral profiles pass (31/32 and
32/32 pair survivors), but mirror survival 23/32 fails three of four complete
reset groups. Three of four cross-policy diagnostics pass; training-snapshot
corner 1 fails at 5/8. General boxing and main adoption remain unapproved.
The earlier two distinct original complete passes plus fresh 29M pass remain a
[limited readiness milestone](results/striking-pilot/gae-readiness-certification/README.md),
not broader certification. Its source milestone is retained; after the new
control's pass/fail screens, version-12 history has ten records and current streak
zero. Version 13 has two failed screens and streak zero.

The supervisor's completed-renderer process notice was independently reviewed:
replay fields match, all 250 MP4 frames decode, original failed state and warning
are preserved. Bookkeeping recovery repeated zero physical controls. No native
physics failure or policy corruption was found; unknown warnings remain rejected.

**Current work:** the [pool-versus-fixed comparison](results/striking-pilot/pool-2064384/README.md)
is completed and independently reviewed: all 32,768 PPO credits, 168 physical
episodes and 78,563 evaluation controls are accounted for. Pool fresh survival
is 8/6/5 versus control 8/6/6 and source 8/6/7. Both final complete gates fail;
there is no demonstrated pool benefit. Freeze both actual finals, without rollback,
adoption, promotion or unchanged PPO extension. No training job is active.

The user correctly questioned whether the videos show punches rather than touches.
The [visual and objective audit](striking-punch-quality-audit.md) finds that the
current low-speed contact criteria do not certify punch mechanics; richer older
motion checks were not fully retained. Seven scorecard tests pass, including a
synthetic low-speed qualifying contact example. The subsequent
[bounded passive replay audit](results/striking-pilot/punch-quality-2080768/README.md)
is completed and independently reviewed: four archived policies, eight known
episodes, 4,000 diagnostic controls and zero PPO updates. Every original trial
field reproduces exactly; fifteen focused tests pass. None of the 19 selected
legacy events simultaneously meets the old speed/forward-cone references; video
source hand excursion is only 5.3–9.1 cm in the preceding 250 ms. These descriptive
references are not universal punch gates. The separate version-1 motion evidence
is now implemented, with per-hand component checks and censored recovery handling.
The [complete retained-demo audit](results/striking-pilot/demo-motion-2080768/README.md)
then exactly reproduced all fifteen trajectories / 7,500 controls. All fifteen
learner contacts fall below both inherited motion references. Twenty-one focused
tests pass. The old teacher stops correcting after its first contact; no validated
positive full strike-cycle target was established. Do not extend unchanged
retention/BC/PPO or restore an earlier actor.

**Latest physical diagnostic:** [jab-cycle feasibility v1](results/striking-pilot/jab-cycle-v1/README.md)
completed and is independently reviewed. All sixteen planned conditions fell;
zero complete candidate episodes passed the unchanged prospective criterion.
The job consumed 4,048 of its at most 8,000 controls; no learner updates.
Nine focused controller/motion tests pass, but the scripted positive-target
hypothesis failed. Reject v1 without imitation, threshold changes or PPO extension.
The [exact full-body replay and workspace review](results/striking-pilot/jab-failure-workspace-v1/README.md)
then reproduced every original native array in another 4,048 controls. Only the
acting body triggered first-fall predicates (eleven low pelvis, one tilt, four
self-leg contacts). Guard endpoints match in cloned static searches; early full
strike endpoints do not match in the two local starts. This is not a global or
dynamic impossibility proof, nor a test of the actual interpolated target at
every instant. Arm action clipping is 19.6%; causal isolation is still pending.
Eight focused tests passed that audit. The subsequent
[matched motor-only versus guard-only comparison](results/striking-pilot/guard-control-v1/README.md)
is completed and independently reviewed: 8/8 motor-only survivors versus 0/8
static-guard survivors. Four unique motor-only resets are duplicated by corner
labels. Only the guard-acting body falls (four low pelvis, four self-leg contacts);
six cases fall without scored contact. The full cycle is unnecessary to reproduce
these failures; the static guard is rejected. Nine focused tests passed.
The separately declared [two-second guard-entry probe](results/striking-pilot/guard-ramp-v1/README.md)
is completed and independently reviewed: 0/8 survivors again, despite median
hand tracking error improving from 5.0 to 0.99 cm. All cases reach the end of the
command ramp before falling. Ten focused tests pass. A final assertion inherited
the wrong episode count; its failed state/traceback are preserved and completion
was recovered after checking all saved trials, with zero additional physics.
Reject entry smoothing as a sufficient repair. Source/array review then confirms
the foundation already carries a learned compact guard; the failed script asks
for an additional 10.3–11.4 cm of hand displacement. The separately declared
[motor-guard cycle v2](results/striking-pilot/motor-guard-cycle-v2/README.md)
is now completed and independently reviewed: preserve that learned guard between
extensions and return through the motor/filter. All eight pairs survive, but
zero physical impacts, left-fist contact substeps or candidate punches occur.
Twelve focused tests pass. Retain it only as a stable scripted control; the
positive punch target is still missing. Next review extension timing/target lag
and prospectively test a bounded terminal-target dwell before further training.
The [terminal-target dwell comparison](striking-terminal-dwell.md) is completed
and independently reviewed: all eight pairs survive, but zero impacts/candidate
punches again. Thirteen focused tests pass and every original outbound prefix
matches exactly through the first changed dwell. Reject 120 ms dwell as a
sufficient repair. The separately declared
[cloned terminal-workspace audit](results/striking-pilot/terminal-workspace-v1/README.md)
is completed and independently reviewed: 24 snapshots, actual median endpoint
error 34.9 cm versus best local static median 22.9 cm. Two starts do not prove
global impossibility; endpoint error is not a collision-surface gap. Next inspect
legal fist/target surface distances and blocking contacts at the recorded/best
poses before selecting a bounded target/authority change. The subsequent
[signed-surface audit](striking-surface-audit.md) is independently reviewed:
recorded legal gaps are positive in all 24 poses, while 20 overlap blocking
surfaces. Of six static-best poses overlapping a legal surface, five also overlap
blockers. This does not prove an entire dynamic path is blocked. A
[6/12 cm lateral-bow comparison](results/striking-pilot/arc-path-v4/README.md)
is independently reviewed after sixteen focused tests passed. Survival is 7/8
and 8/8; neither arm produces any candidate punch or acting-left target contact.
The single old-counter impact is an incidental low-speed right-hand touch.
Reject both as positive targets. Next test the existing lateral command through
the walking motor under a new prospective bounded protocol, not new root/body
authority. The completed lateral-command diagnostic finds no candidate punch; no PPO, imitation, motor unfreezing or gate changes.

Reviewed diagnostic PPO is **3,212,544**, at its completed ceiling after v14 adds
1,048,576 learner credits. Formal diagnostic evaluation is **2,140,974** after
67,172 new controls, under **2,145,802**; unused **4,828 expire**. V14 collection
world controls **1,048,576** are recorded separately from these evaluations.
Final saved-state video adds 1,565 render forwards (cumulative **2,065**), with zero new physical or optimizer transitions.
The historical v14 allocation is closed; the separate prospective v15 allocation above applies. Completed v14 live state is
`/home/pedro/boxing-runs/striking-motion-cycle-v14/audit.json` (completed).
Cloned kinematic evaluations are now **6,349** after 48 producer + 48 independent
surface-audit forwards. Signed-distance queries are separately **960** (480 + 480).
Both new geometry allocations are exhausted; no new cloned/distance allocation.
Unused historical compute expires at review. The declared v14 PPO/evaluation allocation is separate; no new supervised, cloned or distance-query allocation.
Unused historical comparison compute is not an automatic continuation allocation.
The completed audit is `/home/pedro/boxing-runs/striking-punch-quality-2080768/audit.json`;
the later demo-motion review accounts separately for its 7,500 additional controls.
Main boxing **1,048,576**, foundation **474,992,640**, combined main **476,041,216**,
BC 8,000 remain unchanged. Completed live audit and explicit recovery are at
`/home/pedro/boxing-runs/striking-gae-generalization-2064384/audit.json`.
The completed job is `/home/pedro/boxing-runs/striking-pool-2064384/comparison.json`.
No duplicate jobs, blind retries, weakened gates, main adoption or unlimited allocation.

**Current stage (2026-10-22):** all four foundation levels are certified by
[two distinct consecutive complete physical passes](results/curriculum/foundation-certification/README.md).
The foundation finished normally at **474,992,640 consumed main transitions**
under the **475M** ceiling. Full walking passed at 474,562,560; approach then
added 430,080 transitions. The exact actor payload was retained at transition.

**Current operation:** the [opponent-aware striking pilot](striking-pilot.md)
adds trainable arm residuals and independent fist-impact events over the unchanged
certified motor actor. Main boxing PPO remains stopped at **1,048,576 learner
transitions**, without a budget extension or actor adoption. The matched
whole-body authority comparison completed; both controllers had 7/8 paired
survival and zero deterministic impacts. The separate eight-episode saved-Gaussian
checks also found zero impacts. The subsequent
[completed 256-episode holdout and causal contact audit](striking-body-holdout-review.md)
found one reproducible stable two-way exchange, but no robust benefit from body
authority. The first [neutral-partner boxing pilot](striking-partner-curriculum.md)
completed 131,072 diagnostic transitions, with unchanged 6/8 and 8/8 role survival
and no final learner impacts. The [duration continuation](striking-partner-duration-review.md)
stopped after **262,144** additional diagnostic transitions on two stagnant full
checks. No phase passed and the unused allocation is not automatically resumed.
The [completed read-only exploration audit](striking-exploration-review.md) rejected
all four noise profiles: none produced stable own strikes; held/scaled noise
caused 16/16 pair falls. The next [target-directed feasibility diagnostic](striking-directed-feasibility.md)
found four reproducible directed-strike trajectories completing ten seconds, but
poor overall controller survival; it is not a learned policy or a phase pass.
Its fixed variant completed a sixteen-seed holdout with fifteen exact-replayed
ten-second strike trajectories and 49 failures. The completed
[supervised initialization and next comparison](striking-demonstration-initialization.md)
used fifteen demonstrations and 2,000 actor updates. The network learned isolated
stable strikes against a neutral partner, but neither checkpoint passed and
active self-play regressed to 4/8 at 2,000 updates.
The [matched contact-history comparison](striking-contact-history-comparison.md)
completed all 4,000 additional supervised updates and 48 full physical trials.
Neither arm passes: contact history has 5/8 active pair survival, versus 3/8 for
control, and no surviving blue-role own events. No actor is adopted and blind
imitation remains stopped. Inspect
`/home/pedro/boxing-runs/striking-contact-compare-bc2000/comparison.json` and
processes before another launch. The [coverage and channel audits](striking-policy-coverage-audit.md)
are now also completed. Passive traces exactly reproduce all 24 original trials;
selected early falls disappear when high-level arm residuals are removed, with
all learned motor actions and free-body physics retained. These short interventions
are not adopted policies or gates. The [fresh pre-contact review and anchoring
comparison](striking-motor-anchoring.md) verified 32 unique candidate prefixes
and completed a bounded matched BC correction: 1,000 updates per arm and all
120 complete physical episodes. The correction improves some survival but
reduces scoring; neither actor passes the complete criteria. Further identical
imitation is stopped. The [completed PPO comparison and regression review](striking-ppo-anchor-review.md)
verified both fixed 131,072-credit arms and all 120 full physical episodes. Neither
actor passes: both lose surviving neutral strikes and auxiliary anchoring also
regresses survival. No PPO extension or actor adoption is granted. Diagnostic PPO
is now **1,450,240**, formal evaluation controls **512,121** before the next audit,
with **131,072** new PPO collection world controls recorded separately. Cumulative
supervised compute remains 8,000 updates. The completed saved-Gaussian audit
(72 trials) also found no recovery of stable strikes. The subsequent read-only
gradient assay used 1,536 world controls and zero updates. Formal diagnostic
simulation now totals **547,894 world controls**, including that assay.
The [completed demonstration-retention comparison](striking-retention-review.md)
finished both 32,768-credit arms and all 120 full physical trials. Retention
restores legal learned mean-policy strikes, including one fresh ten-second
three-red/two-blue exchange with both robots standing. Neutral-role-1 fresh
survival regresses to 4/8, so neither actor passes the unchanged complete criteria.
No phase is promoted or actor adopted. The selected learned three/two exchange
and all matched stability traces reproduced exactly; four role-1 failures occur
after contact (two low-pelvis, two self-leg). Cumulative diagnostic PPO remains
**1,515,776**, formal diagnostic simulation **611,668 world controls** after the
8,091-control trace audit. The [completed active continuation and next curriculum task](striking-neutral-retention-curriculum.md)
reviewed both further distinct complete generations: neither passes, and no
screen neutral learner scores. Active fresh survival rises to 7/8 with another
surviving two-way episode, but neutral-role-1 fresh survival is 6/8. Stop the
active-only extension. Cumulative diagnostic PPO is now **1,548,544**, formal
diagnostic simulation **656,030 world controls**. The [completed neutral-retention review and credit-horizon comparison](striking-reward-credit-review.md)
verified both further full generations and role ledgers. Both screens fail;
final original/fresh profiles have zero stable events despite 7/8, 8/8, 7/8
survival. No unchanged neutral-training extension is granted. Cumulative
diagnostic PPO remains **1,581,312**, formal diagnostic simulation is now
**712,013 world controls** after the 12,222-control read-only reward audit.
All 26 original physical trials and actual reward reconstruction match exactly.
A bounded matched **gamma 0.99 versus 0.999** comparison is active from the
latest neutral actor, with 32,768 new learner credits per arm (65,536 maximum),
unchanged rewards, replay, one-learner task, physics and full physical criteria.
The isolated variable is temporal discount; critic state and all learned prefixes
are retained. Inspect
`/home/pedro/boxing-runs/striking-credit-horizon-1671168/comparison.json`, its
active child/progress, hashes, raw full evaluations and native logs before new
work. The reward measurement is completed. No rollback, actor adoption, phase
promotion, main ceiling change or automatic further extension is authorized.
Main counters and all physical gates remain unchanged. No actor is adopted.
No gate is relaxed, stage promoted or candidate composite controller certified.

Full walking resumed at 422.89152M main transitions after a
[supervisor log-read fault](curriculum-log-read-recovery.md). The targeted bounded
retry preserves warning inspection and the latest policy; no training or gate
setting changed during that recovery.

**Earlier recovery:** the original shared allocation stopped at
197.30M without repeated gate passes. The [recovery investigation](curriculum-recovery.md)
selects the preserved 181.58M actor, retains all 197.30M consumed transitions,
enables full randomized resets and training disturbances, and reduces PPO's
learning rate. That recovery launched the earlier **400M total** allocation.
The promotion protocol below remains unchanged; the 200M references describe
the initial/default allocation, not an additional per-level requirement.

The [walking tracking comparison](curriculum-tracking-experiment.md) documents
the subsequent directional plateau and a bounded training-only reward experiment.
The [reviewed result](results/curriculum/tracking-comparison-112066560/analysis.md)
supports easy-walking training with tracking width 0.01. The original main policy
resumed at 319.8976M consumed transitions without a diagnostic actor substitution.
At that point the physical benchmark and then-current 400M ceiling were unchanged.

After the lateral displacement plateau persisted, the
[lateral-practice investigation](curriculum-lateral-practice.md) added targeted
sideways commands. That continuation started at 360.36608M consumed transitions
with the same actor and then-current 400M ceiling. The later reviewed 475M ceiling
above is now active.

## Current protocol

The user approved replacing fixed per-stage budgets with performance-based
promotion and gradually mixed standing/walking practice on 2026-10-21.
**200M transitions is now an initial shared ceiling across the foundation,
including the 66,232,320 transitions already trained. It is not a minimum or a
200M allocation for every stage.** It may finish earlier when skills pass their
gates. A budget is not evidence that a skill has been learned.

One full-body actor controls all 29 joints, including the 12 leg joints. Each
level loads the preceding actor, critic and observation statistics after checking
the same model/action/observation contract. No random policy replaces a trained
one. This is a single policy learning progressively harder tasks, rather than
independent agents relearning each task from scratch.

There are no pelvis springs, root welds, gravity compensation, root position
controllers or gait playback. Position servos produce joint torques at 50 Hz;
ten 2 ms MuJoCo steps resolve each action. The policy uses 103 actor inputs and
the critic 216 privileged inputs. The compact boxing guard and all joint actions
remain available throughout. Normalization has variance epsilon 0.01 so new
nonzero commands cannot explode previously constant input dimensions.

## Mixed practice and increasing difficulty

| Level | Training command distribution | Complete physical gate |
| --- | --- | --- |
| Balance | 70% zero commands; otherwise very gentle forward/backward, lateral and yaw requests (x -0.03 to +0.05 m/s, y +/-0.02 m/s, yaw +/-0.05 rad/s) | At least 95% survive 10 s, linear error <=0.15 m/s and yaw error <=0.25 rad/s; repeat with a horizontal impulse at 4 s |
| Walk easy | 40% zero; 30% pure lateral; 30% mixed commands at 25% of full walking ranges | At least 90% survive every direction, linear error <=0.20 m/s, yaw error <=0.25 rad/s and actual directed displacement >=45% of requested speed x10 s |
| Walk | 40% zero commands; x -0.35 to +0.55 m/s, y +/-0.25 m/s, yaw +/-0.5 rad/s | Same physical criteria at full evaluation commands |
| Approach | Navigate toward a fixed marker initially 1.8-2.2 m ahead, stopping around 0.75 m away | At least 90% survive and finish within 0.95 m, with walking tracking-error limits |

Standing practice is retained when walking starts. The air-time reward is zero
for zero commands, so standing trials do not reward unnecessary hopping. The
balance level begins learning small movement corrections before its standing
gate passes. Reward changes and command ranges are recorded in each config.

Evaluation uses seeds **7 and 23**, with **32 reset states per trial**. Walking
tests standing, +0.40/-0.25 m/s forward/backward, +/-0.20 m/s lateral and
+/-0.40 rad/s yaw; easy walking tests one quarter of those commands. Balance
tests both undisturbed trials and a [0.2, -0.2] m/s horizontal velocity impulse
at four seconds. That impulse is a disturbance, never a stabilizing controller.

Every trial ends at its first fall: pelvis below 0.55 m, torso up component
below 0.60, the upstream leg-contact termination, or non-finite state. Failed
trials never reset into the survival count. Tracking errors exclude the first
second of settling; survival includes it. Distances come from simulated qpos,
and a fallen trial contributes zero to the required directed displacement.
The evaluation gate checks every seed, disturbance and requested direction.

## Adaptive advancement and budget

The learner performs a full gate before optimization and approximately every
**5M new transitions**, rounded to checkpoint boundaries. The optimizer stays
resident while a small evaluation subprocess runs. Failed gates reset the
passing streak and training continues; they do not restart PPO or stop it for
a short plateau. No training reward threshold advances a level.

**Two consecutive complete passes on distinct checkpoints are required.** A
preview, fewer than 32 worlds, a missing direction, wrong stage, stale checkpoint
hash or identical policy cannot certify promotion. The supervisor independently
checks the gate history and latest checkpoint before starting the next level.
Promotion ends the current PPO invocation immediately; it does not wait for a
fixed 200M horizon. The next level fine-tunes the same networks and normalizer
with fresh Adam and RNG, as explicitly recorded in its config.

All levels share `--max-total-steps` (default 200M). Earlier foundation work is
included once; the old assisted experiment is separate and is not added. The
learner rounds its requested remaining ceiling DOWN to complete PPO epochs, so
it cannot overshoot the shared ceiling. A small remainder can be unused.
Without two passing gates, exhausting that allocation ends in `budget_reached`,
with no automatic promotion, endless retry or random reset. An explicit later
extension can use `-MaxTotalSteps <larger-total> -WarmStart`.

This protocol does not assert that any particular level needs 200M steps or
that the foundation will succeed within that initial total.

## Preserved training and recovery semantics

The preceding fixed-budget balance run stopped at a durable checkpoint of
**66,232,320** transitions. An immutable pre-migration snapshot and original
config are in `checkpoints/curriculum/archive-fixed-budget-66232320/`. Its last
small preview is not a promotion certificate. Adaptive-v2 initially warm-started
that policy with gentle mixed commands. Recovery-v3 now restores the preserved
181.58M candidate at the 197.30M consumed offset, as described in the recovery report.

**Brax 0.14.2's policy callback does not expose Adam or simulator/RNG state.**
Snapshots include the actor, critic, normalizer and exact contract, but are
marked `optimizer_saved: false`. This one-time migration therefore initializes
Adam and RNG again; it is not a full-state continuation. The same limitation
applies after shutdown or interruption and at a level change. It does not apply
to periodic evaluations within one live PPO invocation: Adam remains resident.

The earlier assisted boxing run stopped at 116,633,600 observed transitions,
with its last durable checkpoint at 115,712,000. That independent experiment
and its real contact-scoring results remain archived in
`checkpoints/skills_200m/stopped-step-115712000/` and `docs/skills-200m.md`.
Its six-action policy is not used as a replacement for learned leg control.

## Operation and published artifacts

The isolated Linux/WSL GPU runtime is `$HOME/venvs/boxing_curriculum`:

```bash
bash tools/setup_curriculum.sh
```

Setup uses uv (also detected in the existing project venv) or Python 3.12 with
venv/pip. It installs `requirements-curriculum.txt` and fetches only the pinned
G1 asset directory. `BOXER_CURRICULUM_MENAGERIE` chooses another asset cache;
the environment rejects a different revision. The default is 1,024 worlds on
the 6 GB laptop, with `JAX_PLATFORMS=cuda,cpu`, preallocation disabled and EGL
rendering. Existing assisted-training environments are not modified.

From PowerShell in the repository, start a persistent hidden WSL host:

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_curriculum.ps1 -Publish -WarmStart -MaxTotalSteps 475000000
# A fresh output without prior state does not need -WarmStart.
```

`-MaxTotalSteps` sets the shared total ceiling, `-GateEvery` sets the evaluation
interval, and `-Worlds` sets the parallel batch. `BOXER_CURRICULUM_PYTHON`
selects another interpreter. Keep Windows and WSL running. The launcher checks
for a duplicate host; the supervisor holds both a run lock and a publication
namespace lock. Stopping the supervisor terminates the entire learner process
group, including an active evaluator, while retaining completed checkpoints.

- `runs/curriculum/state.json`: supervisor status, active level and total at the last supervisor event.
- `runs/curriculum/<level>/progress.json`: live level and total transition counters and PPO metrics.
- `runs/curriculum/<level>/checkpoint/`: numbered atomic, checksum-verified policies.
- `runs/curriculum/<level>/gates/`: exact historical gate results for each evaluated checkpoint.
- `runs/curriculum/<level>/gate_history.json`: full promotion history.
- `runs/curriculum/<level>/training_result.json`: passing streak and promotion/budget status.
- `checkpoints/curriculum/<level>/`: latest published policy and complete contract/config.
- `docs/results/curriculum/`: published progress, gate history and job state.
- `fights/curriculum/<level>.mp4`: one actual simulated trajectory from the latest full gate.

With `-Publish`, completed gates export artifacts, commit and push only the
owned curriculum paths. Git retains previous policies. Unrelated staged files
block publication. Preview files from the original protocol remain historical
records; their own counters/hashes identify them and they cannot promote a level.
Failures retain the last valid checkpoint and are recorded explicitly.

Stop by sending SIGTERM to the Linux supervisor PID from `state.json`:

```powershell
wsl -e kill -TERM <supervisor-pid>
```

For standalone evaluation:

```bash
export PYTHONPATH=. JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
"$HOME/venvs/boxing_curriculum/bin/python" -m boxer.curriculum_eval \
  runs/curriculum/balance/checkpoint --output runs/manual_gate.json
```

## Scope, validation and upstream attribution

The foundation uses the established feet-only G1 locomotion model. It has real
floor contacts and joint dynamics, but does not contain a physical opponent or
the boxing fist/head/torso collision scene. The navigation marker is not a
support or a boxer. After all levels pass, the supervisor ends honestly in
`foundation_completed`. Transfer to two free boxers remains a separate
integration and contact-validation experiment; this job never silently
re-enables pelvis springs or calls navigation boxing.

Before paired training, check joint order, gains, default pose, action scaling,
observations and physical dynamics against the boxing scene. Then evaluate real
legal fist contacts, falls, footwork and engagement from wider spawns, exchanging
corners and using held-out seeds. The two scenes are not assumed interchangeable.

The [CPU transfer audit](curriculum-transfer-audit.md) now records the measured
gain, pose, timestep, and action-scale differences that the adapter must resolve.

Historical smoke tests, pilot metrics and dependency versions remain under
`docs/results/curriculum/`. Adaptive tests additionally verify early exit before
the budget, repeated complete gates, shared accounting, stale/preview rejection,
command mixtures and physical leg authority. These are engineering validations,
not evidence that locomotion or boxing has already been mastered.

The adaptive change passed **20 focused tests**. A separate GPU diagnostic
warm-started the preserved actor into easy walking, executed 20,480 transitions
and completed two 14-trial gates (32 resets per trial, seeds 7/23). Both gates
failed, so it exhausted its deliberately small diagnostic allocation without
promotion. Parameters and losses remained finite. This diagnostic is separate
from the main counter and is not its initialization source. See the
[adaptive validation record](results/curriculum/adaptive-validation.json).

Dependencies are the Apache-2.0 licensed
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) G1 task
at `12ad7f3842791ced39f27e5ac96d066bbe361653` and
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) assets at
`1b86ece576591213e2b666ebf59508454200ca97`. Upstream licenses remain with those
dependencies; no humanoid model or animation was invented for this experiment.
