# Paired striking pilot log (2026-10-22 to 2026-10-29)

Moved from the README on 2026-11-03, unchanged apart from link paths. The
pilot's reports are in [docs](..) and its index in the
[experiment history](../experiment-history.md#decisions-and-evidence).

This section is kept as written during the pilot. Words such as "current",
"next" or "no PPO job is active" describe that time. The pilot was superseded
on 2026-10-30, when the legs were retrained to balance while the arms punch
(see the [README timeline](../../README.md#progress-timeline)).


Latest review (2026-10-29 local): the [first scaled GPU block](../results/striking-pilot/gpu-learning-v1/README.md)
completed 1,048,576 credits with no complete native boxing pass. Final fresh
neutral punches total 27; mirror has 5/1 valid cycles but only 3/8 pair survivors
and no valid reciprocal surviving seed. All checkpoints/traces/evidence verify.
Collection+PPO sustained rate falls from 510 to 371 credits/s by quarter;
growing accounting/history I/O is being corrected without losing attempted steps.
The [matched transfer follow-up](../gpu-throughput-and-transfer-followup.md)
compares actual source/final CUDA behavior with retained native results before
another PPO allocation. No weaker gate, rollback, solver change or main adoption.

Current continuation (2026-10-29): [optimized collection](../gpu-collection-scaling.md)
and [scaled PPO integration](../results/striking-pilot/gpu-batched-scaling-v1/README.md)
are reviewed. Pinned readback improves matched complete collection from 621 to
835 world controls/s; the larger block measures about 870 learner credits/s
across collection and PPO, excluding initialization/evaluation. All 256 updates
replay exactly. The [persistent bounded GPU job](../gpu-bounded-learning.md)
starts with a native baseline, then at most 1,048,576 new credits with full native
gates every 262,144 and a fresh final group. Inspect live state before any restart.
No complete boxing gate passes or reliable reciprocal boxing are newly claimed.

Latest reviewed work (2026-10-29): [batched CUDA mechanics](../results/striking-pilot/batched-gpu-mechanics-v1/README.md)
and the [1,024-credit warm-start PPO integration](../results/striking-pilot/gpu-batched-integration-v1/README.md)
are complete. Full contacts, cycle rewards, first falls and independent randomized
resets are implemented and checked; all original learned state is retained.
Complete frozen collection measured 85.5 / 601.5 world controls/s at 8 / 128 worlds,
excluding PPO. The original eight-world integration does not recover historical
training speed. Its finite checkpoint is archived; no full boxing gate passes.
Next optimize collection and validate explicit larger-batch/update scaling before
a bounded longer GPU learning allocation. Diagnostic PPO is 9,505,024 at its
reviewed ceiling; formal/main counters and physical gates are unchanged.

Earlier work (2026-10-29): [GPU recovery and physical validation](../gpu-training-recovery.md).
A batched current-model prototype reaches thousands of world controls/s including
both actors and the frozen motor; this excludes PPO and full cycle rewards.
Open-loop contact agreement improves with solver/line-search diagnostics, but
reset drift remains. The nine-profile frozen-policy comparison is complete;
none passes complete physical criteria. Solver convergence alone is not a remedy.
The legacy Warp6/7 comparison is also complete: no complete straight cycles in
sixteen reconstructed episodes, and no common-posture survivors from 2 m.
This does not rule out hooks/other punch types. No PPO job is active. Next
[complete batched scoring/rewards/resets and validate GPU PPO integration](../gpu-batched-training-next.md).
Inspect live state before launches; experimental physics cannot replace native gates.

Previous work (2026-10-28): [arm-effort PPO v17](../results/striking-pilot/arm-effort-v17/README.md)
completed its allocation without a full pass. Its final fresh self-play survives
3/8 pairs; both actors strike in one surviving episode. No PPO job is active.
The user requested GPU throughput recovery and a fair legacy-policy comparison;
current-model GPU investigations are recorded above. Independent physical gates remain unchanged.

Previous work (2026-10-28): [contact-rebalance PPO v16](../results/striking-pilot/contact-rebalance-v16/README.md)
completed its 2,097,152-credit allocation. The final policy has 42 valid neutral
cycles on fresh seeds, and both agents strike in one surviving fresh mirror bout.
Fresh survival 3/8, 7/8, 1/8 fails the independent full gate. A separately
bounded [arm-residual diagnostic](../results/striking-pilot/v16-arm-ablation/README.md)
investigates early balance failures; masked actions cannot certify a policy.

Previous review (2026-10-26): v15 completed all 2,097,152 new credits. The final
policy produces **23 complete neutral-practice strikes on fresh seeds**, but
only 2/8 fresh mirror pairs survive, with zero fresh mirror valid cycles.
No full gate passes or general boxing completion. Unchanged PPO is stopped at
its ceiling. A [bounded opponent-distribution diagnostic](../results/striking-pilot/v15-opponent-gap/README.md)
found persistent falls against the exact frozen training adversary. See the
[completed trajectory](../results/striking-pilot/contact-progress-v15/README.md).

Earlier intervention (2026-10-26): [target-coupled contact-progress PPO v15](../striking-contact-progress-training.md)
preserves the actual latest learner/Adam and unchanged physical gates. The audit
finds faster preparation but larger target distances and less legal contact in
v14; the new objective explicitly couples progress to targets/physical contact.
A [qualitative scope estimate](../completion-status.md) is about 40%; learned
stable reciprocal boxing remains outstanding. New budgets are declared
before collection; inspect live state before recovery or another launch.

Intermediate reviewed outcome (2026-10-26): v15 at **1,048,576 new credits** produces
**six complete deterministic punch cycles against neutral partners**, four in
corner 0 and two in corner 1, on two surviving known seeds per corner. The
[independent saved-trace review](../results/striking-pilot/contact-progress-v15/step-1048576-independent-review.json)
verifies checkpoint/native/full hashes, finite arrays and unchanged criteria.
However, mirror pair survival regresses from 5/8 to **0/8**, with no scored contacts
or valid cycles before termination. Neither full gate passes; no promotion or
main adoption. V15 subsequently completed its remaining scheduled gates within
the 2,097,152-credit allocation without resets or weaker gates.

Earlier outcome (2026-10-26): [motion-cycle PPO v14 completed](../results/striking-pilot/motion-cycle-v14/README.md)
all 1,048,576 new learner credits and six full evaluations before the computer
restart. All 144 saved native/full trajectories verify, but **zero valid complete
punch cycles** occur. Fresh final pair survival is 20/24; no boxing milestone or
main adoption. That completed block motivated the separately declared v15
continuation; the finite block does not prove learning impossible.

See the [project trajectory and experiment index](../experiment-history.md)
for reviewed successes, failed approaches, limitations and pending decisions.

Latest milestone (2026-10-24): the longer-GAE candidate passed two distinct
consecutive complete original ten-second gates and a complete fresh set (23/24
paired survivors). Its [scope-limited readiness record](../results/striking-pilot/gae-readiness-certification/README.md)
freezes latest bea8147f. A [broader frozen audit](../striking-gae-generalization.md)
completed new reset/opponent checks: neutral profiles pass, but three of four
full groups fail active combat. Opponent-pool integration now passes 41 focused
tests; the [completed matched comparison](../results/striking-pilot/pool-2064384/README.md)
found no demonstrated improvement and both final full gates fail. A
[visual and objective audit](../striking-punch-quality-audit.md) also confirms
that qualifying contacts do not establish recognizable punches. In the
[completed passive punch-motion replay audit](../results/striking-pilot/punch-quality-2080768/README.md),
all eight episodes reproduce exactly, but 0/19 selected contact events meet both
old speed/direction references. Versioned motion evidence and the
[complete retained-demo audit](../results/striking-pilot/demo-motion-2080768/README.md)
also find no validated positive strike-cycle target. A
[bounded scripted jab-cycle feasibility probe](../results/striking-pilot/jab-cycle-v1/README.md)
also failed all sixteen conditions. An [exact full-body replay and workspace audit](../results/striking-pilot/jab-failure-workspace-v1/README.md)
reproduced those failures: only the acting body terminates, with measured arm
saturation and tracking errors. Static guard endpoints match, but dynamic
stability remains unproven. Next isolate guard-only versus motor-only behavior
before redesign or further PPO. The [reviewed motor/guard comparison](../results/striking-pilot/guard-control-v1/README.md)
finds 8/8 motor-only survivors versus 0/8 static-guard survivors. A
[reviewed two-second entry probe](../results/striking-pilot/guard-ramp-v1/README.md)
also fails 0/8, despite better hand tracking. The
[motor-guard cycle v2](../results/striking-pilot/motor-guard-cycle-v2/README.md)
preserves the already-learned compact guard between extensions: 8/8 pairs survive,
but zero physical impacts or candidate punches occur. Next investigate extension
timing/target lag while retaining that stable control. The
[reviewed 120 ms dwell](../results/striking-pilot/terminal-dwell-v3/README.md)
also survives 8/8 without physical strikes. A
[cloned terminal-workspace review](../results/striking-pilot/terminal-workspace-v1/README.md)
finds unmatched endpoints in bounded local searches; next distinguish collision
surface reach from the commanded point before changing authority. The
[signed-surface audit](../results/striking-pilot/surface-audit-v1/README.md)
finds blocking intersections; the [reviewed 6/12 cm bow comparison](../results/striking-pilot/arc-path-v4/README.md)
still produces no commanded-left target contacts or candidate punches (survival
7/8 and 8/8). Its sole legacy event is an incidental opposite-hand touch.
The [reviewed lateral motor-command comparison](../results/striking-pilot/lateral-command-v5/README.md) has 8/8 versus 2/8 survivors and no candidate punches. The [motor-only isolation](../results/striking-pilot/lateral-isolation-v6/README.md) removes all six falls (8/8 survivors), but produces no punch. The [reviewed temporal handoff](../results/striking-pilot/guard-step-v7/README.md) still has only 4/8 survivors and no punches. Stop unchanged timing/lateral sweeps; next audit arm contracts and joint tracking from saved/source evidence. The [version-14 learned-cycle PPO protocol](../striking-motion-cycle-training.md) completed its bounded 1,048,576-credit continuation with zero valid cycles; inspect the reviewed outcome before any new allocation. General boxing remains
uncertified, and the stopped main policy has not been replaced.

An [earlier striking final](../striking-generalization-audit.md) passed two
distinct consecutive original complete screens, with all 24 pairs surviving in
each. The complete 128-episode reset/opponent audit is reviewed: all four fresh
full groups fail, despite 12 surviving two-way active exchanges. A bounded
[motor-floor diagnostic](../striking-motor-floor-audit.md) found 31/32 motor-only
pair survival. A [matched posture-cost comparison](../striking-posture-cost-comparison.md)
is completed: its extra cost is rejected (15/24 fresh paired survivors versus
20/24 control). All four new full screens fail. A read-only
[objective audit](../striking-objective-audit.md) found a profile-dependent
sampled/mean action gap. A bounded [sampling-profile comparison](../striking-sampling-comparison.md)
found no sustained benefit from half standard deviation. A bounded
[actor KL-guard comparison](../striking-kl-guard-comparison.md) completed without
robust stability (19/24 fresh paired survivors versus 18/24 control and 23/24
source). A bounded [passive failure replay](../striking-kl-failure-trace.md)
completed with exact episode reproduction and mostly post-contact low-pelvis
failures. A bounded [GAE trace-length comparison](../striking-gae-comparison.md)
completed with fresh mirror pair survival 8/8 and six reciprocal episodes,
but all complete physical sets still fail. One [bounded ordinary continuation](../striking-gae-continuation.md)
completed with a complete intermediate pass followed by regression. A bounded
[frozen gradient/critic assay](../striking-gae-gradient-audit.md) investigated
that regression without changing saved policies or adding PPO. It did not
establish retention dominance or a larger final critic error. A bounded
[unchanged consolidation](../striking-gae-consolidation.md) completed with the verified limited milestone above. No policy is
certified as reliable boxing.
The [latest recorded diagnostic video](../../fights/striking-pilot/terminal-dwell-v3-recorded.mp4)
shows both acting corners, seed 7, closed fists and free bodies over 20 seconds.
It reuses saved trajectories with zero valid impacts and no new physical steps;
it is not successful learned boxing. All 500 frames decode and source hashes are
retained in the [video record](../results/striking-pilot/terminal-dwell-v3/video.json).
See the [completion roadmap](../boxing-roadmap.md). The
[verified selected source replay](../../fights/striking-pilot/first-full-screen-pass-1908736.mp4)
illustrates an earlier ten-second exchange; it is not a generalization result.

All four [foundation stages](../curriculum.md) passed their unchanged complete
physical gates twice on distinct checkpoints. Foundation learning finished at
**474,992,640 consumed transitions**; keep it stopped. Both paired robots retain
free roots and legs, without pelvis springs, welds or gravity compensation.

The [striking policy](../striking-pilot.md) adds opponent-aware arm residuals and
movement-command offsets over that frozen learned motor actor. It starts two
robots nominally 2 m apart and credits independent physical fist impacts. It
has not established stable boxing or superiority to the archived unassisted
policies. At **1,048,576 boxing learner transitions**, final evaluation found
6/8 surviving pairs and three impacts concentrated in an episode ending in a
fall. Main PPO stopped at its reviewed ceiling; no further main steps are granted.

The [completed 256-episode holdout and contact audit](../striking-body-holdout-review.md)
found one reproducible ten-second episode with a legal impact from each corner
and both robots still standing. It is rare (one of 64 adaptive sampled trials),
and adaptive body authority regressed survival against control. Neither actor is
adopted. The [actual replay](../../fights/striking-pilot/stable-two-way-exchange-1179648.mp4)
is evidence of that episode, not reliable boxing.

The [neutral-partner curriculum pilot](../striking-partner-curriculum.md) completed
131,072 diagnostic transitions without passing: survival stayed 6/8 and 8/8 by
learner role, with no final learner impacts. The duration continuation stopped
after 262,144 additional steps on two stagnant complete checks. No phase passed.
The [completed exploration audit](../striking-exploration-review.md) found no
stable own strikes under any of four noise profiles. The next
[physical feasibility diagnostic](../striking-directed-feasibility.md) found
four reproducible ten-second directed-strike trajectories, but poor overall
controller survival. It is not a trained boxer or a phase pass. A sixteen-seed
holdout produced fifteen further trajectories, all exactly replayed. The
[completed supervised initialization](../striking-demonstration-initialization.md)
learned isolated stable strikes against a neutral partner, but neither checkpoint
passed and active self-play regressed. The [contact-history comparison](../striking-contact-history-comparison.md)
completed 2,000 further supervised updates per arm and all 48 physical episodes.
Both variants failed; contact feedback reached 5/8 self-play survival versus 3/8
for control, without sufficient own scoring in both roles.
No additional PPO was allocated to that imitation comparison. Both bodies/legs remain free and all contacts
active. No actor is adopted and no physical gate is weakened. Inspect
`/home/pedro/boxing-runs/striking-contact-compare-bc2000/comparison.json`
and processes before another launch; that job is completed. The main run remains
stopped. The [completed coverage/channel audits](../striking-policy-coverage-audit.md)
reproduced every physical result and localized selected early falls to the arm
residual channel. Their short interventions are diagnostic, not adopted policies.
The [fresh sample review and motor-anchoring comparison](../striking-motor-anchoring.md)
validated 32 unique approach prefixes and completed 1,000 supervised updates
per matched arm plus all 120 full physical episodes. The corrected actor has
7/8 active survival on fresh seeds but only one surviving impact; neither arm
passes the complete criteria. No actor or stage is adopted. Further identical
imitation is stopped. The [completed matched PPO comparison](../striking-ppo-anchor-review.md)
preserved the latest BC5000 actor and independent actor/critic Adam histories,
but both 131,072-credit arms failed: stable neutral strikes disappeared and
auxiliary anchoring regressed survival. Neither actor is adopted; no additional
PPO is allocated to that failed objective. The completed saved-Gaussian audit
also found no stable-strike recovery. A short gradient assay showed that the small
approach-only loss does not preserve demonstrated strikes. The
[completed demonstration-retention comparison](../striking-retention-review.md)
restores learned mean-policy impacts: one fresh ten-second episode has three red
and two blue valid strikes with both robots standing. It is one selected episode,
and neutral-role-1 survival regresses to 4/8; neither actor passes full criteria.
No phase advances or actor is adopted. The selected exchange and matched failure
traces reproduce exactly; the video uses actual recorded coordinates and closed
fists. The [completed follow-up active continuation](../striking-neutral-retention-curriculum.md)
also fails both complete screens: stable active contacts persist, while neutral
practice remains unsuccessful. The [completed neutral-partner task](../striking-reward-credit-review.md)
also fails both complete screens: final survival is 7/8, 8/8 and 7/8, with no
stable legal events. No actor is adopted or phase promoted. Training receives no
unchanged extension. The [completed reward-credit audit and matched horizon comparison](../striking-reward-credit-review.md)
verified all raw physical gates and checkpoints. Gamma 0.999 produces six
surviving active impacts across four fresh seeds, including one deterministic
ten-second two-way episode, versus no fresh impacts at gamma 0.99. Neither
passes: neutral strikes remain absent and some pairs fall. The further
32,768-credit ordinary continuation has now completed: both complete generations
fail. Final neutral-role-1 striking improves, but active survival regresses from
7/8 to 6/8 and surviving active events from five to one on the same fresh seeds.
No more optimizer steps are allocated. All 32 subsequent passive source/final replays
reproduce the original physical results exactly, with finite states and no
numerical/capacity errors. Latest-final active failures are red low-pelvis
terminations after interbody contact; the neutral failure is blue self-leg
contact. The fixed controller ablations have now completed: each
rescues all selected falls (6/6 surviving pairs versus 3/6 unchanged baseline).
Removing command offsets retains two legal strikes; removing arm residuals
produces none. The broader 72-episode holdout is now completed:
zero commands improves original-screen survival, but fresh neutral strikes vanish
and active fresh survival changes 8/8 to 7/8. Neither controller passes a complete
set, so the masked controller is not adopted. The
[matched mixed-opponent comparison](../striking-mixed-partner-comparison.md)
is completed and independently reviewed: mixed fresh survival is 24/24 versus
18/24 neutral control and 22/24 source. Neither arm passes complete criteria;
valid strikes remain sparse. The neutral control severely regresses. The
further 65,536-credit mixed continuation is now completed: neither generation
passes, and matched fresh survival regresses 21/24 to 15/24. The [repeatability outcome and generalization audit](../striking-generalization-audit.md)
retain the latest final and all prefixes. Inspect
`/home/pedro/boxing-runs/striking-kl-guard-1949696/comparison.json`
and its process before any launch. Diagnostic PPO starts 1,983,744, ceiling
2,016,512; reviewed formal evaluation is 1,558,615, ceiling 1,642,615. Main counters and complete
physical criteria remain unchanged. No phase promotion or policy adoption.

The [closed-fist replay repair](../closed-fist-rendering.md) restores the
original visual feature without changing learning or physics. Historical videos
and all earlier policies remain archived. Project code, documentation and labels
are in English. The original training stacks below are historical references,
not evidence that the current experiment has already learned their skills.
