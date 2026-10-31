# Boxing project trajectory and experiment index

This is the navigation record for what worked, what failed, what remains
conditional and why the next experiment was authorized. Detailed protocols,
full physical results, limitations, checkpoints and accounting remain in the
linked reports. This index does not replace those records or the live state.

Latest consolidated review: **2026-10-30**. The [sparring motor](spar-motor.md)
removes the paired-fall bottleneck. The certified motor had only balanced with
its arms in guard; with punching arms it survived 6 % of solo trials. After
39.3M fine-tuning steps under generated punches and shoves it survives 100 %,
and unassisted two-robot bouts go the full 20 s in 8/8 seeds with 18.6 clean
punches per bout. Tactics are still scripted.

Last consolidated review: **2026-10-29**, including retained-policy comparisons,
complete batched GPU mechanics and the conditional warm-start integration.
Reliable general boxing is **not yet certified**. The foundation is certified;
learned complete strikes exist, while stable reciprocal combat remains unresolved.

## Sparring motor and unassisted two-robot bouts, 2026-10-30

| Work | Reviewed outcome | Decision and limitation |
| --- | --- | --- |
| [Arm-motion diagnosis](spar-motor.md#diagnosis) | Certified `approach` motor with generated punching arms, solo, no opponent: mean survival 0.06, mean 2–5 s. | The frozen leg controller, not the arm objective, caused the v12–v17 paired falls. Arm residual PPO over it is retired. |
| [`spar` stage fine-tune](spar-motor.md#change) | 39.3M steps from `approach`, ≈ 30 min on the RTX 3060. Gates: 10.8M 0.994 (tracking fail), 21.6M pass, 32.4M fail (one tracking error 0.216), 39.3M pass with 1.000 in all 28 trials. | `checkpoints/spar` is the new motor. One pass, not two consecutive; the gate is a solo test. |
| [Learned tactics](tactics.md) | Batched MJX/Warp two-robot env, ≈ 25k PPO steps/s. Gen‑0 (vs scripted, 20M steps) beats scripted 16–0 natively (20.1 : 5.7 clean punches); learned mirror 37.6 clean punches / 20 s, no falls. | Champion until 2026-10-31, now `checkpoints/tactics-batched`. |
| [Self-play league](tactics.md#results-native-mujoco-16-bouts-per-row) | Gen‑1 beats gen‑0 82–41 batched but loses 3–13 natively; pool + native-style starts: gen‑2 4–10–2, gen‑3 8–8 (10–22 on fresh seeds). Native-scorecard training (v2): 4–12, 3–12–1. | Batched→native transfer of policy-vs-policy skill is the open problem. Select by native evaluation only. |
| [Domain randomization v3](tactics.md#domain-randomization-v3) | Per-world friction, contact softness, masses, joint friction and actuator stiffness. v3 gen‑0 vs champion 21–23–4 over 48 native bouts (best self-play result); gen‑1 12–32–4, gen‑2 9–33–6, still 15–1 vs scripted. Batched gen‑2 beats the champion 137–103 but falls in 56/256. | Randomization narrows the gap for one step; continued self-play learns batched-specific risk-taking. Champion unchanged. Next: native-gated selection, stronger fall penalty, randomized motor. |
| [Native gate + fall penalty v4](tactics.md#native-gate-and-fall-penalty-v4) | Gate on 4 checkpoints per generation vs champion: 6–8–2 (5M), 4–11–1, 5–10–1, 3–13–0 (20M); selected 5M: 16–30–2 over 48 bouts, 2 falls. | Fall penalty removes falls; batched self-play still degrades native strength monotonically. Stopped after one generation. Next: native-physics fine-tuning. |
| [Native-physics fine-tuning](tactics.md#native-physics-fine-tuning-the-new-champion) | PPO on 8 parallel native CPU workers (≈ 630 steps/s), 2M steps from the batched champion. Gate: 8–7–1 → 13–3 → 13–2–1 → 15–0–1; 48 bouts 42–2–4 (21.9 : 15.0); vs scripted 15–1. | **New champion** `checkpoints/tactics`; old one kept as `checkpoints/tactics-batched`. Native strength improves monotonically, so the batched drift was physics mismatch. |
| [Second native round](tactics.md#second-native-round-plateau) | 3M more native steps with the champion in its own pool: gate 5–8–3, 8–7–1, 7–9–0, 7–6–3; pick 18–28–2 over 48, candidate 9 falls vs 5. | Plateau at parity with riskier play; champion unchanged. Next gains need a richer move set. |
| [Hooks and high guard](moves.md) | FK-fitted moves; motor retrained (58 % → 100 % solo survival, two consecutive gate passes). Widened champion, 3M native steps: 33–12–3 over 48 bouts vs previous champion, 15–1 vs scripted (opponent 1.75 clean punches). Uses high guard 34 % of time, no hooks; 6 vs 2 falls. | **New champion** `checkpoints/tactics`; previous kept as `checkpoints/tactics-native-v1`. Hooks need closer footwork or a variety incentive. |
| [Level long hook](moves.md#level-long-hook-v2-active-since-2026-10-31-evening) | Diagnostics: short hook landed 0–6 %; head out of reach for every punch; body hooks touch the torso 2× as often as straights vs a high guard. Level IK-arc hook, motor passes the gate at 0 and 10.8M. Native fine-tune: 37–9–2 vs guard champion, falls 1 vs 4; hooks ~2 % of punches. | **New champion** `checkpoints/tactics`; previous kept as `checkpoints/tactics-guard-v1`. Hooks remain marginal. |
| [Two-robot bouts](spar-motor.md#results) | Native MuJoCo, free roots, scripted tactics: 8/8 go 20 s, 18.6 clean punches/bout, both corners win. Old motor: 0/4, 3 s. | First reliable unassisted reciprocal boxing in the project. Punch choice is scripted; clinching after ≈ 8 s. |

## Historical corrections and retained capabilities

Free-body, unassisted agents and closed fists existed before the current
curriculum. The hand-rendering omission and its corrected commit-level history
are recorded in [closed-fist rendering](closed-fist-rendering.md). Do not describe
free legs or closed fists as first introduced by the current curriculum, or
attribute the visual omission to removing assistance. The visual correction does
not change collision physics, training or scoring.

## Decisions and evidence

| Work, in trajectory order | Reviewed outcome | Retained decision and limitation |
| --- | --- | --- |
| [Foundation recovery and curriculum](curriculum-recovery.md), [certification](results/curriculum/foundation-certification/README.md) | Balance, walk_easy, walk and approach each passed twice on distinct checkpoints; 474,992,640 main transitions consumed. | Retain the certified unassisted motor. This is locomotion certification, not boxing completion. |
| [Paired striking pilot](striking-pilot.md), [whole-body comparison and holdouts](striking-body-holdout-review.md) | Actual paired physics/contact evaluation implemented; expanded body authority did not establish robust benefit. | Preserve independent first-fall/contact criteria; no certified body-authority replacement. |
| [Partner practice](striking-partner-curriculum.md), [duration](striking-partner-duration-review.md), [exploration](striking-exploration-review.md) | Practice/extensions and coverage probes did not establish stable striking. | Do not extend the old allocation blindly; retain the failed evaluations. |
| [Directed feasibility](striking-directed-feasibility.md), [supervised initialization](striking-demonstration-initialization.md) | Physical demonstrations were reproducible, but the directed holdout had only 15/64 successful stable strikes; initial students did not pass. | Demonstrations provide training data, not an adopted autonomous controller or certified policy. |
| [Contact history](striking-contact-history-comparison.md), [motor anchors](striking-motor-anchoring.md), [PPO/anchor comparison](striking-ppo-anchor-review.md) | Observation/initialization/retention mechanisms and physical regression were investigated; neither PPO/anchor final passed. | Preserve explicit checkpoint migrations, optimizer provenance and independent physical checks. Offline loss alone is insufficient. |
| [Demonstration retention](striking-retention-review.md), [neutral-partner curriculum](striking-neutral-retention-curriculum.md), [reward credit](striking-reward-credit-review.md) | Retention restored real learned strikes, including stable reciprocal contact, with incomplete stability. Later curricula/credit investigations remain individually archived. | Partial capability retained diagnostically; no complete stage certification. |
| [Mixed partners](striking-mixed-partner-comparison.md), [policy drift investigation](striking-policy-update-investigation.md), [yaw comparison](striking-approach-yaw-review.md) | Generalization/regression persisted; reduced actor rate and modified yaw did not establish robust physical benefit. | Reject those changes as stability remedies; keep original physical criteria and all failed generations. |
| [Fixed opponent](striking-fixed-opponent-comparison.md), [first complete passes](striking-first-complete-pass.md), [generalization audit](striking-generalization-audit.md) | Two distinct original complete screens passed earlier, but all four fresh full groups failed. | This was a real repeatability milestone on the original screen, not independent generalization or main adoption. |
| [Motor-floor audit](striking-motor-floor-audit.md) | Motor-only paired survival was 31/32; head-dependent failures also existed. | Diagnostic action masks are not adopted policies; the remaining failure prevents claiming a perfect motor floor. |
| [Posture-cost comparison](striking-posture-cost-comparison.md) | Fresh pair survival 15/24 versus 20/24 control. | Reject the extra posture cost; do not weaken gates or increase that penalty blindly. |
| [Objective audit](striking-objective-audit.md), [sampling comparison](striking-sampling-comparison.md) | Sampled/mean gaps depended on profile. Half-noise training and control both finished at 17/24 fresh pair survival. | Reject narrower sampling covariance as a validated robust remedy. A transient original-screen pass did not persist. |
| [Actor KL guard](striking-kl-guard-comparison.md) | Fresh pair survival 19/24 guarded, 18/24 control, 23/24 source; every complete set failed. | Fewer actor updates did not establish stability. No guarded policy adopted. |
| [Exact failure replay and terminal-credit audit](results/striking-pilot/kl-failure-trace-1966080/README.md) | All 72 trials reproduced exactly; nine low-pelvis and three self-leg-contact failures. Eleven followed interbody contact. Direct GAE terminal-credit attenuation measured without new simulation. | Contact-before-fall is association, not causation. Direct terminal credit is not the entire critic-mediated credit horizon. |
| [GAE lambda comparison](striking-gae-comparison.md) | Lambda .99: fresh mirror survival 8/8 with six reciprocal episodes; control 7/8 with four, source 6/8 with four. Neutral survival 6/8 and 7/8 still failed. Aggregate 21/24 only matched source. | Promising specific exchange improvement, not overall robustness or certification. Every complete set failed. |
| [Ordinary GAE continuation](striking-gae-continuation.md) | The +8,192 full screen passed with 24/24 survivors, followed by a failure at +16,384 (4/5/8); final fresh survival 7/7/8 still failed neutral-1. | Stop unchanged PPO. Version-12 history fail/fail/pass/fail, streak zero. Archive the intermediate without restoring it. |
| [Frozen gradient/critic assay](striking-gae-gradient-audit.md) | All 12,288 diagnostic controls and twelve batches reviewed; retention norm .30-.33 versus PPO 6.93-7.99, no larger final critic/GAE error. Zero optimizer updates. | No established retention-dominance/critic-error cause; small frozen batches do not rule out longer-run effects or certify survival. |
| [Unchanged consolidation](striking-gae-consolidation.md) | All 65,536 credits / 144 episodes reviewed; four original gates pass/fail/pass/pass, final fresh 29M complete pass (23/24 pairs). | Freeze latest bea8147f. Version-12 streak two plus one fresh group verifies limited ten-second readiness, not general boxing or main adoption. |
| [Broader frozen audit](striking-gae-generalization.md) | All 130 episodes / 61,934 controls reviewed. Neutral profiles all pass, active mirror 23/32 fails three full groups; three of four cross-policy diagnostics pass. | Broader boxing remains uncertified. Completed-renderer notice recovered without repeated physics; original failed state/logs and validated video retained. |
| [Opponent-pool comparison](results/striking-pilot/pool-2064384/README.md) | Both 16,384-credit arms and all 168 episodes / 78,563 controls reviewed. Pool original fail/fail and fresh 8/6/5; fixed control pass/fail and fresh 8/6/6; source fresh 8/6/7. Exact source/optimizer/sampler provenance, finite states and physical ledgers verified. | No demonstrated pool improvement; freeze both actual finals with no new allocation. Version-13 streak zero; version-12 ten-record history now streak zero, preserving the earlier source milestone. Uneven per-opponent/corner exposure limits inference. |
| [Contact-versus-punch quality audit](striking-punch-quality-audit.md) | User correctly questioned touching in the videos. Frame inspection and code show that .2/.3 m/s onset motion plus .5 N s can qualify without verifying a punch cycle; older code already had richer motion checks. Seven scorecard tests pass, including a synthetic low-speed contact example. | Contact events are not punch certification. No further unchanged PPO; next audit actual velocity, travel, impulse timing and recovery before a prospective objective change. No new physical controls or updates spent by this static audit. |
| [Passive punch-motion replays](results/striking-pilot/punch-quality-2080768/README.md) | All eight selected episodes / 4,000 controls reproduce every original trial field. Fifteen focused tests pass. At contact onset, 0/19 legacy events meet both old speed/forward references; video-source pre-window hand excursion is 5.3–9.1 cm. | Measured small motions support the user's concern, without classifying every punch type. Five recovery windows are censored. No unchanged PPO extension; next implement independent strike-motion evaluation and audit positive demonstrations before a versioned objective/budget. |
| [Versioned evidence and all retained demonstrations](results/striking-pilot/demo-motion-2080768/README.md) | Version 1 records per-hand motion components without punch certification. Twenty-one focused tests pass. All fifteen retention trajectories / 7,500 controls replay exactly; all fifteen acting-hand events are below both old speed and forward references. The old generator exits after its first contact. | No validated positive strike-cycle target; stop unchanged imitation/retention/PPO extensions. This target mismatch is established, while its causal contribution to learned failures remains unisolated. |
| [Jab-cycle feasibility v1](results/striking-pilot/jab-cycle-v1/README.md) | All sixteen conditions and 4,048 controls reviewed. All pairs fell at 4.152–7.238 s; zero complete candidates. Of seventeen acting-hand contact events, thirteen meet the forward reference but only one the speed reference; none meets all prospective requirements. Nine focused controller/motion tests pass. | Reject v1 as a positive training target without threshold relaxation. Next audit guard/strike reachability and first-fall/phase tracking before a bounded redesign. No PPO, imitation, adoption or automatic use of the unused allocation. |

## Latest first-fall and workspace review

Latest reviewed continuation: [exact jab first-fall replay and workspace](results/striking-pilot/jab-failure-workspace-v1/README.md).
All sixteen native trajectories reproduced exactly in 4,048 additional controls.
The acting body alone triggers first fall: eleven low pelvis, one tilt and four
self-leg contacts. Arm requests clip in 19.6% of active channel values; guard endpoints match
in sampled static boxes, whereas early full-strike endpoints do not match in
two local solver starts. Neither static geometry nor these fall predicates
is a causal or dynamic-feasibility proof. Eight focused tests pass. Keep v1
rejected; next isolate motor-only versus guard-only before redesign or PPO.
Formal physical diagnostic consumption is **2,034,975** at its ceiling; separately
recorded cloned geometry is **4,339** evaluations. No learner updates, gate changes,
policy adoption or new allocation from that completed review.

The subsequent [reviewed guard-control comparison](results/striking-pilot/guard-control-v1/README.md)
completed all sixteen episodes / 6,155 controls. Motor-only survives 8/8 versus
static guard 0/8; only the acting body falls, with four low-pelvis and four
self-leg predicates. Six guard cases have no scored contact. All approach
prefixes match; native stderr is empty and nine focused tests pass. Static guard
is sufficient to reproduce failure in this matched sample; a full jab cycle is
not necessary. Reject this as a positive demonstration. The relative roles of
entry transient, final geometry and motor coordination remain unisolated.
Motor-only corner labels duplicate four unique reset conditions.

The next [reviewed smooth guard-entry probe](results/striking-pilot/guard-ramp-v1/README.md)
completed eight cases / 1,942 controls. All eight acting bodies fall again;
the other bodies trigger no first-fall predicates. Median hand tracking improves
from 5.0 to 0.99 cm, so better tracking and slower entry alone do not repair
survival on this sample. Ten focused tests pass. A post-run assertion expected
sixteen instead of eight episodes; preserve the failed state/traceback and correct
the configured-count check. Independent verification recovered completion with
zero new physical controls. No trial was discarded or repeated.

Physical diagnostic total at that review is **2,043,072**, under completed ceiling 2,045,130;
unused compute expires. No PPO/BC, rollback, gate changes, adoption or additional
geometry queries. Reject all three failed guard/cycle controllers as positive
demonstrations. Next inspect guard geometry and learned whole-body responses
against the matched stable motor-only inputs/leg actions before proposing another
bounded intervention; do not blindly unfreeze the motor or repeat failed body
authority experiments.

Source inspection then confirms that the foundation already trains with
`GUARD_ARMS_BOXING`. A reproducible, hash-checked
[array measurement](results/striking-pilot/guard-control-v1/learned-guard-offsets.json)
finds the arbitrary scripted guard asks for 10.3–11.4 cm displacement from the
learned hand position at activation; this is not an isolated causal proof.
The separately declared [motor-guard cycle v2](results/striking-pilot/motor-guard-cycle-v2/README.md)
completed all eight conditions / 4,000 controls. It applies corrections only
during extension, preserving the learned motor guard and filtered return.
**8/8 pairs survive**, versus 0/8 with either artificial guard, but **zero physical
impacts, left-fist contact substeps or qualifying punches**. Twelve focused tests
pass. Sixty-four extension pulses are archived; median extension tracking error
is still 14.7 cm. This improves stability in the sample, not the boxing objective.

Retain v2 as a stable scripted comparison control, without PPO/BC or demo adoption.
Next prospectively compare extension timing/terminal-target dwell using the
retained traces and unchanged full contact/motion/survival criteria. Wider
holdouts remain necessary even if a later positive cycle is found. Diagnostic
physical total is **2,047,072**, at its completed ceiling; PPO 2,163,968, cloned
geometry 4,339, main/BC counters unchanged. Pure array measurements add no physical
or cloned evaluation.

The subsequent [prospective terminal-dwell comparison](striking-terminal-dwell.md)
completed all eight cases / 4,000 controls. Add 120 ms at the unchanged full-extension target, retaining the
original outbound motion and motor-guard return. Eight matched cases, maximum
4,000 new controls, new ceiling 2,051,072 from reviewed total 2,047,072. Thirteen
focused tests pass. Exact full-state agreement with v2 holds through the first
extension before the changed dwell; all complete punch criteria remain unchanged.
All eight pairs survive, but no impacts or candidate punches occur. Reject dwell
as a sufficient repair; no positive-demo adoption or training extension.

The [reviewed terminal-workspace audit](results/striking-pilot/terminal-workspace-v1/README.md)
completed all 24 saved snapshots with zero new physical controls. Actual median
endpoint error is 34.9 cm; the best of two bounded local static starts has median
22.9 cm (range 11.6–52.9). The sampled endpoint is not matched in these searches;
two starts are not a global impossibility or dynamic reachability proof. The
endpoint near the torso center is not its collision surface, so do not infer that
every legal contact is unreachable from endpoint error alone.

Consume 1,890 producer + 24 independent review forwards, cumulative cloned total
**6,253**, under explicit 10,339 ceiling. Durable progress, all source snapshots,
both solver attempts and bounds remain archived. No native errors/nonfinite
values. Physical total **2,051,072**, PPO/main/BC counters fixed; unused budgets
expire. Next measure legal fist/target surface gaps and blocking/inter-arm
contacts at recorded/best poses before a prospective surface-target or authority
intervention. Do not repeat unchanged dwell, enlarge authority blindly or resume
training from these non-contact motions. No job or new allocation is active.

## Where to check current facts

On 2026-10-25, the [signed-surface audit](striking-surface-audit.md) independently
verified all 48 recorded/best poses and ten opposing surfaces per pose. Recorded
poses have positive legal gaps in all 24 cases and blocking overlap in 20. The
old static-best poses overlap a legal surface in six cases, five also intersecting
a blocker. Prior positive-force blocking events exist by each recorded snapshot;
this does not prove the same block is still active or the entire path blocked.
Cloned count increases from 6,253 to **6,349**; **960** signed-distance queries
are separately recorded. No physical or optimizer steps were added.

That evidence motivated the [reviewed lateral-bow comparison](results/striking-pilot/arc-path-v4/README.md).
Both complete arms / sixteen conditions are reviewed: 6 cm survival 7/8, 12 cm
8/8, zero candidate punches and zero acting-left target contact substeps in all
cases. The single old-counter event is an incidental **right-hand** touch at
0.294 m/s with forward cosine .036, not the commanded jab. One 6 cm case fails
the acting-body low-pelvis rule at 9.812 s. Sixteen focused tests pass; original
endpoints, learned guard, authority, collisions and complete criteria are preserved.

Consume **7,991 new controls**, cumulative physical **2,059,063**, below explicit
2,059,072 ceiling; unused nine controls expire. PPO 2,163,968, cloned forwards
6,349, signed distances 960 and main/BC counts unchanged. Reject both bows as
positive demonstrations; no adoption, gate relaxation or unchanged PPO extension.
All these scripted trials keep the existing three movement offsets zero. Next
prospectively vary lateral command through the already-learned walking motor,
with the arm path fixed, to test angle/range adjustment rather than blindly
increase arm/body authority. No new allocation or job remains active.

The user requested the current video after confirming no active training job.
The [recorded terminal-dwell video](../fights/striking-pilot/terminal-dwell-v3-recorded.mp4)
now shows seed 7 without swap, both acting corners in fixed order, 20 seconds
total. All 500 frames decode; source/video hashes and selected indices are in
`results/striking-pilot/terminal-dwell-v3/video.json`. This is saved-state visual
playback with closed fists, zero impacts, zero new physical/optimizer steps and
500 separately recorded rendering forwards, not a new successful boxing result.
The renderer warning log is preserved; full encode/decode completed. The
closed-fist replay coordinate/physics regression passes. No job remains active.

- [Current curriculum status and approved ceilings](curriculum.md).
- [Remaining work and completion criteria](boxing-roadmap.md).
- `docs/results/curriculum/`: full foundation gate histories and certifications.
- `docs/results/striking-pilot/`: raw full evaluation reports, analyses, failure
  histories, traces, progress snapshots and diagnostic accounting.
- `checkpoints/striking-validation/`: immutable evaluated checkpoint payloads,
  full SHA-256 manifests and inherited configurations.
- Individual experiment protocols: exact local live-output path, launcher,
  stop rules and the distinction between completed records and current jobs.
- Git history: implementation, documentation and archived-evidence changes.

Main foundation and boxing counters, diagnostic PPO, physical evaluation,
standalone supervised updates and integrated retention presentations are separate.
Never sum branch lineage counters as newly consumed main training. Historical
budgets are not current authorization; read the latest reviewed decision and live
state. Failed complete profiles remain failures even when a selected clip succeeds.

## Documentation protocol for every continuation

For each new experiment, intervention, failure recovery or budget decision:

1. Record the question, prior evidence, source checkpoint hash, inherited profile,
   optimizer/normalizer/sampler provenance and the exact change being tested.
2. Declare bounded compute, separate consumed counters, full independent physical
   criteria, stop/recovery rules and the live output/process to inspect.
3. Archive the actual results for every arm, failed/surviving episode and distinct
   checkpoint. Preserve complete gate history, numerical/overflow checks and
   limitations; distinguish association, hypothesis and causal evidence.
4. State what worked, what failed or remains inconclusive, and the concrete next
   decision. Explain any new ceiling before spending it. Never replace failures
   with favorable earlier actors or call a pending experiment successful.
5. Update this trajectory index when the reviewed decision changes; keep
   `curriculum.md`, the roadmap and README aligned. Preserve historical protocols
   with their scope rather than erasing the path taken.
6. Run appropriate checks, commit and push project-owned code, documentation and
   durable evidence. Clearly identify any evidence still local or job still
   awaiting review. Keep repository content in English.

## Pending lateral command comparison, 2026-10-25

The [prospective protocol](striking-lateral-command-comparison.md) fixes the
straight v3 arm path and compares original lateral action bounds -1/+1 during
guard/extension through the frozen walking motor. Sixteen conditions, at most
8,000 physical controls, cumulative ceiling 2,067,063 from 2,059,063.
No optimizer or additional geometry allocation. The matched historical control
is retained; both full arms must be reviewed before any continuation decision.
Sixteen focused feasibility/workspace tests pass, including both corners and
command directions, phase release, baseline command consistency and unchanged
physics. Hidden persistent launcher uses existing global locks; no duplicate job.

## Reviewed lateral commands, 2026-10-25

[Complete results](results/striking-pilot/lateral-command-v5/README.md): negative
lateral action 8/8 survivors, positive 2/8, six acting-body falls (five low-pelvis,
one self-leg). Both arms have zero commanded-left target contacts and zero
complete candidate punches. One incidental right touch fails motion criteria.
The 16-case independent review validates prefixes, command/path schedules,
finite native physics, hashes, all events and consumed accounting. All 16
focused tests pass. New 7,244 physical controls, cumulative 2,066,307 below
2,067,063; unused 756 expire. Other compute/checkpoints unchanged.
Reject both as positive targets; do not extend PPO/imitation. Next isolate the
positive-direction schedule with arm requests disabled before any motor change.
No active job or further allocation. The pending entry above records the original
prospective decision, now resolved by this complete review.

## Prospective motor-only lateral isolation, 2026-10-25

[Protocol](striking-lateral-isolation.md): remove all scripted arm residual
requests from the eight positive-direction conditions, preserving frozen motor,
activation/phase schedule and original command channel. Compare to all eight
archived v5 positive cases with exact pre-extension state prefixes. Allocate at
most 4,000 additional physical controls from 2,066,307, ceiling 2,070,307;
all other compute and checkpoint counters retained. This is a causal diagnostic,
not boxing training or a positive demonstration. Review every matched outcome
before attributing failure or changing authority; no automatic learning extension.

## Reviewed lateral/arm isolation, 2026-10-25

[Eight matched interventions](results/striking-pilot/lateral-isolation-v6/README.md)
all survive versus 2/8 archived arm-plus-positive-command controls: all six acting
falls disappear. Motor-only still has no scored impact or candidate punch.
This implicates the arm/locomotion combination on this cohort; it does not prove
a unique joint mechanism or universal command safety. Independent review checks
exact pre-extension prefixes, zero requests/residuals, command schedules, native
physics, every original criterion and source/gate hashes. Eighteen focused tests
pass. New 4,000 physical controls, cumulative 2,070,307, ceiling exhausted. Other
compute/checkpoints unchanged. Next prospectively test guard-only footwork,
released before extension, instead of changing motor authority or extending PPO.

## Prospective temporal handoff, 2026-10-25

[Guard-step protocol](striking-guard-step.md): release positive lateral movement
before extension/dwell, keep original straight arm path and authority. All eight
matched v5 positive controls remain fixed. At most 4,000 new physical controls
from 2,070,307 to 2,074,307. The completed isolation justifies testing overlap;
it does not justify motor unfreezing, changed gates or PPO/imitation extension.
Review complete survival and punch conjunctions before any adoption/next budget.

## Reviewed temporal handoff and stop decision, 2026-10-25

[Guard-step v7](results/striking-pilot/guard-step-v7/README.md): 4/8 survivors,
zero impacts/commanded-left contacts/candidates. Three old positive-control
failures survive, one previous survivor falls. Two low-pelvis and two self-leg
first falls. Twenty focused tests and the full independent trace/source/criteria
review pass. A further 24-trajectory saved-array comparison preserves terminal
windows without new physics; errors near falling states are not causal proof.
New 3,495 controls, cumulative 2,073,802 under 2,074,307; unused 505 expire.
All other compute and checkpoints retained. No job or allocation active.
Reject the handoff as sufficient and stop unchanged dwell/lateral sweeps. Next
audit original/current arm action contracts and actual joint-target tracking in
source/saved trajectories before another prospectively bounded physical change.
No motor unfreezing, larger authority, weaker gates or PPO/BC extension justified.

## Arm audit and learned-cycle objective, 2026-10-25

[Compiled metadata/saved-array audit](results/striking-pilot/arm-contract-v1/README.md)
checks 32 trajectories without new physical controls or selected-state kinematics:
correct action/joint maps, zero native actuator target clipping, measured finite
joint tracking; wrist gain difference was already documented, not isolated as a
failure cause. Short scripted tests reject those scripts, not the possibility of
learning; a perfect scripted teacher is not a prerequisite for all further PPO.

[Prospective version 14](striking-motion-cycle-training.md) resumes actual latest
fixed-control final 3a34aa85 with actor/critic/Adam/RNG/norm intact, explicitly
removes legacy touch payments and zeroes obsolete touch-retention loss, adds
completed-cycle rewards and bounded native forward preparation. Neither shape
reward nor hours elapsed certify punching. Independent full original contact
and per-hand motion gates plus fresh final holdouts remain required.
Allocate at most 1,048,576 new diagnostic learner credits (2,163,968 -> 3,212,544)
and 72,000 formal evaluation controls (2,073,802 -> 2,145,802). First 1,024-credit
integration is included. New training collection world controls recorded apart
from formal evaluation; main remains 476,041,216. No new geometry/BC/render budget.
Do not call zero cycles in early minutes a learned impossibility result; stop
numerical faults immediately and review the full bounded history before extension.

## V14 baseline and publication recovery, 2026-10-25

[Complete 24-condition baseline](results/striking-pilot/motion-cycle-v14/README.md):
7/8 and 8/8 neutral survivors, 6/8 mirror; original complete gate fails and all
motion-cycle counts are zero. This is the unchanged source actor, not PPO failure.
New formal evaluation 11,486, cumulative 2,085,288; zero new PPO before integration.
Twenty-six focused tests pass. A Git-author configuration failure stopped only
publication, before training. Preserve failed state/log, use existing Windows Git
through the WSL launcher, and resume without replaying the baseline or rebuilding
the initial state. All consumed counters, source and optimizer remain intact.

## V14 integration and active bounded learning, 2026-10-25

Ordinary resume retained the 11,486 baseline controls and all source state. The
included 1,024-credit integration verifies finite updates, exact Adam 15,000/
38,672, 18 on-policy audit records, zero initial ratio clipping and empty native
stderr. Saved integration checkpoint d22fe0fe is diagnostic, not a boxing pass.
PPO is now active toward the first 262,144-credit independent full gate within
1,048,576 total new credits. Actual live progress supersedes the archived sample.
The first measured throughput is around 90 learner credits/s, so hours of learning
are plausible; startup/steady-state variation prevents an exact runtime promise.
No early zero-cycle conclusion, automatic extension or main adoption is made.

## V14 full outcome after computer restart, independently reviewed 2026-10-26

[Complete report](results/striking-pilot/motion-cycle-v14/README.md): all 1,048,576
new learner credits were consumed and optimized; final/fresh evaluations completed
at 2026-10-25 22:37:26 America/Sao_Paulo. No active job; this is a completed budget,
not an interrupted learner requiring restart. Final SHA be59681d retained.
All six full gates, 144 native/full trace pairs, hashes, finite states and event
windows verify; native error logs empty. Every complete original contact gate
and motion gate fails. Zero valid cycles in training or evaluation. Intermediate
pair survival reaches 24/24; final original 22/24, fresh 20/24. Original final has
zero scored events; fresh has one low-speed short-excursion contact, not a punch.
The actual latest final is retained without adoption or rollback.

New formal evaluation 67,172, cumulative 2,140,974, ceiling 2,145,802; unused 4,828
expire. Diagnostic PPO 3,212,544 at ceiling; new training collection controls
1,048,576 separately recorded. Main 476,041,216 and other compute unchanged.
No conclusion that learning is impossible follows from this finite block. Do not
blindly extend or repeat: next inspect reward scale/credit and exploration/actual
arm motion before choosing a separately justified continuation. Still missing are
learned complete legal punches, robust reciprocal exchanges and repeated full
independent plus fresh validation. All prior failures remain archived.

## Final learned-checkpoint video, 2026-10-26

Rendered all eight archived fresh-final self-play episodes in order, including
both original early terminations, from exact saved qpos/qvel. No new physics or
policy execution. Closed fists use the existing visual-only graft. All 1,565
frames decode at 25 fps (62.6 seconds); mid-episode and terminal previews inspected.
The original `fall` label includes self-leg contact and need not depict a toppled
body. Zero valid punches is displayed explicitly. Render forwards now 2,065;
formal evaluation 2,140,974 and diagnostic PPO 3,212,544 remain unchanged.
Source hashes, indices and video checksum: motion-cycle-v14/final-video.json.

## Functional scope estimate and target-coupled v15, 2026-10-26

The [approximately 40% estimate](completion-status.md) is qualitative demonstrated
functional scope, not remaining time or learning convergence. Core learned
strikes/reciprocal boxing still lack evidence. The saved-array reward audit shows
after 3 s: learner median hand speed .415 -> .556 m/s, minimum hand/target-center
distance .291 -> .347 m, legal contact fraction .002229 -> zero; replayed
preparation reward 11.68 -> 23.18 on original-seed evaluation cohorts. Saved
Gaussian std remains about .13-.15. Correlation does not isolate every cause;
this is evidence for replacing target-uncoupled preparation, not raising noise
or declaring learning impossible.

[V15 prospective protocol](striking-contact-progress-training.md) preserves actual
latest be59681d, actor/critic/Adam/RNG/norm and teacher coefficient zero. Replace
forward-air payment with bounded terminal-corrected target potential; graded
same-hand real legal contact subgoals supply credit without declaring punches.
Keep delayed complete-cycle payment and every independent original/motion gate.
Allocate at most 2,097,152 new credits (global 3,212,544 -> 5,309,696), plus at most
60,000 new evaluation controls (2,140,974 -> 2,200,974). First 1,024 integration
included. Reuse the exact v14 final full historical baseline without rerunning,
relabeling source hashes or double-counting consumed evaluation. Main and all
geometry/render/BC counters retained. This is an executed next learning
intervention toward the gap, not a claim that the gap has been closed.

## V15 verified integration and active learning, 2026-10-26

Thirty focused tests pass. The included 1,024-credit real integration produces
finite checkpoint 9da56178, actor/critic Adam 31,384/55,056, 18 passive update-audit
records with zero initial ratio clipping and empty native stderr. Graded contact
payment is exercised (1.41765); complete learner cycles remain zero. This is a
mechanical/credit-path check, not evidence of learned valid punches. No source
field or optimizer was reset. Historical baseline reuse consumes zero new
evaluation transitions. The persistent job is collecting/updating toward 524,288
new credits within the 2,097,152 allocation; inspect live progress for current
consumption. Partial milestones must not be presented as closing the boxing gap.

## First complete-cycle training counter and evidence capture, 2026-10-26

V15 progress reports the first learner complete-cycle **0 -> 1** at 404,480 new
credits (optimized lineage 3,533,824), corner 0; opponent cycles stay zero. The
previous/first log records and adjacent policy snapshots 3,531,776 / 3,532,800 /
3,533,824 are archived with hashes. This is a training-monitor signal only: the
first event's raw native window was not retained, exact hand/episode are unknown,
and no held-out full gate has confirmed it. No phase or adoption is advanced.

Keep the live 524,288-target collection running without restart or new allocation.
Add passive bounded-window event/pose/control/policy-version capture for the next
naturally launched training chunks; no retrospective video is invented. Independent
acceptance remains unchanged. Thirty-two focused tests pass, including unchanged
physics with capture on and checksum/provenance serialization. All main counters,
foundation histories and source policies remain verified. No extra physics or
geometry was consumed by this log/checkpoint review.

## V15 first deterministic gate and verified raw training cycles, 2026-10-26

The complete 524,288-credit gate (42f3d004) is independently reviewed: 7/8, 7/8,
5/8 pair survivors; 240 legacy contacts, **zero full valid cycles**. Both original
contact and motion gates fail, so no advancement. New evaluation 9,928, formal
cumulative 2,150,902 within the existing 2,200,974 ceiling. Training continues
productively toward 1,048,576 new credits within the unchanged 2,097,152 allocation.

A frozen raw-window review independently verifies **37 stochastic learner cycles**
(19/18 corners; four active-opponent cases), including real legal impulse,
component conjunctions, full recovery windows, zero applied forces, native timing,
finite state/controls and all source checkpoint versions. All windows and policy
payloads are archived; this is not deterministic generalization or a stage pass.
The first neutral/active windows are rendered at normal/quarter speed with explicit
scope labels. An occluded initial view was preserved and camera-only revision
verified. All 330 frames decode; two impact previews inspected. New render forwards
132, cumulative 2,197. No additional training/evaluation/geometry/BC allocation,
policy alteration, rollback, current-learner interruption or promotion.

## Deterministic neutral strikes and mirror regression, 2026-10-26

At the scheduled **1,048,576 new-credit** gate, checkpoint **b2a2c04c**
(`b2a2c04cd69e9ff8059647bae4ef8750033496f5edcb6ef1b560ea2fb6232df1`)
produces six complete valid cycles without policy sampling: four neutral-0 and
two neutral-1. Each role has valid cycles on surviving known seeds 7 and 23.
Independent saved-data review checks all 24 native/full trace hashes and finite
arrays, reproduces native descriptors and full motion/original contact results,
checks legal impulse/closing/advancement, and validates the exact PPO payload.
No new physics, optimizer, render or geometry controls are consumed by review.

The important failure remains: neutral pair survival is 7/8 and 7/8, while mirror
pair survival falls from 5/8 at 524,288 to **0/8**. All eight mirror episodes end
at first-fall predicates after 0.842–6.552 seconds, before any scored legal contact;
zero mirror valid cycles. Therefore these failures are not evidence of successful
knockouts. The recorded first-fall predicate does not by itself isolate the cause
of instability; changing arm motions/engagement may contribute but is not proven.
The neutral-motion success cannot replace original survival or reciprocal gates.

Training reports 171 learner cycles and zero frozen-opponent cycles at the chunk
boundary; these are sampled training counters, distinct from the six verified
deterministic evaluation events. The parent and exactly one next learner remain
alive, native error logs empty. All earlier failed gates are preserved. Full
foundation histories remain certified and unchanged. No complete v15 pass exists.

Decision: continue the predeclared chunk to **1,572,864 new credits**, ordinary
warm start from the actual latest policy/Adam, without rollback, reward/physics
changes, weakened gates or budget extension. The total v15 ceiling remains
2,097,152 new credits and 60,000 evaluation controls. Review the next full gate
for mirror recovery as well as neutral strike retention before any further
intervention; surviving neutral strikes alone do not justify adoption.

Accounting at this completed gate: new v15 PPO/collection controls **1,048,576**,
global diagnostic PPO **4,261,120**; lineage consumed/optimized **4,177,920**,
lineage world controls **3,555,328**. New gate evaluation **8,635**, cumulative v15
**18,563**, global formal **2,159,537** under 2,200,974. Main **476,041,216**, BC
**8,000**, cloned queries **6,349**, signed-distance calls **960**, renderer
forwards **2,197**, all retained. The live next-chunk counter advances separately.
Evidence: `step-1048576-independent-review.json`; reusable read-only check:
`tools/review_contact_progress_gate.py --label step-1048576`.

## Third v15 full gate: more contacts, weaker recovery, 2026-10-26

Independently reviewed **step-1572864**, exact checkpoint
`39f9e3d39b03d6164518c8418c46fa3eb4b24c92d6b01e8f254ab4daf0589e2d`.
The saved-data audit rechecks all 24 native/full trace hashes, finite arrays,
PPO payload, legal native impulse and unchanged full motion/contact conjunctions;
complete foundation histories/prerequisites remain verified. No new simulation,
optimization, geometry search or rendering is consumed by this review.

Pair survival is **6/8 neutral 0, 7/8 neutral 1, 2/8 mirror**, compared with
7/8, 7/8, 0/8 at the preceding gate. There are **two complete deterministic
neutral cycles**, one corner-0 cycle on surviving seed 31 and one corner-1 cycle
on surviving seed 23. Each neutral role now misses the two-seed motion criterion.
Mirror has zero valid cycles; two surviving episodes exchange legacy contacts.
Neither original nor motion full gates pass; consecutive complete-pass streak
remains zero. This is mixed evidence, not steady improvement or certification.

Across the complete gate, legacy event descriptors increase **271 -> 436**,
onset-speed qualifying events **209 -> 384**, pretravel qualifying events
**65 -> 198**, but return-travel qualifying events fall **80 -> 30**, and full
cycles **6 -> 2**. Counts concern individual components across the complete set,
not matched causal comparisons; they cannot be borrowed across events or hands.
These observations identify recovery consistency as a continuing bottleneck, not
an isolated cause of the survival regression or proof of eventual convergence.

Decision: retain the actual latest checkpoint/Adam and continue the already
approved final chunk toward **2,097,152 new credits**, then original and fresh
full evaluations. Do not roll back to the six-cycle checkpoint, weaken criteria,
add a recovery reward mid-block, or extend the budget now. Review all completed
policies/gates at the ceiling before deciding whether a bounded new intervention
is justified. Exactly one learner and one supervisor are live, all native-error
logs empty; training remains productive with free legs and unassisted physics.

At this completed gate: new PPO/collection **1,572,864**, global diagnostic PPO
**4,785,408**; lineage consumed/optimized **4,702,208**, lineage world controls
**4,079,616**. This gate uses **9,023** evaluation controls; v15 cumulative
**27,586**, global formal **2,168,560**, under the existing **2,200,974** ceiling.
The live final-chunk counter advances separately. Main **476,041,216**, BC
**8,000**, cloned queries **6,349**, signed-distance calls **960** and render
forwards **2,197** unchanged. Evidence: `step-1572864-independent-review.json`.

## V15 completed and next bounded opponent-distribution diagnostic, 2026-10-26

All **2,097,152 new PPO credits** are consumed and optimized. No training or
supervisor remains alive at review. Exact final SHA:
`9132ab4c2c86ed2b2958d551125b001370bf0f4e80f403490d33111ac3bb634a`.
The original final has 13 valid neutral cycles (6/7 by learner corner) and two
corner-1 mirror cycles, but pair survival **5/8, 5/8, 1/8**. Fresh final has
**23 valid neutral cycles** (13/10), spanning three surviving seeds per learner
corner, but zero mirror cycles and pair survival **6/8, 5/8, 2/8**. Neither full
original nor fresh gate passes. No distinct consecutive full passes, adoption or
boxing completion. The retained six-cycle intermediate policy is not restored.

Independent saved-data reviews verify all final/fresh native/full trajectories,
checkpoint hashes, finite arrays and unchanged physical/motion criteria. Of 13
failed mirror episodes across original/fresh, 11 terminate without scored legal
contact. Terminal flags total seven low-pelvis and six self-leg contacts, zero
tilt flags for the fallen bodies. These are measured termination predicates,
not an isolated causal explanation, and not evidence of successful knockouts.
Training ends with 784 sampled learner cycles; this counter is not a held-out
success measure. All native-error logs are empty.

Completed accounting: global diagnostic PPO **5,309,696**; lineage consumed and
optimized **5,226,496**, lineage collection world controls **4,603,904**. New v15
formal evaluation **42,364**, global **2,183,338**. The unused 17,636 of v15's
60,000 evaluation allowance expire with this completed job, not reusable credits.
Main **476,041,216**, BC **8,000**, cloned queries **6,349**, signed distances
**960**, visualization forwards **2,197** retained. All six gates, including the
zero-debit historical baseline and fresh final, remain archived.

Do not extend unchanged PPO at the ceiling: v15 established neutral striking,
but stability and reciprocal strikes remain unresolved. The training active
opponent is still the archived policy with fingerprint **158d7b3d**, frozen long
before v15, while mirror evaluation uses two current actors. Next isolate this
opponent-distribution difference with **at most 8,000 new formal controls**,
zero PPO/BC/render/search: final v15 versus that exact frozen training opponent,
each learner corner, the same four fresh seeds and both swaps (16 episodes,
10 seconds each maximum). Reuse existing final-vs-final fresh records as the
comparison arm, with zero repeated controls. Global formal ceiling for this new
diagnostic **2,191,338** from 2,183,338. This is a separately justified bounded
allocation, not promotion evaluation or a gate substitute. Preserve free bodies,
full action authority, unassisted physics, collision and cycle criteria.

Verify the opposing actor/norm fingerprint equals the frozen payload inside the
actual final checkpoint before collecting. Use global locks and the hidden
persistent WSL launcher; durable attempted-control progress includes failures.
Live diagnostic output: `/home/pedro/boxing-runs/striking-v15-opponent-gap`.
Review both physical roles and saved native cycles before choosing a training
change; better performance against the frozen opponent would support an exposure
mismatch hypothesis, not prove that changing the opponent alone solves balance.
No automatic PPO extension is authorized by this diagnostic result.

## Frozen-opponent result and prospective navigation isolation, 2026-10-26

The opponent-gap diagnostic completed **5,468 new formal controls**, zero PPO.
Independent archive review verifies all 16 episodes/native/full hashes and exact
opponent actor/norm fingerprint. Final-v15 learner corner 0 has **3/8 pair
survivors, five learner falls, zero opponent falls, zero learner valid cycles**;
corner 1 has **4/8 survivors, three learner falls, one opponent fall, one valid
learner cycle on surviving seed 34000005**. Reused current/current fresh mirror
has 2/8 survivors and zero cycles. Thus the frozen/current exposure difference
exists but does not by itself explain or solve the poor active-opponent balance.
The learner also falls against the actual frozen opponent used in collection.
No main adoption, promotion, PPO restart or policy rollback.

Formal cumulative **2,188,806**. The unused 2,532-control diagnostic allowance
expires. Next isolate the learned navigation offsets with **at most 4,000 new
formal controls**, ceiling **2,192,806**, zero PPO: final-v15 mirror on the same
four fresh seeds and both swaps, zeroing only the three high-level command
offsets for both policies. Retain the normal automatic approach, learned motor,
all fourteen arm residuals, free roots/legs, physical contacts and unchanged
measurements. Reuse the existing unmasked mirror arm; repeat zero baseline
controls. This diagnostic modifies the action interface and **cannot pass a
promotion gate**, even if it survives. It tests whether learned command offsets
materially contribute to the active-pair instability, not whether the motor or
arms alone are sufficient. Both actors are masked together, so corner-specific
causes and interactions remain unresolved. Inspect original terminal flags and
valid cycles, not survival alone, before selecting a training intervention.

Launch through the persistent hidden WSL host with global/output locks and
durable attempted controls. Output:
`/home/pedro/boxing-runs/striking-v15-navigation-ablation/state.json`.
Do not start another job while this diagnostic is active. Main/BC/geometry/render
and consumed PPO counters stay unchanged. No open-ended training extension.

## Deterministic neutral bout videos, 2026-10-27

Render two previously reviewed, complete fresh deterministic 10-second episodes:
neutral learner corner 0 / seed 34000004 / unswapped (six valid cycles) and
corner 1 / seed 34000006 / unswapped (four). These are selected examples, not an
unbiased performance estimate; all evaluation outcomes are retained in the full
fresh gate. At 25 fps, saved 50 Hz control states are sampled every other state,
250 frames per episode. Explicitly label deterministic neutral practice and the
role; the neutral partner is still a free body. Model/data are separate and only
mj_forward is used to render stored positions and velocities: no actor execution,
physics stepping, learning or formal evaluation controls. Allocate at most 500
additional render forwards, cumulative **2,697**, from 2,197. Verify trace and
checkpoint hashes, recompute event criteria, decode all frames, inspect visual
previews, and preserve source and per-video metadata. These examples do not
certify active-opponent stability or reciprocal boxing.

Both saved-state renders completed: **250 decoded frames each**, 25 fps, 10 seconds
per video, 500 render forwards in total; cumulative visualization count **2,697**.
Impact previews were visually inspected. The files are
`fights/striking-pilot/v15-neutral-corner-0-fresh-bout.mp4` and
`fights/striking-pilot/v15-neutral-corner-1-fresh-bout.mp4`. The evaluation
checkpoint, event measurements, both source-trace checksums, output hashes, and
exact selected episodes are in `deterministic-neutral-videos.json`. Both are
selected surviving examples, not full gate performance; visual contacts alone do
not replace the independent native-motion criteria.

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

## V16 completed: neutral strike gains, fresh mirror instability, 2026-10-28

V16 consumed/optimized its complete **2,097,152 new PPO credits**; its final
checkpoint SHA-256 is `5bbf6a38aa237633bf1f84893587dd1c369d9bbe8849c171b5735f9d6920a8b5`.
Independent saved-data reviews verify every original/fresh native and full trace,
checkpoint hashes, finite arrays, scored native impulse, complete-cycle criteria
and foundation histories. No native error or overflow was reported.

The original final records **38 neutral cycles and nine self-play cycles**,
with reciprocal cycles on three surviving known seeds; its original pair survival
is **5/8, 6/8, 3/8**. Fresh final records **42 neutral cycles and four self-play
cycles**, with reciprocal cycles in one surviving episode/seed; pair survival is
**3/8, 7/8, 1/8**. Neither original nor fresh full gate passes and there are no
two consecutive passes. V16 demonstrates learned strikes against neutral and
active opponents, while reliable balance and reciprocal boxing remain incomplete.
The fresh and original seed sets differ, so their scores are not a matched causal
comparison, and no weak gate, promotion or rollback is authorized.

All seven fresh mirror failures occur before a scored legal contact; three occur
before any recorded inter-body contact, while four follow other physical contact.
Their fallen-body terminal flags are **two low-pelvis and five self-leg contacts**,
zero tilt. The original mirror has five failures before scoring, including two
before any inter-body contact. Thus contact incentives alone cannot explain all
failures; termination predicates do not identify the causal mechanism. Earlier
zero-navigation-offset isolation also failed 5/8 pairs, and the old posture-only
reward was previously rejected. Preserve the actual final actor/Adam and avoid an
unchanged blind extension at the ceiling.

Total diagnostic PPO **7,406,848**; lineage **7,323,648 consumed/optimized**,
collection **6,701,056**. V16 formal **39,254**, global **2,230,238**. The unused
20,746 v16 evaluation controls expire and do not roll forward. Main **476,041,216**,
BC **8,000**, cloned queries **6,349**, signed distances **960**, render forwards
**2,947** retained. All prior complete gates, including failed and intermediate
reciprocal examples, remain archived.

Next isolate the learned arm residuals: on the same four fresh seeds and swaps,
run an eight-episode deterministic mirror diagnostic with both agents' 14 arm
residuals set to zero. Keep learned navigation commands, motor, free bodies,
collision and termination physics. At most **4,000 new formal controls** from
2,230,238, global ceiling **2,234,238**; zero PPO, BC, render or geometry search.
Reuse saved unmasked fresh mirror as the comparison arm, with no repeated controls.
The masked arm cannot certify punches or promote a checkpoint; removing both
actors' arms changes their behavior and cannot identify one actor's responsibility.
This test asks whether learned upper-body residuals contribute to the early
stability deficit, not whether arms or contacts are the only possible cause.

Run with the hidden persistent WSL host and global/output locks; attempted
controls remain durable on failure. Output:
`/home/pedro/boxing-runs/striking-v16-arm-ablation/state.json`. Review both
unchanged reset sets, terminal and contact evidence before choosing a bounded
training intervention. Do not run a duplicate process or extend PPO based on
an unfinished diagnostic.

## Arm ablation independently reviewed; bounded half-arm diagnostic, 2026-10-28

On identical fresh seeds and swaps, unmasked v16 final mirror survives **1/8**;
with both actors' learned arm residuals zeroed it survives **8/8**. All 16
native/full traces, finite states and original event criteria were independently
recomputed. The zero-arm diagnostic has **zero valid cycles**, so it does not
establish punching capability, and it cannot pass or replace any physical gate.
It strongly implicates learned upper-body residuals in these observed failures;
two actors change together, so no single-joint or single-actor cause is isolated.
The unmasked arm has seven failures (two low pelvis, five self-leg contact), all
before scoring. Correction: **three**, not four, have no recorded inter-body
contact; four follow other contact. Earlier wording and the user-facing count
were incorrect, and the exact saved evidence is retained.

This completed diagnostic consumed **4,000** formal controls; global formal is
**2,234,238**. To test whether some punching authority can coexist with balance,
prospectively evaluate **50% of both actors' learned arm residual outputs** on
the same eight fresh mirror episodes, no other action or physics changes.
Allocate at most **4,000 new formal controls**, global ceiling **2,238,238**;
zero PPO/BC/search/render. Reuse the already-archived 0% and 100% conditions
without repeating them. A scaled-action result is diagnostic, cannot pass a
complete gate, and does not by itself justify a permanently limited arm policy.
Track actual valid strikes and falls rather than survival alone. Use a hidden WSL
host, global/output locks, durable attempted counters and exact source hash.
Output: `/home/pedro/boxing-runs/striking-v16-half-arm/state.json`.
After review, decide the next bounded training intervention; retain v16 final
actor, optimizer and all prior consumed steps, and preserve full unassisted
physical gates. No unchanged or unlimited PPO extension is implied.

Focused evaluator, gate-rejection and rebalance tests: **8 passed**. The first
run exposed an overly broad test assertion on a 32-action interface; correcting
it confirmed only channels 3 through 16 are scaled and other channels unchanged.

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

## Current-model CUDA recovery and bounded comparisons, 2026-10-29

The user authorized reviewing the full project and taking the necessary steps
toward actual boxing without additional permission requests. Source/state review
confirmed v17 had finished and no training process was active. Its final fresh
pair survival remains 5/8, 7/8, 3/8 with 33 neutral cycles and five mirror cycles,
reciprocal in one surviving fresh episode. Historical contact certifications do
not certify complete punches. No unchanged CPU PPO extension or rollback.

Implemented an experimental current-model CUDA physics primitive, batched 103/148
input construction with the inherited engagement profile, and a reference adapter
that preserves native scoring/trace code while advancing GPU physics. Four
asymmetric observation tests match native inputs. The native production step is
unchanged behind a narrow backend hook. Experimental engine/solver reports are
explicitly rejected by the native promotion reviewer. The 41-test focused suite
passed; after adding explicit legacy timestep handling, eight relevant tests passed
with one dependency deprecation warning. Two saved impact poses verify GPU-to-native
force export; a pinned-memory prefetch version also matches direct export exactly.

Nine original-limit/iteration counterfactuals and nine line-search counterfactuals
replay identical controls on both engines for 0.5 s per case. At 30/20, saved impact
windows agree to approximately 6e-6 qpos and 6.2e-4 qvel without solver flags; random
reset drift remains (maximum ankle difference about 0.039 rad). Raising iterations
alone did not resolve line-search limits. This is a measured numerical sensitivity,
not permission to change training physics or weaken native gates. All raw arrays,
source/checksum evidence and the independent saved-array review are retained.

A 45-control current-model prototype includes the actual v17 actor, certified
frozen motor, CPU inference, GPU physics, observations/transfers and per-substep
overflow reads. It measures about 2,239/7,260/14,368 world controls/s at 32/128/512
worlds. These are not PPO training rates: cycle rewards/history, random autoresets
and updates remain to be integrated/validated. The original solver flags are
retained; only repetitive solver printing is disabled in timing, not checks.

The initial dynamics attempt failed an atomic Windows/WSL repository JSON rename;
80 native/GPU attempted substeps are preserved separately. Durable live writes now
use the WSL filesystem. The completed preflight set retains 4,580 native experimental
substeps and 308,109 GPU experimental world substeps, including that failed attempt
and benchmark warmups. Historical plus new GPU world forwards total 1,382. Main
476,041,216, diagnostic PPO 9,504,000 and formal 2,271,001 are the retained before
counters; preflights consumed no new formal controls or learner updates.

The [prospective protocol](gpu-training-recovery.md) declares at most 36,000 new
formal controls (ceiling 2,307,001) for native 30/20, CUDA 3/5 and CUDA 30/20 under
all three unchanged role profiles, exact v17 final and existing fresh seeds.
Historical native 3/5 is reused without debit. The hidden persistent comparison is
running. At the interim review, native 30/20 survival is 6/8, 4/8, 3/8; no complete
pass. CUDA 3/5 neutral-0 matches native 3/5 aggregate 5/8 survival and 17 cycles,
while neutral-1 has 4/8 survival and nine cycles. Full comparison remains pending.

A persistent coordinator waits without running physics, then independently reviews
all complete profiles and publishes owned evidence before a separate legacy
measurement. Retained Warp6/7 200,212,480-step policies will be measured in their
reconstructed own engine at original 0.75 m and separate 2 m spawns on four seeds.
At most 7,680 additional formal controls, with its before counter set to the
completed comparison's actual total; unused previous allowance expires. Explicit
4 ms contact-duration measurement preserves the common impulse/travel/return
criteria. Legacy damage termination and different self-leg geometry limit survival
comparability; no incompatible checkpoint transplant or automatic adoption.

Next decision depends on both complete comparisons. Validate batched scorecard,
motion-cycle rewards/history, full resets and PPO before claiming GPU training
throughput; retain native gates. Then choose a justified bounded learning change
to improve stable repeated reciprocal striking, rather than assuming the latest
frozen-motor architecture or an old policy is inherently superior. Broader fresh
conditions and longer rounds still remain before project completion. Existing
monitoring schedule, checkpoint lineage and all consumed accounting are retained.

The comparison later completed all six native-30/20 and CUDA-3/5 profiles:
CUDA-3/5 mirror has 5/8 pair survivors and valid cycles 3/5, reciprocal on two
surviving fresh seeds, still below full survival criteria. At 16,267 new formal
controls (global 2,287,268), archive overwrite/copy metadata failed during Windows
Git access to an existing trace. Source hashes and all completed profile arrays
were verified intact; this was publication I/O, not a physical evaluation failure.
Added an archive helper that skips identical files, atomically copies changed
results and retries transient Windows access failures. A fault-injection test
passes. Per-profile archive failures no longer stop productive physics. The
explicit resume retains all six completed profiles/counters and runs only the
three remaining CUDA-30/20 profiles. The coordinator's initial early restart saw
the old failed state during verification and stopped without running legacy
physics; it was resumed after the physical supervisor was confirmed running.
Both failure/recovery records are retained. No checkpoint rollback, repeated
evaluation debit, weaker gate or budget extension.

Before the queued legacy comparison, Git/config inspection identified a historical
compatibility trap: Warp6 uses 128 inputs and predates damage/stun; commit b123d8e
introduced those mechanics and the 131-input state used by Warp7. The current
constructor defaults would retroactively add them to Warp6. The diagnostic now
explicitly disables Warp6 stun and uses a damage cap unreachable within 480 controls,
while retaining Warp7's saved cap 3 / stun 10. This restores the recorded historical
contract and leaves all common punch thresholds intact. The focused contract test
passes; no legacy physical controls had yet been consumed.

The frozen-policy backend comparison is now complete and independently reviewed:
new formal controls 24,149, total 2,295,150; unused 11,851 expire. Native 30/20
pair survival is 6/8, 4/8, 3/8; CUDA 3/5 is 5/8, 4/8, 5/8; CUDA 30/20 is 6/8,
4/8, 3/8. Mirror valid cycles are respectively 2/4, 3/5 and 0/0. Every complete
criterion set fails. CUDA 3/5 has reciprocal cycles on two surviving fresh seed
conditions but insufficient survival; convergence alone does not solve learned
boxing. No backend/solver adoption or new PPO follows from these small samples.
The coordinator started the declared legacy measurement with formal before
2,295,150 and separate ceiling 2,302,830. An automatic push raced a foreground
push and got a stale-ref rejection; online verification shows the successful
foreground descendant already contains comparison commit 235630b. No data or
commit loss, rollback, force push, altered gate or budget extension.

The retained legacy comparison completed sixteen episodes and an independent
hash/array/event/counter review. Warp6/Warp7 complete 1/4 and 3/4 original-range
rounds without common-posture failure. From 2 m, neither completes a round in four
seeds each. Zero complete straight cycles occur across all sixteen cases. The
236 independent fist impacts include 229 below the forward cosine reference,
133 below 12 cm pre-travel and 150 below/missing 6 cm return; these overlapping
component failures do not rule out hooks or all other punch types. A separately
recorded, non-stepping posed-model audit verifies that both models' torso IMU
and body forward frames agree, ruling out a fixed frame rotation as explanation.
These are reconstructed own-engine observations, not a current-model full gate
or proof that an architecture is globally superior. All raw traces and reviews
are archived and published by the coordinator (commit 8dbd5d1).

Legacy formal use is 3,359, total **2,298,509**; unused 4,321 expire. The full
reviewed source-v17 checkpoint is retained, with lineage consumed/optimized
9,420,800, world 8,798,208, actor/critic Adam 129,672/153,344. Global diagnostic
PPO remains 9,504,000, main 476,041,216. No active physical job remains. The next
action is implementing complete batched GPU scoring, motion/reward and per-world
reset parity; only after validation is a single 1,024-credit, original-profile
warm-start integration prospectively allowed (global PPO ceiling 9,505,024).
This engineering test is not a convergence judgment or long allocation. See
`gpu-batched-training-next.md`. Longer productive GPU learning remains downstream
of that verified integration; no unchanged CPU restart or rollback.

The existing heartbeat configuration was actually PAUSED when inspected, despite
earlier context reporting it active. It was reactivated under the user's latest
autonomous-continuation request, preserving its name, target and 180-minute
schedule and meaningful-change-only notifications. The prompt now explicitly
requires implementing the remaining GPU mechanics rather than stopping at an
idle-training report. Official OpenAI automation guidance was checked with the
OpenAI Docs skill; actual saved configuration/status was verified locally.

## Complete batched mechanics and GPU PPO integration, 2026-10-29

[Mechanics review](results/striking-pilot/batched-gpu-mechanics-v1/README.md):
implemented unordered per-hand legal contacts, native-equivalent relative
kinematics/impact windows, cycle histories, all v17 rewards, first-fall snapshots
and full independent randomized per-world resets. The failed first comparison
retains 22 GPU substeps. Comparing original native control/measurement/reward on
the same actual CUDA trajectory then passed five checks (zero observation error),
including bitwise survivor invariance and both immutable positive-cycle windows.
Thirty-five focused/regression tests pass. This is semantic parity on identical
states, not proof of independent native/CUDA dynamics equivalence.

The complete frozen collector, including rewards, evidence and resets, measures
85.5 / 601.5 world controls/s at 8 / 128 worlds, excluding Gaussian sampling,
critic/PPO and formal gates. The bare-path thousands/s figure did not include
these costs. One positive learner complete-cycle history against an active
opponent was captured and independently classified; 36/128 worlds fall/reset in
that bounded benchmark. Frozen diagnostics are not new learned progress. Total
development compute is 174,941 GPU world substeps, including the failed attempt.

[The conditional integration](results/striking-pilot/gpu-batched-integration-v1/README.md)
used exactly 1,024 new PPO credits and 10,248 separately counted GPU substeps.
Every source leaf was preserved before updates; final payload/source/RNG/actions/
log probabilities/optimizer counts and metadata independently verify. Final SHA
`be1b5a8d64d88ac5ae2a266f1b975dc79ac39f90f95df1082d438fc86d5327e9`
is the latest experimental GPU prefix: consumed/optimized 9,421,824, world
8,799,232, Adam 129,688/153,360. Global diagnostic PPO **9,505,024** is at ceiling;
main **476,041,216**, formal **2,298,509**, prior BC/cloning/geometry/render counters
and all expired allowances remain retained. No new renderer calls or formal gate.

Collection took 11.588 s and PPO 0.735 s including their first JIT. Only original
solver-limit bits 512/1024 occur; there is no capacity/float/nonfinite failure.
The hidden persistent job stopped for the mandated review, with no duplicate or
unchanged CPU PPO restart. No boxing gate passes. Next optimize complete collector
overhead and test explicit larger-batch/update scaling before declaring a bounded
longer GPU run. Preserve the integration prefix on ordinary continuation and the
native physical gates for advancement. Monitoring frequency remains unchanged.

## Optimized collector, scaled replay and productive GPU continuation

2026-10-29: [the scaling protocol](gpu-collection-scaling.md) measured a complete
frozen-collector improvement from 620.6 to 835.1 world controls/s in a matched
128-world profile. Persistent pinned buffers, grouped readback and cached
pre-substep snapshots remove redundant transfers. No dynamics, action, reward,
first-fall or gate change. Kinematics/history/rewards remain on CPU NumPy; this
is not a claim of fully fused GPU rewards or historical thousands/s recovery.

The test-harness path-variable failure is retained (eleven GPU substeps), followed
by a separate successful five-check semantic regression. Development compute is
164,810 GPU world substeps, plus separately counted reset/boundary forwards and
three exact-readback test forwards. Thirty-six focused tests and the additional
GPU readback/force/snapshot-lifetime test pass. No thresholds were relaxed.

The [scaled integration](results/striking-pilot/gpu-batched-scaling-v1/README.md)
uses 16,384 credits, 256 updates per optimizer and preserves the actual latest
GPU source. All Gaussian samples and all optimizer updates/moments, normalizer,
RNG and samplers reproduce exactly from its stored rollout. Final SHA
`6432029087cb00e73e39bee3a974e07f47f598771bae05c519e1bcad616c2103`,
consumed/optimized 9,438,208, world 8,815,616, Adam 129,944/153,616.
Global diagnostic PPO is 9,521,408 before the next job. Collection/update timing
17.299/1.537 seconds gives about 870 credits/s excluding startup/checkpoint/native
evaluation. No formal gate or learned-behavior improvement follows from integration.

The separately declared [bounded GPU learning protocol](gpu-bounded-learning.md)
now permits 1,048,576 credits (global ceiling 10,569,984) and 72,000 new native
controls (formal ceiling 2,370,509). The hidden persistent job at
`/home/pedro/boxing-runs/gpu-learning-v1/state.json` starts with a new native source
baseline, continues ordinary warm starts on CUDA, evaluates every 262,144 credits
and finishes with a fresh group. Exact counters/processes/partial evaluations in
live state supersede these launch values. Checkpoints/evidence publish at gates.
No budget extension, main adoption, rollback, weaker gate or schedule change.
Hours of learning may still be necessary; a small integration is not convergence
evidence. Stable reciprocal fighting and broader/longer fresh rounds remain open.

## Completed first scaled GPU interval and audited next diagnostic

Reviewed 2026-10-29 local / 2026-10-30 UTC. The full 1,048,576-credit interval
completes with six failing complete native sets. Known neutral cycles 22 to 36;
survival 5/4/3 to 5/5/3. One intermediate has reciprocal valid strikes on one
surviving seed, below the unchanged gate; it is not restored. Final fresh has
27 neutral cycles and mirror 5/1, survival 4/6/3, and no valid reciprocal surviving
seed. This is not robust boxing or a proof that further learning cannot help.

Independent review validates 65 checkpoint hashes, learned-state/Adam prefixes,
144 native/full trajectory pairs and 2,900 training windows (1.7 GB of local
event evidence), including no applied support forces in captures. Final SHA
`639ccec99328b034d7d6c9f1ba08a7305a0edc3e67025aa01cedcc06cf979052`,
lineage consumed/optimized 10,486,784, world 9,864,192, Adam 146,328/170,000.
Global PPO 10,569,984, formal 2,340,326, main 476,041,216; unused 30,183 controls
expire. GPU/forward/discarded/warm-up ledgers are retained; no renderer or promotion.

Collection+PPO rate declines 510/433/395/371 by quarter. Full-history JSON and
native progress rescans occur at every substep in the old supervisor. Stored-data
accounting benchmark: 128 old writes/scans 0.924 s, compact durable writes 0.155 s;
254,077 to 1,356 bytes/write. A small durable frontier plus boundary audit flushes
retains history and tests interrupted-frontier/identity safety. This CPU/disk-only
comparison does not establish a sixfold overall GPU training speedup.

The [follow-up protocol](gpu-throughput-and-transfer-followup.md) prospectively
allows at most 24,000 new CUDA diagnostic controls, formal ceiling 2,364,326,
zero PPO/rendering, using actual source/final, original 3/5 limits and matched
known seeds/placements/roles. Original native observations/contact/motion criteria
remain authoritative; historical native data are reused, not resimulated. New
per-kernel/reset-forward ledgers retain even failed initialization/steps. Inspect
the hidden persistent comparison before any new launch and review every arm
before a bounded learning decision. The existing automation prompt was brought
up to date without changing its three-hour interval or notification intent.
