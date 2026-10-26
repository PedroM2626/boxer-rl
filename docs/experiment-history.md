# Boxing project trajectory and experiment index

This is the navigation record for what worked, what failed, what remains
conditional and why the next experiment was authorized. Detailed protocols,
full physical results, limitations, checkpoints and accounting remain in the
linked reports. This index does not replace those records or the live state.

Last consolidated review: **2026-10-25**, through completed consolidation, limited
ten-second readiness, reviewed broader/pool audits, guard/cycle diagnostics,
signed collision surfaces and the failed lateral-bow comparison. Reliable general boxing is **not yet
certified**. The foundation is certified; paired physical exchanges exist.

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
