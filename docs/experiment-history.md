# Boxing project trajectory and experiment index

This is the navigation record for what worked, what failed, what remains
conditional and why the next experiment was authorized. Detailed protocols,
full physical results, limitations, checkpoints and accounting remain in the
linked reports. This index does not replace those records or the live state.

Last consolidated review: **2026-10-24**, through completed consolidation, limited
ten-second readiness verification, reviewed broader/pool audits, failed jab-cycle
prototype and exact full-body/workspace review. Reliable general boxing is **not yet
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
