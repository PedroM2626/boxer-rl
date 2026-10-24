# Boxing project trajectory and experiment index

This is the navigation record for what worked, what failed, what remains
conditional and why the next experiment was authorized. Detailed protocols,
full physical results, limitations, checkpoints and accounting remain in the
linked reports. This index does not replace those records or the live state.

Last consolidated review: **2026-10-24**, through completed consolidation, limited
ten-second readiness verification, reviewed broader frozen audit and planned
opponent-pool integration. Reliable general boxing is **not yet
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
| [Opponent-pool integration and comparison](striking-opponent-pool-comparison.md) | Version 13 and 41 focused tests verify immutable independently normalized opponents and exact credits. Both native integration blocks pass. Pool final completes +16,384 but fails both original full screens (8/7/7, 6/8/8) and fresh gate (8/6/5); control remains active within the declared allocation. | No demonstrated pool improvement or promotion; await both actual finals before deciding. A read-only monitor backend omission was recovered on CPU without training intervention or new controls; evidence is retained. |

## Where to check current facts

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
