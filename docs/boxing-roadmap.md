# Learned boxing completion roadmap

The [trajectory and experiment index](experiment-history.md) records reviewed
successes, failed approaches and the documentation protocol for every continuation.

Current status: the foundation is certified; paired physical boxing exists,
but reliable reciprocal combat is not certified. Two distinct consecutive
original ten-second full screens passed earlier; independent reset generalization
still fails across all four new full groups. The complete cross-policy audit is
reviewed. The completed [motor-floor audit](striking-motor-floor-audit.md) found
31/32 motor-only pair survival and additional head-dependent failures. A
[bounded training-only posture-cost comparison](striking-posture-cost-comparison.md)
is completed and its extra cost rejected. Those posture-comparison full screens fail; both associated
profile families retain zero pass streaks. A read-only
[objective audit](striking-objective-audit.md) measured terminal reward and matched
action-mode gaps. A bounded [on-policy sampling comparison](striking-sampling-comparison.md)
completed without a sustained benefit from half Gaussian standard deviation.
A bounded [actor KL-guard comparison](striking-kl-guard-comparison.md) completed without robust stability. A bounded [passive failure replay](striking-kl-failure-trace.md)
completed with exact source/endpoint reproduction. A bounded
[GAE trace-length comparison](striking-gae-comparison.md) completed with 8/8 fresh mirror survival and six reciprocal episodes; neutral
profiles still fail. One [bounded ordinary continuation](striking-gae-continuation.md)
completed with one complete pass followed by regression. A bounded
[frozen gradient/critic assay](striking-gae-gradient-audit.md) reviewed
the source, passing checkpoint and latest final without new PPO or adoption.
It did not establish retention dominance or larger final critic error. One
[bounded unchanged consolidation](striking-gae-consolidation.md) completed with two distinct consecutive original complete passes and a complete
fresh 29M pass. The [readiness scope](results/striking-pilot/gae-readiness-certification/README.md)
is limited; a [broader frozen audit](striking-gae-generalization.md) completed
new reset/opponent checks. Neutral profiles pass, but active combat fails three
of four full groups. The explicit pool integration now passes 41 focused tests;
a [bounded pool-versus-fixed comparison](results/striking-pilot/pool-2064384/README.md)
completed with both final full gates failing and no demonstrated benefit.
The user's video observation triggered a [punch-quality audit](striking-punch-quality-audit.md):
current contact counters do not certify punch mechanics. No new PPO job is active.
The [native audit](results/striking-pilot/punch-quality-2080768/README.md) now measures
eight selected episodes exactly: 0/19 events meet both old component references,
with only 5.3–9.1 cm pre-window hand travel in the published clip. Next implement
separate strike-motion evaluation and calibrate positive demonstrations before
declaring a versioned training objective/budget. The version-1 evidence interface
and [complete demo audit](results/striking-pilot/demo-motion-2080768/README.md) are
now implemented/reviewed: fifteen retained targets also fail both references.
The [bounded explicit jab-cycle prototype](results/striking-pilot/jab-cycle-v1/README.md)
failed all sixteen conditions under the same motor/authority. It is not a learned
policy or a usable positive target. The [exact full-body/workspace review](results/striking-pilot/jab-failure-workspace-v1/README.md)
reproduces all sixteen failures: only the acting body terminates. Static guard
endpoints match, while early full-strike endpoints do not match in two local
searches; this does not establish dynamic or global infeasibility. Next compare
motor-only and guard-only behavior under a prospective bounded protocol to
isolate posture versus cycle dynamics before redesign. That
[matched comparison](results/striking-pilot/guard-control-v1/README.md) now finds
8/8 motor-only versus 0/8 static-guard survival, so extension/retraction is
unnecessary to reproduce failure in these cases. A
[reviewed smooth-entry probe](results/striking-pilot/guard-ramp-v1/README.md) also
fails 0/8 despite improved hand tracking. Entry smoothing alone is insufficient.
Source and saved-array inspection confirm the learned motor already carries a
compact guard. The [reviewed motor-guard cycle v2](results/striking-pilot/motor-guard-cycle-v2/README.md)
preserves it between extensions and survives 8/8, but produces zero physical
impacts or qualifying punches. Retain this stable scripted control and next
compare extension timing/terminal-target dwell under a prospective bounded
protocol. The [reviewed terminal dwell](results/striking-pilot/terminal-dwell-v3/README.md)
now retains 8/8 survival but no strikes. The
[terminal workspace review](results/striking-pilot/terminal-workspace-v1/README.md)
finds persistent endpoint errors in two local searches; this is not proof of
collision-surface infeasibility. Next measure actual legal surface gaps and
blocking contacts before another bounded target/authority intervention. The
[signed-surface audit](results/striking-pilot/surface-audit-v1/README.md) now finds
recorded blocking intersections in 20/24 poses. The
[reviewed 6/12 cm bow comparison](results/striking-pilot/arc-path-v4/README.md)
has 7/8 and 8/8 pair survival, no commanded-left target contact and no candidate
punch. Next test angle/range adjustment through the **existing lateral movement
command** with the arm path fixed, before increasing authority or unfreezing the
motor. The [reviewed comparison](results/striking-pilot/lateral-command-v5/README.md) finds 8/8 versus 2/8 survivors and no commanded-left contact or candidate punch. The [motor-only intervention](results/striking-pilot/lateral-isolation-v6/README.md) removes all six falls (8/8 survivors), with no scored impacts. Next test temporal separation of movement and extension under a separately declared allocation.
No positive training target or general boxing certification follows.
These references do not define every
punch type or replace failed gates.

The immediate prerequisite is distinguishing moving touches from purposeful
punches using passive kinematics, impulse timing, arm travel and recovery, with
unchanged historical gates retained. Older code already contains stronger motion
checks; audit that prior work rather than assuming it never existed.

1. Consolidate stable own strikes in both corners against a neutral partner.
   Preserve the existing independent complete physical gates and require two
   distinct consecutive full passes before promotion. Offline imitation error,
   lower KL and individual successful clips are insufficient.
2. Consolidate reciprocal exchanges: approach from separated spawns, initiate
   real strikes, recover guard and continue after receiving contact. Evaluate
   across saved opponents, both corners and fresh conditions to detect brittle
   exploitation. The completed archived-opponent comparison does not establish
   robust benefit; broader combat and punch quality remain unverified.
3. Investigate coordination limits if measured falls justify it. The current
   learned motor is frozen and the boxing head controls commands/arm residuals.
   Any motor adaptation needs a bounded controlled comparison preserving the
   learned base and unassisted free physics. Earlier body-authority experiments
   did not establish robust benefit; do not repeat them blindly.
4. Validate longer rounds and unseen distances/orientations/opponents. Repeated
   physically valid punches by both sides, sustained movement/guard recovery,
   reproducibility and stability are required for completion. Specify broader
   acceptance thresholds before using these new evaluation sets for a decision;
   they must not retroactively replace the current failed gates.

The present ten-second diagnostic gates are prerequisites, not a complete boxing
certification. No fixed large step count guarantees learning. Every extension
must cite measured physical evidence, bounded compute and a stop/review rule.
