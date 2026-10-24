# Contact events do not establish learned boxing punches

**Completed implementation:** the optional native observer now passes fifteen
focused checks. The [separate bounded replay audit](results/striking-pilot/punch-quality-2080768/README.md)
completed all 4,000 controls and exactly reproduced the original physical trial
fields. No selected contact event meets both old speed/direction references.
No training extension is granted. The static findings and zero-compute scope
below remain the record of the initial inspection.

## User observation and verified scope, 2026-10-24

The user reports that repository videos show touching rather than punching.
Inspection of eight frames (2 through 9 seconds) from
`fights/striking-pilot/gae-readiness-2064384.mp4` shows close engagement with short
arm movements and hand contact. This visual check supports the concern; sparse
frames do not measure peak fist velocity or classify every individual event.
The displayed simulation counters count legacy qualifying contact events.
Previous wording treating those counters as proof of learned punches was too
strong. Neither a selected clip nor the limited readiness gates certify punch
mechanics or genuine reciprocal boxing.

## Concrete mismatch in the current objective

`boxer/impact_scorecard.py` requires an actual fist/head-or-torso contact, both
agents upright, onset closing speed at least .3 m/s relative to the target,
onset fist advancement at least .2 m/s relative to the striker's translating and
rotating torso, and .5 N s integrated within the first 80 ms. Separation/rearming
and cooldown prevent scoring an indefinitely motionless sustained contact.
These checks are useful, but do not distinguish all brief moving touches from
recognizable punches. They do not measure preparation, arm travel, elbow
extension, a direction cone relative to the striker, post-contact retraction or
guard recovery. A threshold-qualified contact is not a verified punch cycle.

A synthetic regression example supplies .31 m/s closing, .21 m/s advancement,
and .0128 N s per 2 ms substep for 40 substeps. The scorecard counts one event
(.512 N s). At a constant .21 m/s projection, 80 ms corresponds to only 1.68 cm
of advancement. This is a contract counterexample, not a simulated or measured
robot punch, nor an estimate of the forces in the published video.

The earlier `boxer/env.py` already contains richer `_committed` and
`_forward_drive` checks, including a default 1 m/s torso-relative speed onset
(5 m/s reference), target alignment and forward motion relative to the boxer.
Its comments explicitly describe past failures where leaning, arm lifting or
repeated chest contact inflated punch counts. The newer paired striking pilot
uses a different impact-only scorecard and did not retain every older form
criterion. This is an objective/interface difference, not an inherent consequence
of free legs or unassisted physics. Defaults in the old implementation do not
prove that every historical video used identical settings or mastered boxing.

The current reward also pays engagement/proximity and penalizes falls and arm
residual magnitude. Together with a permissive contact definition, that can
favor cautious short movements. This is a plausible optimization incentive;
the visual/code audit does not isolate it as the cause of every learned behavior.
More opponent diversity alone does not fix this objective mismatch.

## Continuation decision

Freeze both actual finals from the completed
[pool comparison](results/striking-pilot/pool-2064384/README.md). Do not launch an
unchanged PPO extension, roll back the learner, tighten thresholds arbitrarily or
rewrite historical gates to make a new result pass. Keep the original contact
gates and all failures as their historical benchmark, explicitly labeled in scope.

The next engineering task is a separate passive punch-quality audit. Measure
pre-impact own-torso-relative fist velocity and direction, hand travel, arm joint
excursion, impulse time course, contact release, retraction and return toward
guard on selected exact-replayed trajectories. Compare current source/finals
with relevant archived older policies and any usable strike demonstrations;
do not assume a directed teacher is competent boxing. Verify that added
instrumentation leaves actions, contact events and outcomes unchanged.

Use those distributions and the actual G1 mechanics to specify prospective
acceptance criteria for different punch motions, then test a versioned objective
change in a bounded controlled experiment. Full elbow extension is not a universal
definition of all punch types. Preserve unassisted roots/legs, separated spawns,
physical contacts, stability criteria, independent evaluation and consumed steps.
No new PPO or physical replay allocation is declared by this static audit.

This inspection and synthetic unit test consume zero new physical controls or
optimizer updates. The completed comparison retains diagnostic PPO 2,163,968,
formal physical controls 2,015,379, main boxing 1,048,576, foundation 474,992,640
and standalone BC 8,000. General boxing and punch-quality certification are false.
