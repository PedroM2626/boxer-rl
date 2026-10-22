# Paired unassisted foundation transfer

## Certified starting point

All four foundation stages passed two distinct complete consecutive physical
gates, finishing at 474,992,640 main transitions. The [certification archive](results/curriculum/foundation-certification/README.md)
preserves the exact gates and hashes. The final approach actor is
`b39b1990945e8a374802c3a190e5ac5e026ae39e7fd5ef61c5c2b01ad291c6fc`.
Foundation completion is not boxing completion.

## Adapter and physical contract

`boxer/paired_foundation.py` composes two copies of the actual pinned foundation
scene. This deliberately replaces the earlier proposal to adapt the older boxing
robot by changing a few gains: inspection found that the foundation also uses
different foot shapes, inertia/control parameters, floor contacts and integration.
Preserving its original model avoids silently changing learned locomotion.

The shared scene has two free roots and 58 position actuators, with all 29 joints
per corner, including all twelve leg joints, available to policy actions. Each
corner retains foundation gains, bias parameters, damping, armature, explicit
body inertia, default compact guard, 0.5-rad action scale, 2 ms Euler integration,
and 20 ms control. No root forces, springs, welds, gravity compensation, trajectory
playback or scripted punch controller are used. The nominal spawn separation is
2 m, with small independent positional jitter and randomized joint/root velocity
resets. Corner swapping changes initial sides and headings; it does not mirror
joint actions incorrectly.

The original feet-only collision model is retained. Massless head/torso and
blocking collision surfaces from the existing boxing scene, plus cross-robot
contacts, make the opponent physically tangible. There are 158 explicit pairs.
Original body inertials are unchanged. Floor/contact sensors removed during
composition are not actor inputs; their first-fall self-leg collision conditions
are checked from actual contacts. Either robot falling ends the shared round;
there is no reset inside an episode.

The 103-input noiseless actor interface preserves local pelvis velocity/gyro,
projected gravity, requested velocity, 29 joint offsets/velocities, previous
29 actions and four phase values. A numerical test compares it directly with
Playground's original observation implementation. The frozen actor uses the
original normalizer and weights. Dummy privileged values are supplied only to
satisfy the inference dictionary shape; the actor reads `state`, and no critic
is evaluated or trained in this harness. Navigation requests are computed in
each pelvis frame toward the other body, stopping around 0.75 m; only joint
torques from the learned actor can physically move either root.

## Diagnostic protocol

`tools/evaluate_paired_foundation.py` runs standing and mutual approach for ten
seconds, seeds 7/23 plus holdouts 31/47, both corner arrangements and two reset
samples per combination: 32 paired episodes. These are native MuJoCo CPU
trajectories, not GPU PPO gates. They test transfer to shared-contact physics;
results do not replace any curriculum certificate. Physics execution and reset
sampling differ from the frozen foundation benchmark and are documented here.

The smoke diagnostic completed four two-second episodes (seed 7, both modes and
corners) with both robots upright and finite. Mutual approach reduced initial
separation 2.072/1.928 m to 0.892/0.800 m; this short test does not establish
long-round robustness. It used 400 diagnostic world transitions and zero
optimization transitions. A separate physical test positions real fist/head
surfaces in contact and observes positive MuJoCo constraint force. That is a
contact-engine test, not evidence of a learned punch.

The wider diagnostic runs through a hidden persistent Windows WSL host:

```bash
bash tools/run_paired_transfer.sh \
  --output runs/paired-transfer/evaluation.json \
  --video runs/paired-transfer/approach.mp4
```

State, PID, logs, output and actual video are under `runs/paired-transfer/`.
The original foundation remains stopped and certified. Do not launch a second
transfer diagnostic while that host/process is active. Diagnostic simulation
and any future boxing optimization are recorded separately from consumed
foundation transitions. The 475M foundation ceiling is not an unlimited boxing
budget.

## Remaining boxing work

This adapter currently validates two frozen locomotion actors. It does not train
punching, add opponent features to the original actor, or score punches by
proximity. Its raw integrated legal fist/head-or-torso contact force is reported
as impulse, explicitly not as a hit score: sustained leaning contacts must not
earn repeated punches. Before boxing PPO, implement and independently validate
closing velocity, contact-event/cooldown handling, impacts on legal targets,
fall penalties, engagement and opponent-aware policy/training integration.
The learned locomotor must keep control of free legs while strike policy
outputs learn actual arm actions; replaying scripted punch cycles is excluded.
Review full paired rollouts, held-out seeds, side exchange and numerical warnings
before committing a bounded boxing training allocation. Any transfer falls or
standing drift must be measured and addressed, not hidden by root assistance.

## Completed ten-second pilot and next decision

The 32 paired episodes completed with finite state throughout. Both robots
survived **16/16 standing** and **15/16 mutual-approach** episodes. Holdout seeds
31/47 survived all eight episodes in each mode. Surviving approach endpoints
are recorded in the [raw results](results/paired-transfer/evaluation.json) and
[summary](results/paired-transfer/summary.json); the actual navigation video is
`fights/paired-transfer/approach.mp4`. It was visually inspected and shows two
unsupported robots approaching and facing each other. No fists landed on legal
targets in these rollouts, so no learned punching is claimed.

The failed episode (seed 7 plus reset offset 1009, swapped corners) was
replayed with the native numerical-warning guard. It reproduced a red
pelvis collapse at **1.992 s**, height **0.548 m**, torso up **0.860**. There were
**zero cross-robot contacts** before the fall and no numerical warning. This is
an observed locomotion/command robustness failure, not a demonstrated impact
failure. It remains in the denominator. A standing episode also drifted as much
as 0.730 m despite a zero command; surviving is not precise footwork.
This small sample does not establish long-round robustness.

The harness fails on native BADQPOS/BADQVEL/BADQACC or contact/constraint capacity
warnings, preventing MuJoCo's invalid-state reset from appearing as survival.
Six focused adapter tests passed, covering exact observations, physical gains
and inertia, first fall, side swap, invalid actions, actual solver contact force,
and injected numeric warning. The final combined CPU suite passed **47 tests**, including the new
numerical-warning guard and the existing curriculum and promotion checks.
The full diagnostic logs contain no physics warning, overflow, NaN or traceback.
The known JAX/os.fork warning arose in video encoding; encoding completed and the
video was inspected successfully.

This work used **16,100 diagnostic world-control transitions** (400 smoke,
15,600 full pilot, 100 first-fall replay), with zero PPO updates. Unit-test
physics operations are engineering validation. Foundation consumed accounting
remains 474,992,640. Earlier diagnostics retain separate published accounting.

**Continuation decision:** build and test opponent-aware striking integration
on this measured baseline while retaining fall/drift limitations. Implement an
independent fist-only contact event scorecard, then a trainable opponent-aware
policy interface with explicit foundation provenance. Validate neutral outputs
and bounded exploration before a reviewed, bounded boxing PPO pilot. Do not
substitute the old supported controller, add pelvis assistance, treat navigation
as boxing, or start long training before the new action/reward interface is
verified. The existing 30-minute monitor continues this authorized implementation.
