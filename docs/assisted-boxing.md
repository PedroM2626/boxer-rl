# Assisted boxing

The earlier policies often turned a round into a fall or a clinch. Increasing
the training budget did not remove that attractor. This environment separates
boxing decisions from motor execution and provides explicit physical balance
assistance so an episode can contain repeated exchanges.

## Control contract

`SkillsBoxing` wraps the same Menagerie G1 scene used by `WarpBoxing`.
Both corners use a shared PPO actor, with six actions per corner:

| Index | Decision | Effect |
| --- | --- | --- |
| 0 | Lead punch | Request a left-arm strike |
| 1 | Rear punch | Request a right-arm strike |
| 2 | Aim | Continuous body-to-head target selection |
| 3 | Advance | Move the support target toward/away from the opponent |
| 4 | Circle | Move the support target around the opponent |
| 5 | Guard | A high value suppresses new strikes and keeps the compact guard |

Lead/rear name the two motor skills; a learned orthodox/southpaw foot stance
is not implemented. A request over 0.15 starts the stronger requested punch
when the corner is ready. Its hand and aim are latched. At the standard 50 Hz
control rate it extends over 0.18 s, retracts over 0.26 s, and settles until
0.76 s before the next strike can launch. A stunned corner cannot launch a
new strike. The observation is 140 floats: the existing 131 physical/fight
entries, three skill-state entries and six previous decisions.

The guard is a named pose, `boxing`, fitted through the actual robot's forward
kinematics. Unlike the old positive-elbow-only fit, it uses the model's allowed
negative elbow angle to bring the fist back toward the upper chest without
cranking the wrist. Historical guards are preserved for their checkpoints.

## What is physical, and what is assisted

The 58 original joint actuators drive the arms and legs. Contact dynamics,
blocking and impacts use MuJoCo Warp. There is no pose playback or root
teleportation during a round. The support controller applies a bounded spring
and damper to each floating pelvis: up to 500 N per translation axis and
120 N m per rotation axis. It also turns the root toward the opponent.
Support targets move at up to 0.10 m/s in the forward direction and 0.07 m/s
sideways. This can slide the feet: it is not a learned locomotion controller.

Only fist contacts with the head or torso can score. The 50 N threshold and
1 m/s approach gate are checked every 4 ms, including the velocity immediately
before the impact, since a clean impact can stop the fist within that substep.
The original retraction, rearming, damage and stun rules remain in use. Head
hits are worth 1.0, torso hits 0.5. The final training run uses a damage cap of
6.0 and a 12 s round clock. Timeouts with both corners standing are reported
as undecided by the original arena win columns. The long-run evaluator adds
separate points-decision readings, without changing the training reward or
overwriting those historical outcome columns.

The assisted curriculum reduces the dense posture income and adds four reward
units per clean hit value to the existing competitive hit reward. It also
initializes the actor with an active body-punch prior. All six outputs remain
trainable. `--no-skill-prior` disables that initialization. These choices
encourage actual exchanges; they do not establish competitive superiority.

## Training and measurement fixes

- The observation normalizer stays fixed from rollout collection through PPO's
  updates. Previously, changing it before SGD changed the likelihood ratio
  before any actor update.
- Every round ending masks the value bootstrap, including the clock. The
  returned observation is already a respawn, whose value belongs to another
  episode.
- Per-round punch and duration metrics accumulate across training iterations
  and flush only when those specific worlds close a round. A window with zero
  closed rounds reports zero for those fields; `rounds` must be read with them.
  Per-1,000-step hit values remain available for windows without closures.
- Training, video, policy probes and the arena use one environment factory.
  It restores the control space, guard, damage cap and stun duration. A direct
  policy cannot be evaluated as a six-action skills policy.
- Video capture preserves the terminal physical pose before autoreset, streams
  frames to the encoder, labels the controller and assistance, and writes a
  JSON sidecar containing both corners' clean hits, falls and knockouts.

## Verification

`tests/test_skills.py` checks idle stability without scoring, repeated strikes
and retraction by both corners without falls, stun, action validation, partial
autoreset, terminal rendering state and configuration restoration.
`tests/test_warp_training.py` checks value bootstrapping at both terminal and
clock endings, NumPy/Flax inference parity and seeded action sampling.

The original direct environment's own physical self-check also passes,
including its force-gated jab, damage, stun, knockout, autoreset, rearming
and both available imitation references. Historical direct-control results
remain in the main README; assisted results must not be pooled into that league.

## Measured run: 2026-10-21

`runs/skills2/checkpoint/step-2109440.pkl` is a shared policy trained in
self-play for 2,109,440 learner transitions across 256 worlds. The run took
550.7 seconds on the laptop RTX 3060 (6 GB), including validation workloads
running alongside it: 3,830 learner transitions/s overall, with a median of
4,222 per iteration after the first iteration. All 103 training records are
finite. This is learner throughput; world throughput is half that number.

The [held-out evaluation](results/assisted_selfplay_eval.json) uses seed 7,
64 worlds, 600 control steps, deterministic actions, and exchanged corners:
128 completed rounds, none dropped.

| Measurement | Side A | Side B |
| --- | --- | --- |
| Clean punches per round | 3.359 | 3.594 |
| Head hits per round | 0.531 | 0.484 |
| Torso hits per round | 2.828 | 3.109 |
| Accuracy | 27.6% | 30.4% |
| Fraction of steps on the floor | 0.000 | 0.000 |

Every round lasts more than 3 seconds; mean duration is 11.984 seconds.
One round ends on damage and 127 end on the clock, reported as undecided.
This establishes repeated physical exchanges, not a strong competitive boxer.
The learned behaviour remains close-range work, and circling can slide the feet.

The [delivered video](../fights/assisted_ppo_selfplay.mp4) uses the same final
checkpoint with deterministic actions at seed 0. In its 12-second round the
red corner lands five clean torso hits and the blue corner lands three,
with zero falls and no knockout. The [JSON sidecar](../fights/assisted_ppo_selfplay.json)
records those numbers. The video is physics driven; the separate `--scripted`
baseline is not used in this result.

Validation completed: 8 new Warp/learner tests, 45 existing scene, policy,
training, arena and alternating-mode tests, plus the original Warp physical
self-check. The original full MJX environment suite was not rerun.
