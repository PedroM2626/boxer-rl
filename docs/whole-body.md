# Whole-body policy (2026-11-11)

Every boxer before this was a hierarchy of separately trained parts. A brain
(tactics or striker) sent a footwork command to a leg motor and set the
arms, and the motor turned the command into leg and waist targets. Each part
trained with the others frozen, and changing one broke another: when the
striker's arm style changed, its legs began to fall
([striker](striker.md#arm-motion-cost-2026-11-03)). The whole-body policy
(`boxer/whole_body.py`) removes the hierarchy: one network sets every joint
target the robot has, every control step.

## Interface

| | Size | Contents |
| --- | --- | --- |
| Observation | 174 | the striker observation (74: opponent, ring, both arms' motion) and the motor's proprioception without the footwork command (100: pelvis velocity, gyro, gravity, joint positions and velocities, last actions, gait clock) |
| Action | 23 | 15 leg and waist targets in the motor's units, and 8 shoulder and elbow targets in the striker's units (the pose box around every scripted pose, moved at most `striker.RATE` per step, wrists held) |

The network is 512–256–256 for both the policy and the value function.
`spar_match` recognizes a whole-body checkpoint by its sizes (174 → 23)
wherever it takes tactics. Such a corner has no motor: its leg and waist
targets replace the motor's row for that robot. `native_finetune --whole-body`
trains one, and a whole-body policy can also be an opponent in the native
pools.

## Imitation: one network for striker and legs

`python -m boxer.whole_body_train collect` plays 800k native ring steps with
the striker (`checkpoints/striker`) on its legs (`checkpoints/spar-native`). The
opponents are the champion, the high-guard champion, both strikers, and a
scripted brain in 30 % of bouts. At every step it records the whole-body
observation and what the two layers did together: the motor's 15 executed
leg targets and the striker's 8 arm targets. `clone` fits the policy mean to
those labels, with leg errors weighted 4× because they cost falls, and the
value function to discounted native returns. It runs for 40 epochs, starting
the policy scale at 0.1. Held-out mean squared error is 0.0014 for the legs
and 0.0010 for the arms.

The clone (`checkpoints/whole-body-clone`) boxes end to end with no motor.
In 32 bouts of 120 s (seeds 0–15, corners swapped), the clone on no motor
against each opponent on its own legs:

| Opponent | Record | Clean punches per bout | Knockdowns (clone : opponent) |
| --- | --- | --- | --- |
| champion stack | 32–0 | 259 : 88 | 6 : 2 |
| striker (its teacher) | 16–16 | 256 : 252 | 13 : 8 |

The clone matches its teacher's punching but falls more often. The striker
itself goes 1 : 4 against the champion and suffers 6 knockdowns in 32
striker brawls, as measured below. Imitation errors in the legs add up over
120 s.

## Native PPO: more punches, more falls

Two runs, each 2M native transitions starting from the clone. Opponents:
champion, high-guard champion, both strikers, and scripted bouts (30 %).
Arm-motion cost 0.05, minimum policy std 0.05, gate on the last four
checkpoints. The long check ran against the clone, the champion tactics and
the striker.

| Run | Settings | Long check (120 s, seeds 24–31) | Gate |
| --- | --- | --- | --- |
| 1 | lr 5e-5, 4 epochs, fall penalty 30 | 1M: 12 : 5 knockdowns vs the clone; 1.5M: 11 : 10 vs the clone, then 15 : 0 vs the champion (300 clean punches per bout to 20) | rejected |
| 2 | lr 1.5e-5, 2 epochs, fall penalty 60 | 1M: 7 knockdowns vs the clone (431 clean punches per bout to 109); 1.5M: 16 vs the clone | rejected |

In run 1 the updates were large: KL 0.1–0.25 per iteration, with 64 % of
probability ratios outside the PPO clip. Run 2 brought both down (KL 0.02,
21 % clipped). Both runs show what the arm-only experiments showed. PPO in
30 s training episodes raises hits by 1.5–2× and wins the 20 s gate bouts
(14–2, 15–1 against the clone), and the falls appear only in long bouts.
The long-bout gate is what caught it. Neither run was promoted.

Commands (run 2):

```bash
python -m boxer.whole_body_train collect --output runs/whole-body-data
python -m boxer.whole_body_train clone --data runs/whole-body-data --output runs/whole-body-clone
python -m boxer.native_finetune --whole-body --init-from runs/whole-body-clone/checkpoint \
  --output runs/whole-body-native2 --motor checkpoints/spar-native --ring --arm checkpoints/arm-skill \
  --pool checkpoints/tactics checkpoints/tactics-highguard-v1 checkpoints/striker checkpoints/striker-v1 \
  --scripted-fraction .3 --high-guard-fraction .6 --steps 2000000 --checkpoint-every 500000 \
  --min-std .05 --learning-rate 1.5e-5 --epochs 2 --fall-penalty 60 --arm-motion-cost .05 \
  --native-gate 4 --long-gate 2 --long-gate-opponents checkpoints/tactics checkpoints/striker
```

## What is open

The end-to-end interface works: one network can stand, step, turn and punch
for a full 120 s bout. What it lacks is training that values staying up over
a whole bout. Directions not tried yet:

- Episodes as long as the bouts that are judged (120 s), or a value horizon
  that sees a fall coming.
- A per-step penalty on the body's tilt and tilt rate, like the leg motor's
  own reward, instead of only a terminal fall penalty.
- Keeping the leg outputs close to the clone (a KL or L2 anchor) while the
  arms explore.
