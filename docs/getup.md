# Getting up after a knockdown (2026-11-12): not learned yet

After a knockdown the referee resets both robots standing (`spar_match
--knockdowns N`). A boxer that could get up by itself would remove that
reset. This page records the first two attempts. Both failed; the stage, its
floor contacts and the evaluator are kept for the next attempt.

## The `getup` stage

`UnassistedG1("getup")` (`boxer/curriculum_env.py`) is a GPU (MJX/Warp)
curriculum stage with the usual 29-joint G1 action and observation contract:

- **Floor contacts.** The upstream model lets only the legs and hands touch
  the floor, so a fallen robot would sink through it. The stage adds massless
  colliders (density 0; masses unchanged, checked) for the pelvis, torso,
  shoulders, head, upper arms and forearms (`FLOOR_BODY_GEOMS`). They, the
  thighs, shins and hands touch only the floor.
- **Starts.** 50 % lying (tilted 1.3–1.85 rad about a random horizontal axis:
  back, front or side), 30 % part-way through a fall (0.4–1.1 rad), 20 %
  standing. The joints are near the guard pose, and the robot drops onto the
  floor.
- **No fall termination**, no shoves and no footwork commands. Episodes last
  10 s.
- **Reward** (`getup_reward`, per step × dt):
  - pelvis height, from 0.15 m (0) to 0.75 m (1);
  - from attempt 2, head height (0.12 m to 1.2 m);
  - torso uprightness;
  - a standing bonus with a pull towards the guard pose;
  - a small joint-velocity cost.

`python -m boxer.getup_eval CHECKPOINT` runs 128 starts for 10 s and reports,
per start kind, the share that is ever up (pelvis ≥ 0.72 m, torso within about
25° of upright) and up at the end, with peak and final heights.

## Results

Lying and falling starts are 98 of the 128:

| Policy | Training | Lying or falling: ever up | Lying: peak / final pelvis height | Lying: peak uprightness | Standing starts up at the end |
| --- | --- | --- | --- | --- | --- |
| spar motor (`checkpoints/spar`, never trained to get up) | — | 0 / 98 | 0.30 / 0.11 m | 0.35 | 70 % |
| attempt 1: from the spar motor; pelvis height, upright, standing | 150M steps | 0 / 98 | 0.30 / 0.18 m | 0.37 | 73 % |
| attempt 2: from scratch; head height added | 200M steps | 0 / 98 | 0.30 / 0.10 m | 0.98 | 0 % |

- **Attempt 1** inherited the spar motor's narrow action distribution (some
  scales near 0.001) and barely moved on the floor. The pelvis height
  term pays nothing until the pelvis leaves the floor, which needs most of
  getting up already done.
- **Attempt 2** explored from scratch and found a local optimum. It sits
  with the torso upright and the pelvis on the floor, which earns the
  uprightness and about half the head-height reward with no risk. Even
  robots that start standing sit down.

## What it would likely take

- An assist that fades to zero: a lifting force on the torso, scheduled down
  as the policy improves, so that standing up is reached and rewarded before
  it must be done alone.
- Rewards in stages (sit → kneel or squat → stand), each paid only once the
  previous one is reached, so that sitting is no longer a resting place.
- Then, in bouts: the same floor colliders in the native two-robot scene, and
  a count during which the fallen boxer runs the get-up policy while the
  other waits.

Commands (attempt 2):

```bash
python -m boxer.curriculum_train --stage getup --output runs/getup2 --steps 200000000 --worlds 1024 --checkpoints 30 --seed 51
python -m boxer.getup_eval runs/getup2/checkpoint --output runs/getup2_eval.json
```
