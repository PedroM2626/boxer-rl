# Sparring motor: balance that survives its own punches

> Update 2026-10-31: `checkpoints/spar` is now the motor retrained for hooks
> and the high guard ([moves](moves.md)). The straight-only motor described
> here is `checkpoints/spar-straight`.

Date: 2026-10-30.

## Diagnosis

From 10-21 to 10-29 the striking work kept the certified `approach` motor
frozen and asked PPO arm residuals to add punches. Every variant (v12–v17,
GPU batching, KL guards, GAE, opponent pools) ended with the same failure:
pairs fall once both robots act. The leg controller was the cause. It had
been trained with its arms held in the guard pose, so arm motion and
received punches were dynamics it had never seen.

Measured directly with the new `spar` stage (below), the certified motor
alone, with no opponent, survives only **0–19 %** of 10-second trials once its
arms throw punches (mean survival 2–5 s). Its own `approach` gate is ≥ 90 %. No
arm-residual objective can fix that from above.

## Change

`UnassistedG1("spar")` (in `boxer/curriculum_env.py`) is the walking task
with these changes:

* `punch_arms` replaces the 14 arm actions with randomized straight punches.
  Each punch picks a hand and a height between body and head, with ±0.12 rad
  endpoint jitter and 0.8–1.25× speed, using the skills controller's
  FK-checked targets and 0.18 s / 0.26 s extension/retraction. Pauses of
  0.1–1.2 s allow combos. The policy still controls legs and waist and sees
  the arm state through its normal joint observations and last action.
* Shoves every 1–3 s at 0.1–0.5 m/s stand in for received punches. The
  walk stage uses 3–5 s at 0.1–0.3.
* Footwork commands come from the walk ranges, with 50 % standing. The
  evaluation uses half-speed walking commands and the balance stage's push
  trial.

Fine-tuning starts from `checkpoints/curriculum/approach` with the unchanged
curriculum trainer:

```bash
bash tools/run_spar.sh boxer.curriculum_train --stage spar \
  --init-from checkpoints/curriculum/approach --output runs/spar/train \
  --steps 40000000 --worlds 1024 --checkpoints 40 --adaptive-gates --gate-every 10000000
```

## Results

Solo `spar` gate (2 seeds × 7 commands × push on/off, 32 worlds, 10 s):

| Motor | Mean survival | Worst trial | Gate |
| --- | --- | --- | --- |
| certified `approach` (frozen) | 0.06 | 0.00 | fail |
| `spar` @ 10.8M (≈ 8 min of training) | 0.994 | 0.969 | fail: standing drift 0.21 m/s vs 0.20, weak left step |
| `spar` @ 21.6M | 0.999 | 0.969 | **pass** |
| `spar` @ 32.4M | 1.000 | 1.000 | fail: one tracking error 0.216 |
| `spar` @ 39.3M (final, now `checkpoints/spar-straight`) | **1.000** | 1.000 | **pass** |

Two-robot bouts (`boxer/spar_match.py`): native MuJoCo, free roots, no pelvis
support, scripted ring tactics, 20 s or first knockdown, seeds 0–7:

| Motor | Went the distance | Mean length | Clean punches / bout | Thrown / bout |
| --- | --- | --- | --- | --- |
| certified `approach` (15 s, seeds 0–3) | 0/4 | 3.0 s | 0.75 | 3 |
| `spar` @ 10.8M | 8/8 | 20.0 s | 8.9 (both corners score) | 58.5 |
| `spar` @ 39.3M (final) | **8/8** | 20.0 s | **18.6** (red 2 wins, blue 5, 1 draw) | 64.6 |

Video: [`fights/spar_g1_vs_g1.mp4`](../fights/spar_g1_vs_g1.mp4) (final motor, seed 0), with a [JSON sidecar](../fights/spar_g1_vs_g1.json). Both robots approach, exchange punches and
stay up. Weakness: from about 8 s they drift into a chest-to-chest clinch and
rotate, so the scripted range control needs work.

```bash
bash tools/run_spar.sh boxer.spar_match --checkpoint checkpoints/spar-straight \
  --seeds 0 1 2 3 4 5 6 7 --seconds 20 --video runs/spar/match.mp4
```

## What this is and is not

* It is unassisted, two-sided physical boxing. Both bodies are free, every
  joint is driven by its actuator, and contact and scoring use the existing
  impact scorecard.
* Punch *selection* is scripted (`RingBrain`). The motor is learned; the
  punches are fixed primitives. The next step is to replace `RingBrain` with
  a self-play PPO policy that has the 6-decision interface of the archived
  assisted mode (lead, rear, aim, advance, circle, guard). That mode learned
  real exchanges in 2M transitions once balance was not the bottleneck.
* Blocking, slipping and punch types other than straight punches are not
  trained yet. Adding hooks and guard movements to `punch_arms` is the same
  kind of change as this one.
