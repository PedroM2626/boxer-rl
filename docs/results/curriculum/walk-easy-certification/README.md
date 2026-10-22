# Easy-walking certification and full-walking transition

Easy walking earned two consecutive complete physical passes on distinct
checkpoints on 2026-10-22. The supervisor then promoted the same learned policy
to `walk` automatically. There was no manual threshold change or forced advance.

| Walking-stage transitions | Main consumed total | Passing trials | Minimum trial survival |
| --- | --- | --- | --- |
| 209,387,520 | 412,712,960 | 14/14 | 30/32 |
| 214,548,480 | 417,873,920 | 14/14 | 31/32 |

Every trial passed survival, linear/yaw tracking, and required physical
displacement. Both seeds (7/23), all seven requested directions, 32 reset states
per trial, and the original ten-second horizon were retained. The independent
`verified_gate`, `promotion_ready`, and supervisor `certified` checks passed;
both historical policy payload checksums were verified against their gates.
This certifies the defined benchmark, not universal locomotion or boxing skill.

The four lateral directed distances at the second gate were 0.30386, 0.24804,
0.28529, and 0.23950 m, each exceeding the unchanged 0.225 m minimum. Falling
trials were still counted as failures, with zero displacement credit. Standing,
forward/backward translation, and yaw trials passed at both checkpoints.

## Exact policy continuity

The promoted actor, critic, and observation normalizer have SHA-256
`f1e0e6f023ff4ac477171382069fe132af9180ae818b509a18c2572eca0200ee`.
The initial full-walking `step-0.pkl` has the **same payload hash**, verified
independently. No random actor replaced the learned controller. The new stage
records its source as the certified easy-walking checkpoint. Adam/RNG are fresh
at the stage change, as required by the existing checkpoint limitations.

Full-walking training uses its original ranges: x -0.35 to +0.55 m/s, y +/-0.25
m/s, yaw +/-0.5 rad/s, with 40% stationary practice. Its tracking width is 0.25;
the easy-only lateral-practice and 0.01 tracking overrides do not leak into this
stage. Full evaluation commands are four times the easy values. One supervisor,
one full-walking learner, and its initial evaluator were observed. No numerical
or physics overflow diagnostics were found in the new stage logs.

The shared ceiling remains 450M, with 32,126,080 main transitions available at
the stage boundary before PPO rounding. No additional allocation was added by
promotion. Full walking must earn its own two complete passes, followed by
approach; paired free-body boxing remains a separate integration/training task.

## Artifacts and video clarity

`first-complete-pass.json` and `second-complete-pass.json` contain the full
physical trials. `certification.json` records verified hashes and survival;
`next-stage-config.json` and `transition.json` preserve the new configuration and
exact policy continuity. Existing scoped publication saved and pushed the
certified policy, history, and training result.

Walking gate video selection previously captured the first stationary trial.
Future evaluator invocations instead capture the existing forward-command trial
and display its command, without changing trial order, physics, actions, seeds,
or grading. The already-running full-walking evaluator retains the previous
video selection until its next invocation. Older videos remain historical.
Sixteen focused CPU protocol/recovery/profile tests passed after that
presentation-only edit. No extra GPU evaluator or learner was started.
