# Full-walking reward investigation, 2026-10-22

## Why the main run stopped

The 450M allocation ended normally at **447,467,520 main consumed transitions**,
including **29,593,600** in full walking. Equal PPO epoch/block rounding left an
unused remainder below the ceiling. There was no new runtime failure and no
full-walking promotion. Balance and easy walking remain certified.

Full-walking gates passed 5, 10, 9, 6, 4, 4, and 4 of 14 individual trials at
stage counters 0, 5.02M, 10.06M, 15.09M, 20.13M, 25.17M, and 29.59M. Survival
recovered from zero in initial backward trials to at least 31/32 in every trial
at each trained checkpoint. However, lateral translation and yaw tracking
regressed while recorded PPO losses remained finite.

At the final gate, negative-lateral displacement was only 0.237/0.229 m against
0.9 m required. Positive lateral displacement was 0.950/0.868 m, with seed 23
below the distance minimum; tracking errors also affect trial approval. All four
yaw-command errors were about 0.290-0.365 rad/s, above 0.25.
More survival and lower optimization loss therefore do not establish mastery.

The earlier 18:09 UTC review prepared the comparison code but did not launch it.
On the subsequent explicit continuation, live inspection found no active
learner. The stopped checkpoint, config, full history, evaluation, progress,
training result, and supervisor state were preserved under
`checkpoints/curriculum/archive-ceiling-450m-step-447467520/`.
Its SHA-256 was checked against both the payload and final gate:
`606096818b9828f33c0da33b050fd28f074858886bb40d7af56697b91d4a33f5`.

## Controlled experiment now running

The inherited tracking kernel `exp(-squared_error / 0.25)` may provide weak
discrimination for some full-walking errors. This is a hypothesis, not a proven
explanation of regression. A bounded comparison changes only its shared
linear/angular width:

- Control width: **0.25**, restored from the current checkpoint metadata.
- Experimental width: **0.04**, broader than easy walking's 0.01 and narrower
  than the current full-walking setting.
- Same immutable actor, critic, normalizer, seed 43, 1,024 worlds, command
  distribution, pushes, model, and remaining PPO settings in both arms.
- At most **10M requested transitions per arm**, with actual rounded diagnostic
  counts recorded separately. Both arms restart Adam/RNG, controlling for that
  restart. Neither overwrites the main policy.
- One shared unchanged full baseline gate and one unchanged full gate after
  each arm. Full-walking commands, seeds 7/23, 32 resets, ten seconds, required
  displacement, survival, and tracking thresholds remain frozen.

The comparison runner now supports both walking stages and rejects a mismatched
stage/direction set, preview, wrong required distance, checksum mismatch,
identical reward widths, or missing preceding stage certification. It retains
the source command profile in both arms. The main/publication locks prevent
another learner from overlapping it; existing bounded ENODATA and overflow
guards supervise its children.

The persistent hidden WSL host runs:

```bash
bash tools/run_tracking_comparison.sh \
  --output runs/diagnosis/full-walk-tracking-29593600 \
  --sharp-sigma 0.04 --resume-main
```

Its state, baseline, per-arm progress/checkpoints, and full physical results are
under that output. Host logs are `runs/full_walk_comparison_host*.log` and the
shared host record is `runs/curriculum_host.json`. The initial live process check
confirmed one comparison supervisor and one baseline evaluator on the intended
full-walking stage. The outcome is pending.

## Continuation rules and accounting

No candidate is adopted automatically. After the comparison, the existing
runner resumes the original main policy under the unchanged **450M ceiling**;
only its unused rounded remainder may remain, so it can stop again shortly.
This is not a new budget extension. The earlier 19,660,800 diagnostic transitions
remain recorded, and this comparison adds its actual counts separately.

The next automatic review must analyze both complete gates and preserve both
arms before selecting any continuation. Compare each direction, yaw errors,
standing retention, and falls; do not choose from reward alone. If a different
full-walking reward is supported, add explicit full-stage profile forwarding
and resume persistence before using it in the main supervisor. The existing
easy-only launcher flag is not a full-walking setting. Further budget requires
a new bounded decision backed by these results. Physical gates stay unchanged.

Twenty-five focused CPU tests passed, covering both stage-specific comparisons,
shared source/seed/profile, preserved original checkpoints, separate diagnostic
accounting, invalid-gate rejection, distinct widths, prerequisite stages, locks,
read retries, overflow termination, promotion, and profile handling. No extra GPU
test learner was launched. Project artifacts and configuration remain in English.

The follow-up automation was updated to read current stages, budgets, and
diagnostic paths from live state and documentation instead of repeating its
historical easy-walking/400M assumptions. Its existing 30-minute schedule and
quiet-on-unchanged notification intent remain unchanged.
