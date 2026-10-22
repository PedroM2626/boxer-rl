# Targeted lateral command practice, 2026-10-22

## Investigation

The sharper tracking reward stabilized forward/backward tracking and standing,
but seven successive full gates after the reward change remained at 10/14
individually passing trials. At the latest completed gate (151,982,080 walking
transitions), all trials survived ten seconds, but lateral directed distances
were only 0.0373-0.0583 m against the unchanged 0.225 m minimum. No complete
gate passed. Low average tracking error is not sufficient evidence of footwork.

The training sampler previously drew all three moving-command components
together from continuous ranges. A CPU audit sampled 100,000 commands with seed
606. Only **0.409%** met a predominantly lateral request: absolute forward
speed below 0.01 m/s, absolute yaw below 0.02 rad/s, and absolute lateral speed
above 0.03 m/s. Exactly pure lateral requests were absent, as expected from
continuous sampling. This is a demonstrated coverage gap relative to the
axis-isolated gate; it does not establish that coverage alone explains learning.

Reproduce the command-only diagnostic without another GPU learner:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu "$HOME/venvs/boxing_curriculum/bin/python" \
  tools/diagnose_curriculum_commands.py
```

## Training change

For easy walking only, half of the already-nonzero commands are replaced by
pure lateral requests. Each direction is sampled with equal probability, with
speed magnitude uniformly between 0.03125 and 0.0625 m/s. This requests a
velocity; the same learned 29-joint controller must move and balance the body.
There is no new joint target, gait playback, root force, or physical constraint.

The expected mixture is **40% stationary / 30% lateral / 30% original mixed
motion**. The standing mask is preserved exactly for the same input RNG keys.
In the 100,000-command audit, standing remained 40.114%; pure lateral requests
became 29.924% (15.047% positive and 14.877% negative). Predominantly lateral
coverage became 30.134%. Speeds vary around the gate's requested value instead
of reproducing only that exact command. This verifies sampling, not locomotion.

The default sampler remains unchanged when this profile is disabled. The
evaluator does not receive the profile, and still runs the original directions,
seeds, physics, thresholds, and two-distinct-checkpoint promotion requirement.
Full walking and approach retain their original command distributions.

## Preserved continuation and operation

The main checkpoint was preserved at **157,040,640 walking / 360,366,080 main
consumed transitions**, SHA-256
`5d9723fda4744c73751dc33fe9b64990044ec50f2c1b5bfd5f2c71c243c40b40`.
Its policy, config, and gate history are archived under
`checkpoints/curriculum/archive-before-lateral-practice-157040640/`.
The in-progress 157M evaluation was interrupted; the 151.98M result above is
the latest complete gate before this change. Partial evaluations do not count.

The original latest actor/critic/normalizer warm-start the continuation. There
is no rollback or diagnostic-policy import; Adam/RNG restart explicitly. The
tracking width stays 0.01, and the **400M main ceiling remains unchanged**,
leaving at most 39,633,920 main transitions before rounding. Previously recorded
19,660,800 comparison transitions remain separate diagnostic accounting.

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_curriculum.ps1 `
  -Publish -WarmStart -MaxTotalSteps 400000000 `
  -WalkEasyTrackingSigma 0.01 -WalkEasyLateralPractice 0.5
```

`-WalkEasyLateralPractice` is the fraction of *moving* commands to replace,
not of all commands. Set it explicitly to zero to disable the profile. The
supervisor inherits a recorded `lateral_practice_probability` on ordinary
easy-walking resumes if no explicit value is supplied. The standalone learner
flag is `--lateral-practice-probability`. Other stages do not inherit this
override. Configs and checkpoint manifests record its value.

Judge the next full gates by lateral displacement, both direction signs, and
retention of standing/forward/backward performance. Do not substitute sampler
coverage or reward for those physical results. If the remaining allocation
expires without promotion, analyze the evidence before another experiment or
any budget extension. Outcome is pending; no new skill is claimed here.

## Validation and artifacts

Thirty-five focused CPU tests passed; two physical integration cases were not
rerun alongside the active GPU learner. Tests cover the unchanged stationary
mask, balanced directions, velocity limits, unaffected mixed commands,
profile inheritance and isolation across stages, invalid-probability rejection,
unchanged evaluator configuration, promotion, and process guards. The 100,000
command diagnostic and PowerShell syntax validation also passed. No extra GPU
learner was started for validation.

The measured command audit and continuation manifest are in
`docs/results/curriculum/lateral-practice/`. The existing persistent launcher
and overflow guard supervise the relaunched main job. Continued full physical
gates determine whether this practice improves actual footwork.

The [first physical follow-up](results/curriculum/lateral-practice/followup-2026-10-22-1539.md)
records a transient improvement to 12/14 passing trials, then 10/14. Lateral
displacement increased, but neither full gate passed; the original bounded
continuation remains active without another training change.
