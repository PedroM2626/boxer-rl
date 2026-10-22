# Walking tracking reward comparison

## Motivation and current status

At the automatic review beginning 2026-10-22 13:08 UTC, the active walking
learner had consumed **88,473,600 stage transitions / 291,799,040 total**.
Fifteen completed gates, through stage step 82,575,360, had failed; the next
gate was running. The latest five completed gates passed 6, 6, 10, 7, and 6
individual trials out of 14. Backward displacement remained opposite to the
request in both seeds. No new physics overflow or nonfinite diagnostic was
found in the walking logs. One supervisor, one learner, and its evaluator
remained active. No interruption, rollback, or budget extension was performed.

The inspected checkpoint checksum was verified independently:
`8c70c7875d34883ec4413161830d6e06b4650251a561a29d7566d665178e876a`.
This review does not claim a new physical skill or a cause for the plateau.

The previous [review](results/curriculum/monitoring-2026-10-22-1238.md) identifies
weak reward discrimination at the small easy-walking commands. The learner now
accepts an optional **training-only** `--tracking-sigma` value, enabling a
controlled comparison without changing task defaults or the physical evaluator.
The value changes both linear and angular exponential tracking kernels, so an
experiment must be interpreted as joint velocity-reward sensitivity, not a
linear-only intervention. The comparison was subsequently launched as recorded
below; no experimental candidate has been adopted.

## Bounded comparison protocol

If the same directional failures persist around 100M walking transitions:

1. Preserve an immutable, checksum-verified walking checkpoint, live consumed
   counter, configuration, and gate history before pausing the main process.
   Retain the persistent WSL host discipline and avoid concurrent GPU learners.
2. Use the same actor, critic, normalizer, training seed, and 1,024-world setup
   in separate diagnostic outputs. Both arms reset Adam and RNG, so the default
   arm controls for that restart. Neither arm replaces the main checkpoint.
3. Keep all settings identical except tracking width: the existing 0.25 versus
   an explicitly experimental 0.01. Begin with a bounded 10M-transition ceiling
   per arm. Record actual rounded transitions separately as diagnostic compute;
   do not erase or rewind the main consumed counter.
4. Run the unchanged full walking gate before and after each arm. Compare each
   requested direction, standing retention, survival, tracking errors, and real
   displacement. A higher training reward is not evidence of better footwork.
   Keep negative or inconclusive results and both checkpoint hashes.
5. Adopt no candidate automatically from one favorable trial or one passing
   gate. Explain the comparison before selecting a continuation, retain the
   original policy, and require the existing two distinct complete passes for
   promotion. Resume the main job normally if the experiment is inconclusive.

The experiment flag is provided directly to `boxer.curriculum_train`; the
supervisor and PowerShell launcher still use the existing default. Its value
is written into the output `config.json` under
`environment.reward_config.tracking_sigma`. Omission always uses the task
default, including on a diagnostic warm start; repeat the chosen flag when
resuming an experimental arm. Preserve each arm's original configuration.
The evaluator constructs its own frozen configuration and ignores this flag.

## Engineering validation

Fourteen focused CPU tests passed on 2026-10-22, including invalid/nonfinite
width rejection before initialization, recording the training override, keeping
the default unchanged, and preserving the evaluator's original reward width,
90-row allocation, and disabled training pushes. Existing adaptive promotion,
rollback-accounting, and process-overflow guard tests also passed. No extra GPU
training was started and no claim about experimental learning outcomes follows
from these tests.

## Launched comparison, 2026-10-22 13:38 UTC review

Directional failures persisted beyond 100M walking transitions. The main job
was deliberately stopped after the complete 112,066,560 walking gate, retaining
**315,392,000 main consumed transitions** and its existing 400M ceiling. Its
checkpoint, config, and gate history are archived in
`checkpoints/curriculum/walking-comparison-seed-112066560/`, with policy SHA-256
`3866fd8b00d815b3d70b0c4cb5f2f235e5cd2ebf6b94af128c4f9e4c25814574`.
The supervisor's generic `stopped_by_user` signal status records this authorized
automatic intervention, rather than a new human stop request.

The hidden persistent WSL host now runs `boxer.curriculum_compare`. It holds
both the main run lock and publication namespace lock, preventing an overlapping
main supervisor. Its local output is `runs/diagnosis/tracking-112066560/`:

- One shared unchanged full baseline gate evaluates the immutable source.
- The `default` and `sharp` arms each request at most 10M fresh transitions,
  using the same policy, normalizer, critic, seed 42, and 1,024 worlds.
- Each arm receives its own unchanged full post-training physical gate. The
  runner rejects mismatched hashes, previews, missing directions, and invalid
  gate metadata. It records actual rounded diagnostic transitions separately.
- After success or an ordinary experiment error, the host automatically warm
  starts the **original main policy**, with its original shared ceiling and
  publication enabled. No candidate is selected automatically. Explicit SIGTERM
  or SIGINT instead stops the runner without starting a new learner.

The comparison's live phase, diagnostic accounting, errors, and final status
are in its `state.json`; `baseline.json` and each arm's `evaluation.json` hold
physical results. Host logs are `runs/tracking_comparison_host*.log`. The shared
`runs/curriculum_host.json` identifies this persistent host, which later becomes
the main supervisor through exec. This review launches the experiment; its
outcome remains pending and must be assessed before any training change.

Eighteen focused CPU tests passed for the launched runner and existing safety
contracts. They include identical source/seed across arms, preserved original
policy bytes, separate diagnostic accounting including partial failures,
preview rejection, and refusal to overlap an active main lock. Bash syntax
validation passed. These checks do not substitute for physical outcomes.

## Completed and reviewed, 2026-10-22 14:08 UTC review

The comparison completed and the original main policy resumed automatically.
The [outcome and continuation decision](results/curriculum/tracking-comparison-112066560/analysis.md)
record improved backward tracking and standing retention for width 0.01, but
continued failure of the lateral displacement gates. Both arms and the main
policy are preserved. The reviewed change applies width 0.01 only to easy-walking
training, continuing the main actor at its newer 319.8976M consumed counter.
No diagnostic candidate is imported and no stage is promoted.

Use the normal launcher with `-WalkEasyTrackingSigma 0.01` to select this profile
explicitly. Ordinary supervisor resumes also inherit `training_tracking_sigma`
from the easy-walking checkpoint when the option is omitted; standalone learner
experiments still require their explicit flag as described above. Full walking
and approach keep their original defaults. The frozen evaluator remains unchanged.
