# Long training protocol: assisted self-play to 200M

The two-million-transition result is an engineering smoke test, not a conclusion
about what reinforcement learning can learn. This continuation gives the same
boxing curriculum 200,000,000 learner transitions in total, including its
2,109,440-transition seed. Self-play counts two learners per world step.
PPO rounds the target up to a complete iteration: with this batch geometry the
final checkpoint is expected at 200,007,680 transitions.

## Fixed experiment

- Seed bundle: `checkpoints/skills_seed_2m`, preserved in Git with weights,
  optimizer, observation statistics and environment configuration.
- 256 worlds, 140 observation entries, six decisions per corner, shared actor.
- Policy: two 64-unit layers. Critic: two 128-unit layers. Learning rate 0.0003.
- Four PPO epochs, four minibatches, batch size 256, unroll length 20,
  discount 0.99. Each iteration collects 20,480 learner transitions.
- The same assisted balance, motor skills, contact-force scoring, retraction,
  damage, stun and rewards as the seed. No geometry/reward tuning mid-run.
- Milestones: 10M, 25M, 50M, 100M, 150M and 200M, each rounded up to an iteration.
- Save every 50 iterations (1,024,000 transitions), plus every milestone.

**Patience is part of the protocol.** A short-term plateau, lower training
reward or a worse evaluation is recorded and does not stop or reset learning.
The job stops for non-finite data, subprocess failures or an external stop,
retaining its last valid checkpoint. Continuing a failed run is explicit and
does not silently restart its actor.

At the seed run's measured throughput, the remaining budget is approximately
13-16 hours of training on this laptop, plus milestone evaluation/rendering.
This is an estimate from local measurements, not a completion guarantee.

## Run, inspect, stop and resume

Install `requirements-warp.txt` in the Warp environment. On Windows, start
from PowerShell in the repository root:

```powershell
.\tools\start_skills_200m.ps1 -Publish
```

This launches a hidden, persistent Windows WSL host. A detached Linux shell
alone did not survive the WSL command's exit in this setup, so the wrapper
keeps the supervisor in the foreground of its own Windows host process.
`runs/skills_200m_host.json` records that host's PID and start time. The Linux
PID and live training phase are recorded separately in `job_state.json`.

Inside native Linux, or an already persistent WSL host, inspect the plan and
run the supervisor:

```bash
bash tools/run_skills_200m.sh --plan-only
bash tools/run_skills_200m.sh --publish
```

`BOXER_PYTHON` overrides the default `$HOME/venvs/warp_test/bin/python`.
Native Linux also supports `--background`; WSL requires the Windows wrapper
for detached execution. Without
`--publish`, it does not commit or push. On this Windows/WSL setup the wrapper
uses Git for Windows and its configured SSH alias to access the remote.
On Linux, use `--git-executable`, `--git-worktree` and `--branch` if needed.
The supervisor and the computer must remain running; closing this chat does
not stop the background process, but sleep, shutdown or stopping WSL does.

Inspect:

```bash
cat runs/skills_200m/job_state.json
tail -n 1 runs/skills_200m/progress.jsonl
tail -n 20 runs/skills_200m_supervisor.log
```

The job state identifies its PID, phase and current milestone. Progress JSONL
has the live learner-step count and measured throughput. During evaluation
and rendering the training counter stays fixed while the phase changes.

To stop gracefully, send SIGTERM to the PID in `job_state.json`; the supervisor
forwards termination to its active child, records the interruption and retains
the last checkpoint. Restart with the same launch command. A run lock rejects
duplicate supervisors. Unfinished training resumes from its verified checkpoint;
completed milestones are not retrained. An evaluation interrupted mid-stage is
rerun from the same pinned candidate.
The lock covers the shared publication namespace, including launches that use
a different local output directory.

## Checkpoints and recovery

New checkpoints retain the original five-tuple payload so the existing readers
remain compatible. A `.state.json` sidecar stores the sampling key, step and
SHA-256. Writes use a temporary file, flush/fsync and atomic replacement. The
latest pointer moves only after both payload and state are durable. Resume
verifies the checksum and payload step before replacing run metadata.

The original seed predates sampling-key persistence; its first continuation
folds the checkpoint step into the key instead of replaying the original random
stream. Later continuations restore the saved key. Physical simulator state is
not checkpointed, so each milestone segment starts fresh rounds. This segmented
protocol is documented and is not bit-identical to one uninterrupted rollout.

## Evaluation and publication

Each milestone exports an immutable bundle under `checkpoints/skills_200m/`.
Self-play and candidate-versus-seed run at seeds 7 and 23, with 64 worlds and
exchanged corners: 128 closed rounds per matchup/seed. Both policies are loaded
from explicit exported checkpoint paths. Evaluation records hits by target,
accuracy, falls, duration, knockout rate and damage margin. A deterministic
12-second video uses the same candidate, with a JSON sidecar.

The original arena win columns remain unchanged. Additional `*_win_points`
columns decide standing timeouts by legal punch value; `*_win_with_points`
combine those decisions with the original physical outcomes. A knockout cannot
be overturned by points. This changes evaluation reporting, not training physics
or rewards. A checkpoint is not called better merely because it is newer.

With `--publish`, the supervisor commits and pushes completed milestone bundles,
reports and videos to `origin` on `master`. It stages only these owned paths:

- `checkpoints/skills_200m/`
- `docs/results/skills_200m/`
- `fights/skills_200m/`
- `docs/skills-200m.md`

Unrelated staged files or a branch change block automatic publication.
Publication failures are recorded, do not cancel the remaining training budget,
and are retried at the next milestone and completion. Raw logs, intermediate
saves and live process state stay in ignored `runs/`; published bundles contain
the data needed to reproduce the measured milestones.

## Prelaunch verification

The Warp test run passed 19 tests covering physical boxing, PPO termination,
NumPy inference, checkpoint integrity and the long-run supervisor. Supervisor
tests exercise completed-milestone recovery, child-process failure, explicit
publishing scope and continuing the compute budget after a remote failure.
The Windows-compatible checkpoint/pure-supervisor tests passed seven cases;
three Linux process-lock cases are intentionally skipped on native Windows.

Two real GPU resumes were also completed in the ignored
`runs/skills_resume_smoke` experiment: 2,109,440 -> 2,150,400 -> 2,170,880.
The new checkpoint checksum and saved random key were verified. The long run
starts from the preserved 2,109,440 seed, not from this temporary validation run.

A real 16-round, corner-swapped integration evaluation caught an initially
unpaired signed-margin field. Both corner margins are now reported, and a
regression test verifies their signs and point-win attribution after pooling.
The five learner/arena unit tests passed after the correction. The successful
[preflight evaluation](results/skills_200m/preflight_seed_eval.json) verifies
that judged outcomes add to one and swapped margins remain negatives of each
other. This is an integration check, not evidence of improvement from training.

The Windows host wrapper was validated by observing a persistent Windows WSL
process, an active Linux supervisor/learner and increasing progress records.
