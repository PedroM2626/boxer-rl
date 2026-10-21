# Assisted boxing: 200M continuation

Status: **running**. Updated: 2026-10-21T18:26:45.661315+00:00.

Requested budget: 200,000,000 learner transitions total, including the 2,109,440-transition seed. A world step counts twice.

The environment, motor controller and PPO hyperparameters stay fixed. This is the assisted curriculum; balance and locomotion are not learned.

Training is allowed to use the full budget. Short-term plateaus, lower reward or a checkpoint losing an evaluation do not cancel it. Non-finite data, an external interruption or a failed subprocess stop the job with the last valid checkpoint preserved.

Milestones are evaluated with deterministic self-play and against the pinned 2.11M seed, at seeds 7 and 23, exchanging corners. All reports and videos identify the exact checkpoint. The latest checkpoint is not automatically called the best. Points readings settle standing timeouts using legal punch value; the original KO/fall win readings are retained.

| Requested milestone | Actual checkpoint | Self-play hits A / B | Self-play head hits A / B | Win with points vs seed |
| --- | --- | --- | --- | --- |
| 10,000,000 | 10,014,720 | 4.590 / 4.500 | 0.000 / 0.000 | 0.363 |
| 25,000,000 | 25,006,080 | 4.578 / 4.445 | 0.000 / 0.004 | 0.539 |

Machine-readable records: [milestones](results/skills_200m/state.json).
Weights: `checkpoints/skills_200m/`. Videos: `fights/skills_200m/`.

Raw progress, subprocess logs and the live job state remain under `runs/skills_200m/` and are ignored by Git. Published milestones are small reproducible bundles; intermediate saves remain local.

A resume preserves actor, critic, optimizer, normalization and the saved PPO sampling key. Each milestone starts fresh physical rounds; simulator state is not restored. Legacy seeds without a saved key fold the seed step into the new key. This is not a bit-identical uninterrupted rollout.

The computer and WSL must remain running. The job uses an exclusive run lock, atomic checkpoint writes and explicit output-path ownership for automatic Git commits. Unrelated staged changes block publishing.
