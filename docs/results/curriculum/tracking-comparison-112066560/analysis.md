# Tracking reward comparison outcome, 2026-10-22

Both arms completed 9,830,400 transitions from the same immutable source with
training seed 42, 1,024 worlds, and fresh Adam/RNG. The only requested difference
was the shared linear/angular tracking width: 0.25 versus 0.01. Total separate
diagnostic training was **19,660,800 transitions**. Gates used the unchanged
14-trial protocol, 32 resets per trial, seeds 7/23, ten seconds, and physical
displacement criteria. Checkpoint payload checksums were verified before export.

| Evaluation | Individually passing trials | Mean linear error (m/s) | Minimum trial survival | Complete gate |
| --- | --- | --- | --- | --- |
| Original 112.07M walking policy | 6/14 | 0.19480 | 31/32 | Failed |
| Default-width continuation | 6/14 | 0.21171 | 31/32 | Failed |
| Sharp-width continuation | 10/14 | 0.02630 | 32/32 | Failed |

The sharp arm's backward directed displacement was 0.479/0.511 m across seeds,
above the required 0.28125 m. The default arm remained negative
(-0.103/-0.011 m). Under zero command, sharp linear errors were 0.0099/0.0133
m/s versus 0.4466/0.4694 for the default arm. All 448 sharp-arm reset trials
survived ten seconds. These are physical observations on the specified sample,
not a general guarantee of balance or locomotion.

**Both lateral directions still fail:** sharp directed distances were only
0.0496-0.0696 m against the required 0.225 m. No stage is promoted. Reducing
unwanted motion improves tracking error without establishing commanded lateral
footwork; that distinction remains visible in the unmodified displacement gate.

The paired result supports a bounded main-policy continuation with the sharper
reward. It is one training-seed comparison, with no held-out seed replication;
long-term stability, full walking commands, and transfer to boxing remain
unestablished. Reward width was changed for linear and angular tracking together,
so the experiment does not isolate the linear term alone.

## Reviewed continuation

The comparison host correctly resumed the original main policy automatically.
It then advanced another 4,505,600 transitions before this review deliberately
stopped it at **116,572,160 walking / 319,897,600 main total**. That exact actor,
critic, and normalizer were archived with config before changing the reward.
The main checkpoint SHA-256 at the change was
`4bbfe670e386ea12ab965ba1d087673d4e309a083a9a5501a85cfd828dffc484`.

No diagnostic candidate was imported and no actor rollback was performed.
The main learner warm-starts its own latest checkpoint with
`--walk-easy-tracking-sigma 0.01`; Adam/RNG restart is explicit. Main accounting
remains 319,897,600 at restart, and diagnostic accounting remains separately
19,660,800. The shared main ceiling stays 400M. Review direction-specific results
after approximately 20M additional main transitions, without treating this as a
minimum or automatically extending the budget.

The profile is limited to easy walking. The supervisor records it in checkpoint
metadata and inherits it on ordinary easy-walking resumes, preventing silent
reversion. Other stages retain their original training defaults. Evaluation
commands, seeds, physics, reward configuration, thresholds, and the requirement
for two distinct complete passing checkpoints are unchanged. Periodic gate
checks continue with Adam resident.

Twenty-five focused CPU tests passed, covering explicit profile forwarding,
inheritance on resume, isolation from the full-walking stage, unchanged frozen
evaluation, bounded comparison accounting, checkpoint preservation, promotion,
and process guards. PowerShell syntax validation passed. The relaunched hidden
host was observed with one supervisor and one learner using the intended 0.01
flag. No new post-change learning outcome is claimed by this review.

Raw measurements are in `baseline.json`, `default-evaluation.json`, and
`sharp-evaluation.json`; hashes and counts are in `completed.json` and
`summary.json`. `selection.json` records the reviewed continuation and archive.
Both diagnostic policies/configs and the unchanged main policy are retained
under `checkpoints/curriculum/`.
