# Reviewed continuation to a 450M shared ceiling, 2026-10-22

## Closing the 400M allocation

The supervisor stopped normally at **397,230,080 consumed main transitions**:
193,904,640 in easy walking plus 203,325,440 previously consumed in balance.
Equal PPO epoch/block rounding left 2,769,920 unused transitions below 400M;
this was a budget stop, not a runtime failure. Balance remains certified, but
easy walking has no two consecutive complete passes. There is no promotion.

The final gate passed 11/14 individual trials, with at least 31/32 survivors in
every trial. Standing, forward/backward, and both yaw commands passed in both
seeds. Only lateral directed displacement failed:

| Lateral command | Seed 7 | Seed 23 | Required in every trial |
| --- | --- | --- | --- |
| +0.05 m/s | 0.2344 m | 0.2137 m | 0.225 m |
| -0.05 m/s | 0.1919 m | 0.2037 m | 0.225 m |

Negative lateral distances improved across the last three checkpoints from
0.1624/0.1721 to 0.1693/0.1807 and 0.1919/0.2037 m. Positive lateral trials
passed in both seeds at two recent checkpoints, but one regressed at the end.
Before targeted practice, lateral distances were only about 0.043-0.066 m.
This is nonmonotonic physical progress supporting further practice, not a reason
to relax thresholds or claim competence.

## Bounded decision and preserved data

The authorized review selects **450M as the next shared main ceiling**. It
provides at most 52,769,920 additional transitions from the actual counter,
rather than another 450M allocation or 200M per phase. PPO rounding schedules
**49,152,000** new transitions for this invocation if it does not promote early;
it can stop below the requested ceiling again. This is one reviewed extension,
not approval for unlimited further increases.

The final policy, config, full gate history, evaluation, progress, training
result, and supervisor state are preserved under
`checkpoints/curriculum/archive-ceiling-400m-step-397230080/`. The independently
verified payload hash matched the final full gate before relaunch. The decision
record in `docs/results/curriculum/budget-450m/decision.json` includes that hash,
failed trials, profile, counters, and unchanged thresholds.

The latest original actor, critic, and normalizer warm-start the continuation;
there is no rollback, random actor reset, or diagnostic candidate import.
Adam/RNG restart explicitly because these snapshots do not contain them.
All main consumed steps are retained. The earlier 19,660,800 comparison steps
remain separately recorded diagnostic compute, without relabeling or erasure.

## Operation and validation

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_curriculum.ps1 `
  -Publish -WarmStart -MaxTotalSteps 450000000
```

The supervisor inherited tracking width **0.01** and lateral-practice probability
**0.5** from the easy-walking checkpoint. Both were verified in the running
learner command and saved config. One persistent hidden host, one supervisor,
and one learner were observed. No forced-restart or recovery-source flag was
supplied. All frozen evaluation settings and the two-distinct-pass rule remain
unchanged. Process guards and scoped automatic publication remain enabled.

No code changed for this extension. Validation was stopped-process inspection,
checkpoint/gate hash verification, final trial analysis, archival preservation,
and live inherited-profile/config/process checks.

Review both lateral signs and retention of the other skills in the next gates.
A plateau, survival regression, numerical failure, or another budget stop needs
a new evidence-based decision. Full walking, approach, and paired boxing remain
downstream and are not certified by this extension.
