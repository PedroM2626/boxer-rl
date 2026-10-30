# Completed bounded scaled-GPU learning

Reviewed after completion on 2026-10-29 local (2026-10-30 UTC). The job consumed
all **1,048,576** new credits and stopped at its ceiling. Independent review checks
65 finite checkpoints with exact retained metadata/optimizer counts, all **144
native/full trace pairs** and **2,900** qualified training windows/hashes.
No complete physical pass or main adoption. The launch notes below are history.

| Gate | Pair survivors, neutral 0 / neutral 1 / mirror | Neutral valid cycles | Mirror valid cycles, corners 0 / 1 | Complete |
| --- | --- | --- | --- | --- |
| Source baseline | 5 / 4 / 3 | 22 | 2 / 0 | Fail |
| +262,144 | 5 / 3 / 2 | 24 | 3 / 0 | Fail |
| +524,288 | 5 / 4 / 3 | 40 | 2 / 3 | Fail |
| +786,432 | 4 / 4 / 3 | 34 | 1 / 0 | Fail |
| +1,048,576 | 5 / 5 / 3 | 36 | 3 / 0 | Fail |
| Final fresh | 4 / 6 / 3 | 27 | 5 / 1 | Fail |

The +524,288 intermediate has valid reciprocal strikes on one surviving seed;
neither the two-seed requirement nor survival passes. It is not restored. Final
fresh mirror has no surviving seed with valid strikes from both corners. The
known neutral increase is not evidence of robust reciprocal fighting. Small
survival changes do not establish a statistically reliable improvement.

Final SHA `639ccec99328b034d7d6c9f1ba08a7305a0edc3e67025aa01cedcc06cf979052`,
consumed/optimized **10,486,784**, world controls **9,864,192**, Adam
**146,328 / 170,000**. Global diagnostic PPO **10,569,984**, formal **2,340,326**
(41,817 new), main **476,041,216**. Unused 30,183 native controls expire. GPU
substeps 10,485,888 include 5,974 discarded terminal-neighbor and 128 warm-up;
reset forwards 300,288 GPU / 3,384 native remain separate. Only unchanged solver
limit bits 512/1024 occur; no capacity/nonfinite flag, no new renderer calls.

Collection+PPO rate by quarter is **510 / 433 / 395 / 371 credits/s**, excluding
native evaluation, evidence publication and other outside-loop costs. Totals are
2,432.742 s collection and 56.658 s PPO. Complete wall time also includes native
gates, checkpoint/evidence I/O and Git. The one-block 870 figure is not sustained
production throughput. Full audit/history grew to 254,077 bytes rewritten at each
substep. A CPU/disk-only benchmark measures 128 old writes/scans in 0.924 s versus
128 compact durable writes in 0.155 s (1,356 bytes/write). This is an accounting
cost comparison, not a sixfold GPU-training speed claim.

Compact attempted-frontier accounting is tested for interrupted writes, stale
audit files, checkpoint mismatch and changed ceilings; full history is retained
at rollout/phase/publication boundaries. Native child ledgers remain durable and
are refreshed on evaluation boundaries. Resumes must reconcile both files and
reject unoptimized attempts rather than rewind. Thirty-nine focused/regression
tests pass, including debit-before-failed-kernel checks; no physical tolerance or
gate changed. Failure timing in stored traces shows several early approach falls
without prior interbody contact; this is association, not an established cause.

Next: the separately bounded [matched transfer diagnostic](../../../gpu-throughput-and-transfer-followup.md)
uses actual source and actual final on CUDA 3/5 with the same measurement code,
reusing these immutable native results. No additional PPO allocation yet. Preserve
the actual final, all consumed ledgers and the existing monitoring interval.

## Historical launch review

Launched 2026-10-29 under `docs/gpu-bounded-learning.md`, implementation
`f0cb0fd`, from the reviewed scaled prefix SHA
`6432029087cb00e73e39bee3a974e07f47f598771bae05c519e1bcad616c2103`.
The exclusive hidden WSL job is productive; inspect live state at
`/home/pedro/boxing-runs/gpu-learning-v1/state.json` and the recorded host PID/
start time before any launch or intervention. The existing monitor is unchanged.

The complete independent **native baseline** consumed 6,996 controls. Formal
total after baseline is **2,305,505**. All three profiles' raw contact/motion and
full-state traces, hashes and unchanged criteria were checked and published.

| Profile | Pair survivors | Valid straight cycles by corner | Complete result |
| --- | --- | --- | --- |
| Neutral, learner corner 0 | 5/8 | 13 / 0 | Fail on survival |
| Neutral, learner corner 1 | 4/8 | 0 / 9 | Fail on survival |
| Mirror self-play | 3/8 | 2 / 0 | Fail on survival and reciprocal valid cycles |

These are the actual source baseline results before the new learning allocation,
not an improvement credited to the continuing run. Distinct surviving known seeds
with neutral valid punches are four and three respectively. Mirror has no seed
with valid reciprocal straight cycles by both agents. Stable boxing remains open.

After baseline publication (`7821b9a`), the job enters real CUDA collection and
PPO. The first two 16,384-credit rollouts completed finite checkpoint validation
without capacity/nonfinite failure or duplicate jobs. At that reviewed prefix,
new optimized credits are **32,768**, consumed/optimized **9,470,976**, and SHA
`bbd1fd026b8483113ee7ba32976820ed4bbdfc371a7e81fee3b6e886bc8b05be`.
Live attempted/optimized counters continue beyond this launch note. Stochastic
training cycle counts include learners and frozen opponents and cannot replace
native held-out results. Sustained costs also include durable checkpointing/reset
and supervision overhead; the one-block integration rate is not a guaranteed
production throughput.

The first scheduled post-training native gate is at **262,144 new credits**;
then 524,288, 786,432 and 1,048,576, plus a fresh final group. The job stops at its
declared ceiling or earlier on two distinct complete passes before fresh checks.
It preserves every source/optimizer/sampler prefix and attempted transition.
Checkpoints and physical results publish automatically at gates; failures and
partial evaluation ledgers must be investigated before ordinary resume. No
automatic budget extension, forced rollback, gate change or main adoption.
