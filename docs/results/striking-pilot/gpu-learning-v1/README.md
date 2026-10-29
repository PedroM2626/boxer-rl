# Active bounded scaled-GPU learning

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
