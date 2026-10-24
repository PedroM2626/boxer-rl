# Bounded GAE trace-length comparison

## Evidence and hypothesis

The [72-episode exact passive replay](results/striking-pilot/kl-failure-trace-1966080/README.md)
has twelve failures: nine low-pelvis, three self-leg contact, no tilt termination.
Eleven occur after interbody contact; one is an inactive partner's early leg
contact. Source survival is 23/24, guarded final 19/24 and latest control 18/24.
All complete sets fail. Temporal association does not establish contact causation.

The zero-simulation `terminal-credit.json` audit verifies all trace hashes and
checks actual saved critics. Gamma remains .999 but GAE lambda remains .95.
For a direct terminal reward within a training fragment, its coefficient fifty
controls (one second) earlier is .07319, and one hundred controls earlier .00536.
Lambda .99 changes these to .57549 and .33118. Direct-credit half-life changes
from 13.25 to 62.72 controls. This is not the total effective credit horizon:
critic predictions can propagate value across earlier TD errors and fragments.
Reset-aligned offline fragments differ from actual staggered training boundaries;
deterministic traces are off-policy, and longer GAE can increase variance.

The earlier gamma .99/.999 experiment did not vary lambda .95. This comparison
tests a distinct advantage-estimation hypothesis; it does not repeat the rejected
KL guard, posture reward, noise reduction, yaw scaling or body authority change.
Lower attenuation alone is not evidence of better boxing.

## Exact continuation

Start both arms from the **latest original control** 24e23288, full SHA-256
`24e23288b073843637d5210a5a5d4a03f3383c47580fe004832cfd0d3a50d423`,
at 1,966,080 consumed/optimized learner credits and 1,343,488 world controls.
Retain all actor/critic weights, Adam moments/counts 13,192/36,864, normalizer,
RNGs, retention provenance, mixed corner ledger and original frozen adversary.
No earlier actor restoration, optimizer reset or main adoption.

Version 12 sets only training GAE lambda .99, with explicit immutable transition
metadata. Control keeps version 8 / lambda .95. Both retain gamma .999, original
Gaussian covariance, rates .0001, all sixteen actor and critic updates per
1,024-credit rollout, retention weight 1, eight worlds and unroll 128. The
physics, free roots/legs, motor, closed fists, action channels, reward, separated
spawns, real contact scoring, first-fall termination and evaluation gates remain
unchanged. Ordinary warm starts must retain this profile without reapplying the
transition. Passes from other profile families are not inherited.

At most **16,384 credits per arm / 32,768 diagnostic PPO and physical collection
controls total**, plus 131,072 integrated retention presentations separately.
Include the first 1,024-credit integration block and update observer. Hash initial
normalized observations, sampled raw actions and log probabilities across arms:
these must match. Full observer batch hashes may differ because advantage/return
targets intentionally differ. The observer changes no training state.

Original complete screens at +8,192/+16,384; untouched source/final full sets on
26,000,003..26,000,006 seeds, both swaps and all three profiles. At most 168
episodes / 84,000 evaluation controls. Stop an arm if any full original profile
has at most three paired survivors. Stop all on nonfinite state, native errors,
physics/constraint/contact overflow, invalid provenance/counters or initial
sampling mismatch. Preserve attempted compute on failure. Never silently retry.

Diagnostic PPO begins **2,016,512**, new ceiling **2,049,280**. Formal evaluation
begins **1,666,653**, new ceiling **1,750,653**. Foundation 474,992,640, main boxing
1,048,576, combined main 476,041,216 and standalone BC 8,000 remain unchanged.
The previous replay consumed 33,219 controls; unused older allocations are not
extra PPO and are not added to this new limit.

Judge complete physical survival, surviving own-strike breadth and reciprocal
events versus source/control. Two distinct consecutive complete passes plus
independent generalization are prerequisites, not guaranteed adoption. No
automatic promotion, rollback, longer budget or next hyperparameter search.
On completion archive both arms and make an explicit physical-evidence decision.

Use hidden persistent `tools/start_striking_gae_compare.ps1`; inspect
`/home/pedro/boxing-runs/striking-gae-1966080/comparison.json` and both child
outputs/processes before any launch. Preserve the existing monitoring schedule.

## Checks

Thirty-nine focused tests pass. Checks cover lossless source transition, rejected implicit/mislabeled
profiles, terminal/bootstrap and fragment boundaries, and a longer direct GAE
component without next-episode leakage. Existing optimizer, fixed-opponent,
retention, independent evaluator and passive-equivalence checks pass. The initial
native block must additionally validate actual configuration, complete source
retention and counters; software checks do not certify physical improvement.

Python/Bash syntax and PowerShell parsing pass. Initial sampling fingerprints
ignore intentionally different GAE targets while detecting changed raw actions.

The [first native integration block](results/striking-pilot/gae-1966080/README.md)
is verified: 1,024 included learner/world credits, sixteen actor/critic steps,
finite state, version 12 / lambda .99 and unchanged frozen adversary. The first
sampled-batch comparison remains pending until control runs. Source fresh 26M
survival is 8/7/6, a complete failure. No improvement or promotion is inferred.

## Completed comparison and bounded decision

The [full independent review](results/striking-pilot/gae-1966080/README.md)
verifies all 32 finite update payloads, matching initial sampling hash and 168
episodes / 79,297 evaluation controls. Lambda .99 has original mirror survival
7/8 then 8/8 with six reciprocal episodes each, and fresh mirror 8/8 with six,
versus 7/8 with four control and 6/8 with four source. Neutral fresh survival
6/8 and 7/8 still fails; aggregate 21/24 only matches source (control 20/24).
Both original complete screens fail, and every full fresh set fails. No adoption.

These active-exchange results justify only one [bounded ordinary continuation](striking-gae-continuation.md)
from latest 95dbdff0, retaining version 12 and all learned/optimizer/sampler state.
At most 16,384 new credits, full screens at 8,192/16,384 and new 27M source/final
sets, at most 48,000 evaluation controls. Stop at the first screen if severe
regression or no source-relative full-screen measurement improves. New diagnostic
PPO ceiling 2,065,664; formal evaluation ceiling 1,793,950. No rollback, weaker
neutral criteria, automatic promotion or further allocation.
