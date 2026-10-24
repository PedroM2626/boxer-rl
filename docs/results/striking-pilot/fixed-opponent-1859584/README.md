# Bounded fixed-opponent versus current-opponent training comparison

See `docs/striking-fixed-opponent-comparison.md` for evidence, exact contract
and limitations. Both copy latest original-profile 6bf3273a; candidate freezes
only the opposing actor/normalizer with an exact checked fingerprint. Learner,
critic and all Adam/RNG/sampler/credit prefixes are retained. Original physics,
commands, rewards, actions and complete independent gates remain unchanged.
Max 16,384 controls per arm; included first 1,024 guards, full screens at
8,192/16,384, severe-regression/stagnation stops, source/final fresh 18M tests.
Max 32,768 collection, 131,072 replay presentations and 84,000 evaluation controls.
Global diagnostic ceilings: 1,877,248 PPO / 1,223,761 formal evaluation.
No main adoption, promotion or automatic follow-on allocation.

Live state: `/home/pedro/boxing-runs/striking-fixed-opponent-1859584/comparison.json`.
Launcher: `tools/start_striking_fixed_opponent_compare.ps1`. Preserve monitoring.

## Candidate integration evidence

Thirty-five tests and syntax checks pass. Initial source leaves (except explicit
task mode) match exactly. The included candidate first 1,024-control block is
finite with empty native error log and unchanged frozen-opponent fingerprint.
Actor/critic Adam counts 11,544/35,216 and mode/corner prefixes are verified;
updated SHA 4a49ae4624c1bb71d17d2c0b8e602b499f28f1b64aabe3a29fdba2d31e4ac01a.
Exact initial/updated models are archived in the corresponding checkpoint folder.
Native guard proves integration, not physical progress; cross-arm equality is
pending the control arm. Full source fresh 18M reports and their independent
analysis are archived (11,234 evaluation controls). Launch snapshots are historical;
inspect live process and child progress before interventions or accounting updates.

## Completed independent review

All 168 episodes / 77,764 evaluation controls, four full screen generations,
three fresh sets, four exact gate models, 32 finite metric blocks, empty native
error logs and main/foundation histories/hashes are verified and archived.
First-block learned/optimizer/normalizer/RNG/sampler leaves match exactly across
arms. The frozen actor/normalizer fingerprint stays unchanged through both gates.
Fresh source/control/fixed survivors total 20/24 / 12/24 / 23/24. No complete
set passes: fixed original-screen survival regresses 7/7/7 to 5/8/5, and final
fresh neutral-1 is 7/8 rather than the required 8/8. Latest candidate, not the
more favorable intermediate, is retained. Diagnostic totals: 1,877,248 PPO /
1,217,525 formal controls; new collection/replay 32,768 / 131,072 separately.
`analysis.json` records one bounded 32,768-credit ordinary candidate continuation,
with its original frozen adversary unchanged. No adoption or phase promotion.
