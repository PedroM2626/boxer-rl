# Short original-screen repeatability and holdout check

See `docs/striking-first-complete-pass.md` for evidence and exact contract.
Latest source 8003a7bf has one complete original-screen pass, failed fresh 19M
holdout and no promotion. Ordinary version-8 resume retains every learned prefix
and original frozen adversary. At most 8,192 new learner/world controls, one full
original gate, source/final fresh 20M full sets and seed-7 source replay both swaps.
Maximum 74 episodes / 37,000 evaluation controls. Video is selected illustrative
evidence, not reliable boxing. No automatic phase promotion or further allocation.
Diagnostic ceilings: 1,918,208 PPO / 1,297,949 formal controls; main/BC unchanged.

Live state: `/home/pedro/boxing-runs/striking-fixed-repeat-1908736/continuation.json`.
Launcher: `tools/start_striking_fixed_repeatability.ps1`. Preserve monitoring.

## Verified selected replay and recovery

Seventeen focused tests and syntax checks pass; source clone/configuration are
verified exact with the original frozen adversary retained. `replay-proof.json`
records exact original physical fields for both seed-7 trials, 1,000 controls,
and a 250-frame / 25-fps / ten-second clip (red 1 / blue 2, both standing).
Start/middle/end frames were inspected. Video is archived at
`fights/striking-pilot/first-full-screen-pass-1908736.mp4`; it illustrates one
selected successful episode and cannot certify broad stability.

The renderer completed but warned about fork after JAX initialization. The
strict supervisor stopped before training; its state/warning are archived.
Explicit recovery reuses the completed replay only after exact report, decoded
video and warning-hash verification; no simulation is repeated or erased.
No numerical, overflow or training-warning exemption is introduced. The original
37,000-control cap and all 1,000 consumed controls are retained. Launch records
are historical; inspect live resumed state and child progress before decisions.
