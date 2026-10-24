# Bounded training-only posture-cost comparison

## Evidence and hypothesis

The [matched motor-floor review](striking-motor-floor-audit.md) completed all
56 episodes / 23,933 controls with exact passive physical agreement, finite
traces and empty native logs. Motor-only pair survival is 31/32 versus full-head
25/32 neutral-0, 26/32 neutral-1 and 24/32 mirror on the identical 32 resets.
The motor-only failure is an inherited precontact self-leg collision. Masking
also worsens one previously surviving reset in neutral-1 and mirror. It is not
adopted as a boxing controller.

On the preselected eight-reset replay subset, full-head survival is 6/8, 5/8,
4/8; first-fall traces include two, one and two precontact failures respectively.
Failures include both self-leg collision and low pelvis. These temporal and
matched-intervention results implicate head-dependent trajectories; they do not
prove a particular command/arm channel or a learned motor update will solve them.

The training reward currently signals terminal falls but provides no explicit
pre-fall posture margin. Test whether a small continuous learner-only safety
cost improves stable strikes while retaining the latest motor, all learned
action channels, original frozen opponent, optimizer and sampling histories.
This is a new bounded reward comparison, not another identical PPO extension.
Earlier reach shaping, extra imitation, body authority and yaw changes remain
rejected; no modified physics or assistance is introduced.

## Exact change and inherited state

Both arms begin at exact latest 670bed08, lineage 1,916,928 learner / 1,294,336
world, Adam actor 12,424 / critic 36,096. Retain the frozen 6bf3273a adversary,
148/17 interface, all sampler/RNG/normalizer state, gamma .999, retention 1,
actor/critic rates .0001, eight worlds / 128 controls and active/neutral 50/50.
The cost arm adds only explicit version-9 reward metadata; all source leaves
must remain equal at the transition. Ordinary resumes require the matching flag
and profile; the transition cannot be repeatedly applied or used for rollback.

After a native step, subtract from the learner reward only:

`dt * (clip((0.68 - pelvis_height) / 0.13, 0, 1)^2 + clip((0.90 - upright) / 0.30, 0, 1)^2)`.

With dt .02, the cost is zero above both margins and bounded by .04 per control.
It reads the evaluated learner's physical pelvis/upright; it writes no action,
observation, force, state, contact or termination. The neutral/frozen opponent
reward does not enter the learner cost. Original rewards and thresholds remain
unchanged in the independent evaluator, including the original complete gates.
The unchanged control retains version 8 with no added cost.

## Allocation, checks and decision rule

Permit at most **16,384 learner/world credits per arm / 32,768 total**. Each arm
has an included 1,024-credit native integration block, then complete original
screens at 8,192 and 16,384. Use ordinary warm resumes between blocks, retaining
all 16 PPO minibatch updates per rollout. Verify finite learned state, immutable
opponent, exact credit and Adam prefixes after every block. Stop the arm if any
full profile has at most three paired survivors; stop all on invalid physics,
nonfinite state, provenance or consumed accounting errors.

Before training evaluate the source on untouched **23,000,003..23,000,006**, both
swaps and all three full profiles; evaluate each arm's latest final on the same
set. Maximum **168 complete episodes / 84,000 evaluation controls**: source 24,
two original screens per arm 96, two final fresh sets 48. No opponent/reset
substitution or weakened averaged criterion is allowed. Every partial attempt
remains counted. Frozen-motor failures are still pair failures.

Judge by the unchanged full physical criteria, surviving neutral own-strike
breadth, paired stability and surviving reciprocal impacts, not cost/loss alone.
Any apparent benefit must be reviewed against both the latest source and the
matched control; survival gained by suppressing strikes is insufficient. Record
all failed generations. Cost-arm passes form a separate task lineage and cannot
inherit the old version-8 pass streak. No automatic adoption, phase promotion,
further budget or opponent-pool expansion follows this experiment.

Reviewed formal evaluation starts at **1,375,474**, new ceiling **1,459,474**.
Diagnostic PPO starts at **1,918,208**, ceiling **1,950,976**; added world
collection at most 32,768 and retention presentations at most 131,072 are separate.
BC 8,000, main boxing 1,048,576, foundation 474,992,640 and combined main
476,041,216 stay unchanged. Unused allocations remain unused.

Launch `tools/start_striking_posture_compare.ps1`, hidden persistent WSL.
Read `/home/pedro/boxing-runs/striking-posture-1916928/comparison.json` and its
process before any new job. Preserve the monitoring schedule; independently
archive all arms, checkpoints, metrics, complete screens and fresh trials.
