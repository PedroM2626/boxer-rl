# Restoring closed fists in paired replays

The original closed-fist visual graft was implemented in commit `768b326`
(2026-10-14). That scene supported free-body, unanchored agents before the later
assisted experiments and before the current foundation curriculum. Closed fists
were already an established project feature, not a consequence of assistance.

The omission occurred in commit `9f3cc67`: the new paired-foundation builder
loaded the foundation XML directly and copied collision surfaces, but did not
reuse `scene._graft_fists`. Its renders consequently displayed the upstream rigid
open rubber-hand meshes. The earlier explanation attributing this to removing
assistance or first freeing the legs was historically incorrect.

The paired replay renderer now builds a separate visual model with the existing
closed-fist graft applied to both hands of both robots. It displays the recorded
qpos through forward kinematics only. This model never steps simulation, produces
actor observations, rewards or scores, or replaces the learner's resident model.
The default paired builder and all active training paths remain unchanged.
The fingers remain rigid visual meshes; this restores the original appearance,
not a newly learned finger-closing behavior. Actual hand collision capsules and
the independent physical impact scorecard remain unchanged.

## Verification

Eight paired-foundation tests passed. The new regression test verifies four
complete hands, mirrored finger curling (distal links within 0.08 m of the palm),
absence of rubber-hand meshes in replay, and zero collision masks on the graft.
It also checks identical coordinate dimensions/defaults, body masses/inertias,
damping, armature, gains, actuator biases and explicit contact-pair names.

The archived 97,280-step actor was replayed for seed 31 in both side arrangements,
ten seconds each. Every previously recorded trial field matched exactly,
including survival, residuals, raw impulses, final distance and blue's two scoring
timestamps, 4.046/4.104 s. The new video renders closed fists over the actual
trajectory. This repeated input is not another independent success or evidence
that the current actor recovered boxing proficiency.

- [Replay video](../fights/striking-pilot/closed-fists-replay-97280.mp4)
- [Four-hand close-up](results/visual-fist-recovery/four-fists.png)
- [Validation and accounting](results/visual-fist-recovery/validation.json)
- [Exact replay results](results/visual-fist-recovery/replay.json)

The replay adds 1,000 separately counted diagnostic world controls. Formal
striking-interface diagnostics now total 43,656 world controls; diagnostic PPO
remains 768 learner transitions. No new optimization or static reach sweep was
performed. Historical videos are preserved as originally rendered; future paired
evaluation videos use the repaired renderer.

The ongoing learner was not interrupted or restarted. Its checkpoint hash and
finite learning state were verified at 828,416 consumed boxing transitions,
under the existing 1,048,576 ceiling and inherited 0.45 m profile. Complete gate
histories and latest hashes still certify all four foundation stages. Exactly
one boxing learner and no foundation or diagnostic learner were present. No
physics warnings were found. See the [monitor snapshot](results/visual-fist-recovery/monitor.json).
The existing monitoring schedule and next duration-review decision remain intact.
