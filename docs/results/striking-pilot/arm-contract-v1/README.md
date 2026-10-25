# Arm contract and saved tracking audit

Compiled metadata and **32 complete saved trajectories** (v3/v5/v7) were checked
without stepping physics or evaluating selected-state kinematics. No learner,
geometry search, gains or checkpoint was changed; all consumed counters retained.
Reproduce with `tools/audit_striking_arm_contract.py` in the curriculum CPU runtime.

All applied left-arm targets equal recorded motor baseline plus .8 rad filtered
residual within numerical precision; native actuator control clipping is **zero**.
This is separate from the already measured residual-action clipping. Joint maps
are resolved by compiled actuator/joint names for both physical corners.
Shoulder/elbow gains are 75 in both scenes. Wrist gains are 2 in the current
foundation and 20 in the legacy boxer, an already documented difference.

In stable straight v3 episodes the median absolute shoulder-pitch target error
is .120 rad before control, .082 after; elbow .061 to .021 rad. Wrist-pitch is
about .100 rad before and after. These observations do not prove either gains or
contacts are the unique cause of failed punches; endpoint error is not a joint-
order bug or evidence that enough RL training cannot solve the task.

The scripted tests reject specific trajectories. They must not become an
indefinite requirement for a perfect teacher before allowing exploration.
The next explicit [learned-cycle protocol](../../../striking-motion-cycle-training.md)
changes the touch objective and removes old touch-imitation weight while preserving
the actual latest compatible learner/optimizer. A separate bounded allocation,
independent motion criteria and complete gates govern that learning.
