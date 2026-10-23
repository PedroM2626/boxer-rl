# CMU jab reference

`cmu-13-17-jab.json` preserves the existing locally retargeted left jab from
CMU Graphics Lab Motion Capture Database subject 13, trial 17. It is copied from
the project's prior `runs/reference/jab_targets_cmu.json`, used by the older
unassisted Warp policies. No new fit, synthetic strike or Mixamo data is added.
The `strike_window` selects thirteen four-joint poses, not a timed controller.

The data used in this project was obtained from mocap.cs.cmu.edu. The database
was created with funding from NSF EIA-0196217. See [third-party terms and the
conversion provenance](../../LICENSES.md). Diagnostic profile records pin the
reference SHA256. The current test uses similarity to any pose as training reward
only, gated by own fist extension relative to the moving/rotating torso. Nothing
replays these angles into actuators or supports the robot's root.
