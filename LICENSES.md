# Licences and third-party data

Everything here is either a library pulled from a package index at install time or
motion data whose terms allow redistribution. Nothing is vendored into the
repository without a licence that permits it.

## Robot description: Unitree G1 (MuJoCo Menagerie)

`unitree_g1` is Apache-2.0/BSD-licensed upstream and is installed from PyPI as
`mujoco-menagerie` (see `tools/warp_deps.sh`, `boxer/scene.py`). It is never
copied into this repository.

## Motion reference: CMU Graphics Lab Motion Capture Database

The human punch that shapes the imitation channel comes from the CMU database.
Its FAQ (http://mocap.cs.cmu.edu/faqs.php) states, verbatim:

> How can I use this data?
> The motion capture data may be copied, modified, or redistributed without permission.

and the homepage adds: "This dataset of motions is free for all uses."

CMU's requested acknowledgement (from `READMEFIRST.txt`):

> The data used in this project was obtained from mocap.cs.cmu.edu. The database was
> created with funding from NSF EIA-0196217.

Used trials, fetched by `tools/cmu_fetch.sh` into `runs/reference/cmu/`, which is
git-ignored:

- subject 13, trials `13_17`, `13_18` (boxing sessions) — the working reference;
- subject 144, trial `144_13` (a punching session) — kept locally as a candidate.

`tools/bvh_to_dirs.py` reads a **BVH conversion** of the CMU trials rather than
CMU's own ASF/AMC pair, because the Vicon-dialect ASF shipped by CMU leaves the
bone-axis convention ambiguous: a forward kinematics built from it keeps the
elbow deflection pinned at 90 degrees through an entire punch trial, which no
recorded strike can do. The conversions were fetched from a public repository
that carries no licence file of its own. The reasoning recorded here: the motion
data remains CMU's and CMU permits redistribution; a format conversion adds no
authorship; and none of that repository's code is vendored, only the data it
publishes. If that position is ever challenged, the fallback is to convert
CMU's own ASF/AMC locally once the axis convention is pinned down — the failing
attempt and its symptom are documented in `tools/bvh_to_dirs.py`.

## Why the project does not use Mixamo animation files

An earlier revision of the measurement tools analysed a Mixamo `Body_Jab_Cross.fbx`
clip (`tools/fbx_trace.py`, `tools/fbx_arm_dirs.py`) to establish that a recorded
human jab travels essentially flat. Those findings stand and are cited in
`boxer/warp_env.py`. The clip itself, and any curve data derived from it, stay out
of this repository, because Adobe's General Terms of Use, section 3.6 ("Content
Files"), grant:

> a personal, non-exclusive, non-sublicensable … and non-transferable license to
> use the Content Files to create your end use (i.e., the derivative application or
> product authored by you) into which the Content Files, or derivations thereof, are
> embedded … You may modify the Content Files prior to embedding them in the End Use.
> You may reproduce and distribute Content Files only in connection with your End Use,
> however, **under no circumstances can you distribute the Content Files on a
> stand-alone basis, outside of the End Use**.

In practice that means:

- **allowed and published here** — the trained policy weights, the rendered fight
  videos, and the tooling, since they are the product authored here;
- **not allowed** — committing an `.fbx`, or the per-frame position/angle dumps
  (`arm_dirs.json`, `jab_targets.json`) that are derivations of it, which is why
  `runs/` is git-ignored and the tools take a clip path as an argument so each
  user supplies their own motion file.

If you use the Blender tools on a Mixamo clip, download it yourself with an Adobe
ID and keep it local.
