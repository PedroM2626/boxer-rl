# Boxing videos

`assisted_ppo_selfplay.mp4` is the verified six-action PPO self-play at
2,109,440 learner transitions. Its JSON sidecar records five red and three
blue clean hits, no falls, seed 0 and 12 seconds of physical simulation.
The overlay explicitly labels balance assistance. Its reproducible checkpoint
is bundled under `checkpoints/skills_seed_2m`.

The `warp1`, `warp4`, `warp5`, `warp6` and `warp7` videos are historical exports
from the earlier direct-control experiments, retained with their existing
filenames. They were present locally before the assisted continuation and
have no machine-readable generation sidecars. Their names alone do not prove
a training budget or an exact evaluation configuration. The main README
documents the historical measurements and their corrections.

`arm3_twin_free_fall.mp4` and `arm3_vs_pinned_twin.mp4` belong to the earlier
free-root/anchored-opponent diagnosis. These files are reference material;
their measurements must not be pooled with assisted-boxing results.
