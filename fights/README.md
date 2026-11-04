# Boxing videos

## README preview

`striker_vs_champion_preview.mp4` is a 12-second, seed-0 bout of the current
Striker (red fists) against Champion (blue fists), rendered at 25 fps. The
GIF uses the same footage at 15 fps and 640 px wide for inline README playback.
All four ring posts are neutral grey; the fists identify each boxer.
The bout finishes without knockdowns, with 20 clean punches for red and 5 for
blue; this single preview is not a new evaluation of either policy.

Regenerate the video from the repository root with the curriculum WSL runtime:

```bash
bash tools/run_spar.sh boxer.spar_match --ring --seconds 12 --knockdowns 3 --seeds 0 \
  --checkpoint checkpoints/spar-native --red-tactics checkpoints/striker \
  --blue-checkpoint checkpoints/spar --blue-tactics checkpoints/tactics --blue-arm checkpoints/arm-skill \
  --fps 25 --video fights/striker_vs_champion_preview.mp4 --output runs/readme-preview.json
```

Convert the video with FFmpeg:

```bash
ffmpeg -i fights/striker_vs_champion_preview.mp4 \
  -filter_complex "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" \
  -loop 0 fights/striker_vs_champion_preview.gif
```

## Historical videos

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
