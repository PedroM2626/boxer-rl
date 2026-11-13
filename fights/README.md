# Boxing videos

## README preview

`whole_body_vs_champion_preview.mp4` is a 12-second, seed-0 bout of the
whole-body policy (`checkpoints/whole-body`, run 5, red fists) against the champion
stack (blue fists), rendered at 25 fps; the GIF is the same footage at 15 fps
and 640 px wide. Red lands 26 clean punches to 0, no knockdowns. A single
preview, not an evaluation (`docs/whole-body.md` has those).

```bash
bash tools/run_spar.sh boxer.spar_match --ring --seconds 12 --knockdowns 3 --seeds 0   --checkpoint checkpoints/spar-native --red-tactics checkpoints/whole-body   --blue-checkpoint checkpoints/spar --blue-tactics checkpoints/tactics --blue-arm checkpoints/arm-skill   --fps 25 --video fights/whole_body_vs_champion_preview.mp4 --output fights/whole_body_vs_champion_preview.json
ffmpeg -i fights/whole_body_vs_champion_preview.mp4   -filter_complex "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"   -loop 0 fights/whole_body_vs_champion_preview.gif
```

(`imageio_ffmpeg.get_ffmpeg_exe()` in the curriculum runtime gives an
`ffmpeg` binary.)

## Whole-body bouts (2026-11-12)

60 s at 60 fps, knockdown rules (3), each opponent on its own legs, the
promoted run-5 policy; seed 0 unless noted (`runs/wb_videos.sh`):

| Video | Clean punches | Knockdowns |
| --- | --- | --- |
| `whole_body_vs_champion_60s.mp4` (seed 1) | 194 : 36 | 0 : 0 |
| `whole_body_vs_striker_60s.mp4` | 222 : 17 | 0 : 0 |
| `whole_body_vs_itself_60s.mp4` | 241 : 248 | 0 : 0 |
| `whole_body_getup_vs_champion.mp4` (seed 138, 50 s, 25 fps, `--getup checkpoints/getup`, `runs/getup_clip.py`): blue down at 3.8 s, red falls while waiting at 4.8 s, both up after the 8 s count and box on | 117 : 20 | 1 : 1 |

## Getting up

`getup_from_the_floor.mp4` and `.gif`: `checkpoints/getup` standing four G1s
up from lying starts, one after another, 4 s each, in the GPU `getup` stage
(no assist):

```bash
python -m boxer.getup_eval checkpoints/getup --worlds 32 --seconds 4 --seed 11 --video fights/getup_from_the_floor.mp4
```

## Earlier preview

`striker_vs_champion_preview.mp4` and `.gif`: the same 12-second, seed-0
setup with the striker in red (20 clean punches to 5), the README preview
until 2026-11-12. The run-3 whole-body videos (`whole-body-v1`) were replaced
by the run-5 ones above on the same day.

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
