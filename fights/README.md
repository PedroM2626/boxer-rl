# Boxing videos

## README preview

`whole_body_vs_champion_preview.mp4` is a 12-second, seed-0 bout of the
whole-body policy (`checkpoints/whole-body`, run 6, red fists) against the champion
stack (blue fists), rendered at 25 fps; the GIF is the same footage at 15 fps
and 640 px wide. Red lands 29 clean punches to 1, no knockdowns. A single
preview, not an evaluation (`docs/whole-body.md` has those).

```bash
bash tools/run_spar.sh boxer.spar_match --ring --seconds 12 --knockdowns 3 --seeds 0   --checkpoint checkpoints/spar-native --red-tactics checkpoints/whole-body   --blue-checkpoint checkpoints/spar --blue-tactics checkpoints/tactics --blue-arm checkpoints/arm-skill   --fps 25 --video fights/whole_body_vs_champion_preview.mp4 --output fights/whole_body_vs_champion_preview.json
ffmpeg -i fights/whole_body_vs_champion_preview.mp4   -filter_complex "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"   -loop 0 fights/whole_body_vs_champion_preview.gif
```

(`imageio_ffmpeg.get_ffmpeg_exe()` in the curriculum runtime gives an
`ffmpeg` binary.)

## Whole-body bouts (2026-11-13)

60 s at 60 fps, knockdown rules (3), each opponent on its own legs, the
promoted run-6 policy; seed 0 unless noted (`runs/wb_videos6.sh`):

| Video | Clean punches | Knockdowns |
| --- | --- | --- |
| `whole_body_vs_champion_60s.mp4` (seed 1) | 218 : 60 | 0 : 0 |
| `whole_body_vs_striker_60s.mp4` | 194 : 16 | 0 : 0 |
| `whole_body_vs_itself_60s.mp4` | 176 : 186 | 0 : 0 |
| `whole_body_getup_vs_champion.mp4` (seed 138, 50 s, 25 fps, `--getup checkpoints/getup`, `runs/getup_clip.py`, run 5 in red, now `whole-body-v2`): blue down at 3.8 s, red falls while waiting at 4.8 s, both up after the 8 s count and box on | 117 : 20 | 1 : 1 |

### A boxing rhythm (2026-11-14)

`whole-body-style` (run 12) in red, 60 s at 60 fps, knockdown rules (3)
(`runs/wb_eval12.sh`): `style_vs_champion_60s.mp4` (seed 1, the champion
in blue on its own legs), 119 : 37 clean punches, no knockdowns;
`style_vs_itself_60s.mp4` (seed 0), 74 : 75, no knockdowns. Compare with
`whole_body_vs_champion_60s.mp4` and `whole_body_vs_itself_60s.mp4`
(run 6, 218 : 60 and 176 : 186).

### Hits more, falls more (2026-11-14)

`run11_vs_run6_120s.mp4` (and `.json`): run 11's 0.125M step (red, rejected,
`runs/whole-body-native11/gate/step-125952`) against run 6 (blue), 120 s at
60 fps, seed 4 of its 32-bout evaluation (`runs/wb_video11.sh`). Red
outpunches blue 349 : 327 but is knocked down at 30.3 s and 43.7 s (referee
restarts); 2 knockdowns are worth 20 points, so blue keeps it close on the
cards (349 : 347). Over all 32 bouts run 11 wins 31 but falls 21 times to
run 6's 7 ([whole body](../docs/whole-body.md#falls-against-itself-runs-7-to-11-rejected)).

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
by the run-5 ones on the same day, and those by the run-6 ones above on
2026-11-13.

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
