# Clean paper crumple (reference-driven deformation)

This attempt fixes the main visual issue in `attempt_flow_driver_001`: **no reference pixels ever appear inside the paper**.

The reference is used only as:
- **motion driver** (dense optical flow per frame → deformation)
- **shading driver** (fold lighting extracted from the reference, but filtered so the original printed poster texture does not leak)
- **paper silhouette / backside visibility** (paper mask + “backside white” mask)

## Debug window (frames 33–44)

Uses `frame_0044.png` as the “poster” so you can directly compare similarity:

```bash
python attempt_flow_driver_002_clean/run.py \
  --ref_dir data/frames_ref \
  --out_dir attempt_flow_driver_002_clean/out_debug_33_44 \
  --frame_start 33 --frame_end 44 \
  --source_frame 44 \
  --poster_from_ref_frame 44 \
  --bg_mode solid
```

Outputs:
- `.../generated_frames/frame_0033.png` … `frame_0044.png`
- `.../compare_grid_33_44.png` (top: reference, bottom: generated)
- `.../diff_report.csv`

## Full cycle for many posters (1–269)

```bash
python attempt_flow_driver_002_clean/run.py \
  --ref_dir data/frames_ref \
  --afisha_dir data/afisha \
  --out_dir attempt_flow_driver_002_clean/out_afisha_full \
  --full_cycle \
  --source_frame 44 \
  --bg_mode solid \
  --l_mode shading
```

Outputs:
- Per poster: `.../posters/<poster_stem>/generated_frames/frame_0001.png` … `frame_0269.png`
- Concatenated sequence for all posters: `.../sequence_all/generated_frames/frame_000001.png` ...

If you want **only the paper** (for compositing into a separate background), add `--transparent` to output RGBA PNGs.

## Key knobs

- `--bg_mode solid|ref|ref_blur` — what to use outside the paper.
- `--transparent` — output RGBA PNGs with alpha=paper mask (paper only, no background).
- `--l_mode none|shading|target` — lighting:
  - `shading`: recommended for real posters (avoids leaking the reference print)
  - `target`: best pixel-match when poster==reference (debug only)
