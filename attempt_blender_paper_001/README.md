# Blender paper crumple/uncrumple (10+10 frames)

Goal: generate a **clean** paper animation (poster on front, white backside) using Blender cloth simulation.

- Fold: `frames_out` frames
- Unfold: `frames_out` frames (reverse playback of the baked simulation)
- The simulation itself runs for `sim_frames` frames for smoother physics, then we sample it down.

## Run (batch posters)

```bash
python attempt_blender_paper_001/run.py \
  --afisha_dir data/afisha \
  --out_dir attempt_blender_paper_001/out_demo \
  --max_posters 2 \
  --frames_out 10 \
  --sim_frames 80 \
  --engine CYCLES \
  --samples 24
```

Outputs:
- Per poster frames: `out_dir/posters/<poster_stem>/frames/frame_0001.png` … `frame_0020.png`
- Contact sheet (first poster): `out_dir/contact_sheet.png`
- Concatenated sequence: `out_dir/sequence_all/frames/frame_000001.png` ...

## Notes

- Use `--engine CYCLES` for nicer lighting (slower).
- In many headless environments `EEVEE` cannot start (“Unable to open a display”), so `CYCLES` is the safe default here.
- For per-poster motion variation, add `--vary_sim` (re-bakes simulation per poster; slow).
- Speed/quality knobs: `--samples`, `--grid_x`, `--grid_y`, `--cloth_quality`, `--sim_frames`.
