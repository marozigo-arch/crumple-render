# Blender flow-warp paper (clean, reference-driven)

This attempt uses the reference frames only as a **motion driver**:
- Dense optical flow maps each target frame back to a chosen **source** frame (default: 44).
- We convert flow to a per-pixel **UV sampling map** so Blender can sample a poster texture with the same deformation.
- A paper mask + backside mask (from the reference) ensures:
  - only the paper is visible
  - the backside stays white

Because we do **no** pixel compositing from the reference, the output stays clean (no “reference leaks”).

## Debug: match frames 33–44

```bash
python attempt_blender_flowwarp_001/run.py \
  --ref_dir data/frames_ref \
  --out_dir attempt_blender_flowwarp_001/out_debug_33_44 \
  --frame_start 33 --frame_end 44 \
  --source_frame 44 \
  --poster_from_ref_frame 44
```

Outputs:
- `.../generated_frames/frame_0033.png` … `frame_0044.png`
- `.../compare_grid_33_44.png` (top ref / bottom gen)
- `.../diff_report.csv`

## Batch posters

```bash
python attempt_blender_flowwarp_001/run.py \
  --ref_dir data/frames_ref \
  --afisha_dir data/afisha \
  --out_dir attempt_blender_flowwarp_001/out_afisha_33_44 \
  --frame_start 33 --frame_end 44 \
  --source_frame 44
```

## Notes

- Uses Blender in this environment (2.82); runs headless with Cycles CPU.

