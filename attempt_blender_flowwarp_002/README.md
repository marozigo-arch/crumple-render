# Blender flow-warp paper (clean + reference-calibrated shading)

This attempt renders a **clean** poster on a paper silhouette driven by the reference motion:

- Dense optical flow maps each target frame back to a chosen **source** frame (default: `44`).
- Flow is converted into a per-pixel **UV sampling map** so Blender can sample the poster with the same deformation.
- A **paper alpha** mask keeps only the paper visible; a **backside** mask forces the back to stay white.
- A **light ratio** map is derived from the reference (target vs warped source) and applied multiplicatively so the result matches reference lighting/creases without “leaking” the reference poster.

## Debug: tune and match frames 33–44

```bash
python attempt_blender_flowwarp_002/run.py \
  --ref_dir data/frames_ref \
  --out_dir attempt_blender_flowwarp_002/out_best_33_44 \
  --frame_start 33 --frame_end 44 \
  --source_frame 44 \
  --poster_from_ref_frame 44 \
  --autotune \
  --renderer opencv
```

Outputs:
- `.../best/generated_frames/frame_0033.png` … `frame_0044.png`
- `.../best/compare_grid_33_44.png` (top ref / bottom gen)
- `.../best/diff_report.csv`
- `.../best_params.json`

## Batch posters

```bash
python attempt_blender_flowwarp_002/run.py \
  --ref_dir data/frames_ref \
  --afisha_dir data/afisha \
  --out_dir attempt_blender_flowwarp_002/out_afisha_full \
  --frame_start 1 --frame_end 269 \
  --source_frame 44 \
  --renderer opencv
```

Notes:
- Batch output folders:
  - `.../posters/<poster>/generated_frames/` per poster.
  - `.../sequence_all/frames/frame_000001.png ...` concatenated sequence across all posters.
- Blender renderer is kept as an option (`--renderer blender`), but the OpenCV renderer is the reference implementation for matching frames 33–44.
