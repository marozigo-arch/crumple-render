# Flow-driven paper crumple (reference-matched)

This folder contains a deterministic, reference-driven generator that reproduces the paper crumple look by:

1) computing dense optical flow from each reference frame back to a chosen **source** reference frame (default: `frame_0044.png`)
2) sampling an input poster through that mapping (so the poster deforms like the reference)
3) taking **lighting/creases** from the target reference frame (L channel) while taking **color/albedo** from the poster (a/b channels)
4) preserving the **white backside** (detected as near-white pixels inside the paper mask)

It is designed to match `data/frames_ref` frames 33–44 first (debug window), and then apply the same deformation to many posters from `data/afisha`.

## Quick start (frames 33–44)

Generate 33–44 using the poster extracted from `frame_0044.png` and create the required 2-row comparison grid:

```bash
python attempt_flow_driver_001/run.py \
  --ref_dir data/frames_ref \
  --out_dir attempt_flow_driver_001/out_debug_33_44 \
  --frame_start 33 --frame_end 44 \
  --source_frame 44 \
  --poster_from_ref_frame 44
```

Outputs:
- `attempt_flow_driver_001/out_debug_33_44/generated_frames/frame_0033.png` ... `frame_0044.png`
- `attempt_flow_driver_001/out_debug_33_44/compare_grid_33_44.png` (top: reference, bottom: generated)
- `attempt_flow_driver_001/out_debug_33_44/diff_report.csv` (per-frame % difference + SSIM)

## Run on many posters

```bash
python attempt_flow_driver_001/run.py \
  --ref_dir data/frames_ref \
  --afisha_dir data/afisha \
  --out_dir attempt_flow_driver_001/out_afisha_33_44 \
  --frame_start 33 --frame_end 44 \
  --source_frame 44
```

Each poster gets its own folder under `out_dir/posters/<poster_stem>/generated_frames/`.

