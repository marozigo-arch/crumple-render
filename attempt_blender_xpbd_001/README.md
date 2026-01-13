# Blender XPBD paper (physics-based, clean posters)

This attempt uses a lightweight **XPBD paper simulation** (runs inside Blender headless) and applies a **two-sided material**:
- front side: poster texture
- back side: solid white

It renders frames `33..44` to mirror the reference segment, but does **not** reuse reference pixels (no “DITCH DOUBT” leaks).

## Run (batch posters)

```bash
python attempt_blender_xpbd_001/run.py \
  --afisha_dir data/afisha-selected \
  --out_dir attempt_blender_xpbd_001/out_afisha_selected_33_44 \
  --frame_start 33 --frame_end 44 \
  --match_ref_area \
  --ref_time_warp_gamma 2.0 \
  --unfold_gamma 2.8 \
  --make_cycle --cycle_sample_n 10 \
  --make_sequence
```

Outputs:
- per poster frames (viewable, composited): `.../posters/<poster_stem>/frames/frame_0033.png` … `frame_0044.png`
- per poster frames (raw RGBA from Blender): `.../posters/<poster_stem>/frames_rgba/frame_0033.png` … `frame_0044.png`
- quick check (frame 44 vs input): `.../posters/<poster_stem>/poster_vs_frame44.png`
- optional 10-step unfold+fold: `.../posters/<poster_stem>/cycle_frames/frame_0001.png` …
- optional combined sequence for all posters: `.../sequence/frames/frame_000001.png` … + `.../sequence/manifest.json`

## Smooth unfold + video

This pipeline is tuned so that the paper coverage grows smoothly across `33..44` (not "4 keyframes + duplicates"):

```bash
python attempt_blender_xpbd_001/run.py \
  --afisha_dir afisha \
  --out_dir attempt_blender_xpbd_001/out_area_match_v19_final \
  --frame_start 33 --frame_end 44 \
  --ref_dir data/frames_ref \
  --match_ref_area \
  --ref_time_warp_gamma 2.0 \
  --samples 24

python attempt_blender_xpbd_001/make_video.py \
  --frames_dir attempt_blender_xpbd_001/out_area_match_v19_final/posters/3/frames \
  --out_mp4 attempt_blender_xpbd_001/out_area_match_v19_final/posters/3/unfold_pause_fold.mp4 \
  --fps 24 --pause_frames 45
```
