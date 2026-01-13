# Blender reference-driven displacement (clean poster)

This attempt renders a **clean** paper (front = poster texture, back = white) in Blender, but drives the crumple/uncrumple motion using a **per-frame displacement map extracted from the reference frames**.

Why this approach:
- No “reference pixels leaking through” (we never composite the reference into the output).
- Very high similarity is achievable because geometry/shading cues are driven by the reference.
- Fully deterministic and batchable for many posters.

## Debug: match frames 33–44

1) Extract a poster from the reference frame 44 (for high similarity testing).
2) Extract displacement maps for frames 33–44 from `data/frames_ref`.
3) Render frames 33–44 in Blender (Cycles) using that displacement sequence.
4) Produce per-frame metrics + a 2-row comparison image (top=ref, bottom=gen).

```bash
python attempt_blender_refdisplace_001/run.py \
  --ref_dir data/frames_ref \
  --out_dir attempt_blender_refdisplace_001/out_debug_33_44 \
  --frame_start 33 --frame_end 44 \
  --poster_from_ref 44 \
  --tune
```

Outputs:
- Generated frames: `.../generated_frames/frame_0033.png` … `frame_0044.png`
- Comparison grid: `.../compare_grid_33_44.png`
- Metrics: `.../diff_report.csv`
- Best params used: `.../best_params.json`

## Batch posters

```bash
python attempt_blender_refdisplace_001/run.py \
  --ref_dir data/frames_ref \
  --afisha_dir data/afisha \
  --out_dir attempt_blender_refdisplace_001/out_afisha_33_44 \
  --frame_start 33 --frame_end 44
```

Each poster gets: `out_dir/posters/<stem>/generated_frames/frame_0033.png` …

## Notes

- Uses Blender available in this environment (`blender` CLI, currently 2.82).
- EEVEE often fails headless; this uses Cycles CPU by default.

