## attempt_blender_cloth_v8_001

Blender (cloth + magnetic core) crumple/uncrumple generator with:
- Front side = poster texture
- Back side = white paper
- Headless render (`blender -b`)
- Output frames numbered to match reference window (default `33..44`)
- Built-in verification artifacts: per-poster `compare_grid_33_44.png` (top = reference, bottom = generated)

### Run

```bash
python attempt_blender_cloth_v8_001/run.py \
  --afisha_dir data/afisha-selected \
  --out_dir attempt_blender_cloth_v8_001/out_afisha_selected_33_44_v1 \
  --frame_start 33 --frame_end 44 \
  --ref_dir data/frames_ref
```

Outputs (per poster) live under `.../posters/<poster_stem>/`.

