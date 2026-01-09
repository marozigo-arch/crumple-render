# Visual transformation (reference frames 33 → 44)

Based on `data/frames_ref/frame_0033.png` … `data/frames_ref/frame_0044.png` (focus: frame 40).

## Timeline summary

### Frame 33 (crumpled “ball”)
- Very compact silhouette (most area hidden by self-occlusion).
- High curvature concentrated everywhere; no single dominant fold direction.

### Frame 34–36 (rapid “pop open”)
- The sheet expands from a compact ball into a large visible surface in ~1–3 frames.
- Motion is dominated by **large-scale opening** rather than small wrinkle relaxation.
- New features: a broad saddle-like warp and a few strong ridges.

### Frame 37–40 (key intermediate state: 40)
- The sheet is mostly open (near full extent), but remains **non-planar**:
  - a prominent **central ridge / valley system** creating large triangular shading regions
  - corners/edges show **rolled / lifted** behavior rather than floppy drape
  - folds are **sharp-ish** (paper-like creases) instead of smooth cloth wrinkles
- Frame 40 is “open” but not “flat”: curvature is still organized into sparse crease lines.

### Frame 41–44 (planarization)
- The sheet approaches a flat rectangle while keeping residual creases.
- Large folds diminish, and remaining deformation becomes faint, mostly visible as subtle shading/texture warps.

## What this implies for simulation

To match frame 40, you need a model where:
- in-plane stretch/shear is strongly suppressed (**near-isometry**)
- bending is allowed but concentrates into sparse **sharp crease lines**
- unfolding is driven by a “return to rest plane” tendency while creases relax more slowly

This is exactly the regime where cloth models struggle: cloth distributes curvature and allows metric distortion, producing softer, rubbery motion.
