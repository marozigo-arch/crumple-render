# Codex-Derived Framework: Paper Crumpling Simulation

## Goal Analysis
To replicate the "uncrumpling" of a paper sheet (Frames 33-44), specifically Frame 40.
**Key Physics challenge**: Paper is an **inextensible** material (Isometric deformation) with **plasticity** (memory of creases). Cloth simulations often look like rubber (stretchy) or silk (too soft).

## Mathematical Formulation
1.  **Metric Preservation constraint**: $ds^2$ (distance between points) must remain constant.
    *   *Implementation*: High `Tension Stiffness`, High `Shear Stiffness` in Blender Cloth.
2.  **Ridge Formation**: Energy $E$ minimizes bending, but when curvature $\kappa$ exceeds a threshold, the material yields (Plasticity).
    *   *Implementation*: Use Blender Cloth **Plasticity** settings (`Plasticity` > 0, `Shrinking` factor).
3.  **Dynamics**: The unfolding is driven by the release of elastic energy stored in the non-plastic regions, damped by air resistance.

## Proposed Framework Structure
**Phase 1: Generic Realistic Crumpling (Current Focus)**
Goal: Generate a high-quality animation of a sheet crumpling into a ball, looking like stiff paper, not cloth.
Algorithm: "Incremental Crumpling" (Flat -> Ball).
1.  **Stage 0**: Flat Sheet.
2.  **Stage 1 - Major Folds**: Apply forces to fold corners inward or create large primary creases. Determine "crease lines" and weaken stiffness there? Or just force geometry?
    *   *Decision*: Use "Pinched Force Fields" or "Collision Objects" to force folds.
3.  **Stage 2 - Densification**: Continue compressing while adding randomized smaller folds.
4.  **Stage 3 - Ball**: Compress into final spherical shape.

**Phase 2: Reference Matching (Future)**
Goal: Guide Phase 1 to match Frame 33 creases.
- Use `extract_creases.py` on reference.
- Project these lines onto the mesh in Stage 1.

## Proposed Framework Structure (Phase 1)
A Python-based standalone script `generate_crumple.py`.

### 1. Simulation Engine (`sim_engine`)
A class wrapping Blender headless execution.
- Generates a scene with:
    - High-res grid (Subsurf levels).
    - **Cloth Modifier**:
        - `Mass`: variable.
        - `Stiffness`: High.
        - `Plasticity`: variable (Key to specific "paper" look).
    - **Forces**: Turbulence + "Crumple Force" (Colliders).

### 2. Comparator (`visual_metric`)
A class for Structural Comparison.
- **extract_creases(image)**: Adaptive Thresholding + Skeletonization.
- **compute_score(ref, gen)**: 
    - **IoU** of Crease Maps (Structure).
    - **Gradient Direction** alignment (Orientation).
    - **SSIM** (Texture quality).

### 3. Optimization Loop (`optimizer`)
Iterative search for parameters $\theta = \{Mass, Bending, Plasticity, Turbulence\}$.
- Uses a Grid Search (robust) or Random Search.
- Logs results to `data/optimization_log.csv`.
- Saves best renders.

## User Review Required
> [!IMPORTANT]
> I will be using **Plasticity** in the Cloth settings. This is a significant change from previous experiments and should produce more realistic "dead" folds that don't spring back completely.

### [System Scripts]
#### [NEW] [generate_crumple.py](file:///workspaces/codespaces-jupyter/generate_crumple.py)
New main script for Phase 1.
- Sets up Cloth with High Stiffness (Paper).
- Implements "Staged Crumpling":
    - **Step 1**: Large folds (Corners).
    - **Step 2**: Compression.
- Uses `Plasticity` to keep creases.

## Verification Plan
### Automated Tests
1.  **Run Simulation**: `blender -b -P generate_crumple.py`.
2.  **Check Output**: Ensure `render_0001.png` to `render_0100.png` are generated.

### Manual Verification
- **Visual Check**: Does it look like paper?
    - Edges start flat.
    - Folds are sharp?
    - Final state is a tight ball?

