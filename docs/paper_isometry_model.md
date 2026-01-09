# Paper (vs Cloth) model for crumple → unfold

This project’s “cloth” attempts approximate the effect well, but the underlying physics is different:

- **Cloth**: low in-plane stiffness → allows metric change (stretch/shear) and produces **soft wrinkles**.
- **Paper**: extremely high in-plane stiffness → almost **isometric** (locally preserves lengths/area) and concentrates deformation into **sharp ridges/creases** (plastic hinges).

Below is a formulation that is practical to implement in Blender via a custom solver (bpy), while explicitly separating:
1) isometry (stretch/shear suppression) and  
2) crease/ridge formation (hinge-like bending with plasticity).

---

## 1) Continuous model (thin sheet)

Let the paper be a midsurface embedding:

\[
\mathbf{x}(u,v)\in \mathbb{R}^3,\quad (u,v)\in \Omega\subset\mathbb{R}^2
\]

### 1.1 Isometry (paper’s “stiffness”)

The intrinsic metric induced by the embedding is:

\[
g_{ij}=\partial_i\mathbf{x}\cdot \partial_j\mathbf{x}
\]

For an **isometric** deformation relative to the flat parameterization:

\[
g_{ij}=\delta_{ij}\quad\text{(or }g_{ij}=g_{ij}^0\text{ for a general rest metric)}
\]

Equivalently, paper’s stretching energy coefficient is effectively “infinite” compared to bending:

\[
E_{\text{stretch}}=\frac{Y h}{2}\int_\Omega \|g-g^0\|^2\,dA,\quad Yh\gg Bh^3
\]

where \(Y\) is Young’s modulus, \(h\) thickness, and \(B\propto Yh^3\) is the bending modulus.

The **paper limit** is: \(E_{\text{stretch}}\to \infty\Rightarrow g\approx g^0\).  
That yields **developable** surfaces except where energy concentrates into ridges/creases.

### 1.2 Bending and creases

For a smooth sheet, bending energy (e.g. Koiter / Helfrich type) is:

\[
E_{\text{bend}}=\frac{B}{2}\int_\Omega \| \mathbf{II}-\mathbf{II}^0\|^2\,dA
\]

But real paper crumpling is not smooth: curvature concentrates into a sparse set of ridges and creases.
Model this by introducing **hinges** (creases) where bending becomes cheap, and add **plasticity**:

- Elastic hinge: prefers a rest dihedral angle \(\bar\theta\).
- Plastic update: \(\bar\theta\) changes when the fold exceeds a yield threshold.

---

## 2) Discrete model (triangle mesh): isometry + hinge creases

Use a triangulated sheet with vertices \(\mathbf{x}_i\) and rest (flat) positions \(\mathbf{x}_i^0\).

### 2.1 Isometry via edge-length constraints

For every mesh edge \(e=(i,j)\), enforce:

\[
C_e(\mathbf{x}) = \|\mathbf{x}_j-\mathbf{x}_i\| - \ell^0_{ij} = 0
\]

This is a standard discrete proxy for preserving the first fundamental form (metric).

### 2.2 Sharp creases via dihedral (hinge) constraints

For an internal edge \(e=(i,j)\) shared by triangles \((i,j,k)\) and \((j,i,l)\), define dihedral angle:

\[
\theta_e = \operatorname{atan2}\Big(\hat{\mathbf{e}}\cdot(\mathbf{n}_1\times \mathbf{n}_2),\ \mathbf{n}_1\cdot \mathbf{n}_2\Big)
\]

with:
- \(\hat{\mathbf{e}}=(\mathbf{x}_j-\mathbf{x}_i)/\|\mathbf{x}_j-\mathbf{x}_i\|\)
- \(\mathbf{n}_1\propto (\mathbf{x}_j-\mathbf{x}_i)\times(\mathbf{x}_k-\mathbf{x}_i)\)
- \(\mathbf{n}_2\propto (\mathbf{x}_i-\mathbf{x}_j)\times(\mathbf{x}_l-\mathbf{x}_j)\)

Hinge energy on selected crease edges:

\[
E_{\text{hinge}} = \sum_{e\in \mathcal{C}} \frac{k_e}{2}(\theta_e-\bar\theta_e)^2
\]

### 2.3 Plasticity (optional but “paper-like”)

If \(|\theta_e-\bar\theta_e|>\theta_{\text{yield}}\), update \(\bar\theta_e\) (“permanent crease”):

\[
\bar\theta_e \leftarrow \bar\theta_e + \eta\,(\theta_e-\bar\theta_e)
\]

where \(\eta\in(0,1]\) is a plastic rate.

---

## 3) Solver choice (practical): XPBD / PBD projection

To keep implementation simple and stable inside Blender Python, use constraint projection per step:

1) Predict positions with forces (gravity, damping, etc.)  
2) Project constraints:
   - edge-length constraints (near-zero compliance → near-isometry)
   - hinge constraints for creases (high stiffness → sharp ridges)
3) Update velocities from corrected positions

Key point: the “paper” character comes from:
- **very stiff edge-length constraints** (almost no stretch)
- **hinge constraints** (few, strong, possibly plastic)
instead of cloth’s smooth, distributed bending.
