import bpy
import bmesh
import os
import sys
import math
import random
from mathutils import Vector, Matrix


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        return {}
    args = argv[argv.index("--") + 1 :]
    out = {}
    i = 0
    while i < len(args):
        k = args[i]
        if not k.startswith("--"):
            i += 1
            continue
        k = k[2:]
        v = True
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            v = args[i + 1]
            i += 1
        out[k] = v
        i += 1
    return out


def _as_int(d, k, default):
    try:
        return int(d.get(k, default))
    except Exception:
        return default


def _as_float(d, k, default):
    try:
        return float(d.get(k, default))
    except Exception:
        return default


def _as_str(d, k, default):
    v = d.get(k, default)
    return str(v) if v is not None else default


def setup_scene(res_x=864, res_y=1104, samples=64):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1)


def setup_camera_and_light(camera_y=-3.5):
    bpy.ops.object.camera_add(location=(0, camera_y, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.object
    const = cam.constraints.new(type="DAMPED_TRACK")
    const.target = target
    const.track_axis = "TRACK_NEGATIVE_Z"

    bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
    sun = bpy.context.object
    sun.data.energy = 5.0
    sun.rotation_euler = (0.6, 0.2, 0.0)


def _build_triangulated_grid(name, aspect, nx, ny, size=2.0):
    """
    Builds a triangulated sheet in the XZ plane (Y=0), centered at origin.
    Uses standard operators to avoid BMesh issues.
    """
    width = size * aspect
    height = size
    
    # Add Grid
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=nx, 
        y_subdivisions=ny, 
        size=max(width, height), # Size is diameter roughly, we scale later or valid
        enter_editmode=False,
        location=(0, 0, 0)
    )
    obj = bpy.context.object
    obj.name = name
    
    # Grid add makes a square, we need to scale it to aspect
    # But wait, primitive_grid_add size is just one number.
    # It creates (subd) cuts.
    # Let's resize it explicitly.
    # Default size=2 gives -1 to 1.
    
    # We want width and height.
    # Current X dim is 2.0, Z (or Y) is 2.0.
    # We want X=width, Y=height.
    # Actually grid is created on XY plane by default. We want XZ.
    
    obj.scale[0] = width / 2.0
    obj.scale[1] = height / 2.0
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Rotate to XZ plane (rotate 90 deg around X)
    # Actually the primitive aligns to Z up (XY plane).
    # We want Y=0 plane (XZ plane).
    obj.rotation_euler = (1.5708, 0, 0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    
    # Triangulate
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj


def _apply_texture_material(obj, image_path):
    img = None
    try:
        img = bpy.data.images.load(image_path)
    except Exception:
        img = None

    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Specular"].default_value = 0.02

    if img is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        nt.links.new(bsdf.inputs["Base Color"], tex.outputs["Color"])

    obj.data.materials.append(mat)

    # Basic UVs: map XZ to UV 0..1
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data

    xs = [v.co.x for v in mesh.vertices]
    zs = [v.co.z for v in mesh.vertices]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    dx = max(max_x - min_x, 1e-9)
    dz = max(max_z - min_z, 1e-9)

    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            co = mesh.vertices[vi].co
            u = (co.x - min_x) / dx
            v = (co.z - min_z) / dz
            uv_layer[li].uv = (u, v)


def _mesh_topology(mesh):
    # rest positions (Vector) in object space
    rest = [v.co.copy() for v in mesh.vertices]

    # unique edges with rest length
    edge_map = {}
    for e in mesh.edges:
        a, b = e.vertices[0], e.vertices[1]
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if k in edge_map:
            continue
        l0 = (rest[k[1]] - rest[k[0]]).length
        edge_map[k] = l0
    edges = [(a, b, l0) for (a, b), l0 in edge_map.items()]

    # build hinge adjacency: map undirected edge -> two opposite verts (k,l)
    # each triangle contributes 3 (edge -> opposite)
    adjacent = {}
    for poly in mesh.polygons:
        if poly.loop_total != 3:
            continue
        a, b, c = poly.vertices[:]
        tri = (a, b, c)
        for (i, j, opp) in ((tri[0], tri[1], tri[2]), (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1])):
            k = (i, j) if i < j else (j, i)
            adjacent.setdefault(k, []).append(opp)

    hinges = []
    for (i, j), opps in adjacent.items():
        if len(opps) != 2:
            continue
        k, l = opps[0], opps[1]
        if k == l or k in (i, j) or l in (i, j):
            continue
        hinges.append((i, j, k, l))

    return rest, edges, hinges


def _dihedral_angle(xi, xj, xk, xl):
    e = xj - xi
    el = e.length
    if el < 1e-12:
        return 0.0
    e = e / el

    n1 = (xj - xi).cross(xk - xi)
    n2 = (xi - xj).cross(xl - xj)
    if n1.length < 1e-12 or n2.length < 1e-12:
        return 0.0
    n1.normalize()
    n2.normalize()

    sinv = e.dot(n1.cross(n2))
    cosv = max(-1.0, min(1.0, n1.dot(n2)))
    return math.atan2(sinv, cosv)


def _rotate_around_axis(p, axis_point, axis_dir, angle):
    if abs(angle) < 1e-12:
        return p
    rot = Matrix.Rotation(angle, 4, axis_dir)
    return axis_point + (rot @ (p - axis_point))


class PaperXPBD:
    def __init__(
        self,
        rest_positions,
        edges,
        hinges,
        crease_map_path,
        seed,
        crease_angle_rad,
        stretch_stiffness,
        hinge_stiffness,
        hinge_max_correction_rad,
        damping,
        gravity,
        attract_strength,
        unfold_strength,
    ):
        rnd = random.Random(seed)

        self.rest = [p.copy() for p in rest_positions]
        self.x = [p.copy() for p in rest_positions]
        self.v = [Vector((0.0, 0.0, 0.0)) for _ in rest_positions]
        self.inv_mass = [1.0 for _ in rest_positions]

        # small noise to break symmetry
        for i in range(len(self.x)):
            self.x[i].y += rnd.uniform(-0.01, 0.01)
            self.x[i].z += rnd.uniform(-0.01, 0.01)

        self.edges = edges
        self.hinges = hinges

        # Load Crease Map
        crease_map_pixels = None
        map_w, map_h = 0, 0
        if crease_map_path and os.path.exists(crease_map_path):
            try:
                img = bpy.data.images.load(crease_map_path)
                map_w, map_h = img.size[0], img.size[1]
                crease_map_pixels = list(img.pixels)
                print(f"Loaded crease map {crease_map_path}: {map_w}x{map_h}")
            except Exception as e:
                print(f"Failed to load crease map: {e}")

        # Get UVs for sampling
        # We assume UVs were created in setup
        # But we need access to them. 
        # Since we are inside the class, and UVs are on the mesh object...
        # We will assume a simple grid UV mapping for now based on vertex coordinates
        # because passing uv_layer list is messy.
        # Or better: Standard Grid UVs.
        xs = [p.x for p in self.rest]
        zs = [p.z for p in self.rest]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        dx_map = max(max_x - min_x, 1e-9)
        dz_map = max(max_z - min_z, 1e-9)

        self.creases = []
        count_creases = 0
        
        for (i, j, k, l) in hinges:
            # Calculate Midpoint UV
            # p_mid = (self.rest[i] + self.rest[j]) * 0.5
            # uv.x = (p_mid.x - min_x) / dx_map
            # uv.y = (p_mid.z - min_z) / dz_map
            
            p_mid = (self.rest[i] + self.rest[j]) * 0.5
            u = (p_mid.x - min_x) / dx_map
            v = (p_mid.z - min_z) / dz_map
            
            is_crease = False
            if crease_map_pixels:
                px = int(u * (map_w - 1))
                py = int(v * (map_h - 1))
                if 0 <= px < map_w and 0 <= py < map_h:
                    idx = (py * map_w + px) * 4
                    val = crease_map_pixels[idx] # Red channel
                    if val > 0.4: # Threshold
                         is_crease = True
            
            # If no map, fall back to random? No, user wants map control.
            # But let's keep a few random ones if map is empty to prevent boring flat sheet.
            if not crease_map_pixels and rnd.random() < 0.05:
                is_crease = True
                
            if is_crease:
                theta0 = rnd.uniform(-crease_angle_rad, crease_angle_rad)
                # Or force sign if we knew M/V. For now random sign is best approximation of chaos.
                if rnd.random() > 0.5: theta0 = -theta0
                
                self.creases.append((i, j, k, l, theta0))
                count_creases += 1

        print(f"Defined {count_creases} creases based on map/logic.")


        self.stretch_stiffness = max(0.0, min(1.0, stretch_stiffness))
        self.hinge_stiffness = max(0.0, min(1.0, hinge_stiffness))
        self.hinge_max_correction_rad = max(0.0, float(hinge_max_correction_rad))
        self.damping = max(0.0, min(0.999, damping))
        self.g = gravity

        self.attract_strength = max(0.0, attract_strength)
        self.unfold_strength = max(0.0, unfold_strength)

        # Anchor points for unfolding (pull sparse points instead of every vertex to preserve isometry).
        xs = [p.x for p in self.rest]
        zs = [p.z for p in self.rest]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        def nearest(target):
            best_i = 0
            best_d = 1e18
            for i, p in enumerate(self.rest):
                d = (p - target).length_squared
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        mid_x = 0.5 * (min_x + max_x)
        mid_z = 0.5 * (min_z + max_z)
        self.unfold_anchors = sorted(
            set(
                [
                    nearest(Vector((min_x, 0.0, min_z))),
                    nearest(Vector((min_x, 0.0, max_z))),
                    nearest(Vector((max_x, 0.0, min_z))),
                    nearest(Vector((max_x, 0.0, max_z))),
                    nearest(Vector((mid_x, 0.0, min_z))),
                    nearest(Vector((mid_x, 0.0, max_z))),
                    nearest(Vector((min_x, 0.0, mid_z))),
                    nearest(Vector((max_x, 0.0, mid_z))),
                ]
            )
        )

    def _project_edges(self):
        for (a, b, l0) in self.edges:
            xa = self.x[a]
            xb = self.x[b]
            d = xb - xa
            ln = d.length
            if ln < 1e-12:
                continue
            c = (ln - l0) * self.stretch_stiffness
            n = d / ln

            wa = self.inv_mass[a]
            wb = self.inv_mass[b]
            wsum = wa + wb
            if wsum < 1e-12:
                continue

            corr = c * n
            self.x[a] = xa + (wa / wsum) * corr
            self.x[b] = xb - (wb / wsum) * corr

    def _project_unfold(self, alpha):
        if alpha <= 0.0:
            return
        for i in self.unfold_anchors:
            self.x[i] = self.x[i].lerp(self.rest[i], alpha)

    def _project_creases(self, crease_scale):
        if self.hinge_stiffness <= 0.0 or crease_scale <= 0.0:
            return
        for (i, j, k, l, theta0) in self.creases:
            xi = self.x[i]
            xj = self.x[j]
            xk = self.x[k]
            xl = self.x[l]
            e = xj - xi
            el = e.length
            if el < 1e-12:
                continue
            axis_dir = e / el

            theta = _dihedral_angle(xi, xj, xk, xl)
            theta_target = theta0 * crease_scale
            err = theta - theta_target
            corr = -self.hinge_stiffness * err
            if self.hinge_max_correction_rad > 0.0:
                corr = max(-self.hinge_max_correction_rad, min(self.hinge_max_correction_rad, corr))

            # rotate opposite verts around the hinge edge to correct dihedral
            self.x[k] = _rotate_around_axis(xk, xi, axis_dir, 0.5 * corr)
            self.x[l] = _rotate_around_axis(xl, xi, axis_dir, -0.5 * corr)

    def step(self, dt, n_iters, crease_scale, unfold_alpha, center=Vector((0.0, 0.0, 0.0))):
        x_prev = [p.copy() for p in self.x]

        # integrate
        for i in range(len(self.x)):
            f = Vector((0.0, 0.0, 0.0))
            f.z += self.g
            if self.attract_strength > 0.0:
                f += -self.attract_strength * (self.x[i] - center)
            self.v[i] = (1.0 - self.damping) * (self.v[i] + dt * f)
            self.x[i] = self.x[i] + dt * self.v[i]

        # project constraints
        for _ in range(n_iters):
            self._project_creases(crease_scale=crease_scale)
            self._project_edges()
            self._project_unfold(alpha=unfold_alpha)

        # update velocities from projected positions (PBD style)
        for i in range(len(self.x)):
            self.v[i] = (self.x[i] - x_prev[i]) / max(1e-9, dt)

    def isometry_error(self):
        max_rel = 0.0
        for (a, b, l0) in self.edges:
            ln = (self.x[b] - self.x[a]).length
            if l0 > 1e-9:
                max_rel = max(max_rel, abs(ln - l0) / l0)
        return max_rel


def main():
    args = _parse_args()

    target_frame = _as_int(args, "frame", 40)
    start_frame = _as_int(args, "start_frame", 33)
    end_frame = _as_int(args, "end_frame", 45)

    seed = _as_int(args, "seed", 7)
    nx = _as_int(args, "nx", 35)
    ny = _as_int(args, "ny", 45)

    pre_roll = _as_int(args, "pre_roll", 50)
    substeps = _as_int(args, "substeps", 3)
    iters = _as_int(args, "iters", 8)

    n_creases = _as_int(args, "n_creases", 240)
    crease_angle_deg = _as_float(args, "crease_angle_deg", 120.0)
    crease_angle_rad = math.radians(crease_angle_deg)

    stretch_stiffness = _as_float(args, "stretch_stiffness", 0.95)
    hinge_stiffness = _as_float(args, "hinge_stiffness", 0.7)
    hinge_max_deg = _as_float(args, "hinge_max_deg", 6.0)
    damping = _as_float(args, "damping", 0.15)
    gravity = _as_float(args, "gravity", -0.15)

    attract0 = _as_float(args, "attract0", 12.0)
    attract_tau = _as_float(args, "attract_tau", 0.06)
    unfold_max = _as_float(args, "unfold_max", 0.22)
    unfold_power = _as_float(args, "unfold_power", 1.6)
    crease_tau = _as_float(args, "crease_tau", 0.35)

    crease_map_path = _as_str(args, "crease_map", "data/crease_map_from_45.png")

    out_dir = _as_str(args, "out_dir", "data/render_paper_xpbd")
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"frame_{target_frame:04d}")

    image_path = _as_str(args, "image", "data/extracted_texture_from_45.png")
    if not os.path.exists(image_path):
        image_path = "data/afisha-selected/3.jpg"

    res_x = _as_int(args, "res_x", 864)
    res_y = _as_int(args, "res_y", 1104)
    samples = _as_int(args, "samples", 64)
    setup_scene(res_x=res_x, res_y=res_y, samples=samples)
    setup_camera_and_light()

    # Determine aspect ratio if image is available via Blender load
    aspect = 864 / 1104
    try:
        img = bpy.data.images.load(image_path)
        if img and img.size[1] > 0:
            aspect = img.size[0] / img.size[1]
    except Exception:
        pass

    obj = _build_triangulated_grid("PaperSheet", aspect=aspect, nx=nx, ny=ny, size=2.0)
    _apply_texture_material(obj, image_path=image_path)
    bpy.ops.object.shade_smooth()

    # Use rest topology directly from the mesh
    mesh = obj.data
    rest, edges, hinges = _mesh_topology(mesh)
    if edges:
        l0s = [e[2] for e in edges]
        print(
            f"Topology: verts={len(rest)} edges={len(edges)} hinges={len(hinges)} "
            f"l0_min={min(l0s):.6f} l0_max={max(l0s):.6f} creases_target={n_creases}"
        )
    else:
        print(f"Topology: verts={len(rest)} edges=0 hinges={len(hinges)} creases_target={n_creases}")

    sim = PaperXPBD(
        rest_positions=rest,
        edges=edges,
        hinges=hinges,
        crease_map_path=crease_map_path,
        seed=seed,
        crease_angle_rad=crease_angle_rad,
        stretch_stiffness=stretch_stiffness,
        hinge_stiffness=hinge_stiffness,
        hinge_max_correction_rad=math.radians(hinge_max_deg),
        damping=damping,
        gravity=gravity,
        attract_strength=attract0,
        unfold_strength=unfold_max,
    )

    fps = _as_float(args, "fps", 24.0)
    dt = 1.0 / fps / max(1, substeps)

    # Pre-roll to reach a compact crumpled state (frame 33)
    for _ in range(max(0, pre_roll)):
        sim.step(dt=dt, n_iters=iters, crease_scale=1.0, unfold_alpha=0.0)

    def schedule(frame):
        u = 0.0
        if end_frame > start_frame:
            u = (frame - start_frame) / (end_frame - start_frame)
        u = max(0.0, min(1.0, u))

        # unfold increases over time (pull to rest plane)
        unfold_alpha = sim.unfold_strength * (u**unfold_power)
        unfold_alpha = max(0.0, min(0.9, unfold_alpha))

        # creases relax slower than unfolding so ridges remain visible mid-way
        # creases relax slower than unfolding so ridges remain visible mid-way
        # New "Math": Transition Logic
        # We want creases to stay strong until the very end, then snap flat.
        # Or transition linearly? User said "broken harder".
        # Let's keep them scale 1.0 until u > 0.8?
        # Actually, exp decay is decent for organic look.
        # Let's add a "Crease Persistence" term.
        
        # Sigmoid-like transition for creases
        # k = 10 (sharpness), u0 = 0.5 (center)
        # scale = 1.0 / (1.0 + exp(k * (u - u0))) ? No, we un-crumple.
        # We want scale ~ 1 at u=0, scale ~ 0 at u=1.
        crease_scale = math.exp(-u / max(1e-6, crease_tau))
        # Clamp
        crease_scale = max(0.0, min(1.0, crease_scale))

        # attraction drops very fast to "pop" open from ball
        attract = attract0 * math.exp(-u / max(1e-6, attract_tau))
        return crease_scale, unfold_alpha, attract

    render_from = _as_int(args, "render_from", target_frame)

    # Run frames up to target
    for f in range(start_frame, target_frame + 1):
        crease_scale, unfold_alpha, attract = schedule(f)
        sim.attract_strength = attract

        for _ in range(max(1, substeps)):
            sim.step(dt=dt, n_iters=iters, crease_scale=crease_scale, unfold_alpha=unfold_alpha)

        # Push sim positions into the Blender mesh
        for i, v in enumerate(mesh.vertices):
            v.co = sim.x[i]
        mesh.update()
        
        if f >= render_from:
            bpy.context.view_layer.update()
            scene = bpy.context.scene
            scene.frame_current = f
            # Make sure we save unique filenames
            current_out_base = os.path.join(out_dir, f"frame_{f:04d}")
            scene.render.filepath = "//" + current_out_base.replace("\\", "/").lstrip("/")
            bpy.ops.render.render(write_still=True)
            print(f"Rendered: {current_out_base}.png")

    iso_err = sim.isometry_error()
    print(f"Isometry max relative edge error: {iso_err:.6f}")


if __name__ == "__main__":
    main()
