import math
import os
import sys

import bpy


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
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    # Render paper with alpha so we can composite a background and also validate 'paper-only' pixels.
    scene.render.film_transparent = True
    # Keep colors stable for poster matching (avoid Filmic tonemapping).
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1)


def setup_camera_and_light():
    bpy.ops.object.camera_add(location=(0, -3.0, 0.05))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.1

    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.object
    const = cam.constraints.new(type="DAMPED_TRACK")
    const.target = target
    const.track_axis = "TRACK_NEGATIVE_Z"

    # Side-ish key light similar to reference
    bpy.ops.object.light_add(type="AREA", location=(-2.2, -2.4, 1.8))
    key = bpy.context.object
    key.data.energy = 900
    key.data.size = 1.6
    key.rotation_euler = (math.radians(65), math.radians(-10), math.radians(25))

    bpy.ops.object.light_add(type="AREA", location=(1.8, -2.8, 1.4))
    fill = bpy.context.object
    fill.data.energy = 220
    fill.data.size = 2.2
    fill.rotation_euler = (math.radians(70), math.radians(0), math.radians(-25))


def build_sheet(aspect, nx, ny, size=2.0):
    width = size * aspect
    height = size
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=nx,
        y_subdivisions=ny,
        size=max(width, height),
        enter_editmode=False,
        location=(0, 0, 0),
    )
    obj = bpy.context.object
    obj.name = "PaperSheet"

    obj.scale[0] = width / 2.0
    obj.scale[1] = height / 2.0
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    obj.rotation_euler = (math.radians(90), 0, 0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Triangulate for hinge-based crease constraints.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")

    # Render smoother surface while keeping the simulation on the base mesh.
    subsurf = obj.modifiers.new(name="Subsurf", type="SUBSURF")
    subsurf.levels = 2
    subsurf.render_levels = 2
    return obj


def apply_uv_map_xz(obj):
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


def apply_two_sided_paper_material(obj, image_path: str):
    img = None
    try:
        img = bpy.data.images.load(image_path)
    except Exception:
        img = None

    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"

    tex = nt.nodes.new("ShaderNodeTexImage")
    if img is not None:
        tex.image = img
        tex.interpolation = "Closest"

    # Backfacing==1 -> white, else poster
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(tex.outputs["Color"], mix.inputs["Color1"])
    mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Specular"].default_value = 0.02
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    emission = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(mix.outputs["Color"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0

    unlit = nt.nodes.new("ShaderNodeValue")
    unlit.name = "UNLIT_FACTOR"
    unlit.outputs[0].default_value = 0.0

    mix_shader = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(unlit.outputs[0], mix_shader.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])
    nt.links.new(emission.outputs["Emission"], mix_shader.inputs[2])
    nt.links.new(mix_shader.outputs["Shader"], out.inputs["Surface"])

    obj.data.materials.append(mat)


def main():
    args = _parse_args()
    poster_path = _as_str(args, "poster", None)
    out_dir = _as_str(args, "out_dir", "attempt_blender_xpbd_001/out_single")

    frame_start = _as_int(args, "frame_start", 33)
    frame_end = _as_int(args, "frame_end", 44)
    fps = _as_float(args, "fps", 24.0)

    nx = _as_int(args, "nx", 35)
    ny = _as_int(args, "ny", 45)
    samples = _as_int(args, "samples", 48)

    # XPBD parameters (tunable)
    pre_roll = _as_int(args, "pre_roll", 50)
    substeps = _as_int(args, "substeps", 3)
    iters = _as_int(args, "iters", 8)
    attract0 = _as_float(args, "attract0", 12.0)
    attract_tau = _as_float(args, "attract_tau", 0.06)
    unfold_max = _as_float(args, "unfold_max", 0.35)
    unfold_power = _as_float(args, "unfold_power", 2.0)
    crease_tau = _as_float(args, "crease_tau", 0.28)
    seed = _as_int(args, "seed", 7)
    unlit_final_frames = _as_int(args, "unlit_final_frames", 2)

    os.makedirs(out_dir, exist_ok=True)

    res_x = _as_int(args, "res_x", 864)
    res_y = _as_int(args, "res_y", 1104)
    setup_scene(res_x=res_x, res_y=res_y, samples=samples)
    setup_camera_and_light()
    bpy.context.scene.render.fps = int(round(fps))

    aspect = res_x / res_y
    if poster_path and os.path.exists(poster_path):
        try:
            img = bpy.data.images.load(poster_path)
            if img and img.size[1] > 0:
                aspect = img.size[0] / img.size[1]
        except Exception:
            pass

    sheet = build_sheet(aspect=aspect, nx=nx, ny=ny, size=2.0)
    apply_uv_map_xz(sheet)
    apply_two_sided_paper_material(sheet, image_path=poster_path if poster_path else "")
    bpy.ops.object.shade_smooth()

    # Reuse the existing XPBD integrator from simulate_paper_xpbd.py by embedding the minimal parts here:
    from mathutils import Vector
    import random
    from mathutils import Matrix

    mesh = sheet.data
    rest = [v.co.copy() for v in mesh.vertices]

    # edges
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

    # hinges
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

    rnd = random.Random(seed)
    v = [Vector((0.0, 0.0, 0.0)) for _ in rest]
    x = [p.copy() for p in rest]

    # Create random crease targets by sampling hinges
    crease_angle_rad = math.radians(110.0)
    crease_targets = {}
    for (i, j, k, l) in rnd.sample(hinges, min(len(hinges), 320)):
        crease_targets[(i, j, k, l)] = rnd.choice([-1.0, 1.0]) * crease_angle_rad

    stretch_stiffness = 0.95
    hinge_stiffness = 0.7
    hinge_max_correction_rad = math.radians(6.0)
    damping = 0.15
    gravity = -0.15

    def dihedral(xi, xj, xk, xl):
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

    def rotate_around_axis(p, axis_point, axis_dir, angle):
        if abs(angle) < 1e-12:
            return p
        rot = Matrix.Rotation(angle, 4, axis_dir)
        return axis_point + (rot @ (p - axis_point))

    def project_edges():
        for (a, b, l0) in edges:
            d = x[b] - x[a]
            ln = d.length
            if ln < 1e-9:
                continue
            err = (ln - l0) / ln
            corr = stretch_stiffness * 0.5 * err * d
            x[a] = x[a] + corr
            x[b] = x[b] - corr

    def project_unfold(alpha):
        if alpha <= 0:
            return
        for i in range(len(x)):
            x[i] = (1.0 - alpha) * x[i] + alpha * rest[i]

    def project_creases(crease_scale):
        if crease_scale <= 0 or not crease_targets:
            return
        for (i, j, k, l), theta0 in crease_targets.items():
            xi, xj, xk, xl = x[i], x[j], x[k], x[l]
            e = xj - xi
            el = e.length
            if el < 1e-12:
                continue
            axis_dir = e / el
            theta = dihedral(xi, xj, xk, xl)
            theta_target = theta0 * crease_scale
            err = theta - theta_target
            corr = -hinge_stiffness * err
            corr = max(-hinge_max_correction_rad, min(hinge_max_correction_rad, corr))
            x[k] = rotate_around_axis(xk, xi, axis_dir, 0.5 * corr)
            x[l] = rotate_around_axis(xl, xi, axis_dir, -0.5 * corr)

    def step(dt, n_iters, crease_scale, unfold_alpha, attract_strength, center=Vector((0.0, 0.0, 0.0))):
        x_prev = [p.copy() for p in x]
        for i in range(len(x)):
            f = Vector((0.0, 0.0, 0.0))
            f.z += gravity
            if attract_strength > 0:
                f += -attract_strength * (x[i] - center)
            v[i] = (1.0 - damping) * (v[i] + dt * f)
            x[i] = x[i] + dt * v[i]
        for _ in range(n_iters):
            project_creases(crease_scale)
            project_edges()
            project_unfold(unfold_alpha)
        for i in range(len(x)):
            v[i] = (x[i] - x_prev[i]) / max(1e-9, dt)

    dt = 1.0 / fps / max(1, substeps)

    # Pre-roll to reach a compact crumpled state by frame_start
    for _ in range(max(0, pre_roll)):
        step(dt=dt, n_iters=iters, crease_scale=1.0, unfold_alpha=0.0, attract_strength=attract0)

    def schedule(frame):
        u = 0.0
        if frame_end > frame_start:
            u = (frame - frame_start) / (frame_end - frame_start)
        u = max(0.0, min(1.0, u))
        unfold_alpha = unfold_max * (u**unfold_power)
        unfold_alpha = max(0.0, min(0.95, unfold_alpha))

        # Keep creases strong for most of the motion, then rapidly relax near the end.
        if u < 0.85:
            crease_scale = 1.0
        else:
            u2 = (u - 0.85) / 0.15
            crease_scale = math.exp(-u2 / max(1e-6, crease_tau))
        crease_scale = max(0.0, min(1.0, crease_scale))

        attract = attract0 * math.exp(-u / max(1e-6, attract_tau))
        if u > 0.92:
            attract = 0.0
        return crease_scale, unfold_alpha, attract

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    mat = sheet.active_material
    unlit_node = mat.node_tree.nodes.get("UNLIT_FACTOR") if mat else None

    for f in range(frame_start, frame_end + 1):
        crease_scale, unfold_alpha, attract = schedule(f)
        for _ in range(max(1, substeps)):
            step(dt=dt, n_iters=iters, crease_scale=crease_scale, unfold_alpha=unfold_alpha, attract_strength=attract)
        if f == frame_end:
            # Guarantee a perfectly flat end state.
            for i in range(len(x)):
                x[i] = rest[i].copy()
                v[i] = Vector((0.0, 0.0, 0.0))
        if unlit_node is not None:
            if f >= (frame_end - max(1, unlit_final_frames) + 1):
                unlit_node.outputs[0].default_value = 1.0
            else:
                unlit_node.outputs[0].default_value = 0.0
        for i, vert in enumerate(mesh.vertices):
            vert.co = x[i]
        mesh.update()
        bpy.context.view_layer.update()
        scene.frame_current = f
        scene.render.filepath = os.path.join(out_dir, f"frame_{f:04d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
