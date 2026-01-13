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


def setup_scene(res_x: int, res_y: int, samples: int):
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
    scene.render.film_transparent = True

    # Keep colors stable for poster matching (avoid Filmic remapping).
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0)


def setup_camera_and_lights(aspect: float):
    # Front-on orthographic camera.
    bpy.ops.object.camera_add(location=(0.0, 0.0, 4.0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.6 * max(1.0, aspect)

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


def build_paper(aspect: float, cuts: int, size: float):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0.6))
    paper = bpy.context.object
    paper.name = "Paper"

    # Scale to poster aspect.
    paper.scale[0] = aspect * (size / 2.0)
    paper.scale[1] = (size / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=cuts)
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.shade_smooth()
    return paper


def apply_uv_map_xy(obj):
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data

    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(max_x - min_x, 1e-9)
    dy = max(max_y - min_y, 1e-9)

    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            co = mesh.vertices[vi].co
            u = (co.x - min_x) / dx
            v = (co.y - min_y) / dy
            uv_layer[li].uv = (u, v)


def build_two_sided_material(obj, poster_path: str):
    img = None
    if poster_path and os.path.exists(poster_path):
        try:
            img = bpy.data.images.load(poster_path)
        except Exception:
            img = None

    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mix_rgb = nt.nodes.new("ShaderNodeMixRGB")
    mix_rgb.blend_type = "MIX"

    tex = nt.nodes.new("ShaderNodeTexImage")
    if img is not None:
        tex.image = img
        tex.interpolation = "Closest"

    # Backfacing==1 -> white, else poster
    nt.links.new(geo.outputs["Backfacing"], mix_rgb.inputs["Fac"])
    nt.links.new(tex.outputs["Color"], mix_rgb.inputs["Color1"])
    mix_rgb.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)

    principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 0.95
    principled.inputs["Specular"].default_value = 0.02
    nt.links.new(mix_rgb.outputs["Color"], principled.inputs["Base Color"])

    emission = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(mix_rgb.outputs["Color"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0

    unlit = nt.nodes.new("ShaderNodeValue")
    unlit.name = "UNLIT_FACTOR"
    unlit.outputs[0].default_value = 0.0

    mix_shader = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(unlit.outputs[0], mix_shader.inputs["Fac"])
    nt.links.new(principled.outputs["BSDF"], mix_shader.inputs[1])
    nt.links.new(emission.outputs["Emission"], mix_shader.inputs[2])
    nt.links.new(mix_shader.outputs["Shader"], out.inputs["Surface"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return mat


def setup_cloth(paper, quality: int):
    bpy.ops.object.select_all(action="DESELECT")
    paper.select_set(True)
    bpy.context.view_layer.objects.active = paper

    bpy.ops.object.modifier_add(type="CLOTH")
    mod = paper.modifiers["Cloth"]
    settings = mod.settings

    settings.mass = 0.3
    settings.quality = quality
    settings.air_damping = 2.0

    settings.tension_stiffness = 500.0
    settings.compression_stiffness = 500.0
    settings.shear_stiffness = 500.0
    settings.bending_model = "ANGULAR"
    settings.bending_stiffness = 80.0

    cols = mod.collision_settings
    cols.use_self_collision = True
    cols.distance_min = 0.02
    cols.self_friction = 5.0

    # Cache range will be set by caller.
    return mod


def create_core(sim_frames: int):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    core = bpy.context.object
    core.name = "Core"
    core.hide_render = True

    bpy.ops.object.modifier_add(type="COLLISION")
    core.modifiers["Collision"].settings.thickness_outer = 0.05

    core.scale = (1.2, 1.2, 1.2)
    core.keyframe_insert(data_path="scale", frame=1)
    core.keyframe_insert(data_path="scale", frame=max(2, int(round(sim_frames * 0.4))))
    core.scale = (0.2, 0.2, 0.2)
    core.keyframe_insert(data_path="scale", frame=sim_frames)
    return core


def create_force_fields(seed: int, sim_frames: int):
    bpy.ops.object.effector_add(type="HARMONIC", location=(0, 0, 0))
    magnet = bpy.context.object
    magnet.name = "Magnet"
    magnet.field.shape = "POINT"
    magnet.field.rest_length = 0.0

    magnet.field.strength = 0.0
    magnet.keyframe_insert(data_path="field.strength", frame=1)
    magnet.field.strength = 100.0
    magnet.keyframe_insert(data_path="field.strength", frame=max(2, int(round(sim_frames * 0.2))))
    magnet.field.strength = 200.0
    magnet.keyframe_insert(data_path="field.strength", frame=max(3, int(round(sim_frames * 0.6))))

    bpy.ops.object.effector_add(type="TURBULENCE", location=(1, 1, 1))
    turb = bpy.context.object
    turb.name = "Turbulence"
    turb.field.strength = 20.0
    turb.field.size = 2.0
    turb.field.noise = 0.65
    turb.field.seed = int(seed)
    return magnet, turb


def map_output_frame_to_sim(frame, anchors_ref_to_sim):
    # anchors_ref_to_sim: sorted by ref frame ascending, e.g. [(33,100),(37,75),...,(44,1)]
    anchors = sorted(anchors_ref_to_sim, key=lambda t: t[0])
    if frame <= anchors[0][0]:
        return float(anchors[0][1])
    if frame >= anchors[-1][0]:
        return float(anchors[-1][1])
    for (r0, s0), (r1, s1) in zip(anchors, anchors[1:]):
        if r0 <= frame <= r1:
            t = (frame - r0) / max(1e-9, (r1 - r0))
            return (1.0 - t) * s0 + t * s1
    return float(anchors[-1][1])


def main():
    args = _parse_args()
    poster_path = _as_str(args, "poster", "")
    out_dir = _as_str(args, "out_dir", "attempt_blender_cloth_v8_001/out_single")
    os.makedirs(out_dir, exist_ok=True)

    frame_start = _as_int(args, "frame_start", 33)
    frame_end = _as_int(args, "frame_end", 44)
    seed = _as_int(args, "seed", 7)

    res_x = _as_int(args, "res_x", 864)
    res_y = _as_int(args, "res_y", 1104)
    samples = _as_int(args, "samples", 48)
    fps = _as_float(args, "fps", 24.0)

    sim_frames = _as_int(args, "sim_frames", 100)
    cloth_quality = _as_int(args, "cloth_quality", 12)
    subdiv_cuts = _as_int(args, "subdivide_cuts", 25)
    paper_size = _as_float(args, "paper_size", 2.4)

    # Final-frame identity controls
    unlit_final_frames = _as_int(args, "unlit_final_frames", 2)

    bpy.context.scene.render.fps = int(round(fps))
    aspect = res_x / res_y
    if poster_path and os.path.exists(poster_path):
        try:
            img = bpy.data.images.load(poster_path)
            if img and img.size[1] > 0:
                aspect = img.size[0] / img.size[1]
        except Exception:
            pass

    setup_scene(res_x=res_x, res_y=res_y, samples=samples)
    setup_camera_and_lights(aspect=aspect)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = sim_frames
    scene.gravity = (0.0, 0.0, -9.81)

    paper = build_paper(aspect=aspect, cuts=subdiv_cuts, size=paper_size)
    apply_uv_map_xy(paper)
    mat = build_two_sided_material(paper, poster_path=poster_path)
    cloth_mod = setup_cloth(paper, quality=cloth_quality)

    core = create_core(sim_frames=sim_frames)
    create_force_fields(seed=seed, sim_frames=sim_frames)

    cloth_mod.point_cache.frame_start = 1
    cloth_mod.point_cache.frame_end = sim_frames

    mesh = paper.data
    rest_positions = [v.co.copy() for v in mesh.vertices]

    # Drive the simulation forward and record evaluated (post-modifier) vertex positions.
    # Note: cloth deformation lives in the evaluated mesh, not in `obj.data` directly.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    positions = []
    for f in range(1, sim_frames + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        depsgraph.update()
        eval_obj = paper.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        positions.append([v.co.copy() for v in eval_mesh.vertices])
        eval_obj.to_mesh_clear()

    # Disable cloth for manual playback (so setting vertices is not overridden by physics).
    paper.modifiers.remove(cloth_mod)
    if core:
        core.hide_viewport = True

    # Reference-time mapping from the existing v8 calibration:
    # sim 1 -> ref 44, 25 -> 42, 50 -> 40, 75 -> 37, 100 -> 33
    anchors_ref_to_sim = [(33, sim_frames), (37, int(round(sim_frames * 0.75))), (40, int(round(sim_frames * 0.5))), (42, int(round(sim_frames * 0.25))), (44, 1)]

    # Render frames numbered like the reference window.
    unlit_node = mat.node_tree.nodes.get("UNLIT_FACTOR") if mat else None
    for out_f in range(frame_start, frame_end + 1):
        if out_f == frame_end:
            verts = rest_positions
        else:
            sim_f = map_output_frame_to_sim(out_f, anchors_ref_to_sim=anchors_ref_to_sim)
            sim_i = int(round(max(1.0, min(float(sim_frames), sim_f))))
            verts = positions[sim_i - 1]
        for i, v in enumerate(mesh.vertices):
            v.co = verts[i]
        mesh.update()
        bpy.context.view_layer.update()

        if unlit_node is not None:
            if out_f >= (frame_end - max(1, unlit_final_frames) + 1):
                # Force near-identity poster match on final frames.
                unlit_node.outputs[0].default_value = 1.0
            else:
                unlit_node.outputs[0].default_value = 0.0

        scene.frame_current = out_f
        scene.render.filepath = os.path.join(out_dir, f"frame_{out_f:04d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
