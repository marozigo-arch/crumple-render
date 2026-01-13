import argparse
import json
import math
import os
import sys

import bpy


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    ap = argparse.ArgumentParser()
    ap.add_argument("--posters_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frames_out", type=int, default=10)
    ap.add_argument("--sim_frames", type=int, default=120)
    ap.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--grid_x", type=int, default=100)
    ap.add_argument("--grid_y", type=int, default=130)
    ap.add_argument("--cloth_quality", type=int, default=8)
    ap.add_argument("--vary_sim", action="store_true")
    return ap.parse_args(argv)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    return scene


def setup_render(scene, engine):
    if engine == "EEVEE":
        scene.render.engine = "BLENDER_EEVEE"
    else:
        scene.render.engine = "CYCLES"
    scene.render.resolution_x = 864
    scene.render.resolution_y = 1104
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True

    if engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = 64
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.preview_samples = 16
    else:
        scene.eevee.taa_render_samples = 32
        scene.eevee.use_soft_shadows = True
        scene.eevee.shadow_cube_size = "1024"
        scene.eevee.shadow_cascade_size = "1024"
        scene.eevee.use_gtao = False
        scene.eevee.use_bloom = False

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Filmic - Medium High Contrast"


def setup_world(scene):
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    nt = scene.world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0)
    bg.inputs[1].default_value = 1.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_camera(scene):
    bpy.ops.object.camera_add(location=(0.0, -3.9, 1.35), rotation=(math.radians(78), 0, 0))
    cam = bpy.context.object
    scene.camera = cam
    return cam


def add_lights():
    # Key side light
    bpy.ops.object.light_add(type="AREA", location=(-2.1, -2.2, 1.9))
    key = bpy.context.object
    key.data.energy = 1200
    key.data.size = 1.6
    key.rotation_euler = (math.radians(65), math.radians(-10), math.radians(25))

    # Soft fill
    bpy.ops.object.light_add(type="AREA", location=(1.8, -2.8, 1.4))
    fill = bpy.context.object
    fill.data.energy = 350
    fill.data.size = 2.2
    fill.rotation_euler = (math.radians(70), math.radians(0), math.radians(-25))

    # Rim
    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.8, 2.0))
    rim = bpy.context.object
    rim.data.energy = 450
    rim.data.size = 1.8
    rim.rotation_euler = (math.radians(120), 0, 0)


def create_paper_mesh(grid_x, grid_y, paper_aspect=864 / 1104):
    # Use a dense grid so folds look paper-like.
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=int(grid_x),
        y_subdivisions=int(grid_y),
        size=1.0,
        location=(0, 0, 1.05),
    )
    paper = bpy.context.object
    paper.name = "Paper"
    paper.scale[0] = paper_aspect
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return paper


def add_cloth(paper, cloth_quality):
    cloth = paper.modifiers.new(name="Cloth", type="CLOTH")
    s = cloth.settings
    s.quality = int(cloth_quality)
    s.mass = 0.18
    s.tension_stiffness = 30
    s.compression_stiffness = 30
    s.shear_stiffness = 25
    s.bending_stiffness = 6
    s.tension_damping = 5
    s.compression_damping = 5
    s.shear_damping = 5
    s.bending_damping = 0.8
    s.air_damping = 1
    s.use_pressure = False

    c = cloth.collision_settings
    c.use_collision = True
    c.collision_quality = 4
    c.distance_min = 0.003
    c.use_self_collision = True
    c.self_friction = 4
    c.self_distance_min = 0.003
    if hasattr(c, "self_collision_quality"):
        c.self_collision_quality = 4
    return cloth


def add_colliders():
    # Central collider to keep the crumple ball volume.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.28, location=(0.0, 0.0, 0.85))
    sphere = bpy.context.object
    sphere.name = "CrumpleCore"
    bpy.ops.object.modifier_add(type="COLLISION")
    sphere.hide_render = True

    # Invisible floor for stability.
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "Floor"
    bpy.ops.object.modifier_add(type="COLLISION")
    floor.hide_render = True
    return sphere, floor


def add_force_fields(seed):
    # Main attractor: pulls the sheet inward.
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.85))
    att = bpy.context.object
    att.name = "Attractor"
    att.field.type = "FORCE"
    att.field.strength = 0.0
    att.field.flow = 1.0
    att.field.falloff_type = "SPHERE"
    att.field.distance_max = 2.2
    att.field.use_min_distance = True
    att.field.distance_min = 0.15

    # Turbulence: adds small wrinkles.
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 1.0))
    turb = bpy.context.object
    turb.name = "Turbulence"
    turb.field.type = "TURBULENCE"
    turb.field.strength = 0.0
    turb.field.size = 0.35
    turb.field.flow = 1.0

    # Randomize slightly per seed.
    offs = ((seed % 17) - 8) * 0.03
    turb.location.x += offs
    turb.location.y -= offs * 0.7
    return att, turb


def animate_controls(scene, att, turb, sim_frames):
    scene.frame_start = 1
    scene.frame_end = sim_frames

    def kf(obj, prop, frame, value):
        setattr(obj, prop, value)
        obj.keyframe_insert(data_path=prop, frame=frame)

    def kf_field(obj, prop, frame, value):
        setattr(obj.field, prop, value)
        obj.field.keyframe_insert(data_path=prop, frame=frame)

    # Attractor ramps in, holds, then eases off slightly (prevents over-collapse).
    kf_field(att, "strength", 1, 0.0)
    kf_field(att, "strength", int(sim_frames * 0.25), -260.0)
    kf_field(att, "strength", int(sim_frames * 0.70), -340.0)
    kf_field(att, "strength", sim_frames, -300.0)

    # Turbulence kicks in early then fades.
    kf_field(turb, "strength", 1, 0.0)
    kf_field(turb, "strength", int(sim_frames * 0.20), 18.0)
    kf_field(turb, "strength", int(sim_frames * 0.55), 10.0)
    kf_field(turb, "strength", sim_frames, 4.0)


def create_paper_material():
    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.65
    bsdf.inputs["Specular"].default_value = 0.25

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.interpolation = "Cubic"
    tex.extension = "CLIP"
    tex.location = (-800, 120)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-800, -120)

    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)  # backside white
    mix.location = (-420, 60)

    nt.links.new(geom.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(tex.outputs["Color"], mix.inputs["Color1"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, tex


def assign_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def bake_cloth_cache(scene, paper):
    # Bake physics caches for determinism.
    for mod in paper.modifiers:
        if mod.type == "CLOTH":
            pc = mod.point_cache
            pc.frame_start = scene.frame_start
            pc.frame_end = scene.frame_end
            if hasattr(pc, "use_disk_cache"):
                pc.use_disk_cache = False

    bpy.ops.ptcache.free_bake_all()
    bpy.ops.ptcache.bake_all(bake=True)


def sim_frame_for_output(i_out_1based, frames_out, sim_frames):
    if frames_out <= 1:
        return 1
    t = (i_out_1based - 1) / float(frames_out - 1)
    return int(round(1 + t * (sim_frames - 1)))


def render_sequence(scene, paper_tex_node, poster_items, out_dir, frames_out, sim_frames, vary_sim):
    posters_root = os.path.join(out_dir, "posters")
    ensure_dir(posters_root)
    seq_root = os.path.join(out_dir, "sequence_all", "frames")
    ensure_dir(seq_root)

    seq_idx = 1
    for item in poster_items:
        stem = item["stem"]
        path = item["path"]
        seed = int(item.get("seed", 0))
        poster_out = os.path.join(posters_root, stem, "frames")
        ensure_dir(poster_out)

        img = bpy.data.images.load(path, check_existing=True)
        paper_tex_node.image = img

        # Optional: per-poster motion variation by re-baking with a new seed.
        if vary_sim:
            # Move turbulence slightly to change wrinkles
            turb = bpy.data.objects.get("Turbulence")
            if turb:
                offs = ((seed % 29) - 14) * 0.02
                turb.location.x = offs
                turb.location.y = -offs * 0.6
            bake_cloth_cache(scene, bpy.data.objects["Paper"])

        # Fold (1..frames_out)
        for i in range(1, frames_out + 1):
            sf = sim_frame_for_output(i, frames_out, sim_frames)
            scene.frame_set(sf)
            out_path = os.path.join(poster_out, "frame_%04d.png" % i)
            scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            seq_path = os.path.join(seq_root, "frame_%06d.png" % seq_idx)
            try:
                # hardlink if possible, else copy
                if os.path.exists(seq_path):
                    os.remove(seq_path)
                os.link(out_path, seq_path)
            except OSError:
                import shutil

                shutil.copyfile(out_path, seq_path)
            seq_idx += 1

        # Unfold (frames_out+1..2*frames_out) as reverse playback
        for i in range(1, frames_out + 1):
            sf = sim_frame_for_output(frames_out - i + 1, frames_out, sim_frames)
            scene.frame_set(sf)
            out_path = os.path.join(poster_out, "frame_%04d.png" % (frames_out + i))
            scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            seq_path = os.path.join(seq_root, "frame_%06d.png" % seq_idx)
            try:
                if os.path.exists(seq_path):
                    os.remove(seq_path)
                os.link(out_path, seq_path)
            except OSError:
                import shutil

                shutil.copyfile(out_path, seq_path)
            seq_idx += 1


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    poster_items = json.loads(open(args.posters_json, "r", encoding="utf-8").read())

    scene = reset_scene()
    setup_render(scene, args.engine)
    if args.engine == "CYCLES":
        scene.cycles.samples = int(args.samples)
    else:
        scene.eevee.taa_render_samples = int(args.samples)
    setup_world(scene)
    add_camera(scene)
    add_lights()

    paper = create_paper_mesh(grid_x=args.grid_x, grid_y=args.grid_y)
    add_cloth(paper, cloth_quality=args.cloth_quality)
    add_colliders()
    att, turb = add_force_fields(seed=poster_items[0].get("seed", 0))
    animate_controls(scene, att, turb, sim_frames=args.sim_frames)

    mat, tex_node = create_paper_material()
    assign_material(paper, mat)

    # Bake once (default) for speed.
    bake_cloth_cache(scene, paper)

    render_sequence(
        scene,
        tex_node,
        poster_items,
        out_dir=out_dir,
        frames_out=args.frames_out,
        sim_frames=args.sim_frames,
        vary_sim=args.vary_sim,
    )


if __name__ == "__main__":
    main()
