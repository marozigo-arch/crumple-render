import argparse
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
    ap.add_argument("--poster", required=True)
    ap.add_argument("--disp_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_start", type=int, required=True)
    ap.add_argument("--frame_end", type=int, required=True)
    ap.add_argument("--disp_strength", type=float, default=0.22)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--subdiv", type=int, default=5)
    ap.add_argument("--bg", type=float, default=0.02)
    return ap.parse_args(argv)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def setup_render(scene, samples):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(samples)
    scene.cycles.use_adaptive_sampling = True
    scene.render.resolution_x = 864
    scene.render.resolution_y = 1104
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Filmic - Medium High Contrast"


def setup_world(scene, bg):
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    nt = scene.world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg_node = nt.nodes.new("ShaderNodeBackground")
    bg_node.inputs[0].default_value = (bg, bg, bg, 1.0)
    bg_node.inputs[1].default_value = 1.0
    nt.links.new(bg_node.outputs["Background"], out.inputs["Surface"])


def add_camera(scene):
    bpy.ops.object.camera_add(location=(0.0, -3.9, 1.35), rotation=(math.radians(78), 0, 0))
    cam = bpy.context.object
    scene.camera = cam
    return cam


def add_lights():
    bpy.ops.object.light_add(type="AREA", location=(-2.1, -2.2, 1.9))
    key = bpy.context.object
    key.data.energy = 1200
    key.data.size = 1.6
    key.rotation_euler = (math.radians(65), math.radians(-10), math.radians(25))

    bpy.ops.object.light_add(type="AREA", location=(1.8, -2.8, 1.4))
    fill = bpy.context.object
    fill.data.energy = 350
    fill.data.size = 2.2
    fill.rotation_euler = (math.radians(70), math.radians(0), math.radians(-25))

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.8, 2.0))
    rim = bpy.context.object
    rim.data.energy = 450
    rim.data.size = 1.8
    rim.rotation_euler = (math.radians(120), 0, 0)


def create_paper():
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 1.05))
    paper = bpy.context.object
    paper.name = "Paper"
    paper.scale[0] = 864 / 1104
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return paper


def add_subdiv(paper, levels):
    sub = paper.modifiers.new(name="Subdiv", type="SUBSURF")
    sub.subdivision_type = "SIMPLE"
    sub.levels = int(levels)
    sub.render_levels = int(levels)
    return sub


def create_material(poster_path):
    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Specular"].default_value = 0.22

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.interpolation = "Cubic"
    tex.extension = "CLIP"
    tex.image = bpy.data.images.load(poster_path, check_existing=True)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)

    nt.links.new(geom.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(tex.outputs["Color"], mix.inputs["Color1"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def add_displace(paper, disp_dir, frame_start, frame_end, strength):
    tex = bpy.data.textures.new(name="DispTex", type="IMAGE")
    # For compatibility with Blender 2.82, we swap the texture image per frame
    # instead of relying on Image sequence properties.
    first = os.path.join(disp_dir, f"disp_{frame_start:04d}.png")
    tex.image = bpy.data.images.load(first, check_existing=True)

    mod = paper.modifiers.new(name="Displace", type="DISPLACE")
    mod.texture = tex
    mod.texture_coords = "UV"
    mod.strength = float(strength)
    mod.mid_level = 0.5
    return mod, tex


def render_frames(scene, out_dir, disp_dir, frame_start, frame_end, disp_texture):
    ensure_dir(out_dir)
    scene.frame_start = int(frame_start)
    scene.frame_end = int(frame_end)

    for f in range(frame_start, frame_end + 1):
        scene.frame_set(int(f))
        disp_path = os.path.join(disp_dir, f"disp_{f:04d}.png")
        disp_texture.image = bpy.data.images.load(disp_path, check_existing=True)
        disp_texture.image.reload()
        scene.render.filepath = os.path.join(out_dir, f"frame_{f:04d}.png")
        bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    scene = reset_scene()
    setup_render(scene, samples=args.samples)
    setup_world(scene, bg=args.bg)
    add_camera(scene)
    add_lights()

    paper = create_paper()
    add_subdiv(paper, levels=args.subdiv)
    mat = create_material(args.poster)
    if paper.data.materials:
        paper.data.materials[0] = mat
    else:
        paper.data.materials.append(mat)

    _, disp_tex = add_displace(
        paper,
        disp_dir=args.disp_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        strength=args.disp_strength,
    )

    render_frames(
        scene,
        out_dir=args.out_dir,
        disp_dir=args.disp_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        disp_texture=disp_tex,
    )


if __name__ == "__main__":
    main()
