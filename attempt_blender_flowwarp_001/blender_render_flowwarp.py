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
    ap.add_argument("--assets_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_start", type=int, required=True)
    ap.add_argument("--frame_end", type=int, required=True)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--bg", type=float, default=0.02)
    ap.add_argument("--debug", choices=["none", "mask", "uv"], default="none")
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
    scene.render.film_transparent = True
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
    bpy.ops.object.camera_add(location=(0.0, -2.4, 1.05))
    cam = bpy.context.object
    scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.0
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 1.05))
    target = bpy.context.object
    con = cam.constraints.new(type="DAMPED_TRACK")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    return cam


def add_lights():
    bpy.ops.object.light_add(type="AREA", location=(-2.1, -2.2, 1.9))
    key = bpy.context.object
    key.data.energy = 900
    key.data.size = 1.6
    key.rotation_euler = (math.radians(65), math.radians(-10), math.radians(25))

    bpy.ops.object.light_add(type="AREA", location=(1.8, -2.8, 1.4))
    fill = bpy.context.object
    fill.data.energy = 220
    fill.data.size = 2.2
    fill.rotation_euler = (math.radians(70), math.radians(0), math.radians(-25))


def create_paper():
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 1.05), rotation=(math.radians(90), 0.0, 0.0))
    paper = bpy.context.object
    paper.name = "Paper"
    paper.scale[0] = 864 / 1104
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return paper


def create_material(poster_path, debug_mode):
    mat = bpy.data.materials.new(name="WarpMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0

    # Poster texture
    poster = nt.nodes.new("ShaderNodeTexImage")
    poster.image = bpy.data.images.load(poster_path, check_existing=True)
    poster.interpolation = "Cubic"
    poster.extension = "CLIP"

    # Explicit UVs for all driver textures (avoid fallback coord differences).
    uvmap = nt.nodes.new("ShaderNodeUVMap")
    uvmap.uv_map = "UVMap"

    # UV map (per-frame swapped)
    uv_img = nt.nodes.new("ShaderNodeTexImage")
    uv_img.interpolation = "Closest"
    uv_img.extension = "CLIP"
    nt.links.new(uvmap.outputs["UV"], uv_img.inputs["Vector"])

    # Separate RG into UV vector
    sep = nt.nodes.new("ShaderNodeSeparateRGB")
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(uv_img.outputs["Color"], sep.inputs["Image"])
    nt.links.new(sep.outputs["R"], comb.inputs["X"])
    nt.links.new(sep.outputs["G"], comb.inputs["Y"])

    # Feed vector into poster sampling
    nt.links.new(comb.outputs["Vector"], poster.inputs["Vector"])

    # Backside mask (per-frame swapped): mix poster with white
    back = nt.nodes.new("ShaderNodeTexImage")
    back.interpolation = "Linear"
    back.extension = "CLIP"
    nt.links.new(uvmap.outputs["UV"], back.inputs["Vector"])

    mix_back = nt.nodes.new("ShaderNodeMixRGB")
    mix_back.blend_type = "MIX"
    mix_back.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(back.outputs["Color"], mix_back.inputs["Fac"])
    nt.links.new(poster.outputs["Color"], mix_back.inputs["Color1"])

    # Paper alpha mask (per-frame swapped)
    mask = nt.nodes.new("ShaderNodeTexImage")
    mask.interpolation = "Linear"
    mask.extension = "CLIP"
    nt.links.new(uvmap.outputs["UV"], mask.inputs["Vector"])

    # Transparent cutout
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mask.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emit.outputs["Emission"], mix.inputs[2])

    if debug_mode == "mask":
        nt.links.new(mask.outputs["Color"], emit.inputs["Color"])
    elif debug_mode == "uv":
        nt.links.new(uv_img.outputs["Color"], emit.inputs["Color"])
    else:
        nt.links.new(mix_back.outputs["Color"], emit.inputs["Color"])
    if debug_mode == "none":
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    return mat, uv_img, mask, back


def render_frames(scene, out_dir, assets_dir, frame_start, frame_end, uv_node, mask_node, back_node):
    ensure_dir(out_dir)
    uv_dir = os.path.join(assets_dir, "uv")
    mask_dir = os.path.join(assets_dir, "mask")
    back_dir = os.path.join(assets_dir, "backside")

    for f in range(frame_start, frame_end + 1):
        uv_path = os.path.join(uv_dir, f"uv_{f:04d}.png")
        mask_path = os.path.join(mask_dir, f"mask_{f:04d}.png")
        back_path = os.path.join(back_dir, f"back_{f:04d}.png")

        uv_node.image = bpy.data.images.load(uv_path, check_existing=True)
        if hasattr(uv_node.image, "colorspace_settings"):
            uv_node.image.colorspace_settings.name = "Non-Color"
        uv_node.image.reload()

        mask_node.image = bpy.data.images.load(mask_path, check_existing=True)
        if hasattr(mask_node.image, "colorspace_settings"):
            mask_node.image.colorspace_settings.name = "Non-Color"
        mask_node.image.reload()

        back_node.image = bpy.data.images.load(back_path, check_existing=True)
        if hasattr(back_node.image, "colorspace_settings"):
            back_node.image.colorspace_settings.name = "Non-Color"
        back_node.image.reload()

        scene.frame_set(int(f))
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
    mat, uv_node, mask_node, back_node = create_material(args.poster, debug_mode=args.debug)
    paper.data.materials.append(mat)

    render_frames(
        scene,
        out_dir=args.out_dir,
        assets_dir=args.assets_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        uv_node=uv_node,
        mask_node=mask_node,
        back_node=back_node,
    )


if __name__ == "__main__":
    main()
