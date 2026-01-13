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
    ap.add_argument("--poster_interp", choices=["Linear", "Cubic", "Closest"], default="Linear")
    ap.add_argument("--view", choices=["Standard", "Filmic"], default="Standard")
    ap.add_argument("--light_scale", type=float, default=2.0)
    ap.add_argument("--debug", choices=["none", "alpha", "uv", "light"], default="none")
    return ap.parse_args(argv)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def setup_render(scene, samples, view):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(samples)
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x = 864
    scene.render.resolution_y = 1104
    scene.render.resolution_percentage = 100
    scene.render.tile_x = 256
    scene.render.tile_y = 256
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = view
    if view == "Filmic":
        scene.view_settings.look = "Filmic - Medium High Contrast"
    else:
        scene.view_settings.look = "None"


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


def _set_data_colorspace(img):
    if hasattr(img, "colorspace_settings"):
        # Blender 2.82: 'Raw' avoids any unintended conversions for numeric maps.
        img.colorspace_settings.name = "Raw"


def create_material(poster_path, poster_interp, light_scale, debug_mode):
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
    poster.interpolation = poster_interp
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
    sep_uv = nt.nodes.new("ShaderNodeSeparateRGB")
    comb_uv = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(uv_img.outputs["Color"], sep_uv.inputs["Image"])
    nt.links.new(sep_uv.outputs["R"], comb_uv.inputs["X"])
    nt.links.new(sep_uv.outputs["G"], comb_uv.inputs["Y"])

    # Feed vector into poster sampling
    nt.links.new(comb_uv.outputs["Vector"], poster.inputs["Vector"])

    # Backside mask (per-frame swapped): mix poster with white
    back = nt.nodes.new("ShaderNodeTexImage")
    back.interpolation = "Linear"
    back.extension = "CLIP"
    nt.links.new(uvmap.outputs["UV"], back.inputs["Vector"])
    sep_back = nt.nodes.new("ShaderNodeSeparateRGB")
    nt.links.new(back.outputs["Color"], sep_back.inputs["Image"])

    mix_back = nt.nodes.new("ShaderNodeMixRGB")
    mix_back.blend_type = "MIX"
    mix_back.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(sep_back.outputs["R"], mix_back.inputs["Fac"])
    nt.links.new(poster.outputs["Color"], mix_back.inputs["Color1"])

    # Light ratio (per-frame swapped): multiply base color
    light = nt.nodes.new("ShaderNodeTexImage")
    light.interpolation = "Linear"
    light.extension = "CLIP"
    nt.links.new(uvmap.outputs["UV"], light.inputs["Vector"])
    sep_light = nt.nodes.new("ShaderNodeSeparateRGB")
    nt.links.new(light.outputs["Color"], sep_light.inputs["Image"])

    light_scale_node = nt.nodes.new("ShaderNodeMath")
    light_scale_node.operation = "MULTIPLY"
    light_scale_node.inputs[1].default_value = float(light_scale)
    nt.links.new(sep_light.outputs["R"], light_scale_node.inputs[0])

    comb_light = nt.nodes.new("ShaderNodeCombineRGB")
    nt.links.new(light_scale_node.outputs["Value"], comb_light.inputs["R"])
    nt.links.new(light_scale_node.outputs["Value"], comb_light.inputs["G"])
    nt.links.new(light_scale_node.outputs["Value"], comb_light.inputs["B"])

    mul = nt.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    nt.links.new(mix_back.outputs["Color"], mul.inputs["Color1"])
    nt.links.new(comb_light.outputs["Image"], mul.inputs["Color2"])

    # Paper alpha (per-frame swapped)
    alpha = nt.nodes.new("ShaderNodeTexImage")
    alpha.interpolation = "Linear"
    alpha.extension = "CLIP"
    nt.links.new(uvmap.outputs["UV"], alpha.inputs["Vector"])
    sep_alpha = nt.nodes.new("ShaderNodeSeparateRGB")
    nt.links.new(alpha.outputs["Color"], sep_alpha.inputs["Image"])

    # Transparent cutout
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(sep_alpha.outputs["R"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emit.outputs["Emission"], mix.inputs[2])

    if debug_mode == "alpha":
        nt.links.new(alpha.outputs["Color"], emit.inputs["Color"])
    elif debug_mode == "uv":
        nt.links.new(uv_img.outputs["Color"], emit.inputs["Color"])
    elif debug_mode == "light":
        nt.links.new(light.outputs["Color"], emit.inputs["Color"])
    else:
        nt.links.new(mul.outputs["Color"], emit.inputs["Color"])

    if debug_mode == "none":
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    # Color space settings
    _set_data_colorspace(uv_img.image) if uv_img.image else None
    _set_data_colorspace(alpha.image) if alpha.image else None
    _set_data_colorspace(back.image) if back.image else None
    _set_data_colorspace(light.image) if light.image else None

    return mat, uv_img, alpha, back, light


def render_frames(scene, out_dir, assets_dir, frame_start, frame_end, uv_node, alpha_node, back_node, light_node):
    ensure_dir(out_dir)
    uv_dir = os.path.join(assets_dir, "uv")
    alpha_dir = os.path.join(assets_dir, "alpha")
    back_dir = os.path.join(assets_dir, "backside")
    light_dir = os.path.join(assets_dir, "light")

    for f in range(frame_start, frame_end + 1):
        uv_path = os.path.join(uv_dir, f"uv_{f:04d}.png")
        alpha_path = os.path.join(alpha_dir, f"alpha_{f:04d}.png")
        back_path = os.path.join(back_dir, f"back_{f:04d}.png")
        light_path = os.path.join(light_dir, f"light_{f:04d}.png")

        uv_node.image = bpy.data.images.load(uv_path, check_existing=True)
        _set_data_colorspace(uv_node.image)

        alpha_node.image = bpy.data.images.load(alpha_path, check_existing=True)
        _set_data_colorspace(alpha_node.image)

        back_node.image = bpy.data.images.load(back_path, check_existing=True)
        _set_data_colorspace(back_node.image)

        light_node.image = bpy.data.images.load(light_path, check_existing=True)
        _set_data_colorspace(light_node.image)

        scene.frame_set(int(f))
        scene.render.filepath = os.path.join(out_dir, f"frame_{f:04d}.png")
        bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    scene = reset_scene()
    setup_render(scene, samples=args.samples, view=args.view)
    setup_world(scene, bg=args.bg)
    add_camera(scene)
    add_lights()

    paper = create_paper()
    mat, uv_node, alpha_node, back_node, light_node = create_material(
        args.poster, poster_interp=args.poster_interp, light_scale=args.light_scale, debug_mode=args.debug
    )
    paper.data.materials.append(mat)

    render_frames(
        scene,
        out_dir=args.out_dir,
        assets_dir=args.assets_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        uv_node=uv_node,
        alpha_node=alpha_node,
        back_node=back_node,
        light_node=light_node,
    )


if __name__ == "__main__":
    main()
