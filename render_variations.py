import bpy
import os
import math
import sys

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 864
    scene.render.resolution_y = 1104
    scene.cycles.samples = 32
    
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1)

def create_poster_variant(name, image_path, location, axis, angle_deg, label):
    # Load image
    try:
        img = bpy.data.images.load(image_path)
    except:
        # Fallback if already loaded or not found (should be loaded ideally)
        pass
    
    # We might need to check if image is already loaded to avoid duplicates
    img = None
    for i in bpy.data.images:
        if i.filepath == image_path or i.filepath.endswith(os.path.basename(image_path)):
            img = i
            break
    if not img:
        try:
            img = bpy.data.images.load(image_path)
        except:
            return

    w, h = img.size
    aspect = w / h
    
    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=location, rotation=(1.5708, 0, 0))
    plane = bpy.context.object
    plane.name = name
    plane.scale[0] = aspect
    
    # Subdivide
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=10)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Material
    mat = bpy.data.materials.new(name=f"Mat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = img
    
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    plane.data.materials.append(mat)
    
    # Empty for origin
    bpy.ops.object.empty_add(location=plane.location)
    empty = bpy.context.object
    empty.name = f"Origin_{name}"
    
    # Modifier
    mod = plane.modifiers.new(name="Bend", type='SIMPLE_DEFORM')
    mod.deform_method = 'BEND'
    mod.origin = empty 
    mod.angle = math.radians(angle_deg)
    mod.deform_axis = axis
    
    # Add text label above
    bpy.ops.object.text_add(location=(location[0], location[1], location[2] + 1.5))
    txt = bpy.context.object
    txt.data.body = label
    txt.data.align_x = 'CENTER'
    txt.scale = (0.3, 0.3, 0.3)
    txt.rotation_euler = (1.5708, 0, 0)
    
    return plane

def setup_camera():
    # Wider shot to see all variants
    bpy.ops.object.camera_add(location=(0, -10, 2))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.object
    
    const = cam.constraints.new(type='DAMPED_TRACK')
    const.target = target
    const.track_axis = 'TRACK_NEGATIVE_Z'

def main():
    setup_scene()
    
    image_path = "data/extracted_texture_from_45.png"
    if not os.path.exists(image_path):
        image_path = "data/afisha-selected/3.jpg"

    # Define Variations
    # Grid layout: 2x2
    # x: -2, 2
    # z: 2, -2 (relative to center vertical)
    
    variations = [
        {"axis": 'X', "angle": 90, "pos": (-2.5, 0, 2.5), "label": "X / +90"},
        {"axis": 'X', "angle": -90, "pos": (2.5, 0, 2.5), "label": "X / -90"},
        {"axis": 'Y', "angle": 90, "pos": (-2.5, 0, -2.5), "label": "Y / +90"},
        {"axis": 'Y', "angle": -90, "pos": (2.5, 0, -2.5), "label": "Y / -90"},
    ]
    
    for i, v in enumerate(variations):
        create_poster_variant(f"Var_{i}", image_path, v["pos"], v["axis"], v["angle"], v["label"])
        
    setup_camera()
    
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.filepath = "//data/variations_grid.png"
    
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()
