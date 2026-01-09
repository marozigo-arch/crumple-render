import bpy
import os

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 864
    scene.render.resolution_y = 1104
    scene.cycles.samples = 64
    
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1)

def create_poster(name, image_path, location):
    # Load image
    try:
        img = bpy.data.images.load(image_path)
    except:
        print(f"Could not load {image_path}")
        return
    
    # Calculate aspect ratio
    w, h = img.size
    aspect = w / h
    
    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=location, rotation=(1.5708, 0, 0)) # Vertical plane
    plane = bpy.context.object
    plane.name = name
    
    # Adjust scale based on aspect
    plane.scale[0] = aspect
    
    # Material
    mat = bpy.data.materials.new(name=f"Mat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = img
    
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    
    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)
        
    return plane

def setup_camera():
    bpy.ops.object.camera_add(location=(0, -8, 2))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Point at origin
    bpy.ops.object.empty_add(location=(0, 0, 1)) # Look slightly up or at center
    target = bpy.context.object
    
    const = cam.constraints.new(type='DAMPED_TRACK')
    const.target = target
    const.track_axis = 'TRACK_NEGATIVE_Z'

def main():
    setup_scene()
    
    base_path = "data/afisha-selected"
    files = [f for f in os.listdir(base_path) if f.endswith('.jpg')]
    files.sort()
    
    # Place posters
    # Just simplistic placement: one at X=-2, one at X=2
    x_offset = -1.5
    for i, f in enumerate(files):
        path = os.path.join(base_path, f)
        create_poster(f"Poster_{i}", path, location=(x_offset + i*3, 0, 1.5))
        
    setup_camera()
    
    bpy.context.scene.render.filepath = "//data/poster_render_test.png"
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()
