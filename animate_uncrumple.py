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
    scene.cycles.samples = 64
    
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1)

def create_crumpled_poster(image_path, start_frame, end_frame):
    # Load image
    img = None
    try:
        img = bpy.data.images.load(image_path)
    except:
        pass
    
    # Aspect Ratio
    aspect = 1.0
    if img:
        aspect = img.size[0] / img.size[1]
    
    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0), rotation=(1.5708, 0, 0))
    plane = bpy.context.object
    plane.name = "CrumpledPoster"
    plane.scale[0] = aspect
    
    # Heavy Subdivision for displacement
    bpy.ops.object.mode_set(mode='EDIT')
    # Increased cuts even more, and we'll use a Subsurf Modifier for the rest
    bpy.ops.mesh.subdivide(number_cuts=20) 
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add Subdivision Surface Modifier (essential for sharp creases)
    subsurf = plane.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 3
    
    # Smooth shading
    bpy.ops.object.shade_smooth()
    
    # Material
    mat = bpy.data.materials.new(name="PosterMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Roughness'].default_value = 0.8
    bsdf.inputs['Specular'].default_value = 0.1 # Paper is matte
    
    if img:
        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = img
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
        
    plane.data.materials.append(mat)
    
    # Setup Displacement Texture - USING MUSGRAVE FOR SHARP CREASES
    tex = bpy.data.textures.new("CrumpleTex", 'MUSGRAVE')
    tex.musgrave_type = 'RIDGED_MULTIFRACTAL'
    tex.noise_scale = 1.0 
    tex.lacunarity = 2.0
    # In 2.82/API, 'dimension' might be different or 'octaves' used. 
    # Let's try omitting dimension for now or using 'octaves' to ensure detail.
    tex.octaves = 2.0
    
    # Add Displace Modifier
    mod = plane.modifiers.new(name="Crumple", type='DISPLACE')
    mod.texture = tex
    mod.mid_level = 0.5
    
    # Animate Strength - "Uncrumpling"
    # It must look very flattened at 40 in previous attempt, meaning curve was too steep.
    # Ref 40 is still very crumpled.
    # Frame 33: Twisted/Crumpled Ball (Strength 2.5)
    # Frame 40: Opening up but still rough (Strength 1.5)
    # Frame 45: Flat (Strength 0.0)
    
    # Animate Strength - "Uncrumpling"
    # To prevent "Stretching" look (rubber effect), we must shrink the XY plane as it crumples up (Z).
    # Conservation of Surface Area: Higher Z-displacement -> Smaller XY footprint.
    
    # Frame 33: Twisted/Crumpled (Strength 3.0) -> Scale 0.7 (significant contraction)
    mod.strength = 3.0
    mod.keyframe_insert(data_path="strength", frame=start_frame)
    plane.scale[0] = aspect * 0.7
    plane.scale[1] = 0.7
    plane.keyframe_insert(data_path="scale", frame=start_frame)
    
    # Frame 40: Opening up (Strength 1.8) -> Scale 0.85
    mod.strength = 1.8 
    mod.keyframe_insert(data_path="strength", frame=40)
    plane.scale[0] = aspect * 0.85
    plane.scale[1] = 0.85
    plane.keyframe_insert(data_path="scale", frame=40)
    
    # Frame 45: Flat (Strength 0.0) -> Scale 1.0
    mod.strength = 0.0
    mod.keyframe_insert(data_path="strength", frame=end_frame)
    plane.scale[0] = aspect
    plane.scale[1] = 1.0
    plane.keyframe_insert(data_path="scale", frame=end_frame)

    return plane

def setup_camera():
    # Moved closer to match reference size, adjusted for the smaller crumpled scale
    bpy.ops.object.camera_add(location=(0, -2.8, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.object
    
    const = cam.constraints.new(type='DAMPED_TRACK')
    const.target = target
    const.track_axis = 'TRACK_NEGATIVE_Z'
    
    # Add strong lighting to see folds
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.object
    sun.data.energy = 5.0
    sun.rotation_euler = (0.5, 0.2, 0)

def main():
    import sys
    target_frame = None
    if "--frame" in sys.argv:
        idx = sys.argv.index("--frame")
        if idx + 1 < len(sys.argv):
            target_frame = int(sys.argv[idx+1])

    setup_scene()
    
    image_path = "data/extracted_texture_from_45.png"
    if not os.path.exists(image_path):
        image_path = "data/afisha-selected/3.jpg"

    # Timeline 33 -> 45
    create_crumpled_poster(image_path, start_frame=33, end_frame=45)
    setup_camera()
    
    output_dir = "data/render_uncrumple_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene
    scene.frame_start = 33
    scene.frame_end = 45

    if target_frame is not None:
        print(f"Rendering single frame: {target_frame}")
        scene.frame_current = target_frame
        scene.render.filepath = f"//{output_dir}/frame_{target_frame:04d}"
        bpy.ops.render.render(write_still=True)
    else:
        scene.render.filepath = f"//{output_dir}/frame_"
        bpy.ops.render.render(animation=True)

if __name__ == "__main__":
    main()
