import bpy
import os
import math

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

def create_unfolding_poster(image_path, start_frame, end_frame):
    # Load image
    try:
        img = bpy.data.images.load(image_path)
    except:
        print(f"Could not load {image_path}")
        return
    
    w, h = img.size
    aspect = w / h
    
    # Create plane with subdivisions for bending
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0), rotation=(1.5708, 0, 0))
    plane = bpy.context.object
    plane.name = "UnfoldingPoster"
    plane.scale[0] = aspect
    
    # Subdivide
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=10)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Material
    mat = bpy.data.materials.new(name="PosterMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = img
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    plane.data.materials.append(mat)
    
    # Add Simple Deform Modifier (Bend)
    # Need an Empty for control if we want specific axis, or just rely on local axis
    # For simple bend along axis, we usually don't need origin if we set axis right
    # But if we did need it, it cannot be self.
    
    # Create empty for origin control
    bpy.ops.object.empty_add(location=plane.location)
    empty = bpy.context.object
    empty.name = "BendOrigin"
    # Rotate empty to align bend axis if needed - for now keep aligned
    
    mod = plane.modifiers.new(name="Bend", type='SIMPLE_DEFORM')
    mod.deform_method = 'BEND'
    mod.origin = empty 
    # Based on analysis: Axis X
    # Angle: Analysis suggests X. Let's try 90 degrees (Unfolding)
    # Frame 33: 90 deg -> Frame 45: 0 deg
    mod.angle = math.radians(90)
    mod.deform_axis = 'X' 
    
    # Animate modifier angle
    mod.angle = math.radians(90)
    mod.keyframe_insert(data_path="angle", frame=start_frame)
    
    mod.angle = math.radians(0)
    mod.keyframe_insert(data_path="angle", frame=end_frame)
    
    # Also animate rotation/location slightly to match "dynamic" feel
    plane.rotation_euler[2] = math.radians(-10)
    plane.keyframe_insert(data_path="rotation_euler", frame=start_frame, index=2)
    
    plane.rotation_euler[2] = math.radians(0)
    plane.keyframe_insert(data_path="rotation_euler", frame=end_frame, index=2)

    return plane

def setup_camera():
    bpy.ops.object.camera_add(location=(0, -6, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Simple Look At
    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.object
    
    const = cam.constraints.new(type='DAMPED_TRACK')
    const.target = target
    const.track_axis = 'TRACK_NEGATIVE_Z'

def main():
    import sys
    
    # Simple argv parsing since blender consumes some args
    # Look for --frame <n>
    target_frame = None
    if "--frame" in sys.argv:
        idx = sys.argv.index("--frame")
        if idx + 1 < len(sys.argv):
            target_frame = int(sys.argv[idx+1])

    setup_scene()
    
    # Clean output dir
    output_dir = "data/render_unfold_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_path = "data/extracted_texture_from_45.png"
    if not os.path.exists(image_path):
        print(f"Warning: Extracted texture not found at {image_path}, falling back")
        image_path = "data/afisha-selected/3.jpg" # Fallback

    # Match reference timeline: Frame 45 is flat (end), Frame 33 is folded (start)
    # We want to enable rendering frames like 40.
    start_frame = 33
    end_frame = 45
    
    create_unfolding_poster(image_path, start_frame=start_frame, end_frame=end_frame)
    setup_camera()
    
    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    scene.render.filepath = f"//{output_dir}/frame_"

    if target_frame is not None:
        print(f"Rendering single frame: {target_frame}")
        scene.frame_current = target_frame
        # Explicitly format name to match animation output style
        # Frame 40 -> frame_0040.png
        scene.render.filepath = f"//{output_dir}/frame_{target_frame:04d}"
        bpy.ops.render.render(write_still=True)
    else:
        print("Rendering animation sequence")
        scene.render.filepath = f"//{output_dir}/frame_"
        bpy.ops.render.render(animation=True)

if __name__ == "__main__":
    main()
