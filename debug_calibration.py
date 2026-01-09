import bpy
import math

def setup_scene():
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Set resolution based on reference (864x1104)
    scene.render.resolution_x = 864
    scene.render.resolution_y = 1104
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 32
    
    # World Background
    if not scene.world:
        new_world = bpy.data.worlds.new("NewWorld")
        scene.world = new_world

    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes['Background']
    bg_node.inputs[0].default_value = (0.1, 0.1, 0.1, 1)

def create_axes():
    # Create materials
    mat_x = bpy.data.materials.new(name="Material_X")
    mat_x.use_nodes = True
    mat_x.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 0, 0, 1) # Red

    mat_y = bpy.data.materials.new(name="Material_Y")
    mat_y.use_nodes = True
    mat_y.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 1, 0, 1) # Green

    mat_z = bpy.data.materials.new(name="Material_Z")
    mat_z.use_nodes = True
    mat_z.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 0, 1, 1) # Blue
    
    # Create Axis Objects (Cylinders)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=5, location=(2.5, 0, 0), rotation=(0, 1.5708, 0))
    obj_x = bpy.context.object
    obj_x.name = "Axis_X"
    obj_x.data.materials.append(mat_x)
    
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=5, location=(0, 2.5, 0), rotation=(1.5708, 0, 0))
    obj_y = bpy.context.object
    obj_y.name = "Axis_Y"
    obj_y.data.materials.append(mat_y)
    
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=5, location=(0, 0, 2.5), rotation=(0, 0, 0))
    obj_z = bpy.context.object
    obj_z.name = "Axis_Z"
    obj_z.data.materials.append(mat_z)

def setup_camera():
    bpy.ops.object.camera_add(location=(5, -5, 5))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Point camera at origin
    # Using Damped Track as recommended
    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.object
    target.name = "Camera_Target"
    
    constraint = cam.constraints.new(type='DAMPED_TRACK')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    
    return cam

def main():
    setup_scene()
    create_axes()
    cam = setup_camera()
    
    # Render Frame 1
    bpy.context.scene.render.filepath = "//data/debug_render_001.png"
    bpy.ops.render.render(write_still=True)
    
if __name__ == "__main__":
    main()
