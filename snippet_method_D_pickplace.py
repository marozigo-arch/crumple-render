
import bpy
import os
import math

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    
    # 1. Create Floor
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0,0,-2))
    floor = bpy.context.object
    bpy.ops.object.modifier_add(type='COLLISION')
    floor.collision.damping = 0.8
    floor.collision.cloth_friction = 0.8

    # Create Paper
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=80, y_subdivisions=80, size=2, location=(0,0,0.01))
    paper = bpy.context.object
    paper.name = "Paper_PickPlace"
    bpy.ops.object.shade_smooth()
    
    # 2. Physics (Synthetic-Cloth-Data "Paper" params - High Stiffness)
    bpy.ops.object.modifier_add(type='CLOTH')
    cloth_mod = paper.modifiers["Cloth"]
    settings = cloth_mod.settings
    settings.quality = 8 # They use 5, but 8 is safer for paper
    settings.mass = 0.5
    settings.air_damping = 2.0
    settings.tension_stiffness = 50.0 # Max of their range
    settings.compression_stiffness = 50.0
    settings.shear_stiffness = 20.0
    settings.bending_stiffness = 40.0 # High value for paper
    
    # Damping (Blender 2.82)
    settings.tension_damping = 5.0
    settings.compression_damping = 5.0
    settings.shear_damping = 5.0
    settings.bending_damping = 0.5
    
    col = cloth_mod.collision_settings
    col.use_self_collision = True
    col.self_distance_min = 0.003
    col.distance_min = 0.003
    
    # 3. Pin Group for Grasping
    corner_idx = 0 # Bottom left corner
    vg = paper.vertex_groups.new(name="Pin")
    vg.add([corner_idx], 1.0, 'REPLACE')
    # settings.vertex_group_mass = "Pin" # We will animate this weight
    # Actually, we use a Hook modifier with vertex weight mix like they do.
    
    # 4. Animate "Pick and Drop"
    # Create Empty Hook
    bpy.ops.object.empty_add()
    hook_empty = bpy.context.object
    hook_empty.location = paper.data.vertices[corner_idx].co
    
    # Hook Modifier
    hook_mod = paper.modifiers.new(name="Hook", type='HOOK')
    hook_mod.object = hook_empty
    hook_mod.vertex_group = "Pin"
    
    # Animate Empty
    # Frame 1: Start
    hook_empty.keyframe_insert(data_path="location", frame=1)
    
    # Frame 15: Lift up and over
    hook_empty.location = (-0.5, 0.5, 1.5) # Move corner to center-ish high
    hook_empty.keyframe_insert(data_path="location", frame=15)
    
    # Frame 25: Drop (Release Hook influence)
    # We can't easily animate Hook influence in 2.82 directly if not exposed?
    # They used VertexWeightMix modifier to set weight to 0.
    # Let's try that.
    
    mix_mod = paper.modifiers.new(name="WeightMix", type='VERTEX_WEIGHT_MIX')
    mix_mod.vertex_group_a = "Pin"
    mix_mod.mix_set = 'OR'
    mix_mod.mix_mode = 'SET'
    mix_mod.default_weight_a = 0.0 # Default weight for all? No.
    # Actually simpler: Just animate the Pin group stiffness?
    # Or just animate the hook strength.
    hook_mod.strength = 1.0
    hook_mod.keyframe_insert(data_path="strength", frame=1)
    hook_mod.keyframe_insert(data_path="strength", frame=20)
    hook_mod.strength = 0.0
    hook_mod.keyframe_insert(data_path="strength", frame=21) # Release
    
    # 5. Run Simulation
    scene.frame_start = 1
    scene.frame_end = 42
    scene.frame_current = 42
    
    # Camera
    bpy.ops.object.camera_add(location=(0, -2.5, 2.5))
    cam = bpy.context.object
    bpy.ops.object.constraint_add(type='TRACK_TO')
    const = cam.constraints["Track To"]
    const.target = paper
    const.track_axis = 'TRACK_NEGATIVE_Z'
    const.up_axis = 'UP_Y'
    scene.camera = cam
    
    # Light
    bpy.ops.object.light_add(type='AREA', location=(0,0,5))
    
    # Render
    out_path = os.path.abspath("data/benchmarks/method_D_pickplace.png")
    scene.render.filepath = out_path
    
    # Bake/Run?
    # Just render frame 42, Blender will simulate frames 1-42.
    bpy.ops.render.render(write_still=True)
    print("Rendered Method D")
    
if __name__ == "__main__":
    main()
