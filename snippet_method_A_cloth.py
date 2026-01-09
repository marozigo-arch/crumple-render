
import bpy
import os
import sys

# Parameters from Cloth Funnels (PyFlex)
# cloth_stiffness = (0.75, .02, .02) # Stretch, Bend, Shear
# Mass is also high.
# In Blender, "Tension" = Stretch, "Bending" = Bending.
# Shear in Blender is "Shear".
# 0.75 is very high for PyFlex (soft body). In Blender range is 0-X.
# We will use high stiffness values matching the "natural" look they achieved.

def main():
    # 1. Setup Scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    
    # 2. Create Paper Mesh
    # Add Floor
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -2))
    floor = bpy.context.object
    floor.name = "Floor"
    bpy.ops.object.modifier_add(type='COLLISION')
    floor.collision.damping = 0.8
    floor.collision.cloth_friction = 0.8

    # Add Paper
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=100, y_subdivisions=100, size=2, location=(0,0,0))
    paper = bpy.context.object
    paper.name = "Paper_ClothFunnels"
    bpy.ops.object.shade_smooth()
    
    # 3. Add Cloth Modifier
    cloth_mod = paper.modifiers.new(name="Cloth", type='CLOTH')
    settings = cloth_mod.settings
    
    # MAPPING PYFLEX TO BLENDER (Empirical)
    # Stretch = 0.75 -> Tension = 80 (High)
    # Bend = 0.02 -> Bending = 0.5 (Low?? In PyFlex 0.02 is small, but paper is stiff?)
    # Wait, PyFlex 0.02 for bending is very flexible (like silk). 
    # BUT user said "rigid fabric". Maybe they use Scaling?
    # Let's try to interpret "stiffness factor" from generate_tasks.py: default=1.
    # If they simulate document, they might scale it up.
    # Let's stick to the user's insight: "Stiff Cloth".
    
    settings.quality = 12
    settings.mass = 0.3 # heavier paper
    settings.tension_stiffness = 80.0 # Stretch
    settings.compression_stiffness = 80.0 # Compression
    settings.shear_stiffness = 80.0 # Shear (0.02 in PyFlex? Maybe 80 is too high? Let's try high to act like paper)
    settings.bending_stiffness = 15.0 # Paper is stiff. 0.02 PyFlex might be normalized.
    # Damping (Split in Blender 2.82)
    settings.tension_damping = 5.0
    settings.compression_damping = 5.0
    settings.shear_damping = 5.0
    settings.bending_damping = 0.5
    
    # Collision
    cloth_mod.collision_settings.use_self_collision = True
    cloth_mod.collision_settings.self_distance_min = 0.002
    
    # 4. Crumpling Force (Turbulence)
    bpy.ops.object.effector_add(type='TURBULENCE', enter_editmode=False, location=(0, 0, 0))
    force = bpy.context.object
    force.field.strength = 200.0 # Strong turbulence to crumple
    force.field.size = 0.5
    force.field.noise = 10.0
    
    # Animate Force (On/Off)
    force.field.strength = 0
    force.keyframe_insert(data_path="field.strength", frame=1)
    force.field.strength = 500
    force.keyframe_insert(data_path="field.strength", frame=10)
    force.field.strength = 0
    force.keyframe_insert(data_path="field.strength", frame=30)
    
    # 5. Run Simulation
    scene.frame_start = 1
    scene.frame_end = 42
    scene.frame_current = 42
    
    # Bake? for headless we just render.
    
    # 6. Render Frame 42
    scene.render.filepath = os.path.abspath("data/benchmarks/method_A_cloth.png")
    # Camera
    bpy.ops.object.camera_add(location=(0, -3, 2))
    cam = bpy.context.object
    cam.rotation_euler = (1.0, 0, 0)
    scene.camera = cam
    
    # Light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    
    bpy.ops.render.render(write_still=True)
    print("Rendered Method A")
    
if __name__ == "__main__":
    main()
