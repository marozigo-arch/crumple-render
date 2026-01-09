import bpy
import math
import random
import os

# --- Settings ---
VERSION = "v6"
OUTPUT_DIR = f"/workspaces/codespaces-jupyter/data/render_crumple_{VERSION}"
RES_X = 864
RES_Y = 1104
FRAMES = 100

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 4.0), rotation=(0, 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Raking Light for Volume
    bpy.ops.object.light_add(type='AREA', location=(3, 2, 4))
    key = bpy.context.object
    key.data.energy = 2000
    
    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 2))
    fill = bpy.context.object
    fill.data.energy = 800

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.render.resolution_x = RES_X
    bpy.context.scene.render.resolution_y = RES_Y
    bpy.context.scene.frame_end = FRAMES

def create_paper():
    aspect = RES_X / RES_Y
    # Start slightly larger
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0.5)) # Start ABOVE center
    paper = bpy.context.object
    paper.scale[0] = aspect * 1.2
    paper.scale[1] = 1.2
    
    # Geometry: Medium-Low Poly for "Cardboard" folds
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=25) 
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Material
    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Roughness'].default_value = 1.0
    paper.data.materials.append(mat)
    
    bpy.ops.object.shade_flat()
    return paper

def setup_physics(obj):
    bpy.ops.object.modifier_add(type='CLOTH')
    mod = obj.modifiers["Cloth"]
    settings = mod.settings
    
    settings.mass = 0.5 # Heavier "Cardboard"
    settings.quality = 12
    # Damping: High to prevent vibrating on the sphere
    settings.air_damping = 2.0 
    
    # Stiffness: VERY HIGH to resist bending
    settings.tension_stiffness = 800.0
    settings.compression_stiffness = 800.0
    settings.shear_stiffness = 800.0
    
    settings.bending_model = 'ANGULAR'
    settings.bending_stiffness = 150.0 # Force large folds
    
    # Collisions
    cols = mod.collision_settings
    cols.use_self_collision = True
    cols.distance_min = 0.02
    cols.self_friction = 10.0 

def create_core_system():
    # 1. THE CORE SPHERE (The "Mold")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, location=(0,0,0))
    core = bpy.context.object
    core.name = "ShrinkingCore"
    core.hide_render = True
    
    # Collision
    bpy.ops.object.modifier_add(type='COLLISION')
    core.modifiers['Collision'].settings.thickness_outer = 0.05
    core.modifiers['Collision'].settings.thickness_outer = 0.05
    # core.modifiers['Collision'].settings.damping = 1.0 # Use damping instead if needed
    
    # ANIMATION: SHRINK
    # Frame 1: Full size (Paper wraps around ~1.2 radius)
    core.scale = (1, 1, 1)
    core.keyframe_insert(data_path="scale", frame=1)
    
    # Frame 40: Half size (Ref 40)
    core.scale = (0.6, 0.6, 0.6)
    core.keyframe_insert(data_path="scale", frame=40)
    
    # Frame 100: Tiny (Ref 33)
    core.scale = (0.15, 0.15, 0.15)
    core.keyframe_insert(data_path="scale", frame=100)
    
    # 2. SUCTION FORCE (The "Vacuum")
    # Pulls paper onto the core
    bpy.ops.object.effector_add(type='FORCE', location=(0,0,0))
    suc = bpy.context.object
    suc.field.strength = -100.0 # Pull IN
    suc.field.flow = 5.0 # Drag
    
    # 3. TURBULENCE (The "Imperfection")
    # Adds randomness so folds aren't perfect longitude lines
    bpy.ops.object.effector_add(type='TURBULENCE', location=(1,1,0))
    turb = bpy.context.object
    turb.field.strength = 100.0
    turb.field.size = 1.5

def run():
    setup_scene()
    paper = create_paper()
    setup_physics(paper)
    create_core_system()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ref_map = {
        1: 44,
        25: 42,
        50: 40,
        75: 37,
        100: 33
    }
    
    print(f"Starting simulation V6 (Core)... Output: {OUTPUT_DIR}")
    for f in range(1, 101):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        
        if f in ref_map:
            ref_idx = ref_map[f]
            filename = f"val_frame_{ref_idx:04d}.png"
            bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered V6 Frame {f} -> {filename}")

if __name__ == "__main__":
    run()
