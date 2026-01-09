import bpy
import math
import random
import os

# --- Settings ---
VERSION = "v8"
OUTPUT_DIR = f"/workspaces/codespaces-jupyter/data/render_crumple_{VERSION}"
RES_X = 864
RES_Y = 1104
FRAMES = 100

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Camera - TrackTo Core
    bpy.ops.object.camera_add(location=(0, 0, 4.0), rotation=(0, 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Lighting
    bpy.ops.object.light_add(type='AREA', location=(3, 2, 4))
    key = bpy.context.object
    key.data.energy = 2500
    key.data.size = 2.0
    
    bpy.ops.object.light_add(type='AREA', location=(-2, -3, 2))
    fill = bpy.context.object
    fill.data.energy = 800

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.render.resolution_x = RES_X
    bpy.context.scene.render.resolution_y = RES_Y
    bpy.context.scene.frame_end = FRAMES

def create_paper():
    aspect = RES_X / RES_Y
    # Start slightly ABOVE center so it falls/pulls down
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0.5)) 
    paper = bpy.context.object
    paper.scale[0] = aspect * 1.2
    paper.scale[1] = 1.2
    
    # Geometry: 25 cuts (Cardboard stiff)
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
    
    settings.mass = 0.3
    settings.quality = 12
    # Damping: High to stabilize the magnetic pull
    settings.air_damping = 2.0
    
    # Stiffness
    settings.tension_stiffness = 500.0
    settings.compression_stiffness = 500.0
    settings.shear_stiffness = 500.0
    settings.bending_model = 'ANGULAR'
    settings.bending_stiffness = 80.0 
    
    # Collisions
    cols = mod.collision_settings
    cols.use_self_collision = True
    cols.distance_min = 0.02
    cols.self_friction = 5.0 

def create_magnetic_core(cam):
    # 1. THE CORE SPHERE
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0,0,0))
    core = bpy.context.object
    core.name = "Core"
    core.hide_render = True
    
    bpy.ops.object.modifier_add(type='COLLISION')
    core.modifiers['Collision'].settings.thickness_outer = 0.05
    # Fix: Removed friction assignment that caused crash in V6
    
    # Anim: Stand still, then Shrink
    # Start Radius ~1.0
    core.scale = (1.2, 1.2, 1.2)
    core.keyframe_insert(data_path="scale", frame=1)
    
    # Shrink starts at Frame 40
    core.keyframe_insert(data_path="scale", frame=40)
    
    # End at Frame 100
    core.scale = (0.2, 0.2, 0.2)
    core.keyframe_insert(data_path="scale", frame=100)
    
    # 2. THE MAGNET (Harmonic Point Force)
    bpy.ops.object.effector_add(type='HARMONIC', location=(0,0,0))
    magnet = bpy.context.object
    magnet.name = "Magnet"
    
    # Settings
    magnet.field.shape = 'POINT'
    magnet.field.strength = 100.0 # Pulls objects towards origin
    magnet.field.rest_length = 0.0 # Pulls to center point
    
    # Animate Strength
    # Frame 1-20: Weak pulse to start movement
    magnet.field.strength = 0.0
    magnet.keyframe_insert(data_path="field.strength", frame=1)
    
    magnet.field.strength = 100.0
    magnet.keyframe_insert(data_path="field.strength", frame=20)
    
    # Frame 60: Increase pull as core shrinks
    magnet.field.strength = 200.0
    magnet.keyframe_insert(data_path="field.strength", frame=60)
    
    # 3. TURBULENCE (To break symmetry)
    bpy.ops.object.effector_add(type='TURBULENCE', location=(1,1,1))
    turb = bpy.context.object
    turb.field.strength = 20.0 # Subtle
    turb.field.size = 2.0

    # Track Camera
    cons = cam.constraints.new(type='TRACK_TO')
    cons.target = core
    cons.track_axis = 'TRACK_NEGATIVE_Z'
    cons.up_axis = 'UP_Y'

def run():
    setup_scene()
    paper = create_paper()
    setup_physics(paper)
    create_magnetic_core(bpy.context.scene.camera)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ref_map = {
        1: 44,
        25: 42,
        50: 40,
        75: 37,
        100: 33
    }
    
    print(f"Starting simulation V8 (Magnet)... Output: {OUTPUT_DIR}")
    for f in range(1, 101):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        
        if f in ref_map:
            ref_idx = ref_map[f]
            filename = f"val_frame_{ref_idx:04d}.png"
            bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered V8 Frame {f} -> {filename}")

if __name__ == "__main__":
    run()
