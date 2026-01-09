import bpy
import math
import random
import os

# --- Settings ---
VERSION = "v2"
OUTPUT_DIR = f"/workspaces/codespaces-jupyter/data/render_crumple_{VERSION}"
RES_X = 864
RES_Y = 1104
FRAMES = 100

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 4.5), rotation=(0, 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Light
    bpy.ops.object.light_add(type='AREA', location=(3, 2, 4))
    key = bpy.context.object
    key.data.energy = 1500
    key.data.size = 2.0
    
    bpy.ops.object.light_add(type='AREA', location=(-2, -3, 3))
    fill = bpy.context.object
    fill.data.energy = 500

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.render.resolution_x = RES_X
    bpy.context.scene.render.resolution_y = RES_Y
    bpy.context.scene.frame_end = FRAMES

def create_paper():
    aspect = RES_X / RES_Y
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    paper = bpy.context.object
    paper.scale[0] = aspect * 1.2
    paper.scale[1] = 1.2
    
    # Geometry
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=40)
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Pre-crumple
    tex = bpy.data.textures.new("CrumpleTex", 'VORONOI')
    tex.noise_scale = 4.0
    mod_disp = paper.modifiers.new(name='PreCrumple', type='DISPLACE')
    mod_disp.texture = tex
    mod_disp.strength = 0.05
    
    # Material
    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Roughness'].default_value = 1.0
    paper.data.materials.append(mat)
    
    return paper

def setup_physics(obj):
    bpy.ops.object.modifier_add(type='CLOTH')
    mod = obj.modifiers["Cloth"]
    settings = mod.settings
    
    # Fast, light physics
    settings.mass = 0.15 
    settings.quality = 12
    settings.air_damping = 0.5
    
    settings.bending_model = 'ANGULAR'
    settings.bending_stiffness = 3.0 # Sharp folds
    
    # Collisions
    cols = mod.collision_settings
    cols.use_self_collision = True
    cols.distance_min = 0.015
    cols.self_friction = 15.0

def create_hands():
    # We simulate "Hands" as large invisible Collision Spheres.
    # Logic: 
    # 1. Squeeze X (Sides) -> creates vertical/diagonal buckling (Frame 1-40)
    # 2. Squeeze Y (Top/Bot) -> creates horizontal folds (Frame 30-70)
    # 3. Final Compact -> All press in (Frame 70-100)
    
    radius_hand = 1.5 # Large "Palms"
    start_dist = 3.5
    
    hands = []
    
    # --- PAIR 1: Left/Right (X-Axis) ---
    for x_dir in [-1, 1]:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius_hand, location=(x_dir * start_dist, 0, 0.2))
        hand = bpy.context.object
        hand.name = f"Hand_X_{x_dir}"
        hand.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # ANIMATION
        # Frame 1: Start far
        hand.keyframe_insert(data_path="location", frame=1)
        
        # Frame 30: Squeeze IN (to ~0.8)
        hand.location = (x_dir * 0.8, 0, 0.2)
        hand.keyframe_insert(data_path="location", frame=30)
        
        # Frame 40: Hold/Pulse
        hand.location = (x_dir * 0.9, 0, 0.2) # Slight release
        hand.keyframe_insert(data_path="location", frame=40)
        
        # Frame 100: Final Crush
        hand.location = (x_dir * 0.1, 0, 0)
        hand.keyframe_insert(data_path="location", frame=100)
        
        hands.append(hand)
        
    # --- PAIR 2: Top/Bottom (Y-Axis) ---
    for y_dir in [-1, 1]:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius_hand, location=(0, y_dir * start_dist, 0.2))
        hand = bpy.context.object
        hand.name = f"Hand_Y_{y_dir}"
        hand.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # ANIMATION
        # Frame 1: Start far
        hand.keyframe_insert(data_path="location", frame=1)
        
        # Frame 30: Start moving in
        
        # Frame 60: Squeeze IN (to ~0.8) - happens AFTER X squeeze
        hand.location = (0, y_dir * 0.8, 0.2)
        hand.keyframe_insert(data_path="location", frame=60)
        
        # Frame 100: Final Crush
        hand.location = (0, y_dir * 0.1, 0)
        hand.keyframe_insert(data_path="location", frame=100)
        
        hands.append(hand)
        
    return hands

def run():
    setup_scene()
    paper = create_paper()
    setup_physics(paper)
    create_hands()
    
    # Floor
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0,0,-0.05))
    floor = bpy.context.object
    bpy.ops.object.modifier_add(type='COLLISION')
    floor.hide_render = True
    
    # Mappings
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ref_map = {
        1: 44,
        25: 42,
        50: 40,
        75: 37,
        100: 33
    }
    
    print("Starting simulation V2 (Hands)...")
    for f in range(1, 101):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        
        if f in ref_map:
            ref_idx = ref_map[f]
            filename = f"val_frame_{ref_idx:04d}.png" # Use 'val' for validation frames
            bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered V2 Frame {f} -> {filename}")

if __name__ == "__main__":
    run()
