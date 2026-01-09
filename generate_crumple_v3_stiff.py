import bpy
import math
import random
import os

# --- Settings ---
VERSION = "v3"
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
    
    # Lighting
    bpy.ops.object.light_add(type='AREA', location=(3, 2, 4))
    key = bpy.context.object
    key.data.energy = 2000
    key.data.size = 2.0
    
    # Back Rim Light to show edges/translucency illusion
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
    # Start slightly larger
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    paper = bpy.context.object
    paper.scale[0] = aspect * 1.2
    paper.scale[1] = 1.2
    
    # Geometry: LOW POLY to force LARGE TILES
    # High poly = Fabric. Low poly = Cardboard/Paper.
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=20) # Was 40. 20 creates larger facets.
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Pre-crumple: Minimal noise, just to break symmetry
    tex = bpy.data.textures.new("CrumpleTex", 'VORONOI')
    tex.noise_scale = 2.0 # Large scale noise
    mod_disp = paper.modifiers.new(name='PreCrumple', type='DISPLACE')
    mod_disp.texture = tex
    mod_disp.strength = 0.05
    
    # Material - Paper White
    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Roughness'].default_value = 0.8
    bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 0.95, 1)
    paper.data.materials.append(mat)
    
    # Shade FLAT to emphasize the "Tiles" (optional, but requested visual style effectively)
    bpy.ops.object.shade_flat()
    
    return paper

def setup_physics(obj):
    bpy.ops.object.modifier_add(type='CLOTH')
    mod = obj.modifiers["Cloth"]
    settings = mod.settings
    
    # STIFFNESS IS KEY
    settings.mass = 0.3 # Heavy enough to fall, not float
    settings.quality = 12
    settings.air_damping = 1.0 # Slight drag
    
    # Tension/Compression/Shear: MAX to prevent behaving like rubber/cloth
    # Paper doesn't stretch.
    settings.tension_stiffness = 500.0
    settings.compression_stiffness = 500.0
    settings.shear_stiffness = 500.0
    
    # Bending: HIGH to resist folding until forced -> snaps into crease
    settings.bending_model = 'ANGULAR' # Angular is good for creases
    settings.bending_stiffness = 80.0 # High value = Stiff
    
    # Collisions
    cols = mod.collision_settings
    cols.use_self_collision = True
    cols.distance_min = 0.02 # Paper thickness
    cols.self_friction = 5.0 # Paper slides a bit

def create_hands():
    # Large spheres to CRUSH the paper
    hands = []
    
    # We want "Hands" pressing from X then Y
    # But to prevent "Edges spoiling", we need the hands to be large enough to cover the edge,
    # OR push in a way that folds it in.
    
    # Setup 4 hands (Left, Right, Top, Bottom)
    # Move them sequentially but overlapping
    
    positions = [
        (-3.0, 0, 0), (3.0, 0, 0), # X Pair
        (0, -3.0, 0), (0, 3.0, 0)  # Y Pair
    ]
    
    for i, start_pos in enumerate(positions):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, location=(start_pos[0], start_pos[1], 0.5))
        hand = bpy.context.object
        hand.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # Determine axis
        is_x = abs(start_pos[0]) > 0
        
        # Animate
        hand.keyframe_insert(data_path="location", frame=1)
        
        # TIMELINE
        # Frames 1-30: X Hands press
        # Frames 20-50: Y Hands press (Overlap)
        # Frames 50-100: All crush to center
        
        # Target position (near center)
        target_val = 0.2 if start_pos[0] > 0 or start_pos[1] > 0 else -0.2
        target_pos = list(start_pos)
        
        if is_x:
            # X Hands Action
            hand.location = (start_pos[0], start_pos[1], 0.5)
            hand.keyframe_insert(data_path="location", frame=1)
            
            # Press in
            hand.location = (target_val * 3, 0, 0.2) # ~0.6
            hand.keyframe_insert(data_path="location", frame=35)
            
            # Stay/Crush more
            hand.location = (target_val, 0, 0)
            hand.keyframe_insert(data_path="location", frame=100)
            
        else:
            # Y Hands Action (Delayed)
            hand.location = (start_pos[0], start_pos[1], 0.5)
            hand.keyframe_insert(data_path="location", frame=15) # Wait start
            
            # Press in
            hand.location = (0, target_val * 3, 0.2)
            hand.keyframe_insert(data_path="location", frame=50)
            
            # Stay/Crush more
            hand.location = (0, target_val, 0)
            hand.keyframe_insert(data_path="location", frame=100)
            
        hands.append(hand)
        
    return hands

def add_floor():
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0,0,-0.02))
    floor = bpy.context.object
    bpy.ops.object.modifier_add(type='COLLISION')
    floor.hide_render = True

def run():
    setup_scene()
    paper = create_paper()
    setup_physics(paper)
    create_hands()
    add_floor()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Mappings
    ref_map = {
        1: 44,
        25: 42,
        50: 40,
        75: 37,
        100: 33
    }
    
    print(f"Starting simulation V3 (Stiff)... Output: {OUTPUT_DIR}")
    for f in range(1, 101):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        
        if f in ref_map:
            ref_idx = ref_map[f]
            filename = f"val_frame_{ref_idx:04d}.png"
            bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered V3 Frame {f} -> {filename}")

if __name__ == "__main__":
    run()
