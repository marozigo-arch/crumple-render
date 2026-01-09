import bpy
import math
import random
import os

# --- Settings ---
VERSION = "v4"
OUTPUT_DIR = f"/workspaces/codespaces-jupyter/data/render_crumple_{VERSION}"
RES_X = 864
RES_Y = 1104
FRAMES = 100

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 4.3), rotation=(0, 0, 0)) # Slightly closer
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Lighting - High Contrast
    bpy.ops.object.light_add(type='AREA', location=(4, 2, 4))
    key = bpy.context.object
    key.data.energy = 2500
    key.data.size = 2.0
    
    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 2))
    fill = bpy.context.object
    fill.data.energy = 600

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
    
    # Geometry: Low Poly -> Stiff Tiles
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=20) 
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Pre-crumple: Stronger noise to prevent "Football" symmetry
    tex = bpy.data.textures.new("CrumpleTex", 'VORONOI')
    tex.noise_scale = 3.0
    mod_disp = paper.modifiers.new(name='PreCrumple', type='DISPLACE')
    mod_disp.texture = tex
    mod_disp.strength = 0.1 # Strong initial unevenness
    
    # Material
    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Roughness'].default_value = 0.9
    paper.data.materials.append(mat)
    
    bpy.ops.object.shade_flat()
    return paper

def setup_physics(obj):
    bpy.ops.object.modifier_add(type='CLOTH')
    mod = obj.modifiers["Cloth"]
    settings = mod.settings
    
    settings.mass = 0.3
    settings.quality = 12
    # Damping: Low to allow Spring Back
    settings.air_damping = 0.1 
    
    # Stiffness: High
    settings.tension_stiffness = 200.0
    settings.compression_stiffness = 200.0
    settings.shear_stiffness = 200.0
    
    settings.bending_model = 'ANGULAR'
    settings.bending_stiffness = 50.0 
    
    # Friction: Lower to allow sliding/opening
    cols = mod.collision_settings
    cols.use_self_collision = True
    cols.distance_min = 0.02
    cols.self_friction = 2.0 

def create_forces():
    # 1. INITIAL TURBULENCE (The "Chaos" starter)
    # This prevents the "Football" tube by buckling it randomly first.
    bpy.ops.object.effector_add(type='TURBULENCE', location=(0,0,0))
    turb = bpy.context.object
    turb.field.size = 2.0
    turb.field.strength = 0.0
    
    turb.keyframe_insert(data_path="field.strength", frame=1)
    turb.field.strength = 1000.0 # BANG
    turb.keyframe_insert(data_path="field.strength", frame=5)
    turb.field.strength = 0.0
    turb.keyframe_insert(data_path="field.strength", frame=20)

def create_pulsing_hands():
    # 4 Hands that SQUEEZE -> RELEASE -> SQUEEZE
    hands = []
    radius = 1.5
    start_d = 3.5
    
    # 4 Diagonal Hands (X-shape)
    # This avoids axis-aligned "Football" tube
    directions = [
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    
    for dx, dy in directions:
        # Normalize roughly
        loc_x = dx * (start_d / 1.414)
        loc_y = dy * (start_d / 1.414)
        
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(loc_x, loc_y, 0.2))
        hand = bpy.context.object
        hand.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # KEYFRAMES for "Spring Back"
        
        # Frame 1: Out
        hand.keyframe_insert(data_path="location", frame=1)
        
        # Frame 30: Squeeze IN (Pulse 1)
        hand.location = (loc_x * 0.3, loc_y * 0.3, 0.2)
        hand.keyframe_insert(data_path="location", frame=30)
        
        # Frame 45: RELEASE / Spring Back
        # Move hands OUT slightly to let paper expand
        hand.location = (loc_x * 0.5, loc_y * 0.5, 0.2)
        hand.keyframe_insert(data_path="location", frame=45)
        
        # Frame 70: Squeeze Deeper (Pulse 2)
        hand.location = (loc_x * 0.15, loc_y * 0.15, 0.1)
        hand.keyframe_insert(data_path="location", frame=70)
        
        # Frame 100: CRUSH
        hand.location = (0, 0, 0)
        hand.keyframe_insert(data_path="location", frame=100)
        
        hands.append(hand)

def run():
    setup_scene()
    paper = create_paper()
    setup_physics(paper)
    create_forces()
    create_pulsing_hands()
    
    # Floor
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0,0,-0.02))
    floor = bpy.context.object
    bpy.ops.object.modifier_add(type='COLLISION')
    floor.hide_render = True
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ref_map = {
        1: 44,
        25: 42,
        50: 40,
        75: 37,
        100: 33
    }
    
    print(f"Starting simulation V4 (Spring)... Output: {OUTPUT_DIR}")
    for f in range(1, 101):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        
        if f in ref_map:
            ref_idx = ref_map[f]
            filename = f"val_frame_{ref_idx:04d}.png"
            bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered V4 Frame {f} -> {filename}")

if __name__ == "__main__":
    run()
