import bpy
import math
import random
import os

# --- Settings ---
VERSION = "v5"
OUTPUT_DIR = f"/workspaces/codespaces-jupyter/data/render_crumple_{VERSION}"
RES_X = 864
RES_Y = 1104
FRAMES = 100

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Camera with TrackTo
    bpy.ops.object.camera_add(location=(0, 0, 4.5), rotation=(0, 0, 0))
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
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    paper = bpy.context.object
    paper.name = "Paper"
    paper.scale[0] = aspect * 1.2
    paper.scale[1] = 1.2
    
    # Geometry: Low Poly (20 cuts) for stiff tiles
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=20) 
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Pre-crumple
    tex = bpy.data.textures.new("CrumpleTex", 'VORONOI')
    tex.noise_scale = 2.0
    mod_disp = paper.modifiers.new(name='PreCrumple', type='DISPLACE')
    mod_disp.texture = tex
    mod_disp.strength = 0.08
    
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
    # Damping: Moderate
    settings.air_damping = 1.0 
    
    # Stiffness: High
    settings.tension_stiffness = 500.0
    settings.compression_stiffness = 500.0
    settings.shear_stiffness = 500.0
    settings.bending_model = 'ANGULAR'
    settings.bending_stiffness = 50.0 
    
    # Collisions
    cols = mod.collision_settings
    cols.use_self_collision = True
    cols.distance_min = 0.02
    cols.self_friction = 5.0 

def create_forces():
    # 1. INITIAL TURBULENCE (Buckling Phase)
    # Active Frames 1-40.
    bpy.ops.object.effector_add(type='TURBULENCE', location=(0,0,0))
    turb = bpy.context.object
    turb.field.size = 2.0
    turb.field.strength = 0.0
    
    turb.keyframe_insert(data_path="field.strength", frame=1)
    
    # Strong Pulse to create initial large folds
    turb.field.strength = 500.0 
    turb.keyframe_insert(data_path="field.strength", frame=15)
    
    turb.field.strength = 0.0 
    turb.keyframe_insert(data_path="field.strength", frame=45)

def create_slow_hands():
    # 4 Large Spheres that start FAR and close in SLOWLY.
    hands = []
    radius = 2.0 # Very large, "wall-like"
    start_dist = 4.0
    
    # Position: Left, Right, Top, Bottom
    positions = [
        (-start_dist, 0, 0), (start_dist, 0, 0),
        (0, -start_dist, 0), (0, start_dist, 0)
    ]
    
    for i, pos in enumerate(positions):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=pos)
        hand = bpy.context.object
        hand.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # ANIMATION SCHEDULE
        # Ref 42 is Frame 25. We want ONLY folding here, NO crushing.
        # Ref 40 is Frame 50. We want visible crumpledness.
        # Ref 33 is Frame 100. We want a ball.
        
        hand.keyframe_insert(data_path="location", frame=1)
        
        # Frame 30: Reach the "Touching" distance (Radius 2 + Paper Width ~1.2)
        # So hand center needs to be at ~2.5
        # The hands shouldn't touch the paper until ~Frame 30.
        
        x_dir = 0
        y_dir = 0
        if pos[0] != 0: x_dir = 1 if pos[0] > 0 else -1
        if pos[1] != 0: y_dir = 1 if pos[1] > 0 else -1
        
        # Frame 40: Contact & Start Press
        # Target: Just touching edges
        touch_dist = 2.5 
        hand.location = (x_dir * touch_dist, y_dir * touch_dist, 0)
        hand.keyframe_insert(data_path="location", frame=40)
        
        # Frame 100: CRUSH
        # Target: 0.0 (Overlapping in center)
        hand.location = (x_dir * 0.2, y_dir * 0.2, 0)
        hand.keyframe_insert(data_path="location", frame=100)
        
        hands.append(hand)

def fix_camera_tracking(cam, target):
    # Ensure camera always looks at the paper so it doesn't "disappear"
    cons = cam.constraints.new(type='TRACK_TO')
    cons.target = target
    cons.track_axis = 'TRACK_NEGATIVE_Z'
    cons.up_axis = 'UP_Y'

def run():
    setup_scene()
    paper = create_paper()
    setup_physics(paper)
    create_forces()
    create_slow_hands()
    
    # Fix camera to paper center
    fix_camera_tracking(bpy.context.scene.camera, paper)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ref_map = {
        1: 44,
        25: 42,
        50: 40,
        75: 37,
        100: 33
    }
    
    print(f"Starting simulation V5 (Slow)... Output: {OUTPUT_DIR}")
    for f in range(1, 101):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        
        if f in ref_map:
            ref_idx = ref_map[f]
            filename = f"val_frame_{ref_idx:04d}.png"
            bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered V5 Frame {f} -> {filename}")

if __name__ == "__main__":
    run()
