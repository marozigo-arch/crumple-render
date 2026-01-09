import bpy
import math
import random
import os

# --- Settings ---
VERSION = "v7"
OUTPUT_DIR = f"/workspaces/codespaces-jupyter/data/render_crumple_{VERSION}"
RES_X = 864
RES_Y = 1104
FRAMES = 100

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Camera - TrackTo Core
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
    
    return cam

def create_paper():
    aspect = RES_X / RES_Y
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    paper = bpy.context.object
    paper.scale[0] = aspect * 1.2
    paper.scale[1] = 1.2
    
    # Geometry: 20 cuts (Low Poly / Stiff)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=20) 
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
    # Damping: High to prevent explosions in the sandwich
    settings.air_damping = 2.0 
    
    # Stiffness: High (Cardboard)
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

def create_sandwich_system(cam):
    # 1. THE CORE (Inner Mold)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0,0,0))
    core = bpy.context.object
    core.name = "Core"
    core.hide_render = True
    bpy.ops.object.modifier_add(type='COLLISION')
    core.modifiers['Collision'].settings.thickness_outer = 0.05
    
    # Scale Animation: Shrink 1.0 -> 0.2
    # Start shrinking LATER (Frame 40)
    core.scale = (1.5, 1.5, 1.5) # Start BIG to catch the paper
    core.keyframe_insert(data_path="scale", frame=1)
    core.keyframe_insert(data_path="scale", frame=30) # Hold size
    
    core.scale = (0.2, 0.2, 0.2)
    core.keyframe_insert(data_path="scale", frame=100) # Shrink
    
    # Track camera to core (center)
    cons = cam.constraints.new(type='TRACK_TO')
    cons.target = core
    cons.track_axis = 'TRACK_NEGATIVE_Z'
    cons.up_axis = 'UP_Y'
    
    # 2. THE CRUSHER (Outer Shell)
    # 6 Planes forming a box that contracts
    # Just use huge cubes or planes.
    
    # Axis-aligned pushers: +/-X, +/-Y, +/-Z
    # Start Dist large (4.0) -> End Dist small (0.2)
    
    directions = [
        (4.0, 0, 0), (-4.0, 0, 0),
        (0, 4.0, 0), (0, -4.0, 0),
        (0, 0, 4.0), (0, 0, -4.0)
    ]
    
    for i, start_pos in enumerate(directions):
        # Create a large plane facing center
        bpy.ops.mesh.primitive_plane_add(size=10, location=start_pos)
        pusher = bpy.context.object
        pusher.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # Rotate to face center (rudimentary)
        # Or just use Cube faces. Actually UV Sphere is safer/easier collision.
        # Let's use Large Spheres as "Walls"
        
        bpy.ops.object.delete() # Delete plane
        
        radius = 5.0 # Giant wall-sphere
        # Offset pos so surface is at start_pos
        # e.g. at X=4, sphere center should be at X=9 (4+5) so surface touches 4.
        
        x, y, z = start_pos
        offset_dist = radius
        
        cx = x + (offset_dist if x > 0 else -offset_dist if x < 0 else 0)
        cy = y + (offset_dist if y > 0 else -offset_dist if y < 0 else 0)
        cz = z + (offset_dist if z > 0 else -offset_dist if z < 0 else 0)
        
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(cx, cy, cz))
        wall = bpy.context.object
        wall.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # Animation
        
        # Frame 1: Far away
        wall.keyframe_insert(data_path="location", frame=1)
        
        # Frame 40: Touch the Core (which is radius 1.5)
        # So we move in until surface is at ~1.5
        # Target dist = 1.6
        
        move_vec = [0,0,0]
        if x != 0: move_vec[0] = (1 if x > 0 else -1)
        if y != 0: move_vec[1] = (1 if y > 0 else -1)
        if z != 0: move_vec[2] = (1 if z > 0 else -1)
        
        # Move in to Compression Phase 1 (Touch Core)
        target_dist_1 = 1.8 
        t1_x = cx - (x - (move_vec[0] * target_dist_1))
        t1_y = cy - (y - (move_vec[1] * target_dist_1))
        t1_z = cz - (z - (move_vec[2] * target_dist_1))
        
        wall.location = (cx - (x - move_vec[0]*target_dist_1), 
                         cy - (y - move_vec[1]*target_dist_1), 
                         cz - (z - move_vec[2]*target_dist_1))
        
        # Simplified math:
        # Start Loc = (9, 0, 0) (Surf at 4)
        # End Loc needs Surf at 0.2
        # Delta = 4 - 0.2 = 3.8
        # New Loc = 9 - 3.8 = 5.2
        
        # Safer way:
        start_surf = 4.0
        end_surf = 0.2
        delta = start_surf - end_surf
        
        # Animate
        # Frame 30: Start moving
        wall.location = (cx, cy, cz)
        wall.keyframe_insert(data_path="location", frame=25)
        
        # Frame 100: CRUSH
        wall.location = (cx - move_vec[0]*delta, 
                         cy - move_vec[1]*delta, 
                         cz - move_vec[2]*delta)
        wall.keyframe_insert(data_path="location", frame=100)


def run():
    cam = setup_scene()
    paper = create_paper()
    setup_physics(paper)
    create_sandwich_system(cam)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ref_map = {
        1: 44,
        25: 42,
        50: 40,
        75: 37,
        100: 33
    }
    
    print(f"Starting simulation V7 (Sandwich)... Output: {OUTPUT_DIR}")
    for f in range(1, 101):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        
        if f in ref_map:
            ref_idx = ref_map[f]
            filename = f"val_frame_{ref_idx:04d}.png"
            bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered V7 Frame {f} -> {filename}")

if __name__ == "__main__":
    run()
