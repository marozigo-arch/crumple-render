import bpy
import math
import random
import os

# --- Settings ---
OUTPUT_DIR = "/workspaces/codespaces-jupyter/data/render_crumple"
RES_X = 864
RES_Y = 1104
FRAMES = 100

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Camera - Top-down
    bpy.ops.object.camera_add(location=(0, 0, 4.5), rotation=(0, 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Lighting - Low Raking Light for Volume
    bpy.ops.object.light_add(type='AREA', location=(3, 2, 2.5))
    key = bpy.context.object
    key.data.energy = 1200
    key.data.size = 3.0
    
    bpy.ops.object.light_add(type='AREA', location=(-2, -3, 4))
    fill = bpy.context.object
    fill.data.energy = 300
    
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
    
    # Geometry: TRIANGULATED for sharp angular folds
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=35) 
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Displace Texture (Voronoi) - Pre-crumple Geometry
    tex = bpy.data.textures.new("CrumpleTex", 'VORONOI')
    tex.noise_scale = 4.0
    tex.distance_metric = 'MANHATTAN'
    
    mod_disp = paper.modifiers.new(name='PreCrumple', type='DISPLACE')
    mod_disp.texture = tex
    mod_disp.strength = 0.08 # Stronger initial creases (was 0.02)
    
    # Material
    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1)
    paper.data.materials.append(mat)
    
    return paper

def setup_physics(obj):
    bpy.ops.object.modifier_add(type='CLOTH')
    mod = obj.modifiers["Cloth"]
    settings = mod.settings
    
    settings.mass = 0.1 # Lighter paper
    settings.quality = 12 
    
    settings.tension_stiffness = 80
    settings.compression_stiffness = 80
    settings.shear_stiffness = 80
    settings.bending_model = 'ANGULAR'
    settings.bending_stiffness = 2.0 # Allow easier bending
    
    settings.air_damping = 0.1 # REMOVE DRAG. 5.0 was acting like water.
    
    # Collisions
    cols = mod.collision_settings
    cols.use_self_collision = True
    cols.distance_min = 0.015 
    cols.self_friction = 10.0
    
def add_forces():
    # 1. TURBULENCE: Immediate, strong noise to buckle the paper (Sim 1-25)
    bpy.ops.object.effector_add(type='TURBULENCE', location=(0,0,0))
    turb = bpy.context.object
    turb.field.size = 2.0
    turb.field.strength = 0.0
    turb.keyframe_insert(data_path="field.strength", frame=1)
    
    turb.field.strength = 2000.0 # EXPLOSIVE Force to ensure buckling
    turb.keyframe_insert(data_path="field.strength", frame=15)
    
    turb.field.strength = 0.0 
    turb.keyframe_insert(data_path="field.strength", frame=40)

    # 2. FINAL COMPACTOR: Shrink the ball at the end (Sim 80-100)
    bpy.ops.object.effector_add(type='HARMONIC', location=(0,0,0))
    compactor = bpy.context.object
    compactor.name = "Compactor"
    compactor.field.rest_length = 0.0
    compactor.field.strength = 0.0
    compactor.keyframe_insert(data_path="field.strength", frame=1)
    compactor.keyframe_insert(data_path="field.strength", frame=80) 
    
    compactor.field.strength = 100.0 # Suck into a ball
    compactor.keyframe_insert(data_path="field.strength", frame=100)

def create_pushers():
    # 3. PUSHERS: "Crush" the paper (Frames 10-100)
    pushers = []
    count = 8
    radius = 2.5
    
    for i in range(count):
        angle = (i / count) * 2 * math.pi
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius
        
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(x, y, 0))
        sphere = bpy.context.object
        sphere.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')
        
        # Animate
        sphere.keyframe_insert(data_path="location", frame=1)
        
        # Start moving IN immediately.
        # Target: DEEP CRUSH (Almost 0)
        tx = random.uniform(-0.05, 0.05)
        ty = random.uniform(-0.05, 0.05)
        
        sphere.location = (tx, ty, 0)
        # Reach full crush by Frame 100
        sphere.keyframe_insert(data_path="location", frame=100)
        
        pushers.append(sphere)

def run():
    setup_scene()
    paper = create_paper()
    setup_physics(paper)
    add_forces()
    create_pushers()
    
    # Floor with Friction to prevent sliding
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0,0,-0.05))
    floor = bpy.context.object
    bpy.ops.object.modifier_add(type='COLLISION')
    # floor.modifiers["Collision"].settings.friction = 50.0 # Error.
    # Friction is mostly determined by the Cloth's friction setting interacting with this.
    floor.hide_render = True
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Mappings
    # Ref 44 (Flat) -> Sim 1
    # Ref 42 (Buckled) -> Sim 25
    # Ref 40 (Crumpled) -> Sim 50
    # Ref 33 (Ball) -> Sim 100
    ref_map = {
        1: 44,
        25: 42,
        50: 40,
        75: 37,
        100: 33
    }
    
    for f in range(1, 101):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        
        if f in ref_map:
            ref_idx = ref_map[f]
            filename = f"ref_frame_{ref_idx:04d}.png"
            bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered Sim {f} -> Ref {ref_idx}")

if __name__ == "__main__":
    run()
