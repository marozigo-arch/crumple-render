import bpy
import os
import sys

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    scene.render.resolution_x = 864
    scene.render.resolution_y = 1104
    
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1)

def create_cloth_poster(image_path, pressure_val=0.0, shrink_val=0.0, 
                        mass_val=0.3, bending_val=15.0, tension_val=80.0, turbulence_val=20.0):
    # Load image
    img = None
    try:
        img = bpy.data.images.load(image_path)
    except:
        pass
    aspect = 1.0
    if img:
        aspect = img.size[0] / img.size[1]

    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0), rotation=(1.5708, 0, 0))
    plane = bpy.context.object
    plane.name = "ClothPoster"
    plane.scale[0] = aspect
    
    # Subdivide for cloth
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=40)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Material
    mat = bpy.data.materials.new(name="PosterMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Roughness'].default_value = 1.0 
    bsdf.inputs['Specular'].default_value = 0.0
    
    if img:
        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = img
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    plane.data.materials.append(mat)
    
    # Add Cloth Modifier
    mod = plane.modifiers.new(name="Cloth", type='CLOTH')
    settings = mod.settings
    settings.quality = 5
    settings.mass = mass_val
    settings.tension_stiffness = tension_val
    settings.compression_stiffness = tension_val
    settings.shear_stiffness = tension_val
    settings.bending_stiffness = bending_val
    
    settings.use_pressure = False 
    
    # Enable Self Collision (In 2.8x it is often under collision_settings)
    if hasattr(settings, 'collision_settings'):
        settings.collision_settings.use_self_collision = True
        settings.collision_settings.distance = 0.015
    else:
        # Fallback or direct attribute if valid in some versions
        try:
            settings.use_self_collision = True
            settings.self_collision_distance = 0.015
        except:
            pass
    
    bpy.ops.object.effector_add(type='TURBULENCE', location=(0, -1, 0))
    turb = bpy.context.object
    turb.field.strength = turbulence_val
    turb.field.size = 0.5
    turb.field.noise = 1
    
    # Animate Turbulence strength to stop?
    
    return plane

def setup_camera():
    bpy.ops.object.camera_add(location=(0, -3.5, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.object
    const = cam.constraints.new(type='DAMPED_TRACK')
    const.target = target
    const.track_axis = 'TRACK_NEGATIVE_Z'
    
    # Light
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 5))
    sun = bpy.context.object
    sun.data.energy = 5.0

def main():
    setup_scene()
    image_path = "data/extracted_texture_from_45.png"
    
    pressure = 0.0
    shrink = 0.0
    
    # Defaults
    mass = 0.3
    bending = 15.0
    tension = 80.0
    turbulence = 20.0
    
    # Parse args
    args = sys.argv
    if "--pressure" in args:
        pressure = float(args[args.index("--pressure")+1])
    if "--mass" in args:
        mass = float(args[args.index("--mass")+1])
    if "--bending" in args:
        bending = float(args[args.index("--bending")+1])
    if "--tension" in args:
        tension = float(args[args.index("--tension")+1])
    if "--turbulence" in args:
        turbulence = float(args[args.index("--turbulence")+1])
        
    print(f"Simulating with: Mass={mass}, Bend={bending}, Tension={tension}, Turb={turbulence}")
    
    plane = create_cloth_poster(image_path, pressure, shrink, mass, bending, tension, turbulence)
    setup_camera()
    
    # BAKE SIMULATION
    # We want Frame 40 relative to our video.
    # Our simulation: Frame 1 (Flat) -> Frame 20 (Crumpled).
    # Video: Frame 33 (Crumpled) -> Frame 45 (Flat).
    # So we need the state at ~Frame 20 of Sim to correspond to Frame 33 of Video.
    # And state at ~Frame 1 of Sim to correspond to Frame 45.
    
    # Wait, Physics runs forward.
    # 0 -> 20: Crumpling.
    # User calls for "Frame 40" (which is mostly flat).
    # Mapping:
    # Video Frame 33 (Start of action) = Sim Frame 20 (Crumpled)
    # Video Frame 45 (End of action)   = Sim Frame 1 (Flat)
    
    # Target Frame 40 is Closer to 45 (Flat) than 33 (Crumpled). 
    # (45 - 33) = 12 frames range.
    # (45 - 40) = 5 frames from Flat.
    # (40 - 33) = 7 frames from Crumpled.
    # So Frame 40 is roughly 40% into the crumpling process (if we play backwards).
    # Sim Frame equivalent = 20 * (7/12) ~= Frame 11.
    
    # Let's bake 20 frames.
    bpy.context.scene.frame_end = 20
    
    # Point cache bake (headless needs special handling usually, but ops.ptcache.bake with 'ALL' might work)
    # In headless, simple timeline playback is often safer to trigger sim step-by-step
    
    scene = bpy.context.scene
    for f in range(1, 21):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        
    # Render Specific "Equivalent" Frame
    # Frame 40 (Video) ~ Frame 8-10 (Sim)
    scene.frame_set(8) 
    
    bpy.context.scene.render.filepath = "//data/render_cloth_test/frame_0040"
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()
