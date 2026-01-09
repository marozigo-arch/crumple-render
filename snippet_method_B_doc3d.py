
import bpy
import os
import sys

def main():
    # 1. Setup Scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    
    # 2. Import Doc3D Mesh
    obj_path = os.path.abspath("data/external_repos/doc3D-renderer/obj/1_1.obj")
    if not os.path.exists(obj_path):
        print(f"Error: {obj_path} not found")
        return
        
    bpy.ops.import_scene.obj(filepath=obj_path)
    paper = bpy.context.selected_objects[0]
    paper.name = "Paper_Doc3D"
    
    # 3. Scale and Position (Normalize)
    # Doc3D meshes seem to be in [-1, 1] range roughly.
    # We want it visible.
    
    # 4. Camera
    bpy.ops.object.camera_add(location=(0, -2.5, 2.5))
    cam = bpy.context.object
    # Point at paper
    bpy.ops.object.constraint_add(type='TRACK_TO')
    cam.constraints["Track To"].target = paper
    cam.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints["Track To"].up_axis = 'UP_Y'
    scene.camera = cam
    
    # 5. Light
    bpy.ops.object.light_add(type='POINT', location=(2, -2, 3))
    light = bpy.context.object
    light.data.energy = 1000.0
    
    # 6. Render
    out_path = os.path.abspath("data/benchmarks/method_B_doc3d.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered to {out_path}")

if __name__ == "__main__":
    main()
