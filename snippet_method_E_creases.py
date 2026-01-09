
# Method E: XPBD with Crease Map from Frame 45

import bpy
import bmesh
import os
import sys
import math
import random
from mathutils import Vector, Matrix

# PARAMS
SEED = 8282795
NX = 32
NY = 42
HINGE_STIFFNESS = 0.8  # Stiff hinges for creases
FLAT_STIFFNESS = 0.4   # Weaker for non-creases
CREASE_ANGLE_DEG = 140.0
CREASE_MAP_PATH = "data/crease_map_from_45.png"

# ... (Standard geometry functions: _build_triangulated_grid, _mesh_topology, _dihedral_angle, _rotate_around_axis included for standalone) ...

def _build_triangulated_grid(name, aspect, nx, ny, size=2.0):
    width = size * aspect
    height = size
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=nx, y_subdivisions=ny, size=max(width, height), enter_editmode=False, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.scale[0] = width / 2.0
    obj.scale[1] = height / 2.0
    bpy.ops.object.transform_apply(rotation=False, scale=True)
    obj.rotation_euler = (1.5708, 0, 0)
    bpy.ops.object.transform_apply(rotation=True, scale=False)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # UV Map (Project from View - Bounds)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.001)
    # Or simplified: The grid has default UVs? yes "UVMap"
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj

def _mesh_topology(mesh):
    rest = [v.co.copy() for v in mesh.vertices]
    # Simple edge constraints
    edge_map = {}
    for e in mesh.edges:
        a, b = e.vertices[0], e.vertices[1]
        k = (a, b) if a < b else (b, a)
        # Store rest length
        if k not in edge_map:
            l0 = (rest[k[1]] - rest[k[0]]).length
            edge_map[k] = l0
    edges = [(a, b, l0) for (a, b), l0 in edge_map.items()]
    
    # Hinge constraints
    adjacent = {}
    for poly in mesh.polygons:
        if poly.loop_total != 3: continue
        tri = poly.vertices
        for (i, j, opp) in ((tri[0], tri[1], tri[2]), (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1])):
            k = (i, j) if i < j else (j, i)
            adjacent.setdefault(k, []).append(opp)
            
    hinges = []
    for (i, j), opps in adjacent.items():
        if len(opps) == 2:
            k, l = opps[0], opps[1]
            if k != l and k not in (i, j) and l not in (i, j):
                hinges.append((i, j, k, l))
    return rest, edges, hinges

def _dihedral_angle(xi, xj, xk, xl):
    e = xj - xi
    if e.length < 1e-12: return 0.0
    e = e.normalized()
    n1 = (xj - xi).cross(xk - xi)
    n2 = (xi - xj).cross(xl - xj)
    if n1.length < 1e-12 or n2.length < 1e-12: return 0.0
    n1.normalize(); n2.normalize()
    sinv = e.dot(n1.cross(n2))
    cosv = max(-1.0, min(1.0, n1.dot(n2)))
    return math.atan2(sinv, cosv)

def _rotate_around_axis(p, axis_point, axis_dir, angle):
    if abs(angle) < 1e-12: return p
    rot = Matrix.Rotation(angle, 4, axis_dir)
    return axis_point + (rot @ (p - axis_point))

class PaperXPBD:
    def __init__(self, rest, edges, hinges, uv_layer, crease_map_path, seed):
        rnd = random.Random(seed)
        self.rest = [p.copy() for p in rest]
        self.x = [p.copy() for p in rest]
        self.v = [Vector((0,0,0)) for _ in rest]
        self.inv_mass = [1.0 for _ in rest]
        
        # Load Crease Map using Blender API (to avoid cv2 dependency in Blender)
        self.crease_map_pixels = None
        self.map_width = 0
        self.map_height = 0
        
        if os.path.exists(crease_map_path):
            try:
                img = bpy.data.images.load(crease_map_path)
                self.map_width = img.size[0]
                self.map_height = img.size[1]
                # Pixels are RGBA float flat list [r,g,b,a, r,g,b,a, ...]
                # or [r,g,b,a] * width * height
                self.crease_map_pixels = list(img.pixels) # Cache it
                print(f"Loaded crease map via Blender: {self.map_width}x{self.map_height}")
            except Exception as e:
                print(f"Failed to load crease map: {e}")
        
        self.edges = edges
        self.hinges = []
        
        # Parse Hinges and Check UVs
        vertex_uvs = uv_layer 
        
        count_creases = 0
        for (i,j,k,l) in hinges:
            uv_i = vertex_uvs[i]
            uv_j = vertex_uvs[j]
            uv_mid = (uv_i + uv_j) * 0.5
            
            is_crease = False
            if self.crease_map_pixels is not None:
                h, w = self.map_height, self.map_width
                # UV (0,0) is Bottom-Left in Blender.
                px = int(uv_mid.x * (w - 1))
                py = int(uv_mid.y * (h - 1)) 
                
                if 0 <= px < w and 0 <= py < h:
                    idx = (py * w + px) * 4 # RGBA
                    # Check Red channel (intensity)
                    val = self.crease_map_pixels[idx] 
                    if val > 0.5: # Float 0..1
                        is_crease = True
            
            if is_crease:
                # Permanent Crease
                sign = 1.0 if rnd.random() > 0.5 else -1.0
                target_angle = sign * math.radians(CREASE_ANGLE_DEG)
                stiffness = HINGE_STIFFNESS
                count_creases += 1
            else:
                target_angle = 0.0 
                stiffness = FLAT_STIFFNESS
                
            self.hinges.append({
                'indices': (i,j,k,l),
                'angle': target_angle,
                'stiffness': stiffness
            })
            
        print(f"Found {count_creases} creases out of {len(hinges)} hinges.")
        
        # Anchors (Simplified from C)
        self.anchors = [0, 1] # Pin top corners? Or rely on physics.
        # Actually better to use the Method C unfolding logic (attractors).
        xs = [p.x for p in self.rest]
        self.attract_strength = 0.0

    def step(self, dt):
        x_prev = [p.copy() for p in self.x]
        
        # Integrate (Gravity + Damping)
        for i in range(len(self.x)):
            f = Vector((0,0, -0.2)) # Gravity
            self.v[i] = (1.0 - 0.1) * (self.v[i] + dt*f) # Damping 0.1
            self.x[i] += dt * self.v[i]
            
        # Project Constraints
        for _ in range(10): # Iters
            # Edges
            for (a, b, l0) in self.edges:
                diff = self.x[b] - self.x[a]
                if diff.length < 1e-12: continue
                corr = (diff.length - l0) * 1.0 * diff.normalized() # Stiffness 1.0
                self.x[a] += 0.5 * corr
                self.x[b] -= 0.5 * corr
                
            # Hinges
            for h in self.hinges:
                i, j, k, l = h['indices']
                target = h['angle']
                stiff = h['stiffness']
                
                # Standard Hinge Logic
                e = self.x[j] - self.x[i]
                if e.length < 1e-12: continue
                axis = e.normalized()
                current = _dihedral_angle(self.x[i], self.x[j], self.x[k], self.x[l])
                diff = current - target
                corr = -stiff * diff
                
                self.x[k] = _rotate_around_axis(self.x[k], self.x[i], axis, 0.5*corr)
                self.x[l] = _rotate_around_axis(self.x[l], self.x[i], axis, -0.5*corr)
                
        # Update Velocity
        for i in range(len(self.x)):
            self.v[i] = (self.x[i] - x_prev[i]) / dt

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    
    # 1. Floor
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0,0,-2))
    
    # 2. Paper Mesh
    obj = _build_triangulated_grid("Paper", 32.0/42.0, NX, NY)
    mesh = obj.data
    
    # Extract UVs for Solver
    # Need to bake UVs to vertex list.
    # We iterate vertices, find a loop using it, get UV.
    # Since it's a grid (generated from primitive), the UVs are standard 0..1.
    # Vertex i (row r, col c) -> u=c/nx, v=r/ny.
    vertex_uvs = [Vector((0,0)) for _ in mesh.vertices]
    # Reconstruct from grid logic:
    # vertices are ordered row by row usually.
    # Actually, simpler to read from split loops if needed, but for grid, manual calc is robust.
    # Let's read from Blender UV layer.
    uv_layer = mesh.uv_layers.active.data
    # Map vert index to uv
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop = mesh.loops[loop_idx]
            uv = uv_layer[loop_idx].uv
            vertex_uvs[loop.vertex_index] = uv
            
    rest, edges, hinges = _mesh_topology(mesh)
    
    sim = PaperXPBD(rest, edges, hinges, vertex_uvs, CREASE_MAP_PATH, SEED)
    
    # Run Sim
    scene.frame_start = 1
    scene.frame_end = 42
    
    # Pre-roll steps?
    # Just standard steps.
    dt = 1.0/24.0/5.0
    
    for f in range(1, 43):
        scene.frame_set(f)
        for _ in range(5):
            sim.step(dt)
        
        # Update Mesh
        for i, v in enumerate(mesh.vertices):
            v.co = sim.x[i]
        mesh.update()
        
    bpy.ops.object.shade_smooth()
    
    # Visuals
    mat = bpy.data.materials.new(name="PaperMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    
    # Add Image Texture node with Frame 45? Or Frame 40?
    # No, just white paper for structure check.
    
    obj.data.materials.append(mat)
    
    # Light & Camera
    bpy.ops.object.light_add(type='AREA', location=(0,0,5))
    bpy.ops.object.camera_add(location=(0, -2.5, 2.5))
    cam = bpy.context.object
    bpy.ops.object.constraint_add(type='TRACK_TO')
    const = cam.constraints["Track To"]
    const.target = obj
    const.track_axis = 'TRACK_NEGATIVE_Z'
    const.up_axis = 'UP_Y'
    scene.camera = cam
    
    out_path = os.path.abspath("data/benchmarks/method_E_creases.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print("Rendered Method E")

if __name__ == "__main__":
    main()
