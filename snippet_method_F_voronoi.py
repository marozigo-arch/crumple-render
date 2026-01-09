
# Method F: XPBD with Procedural Voronoi Creases (Large Facets)

import bpy
import bmesh
import os
import sys
import math
import random
from mathutils import Vector, Matrix

# PARAMS
SEED = 42 # Variable
NX = 32
NY = 42
VORONOI_SEEDS = 8 # Low number = Large tiles
HINGE_STIFFNESS = 0.8
FLAT_STIFFNESS = 0.4
CREASE_ANGLE_DEG = 140.0

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
    
    # UV Map
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj

def _mesh_topology(mesh):
    rest = [v.co.copy() for v in mesh.vertices]
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
    def __init__(self, rest, edges, hinges, uv_layer, seed):
        rnd = random.Random(seed)
        self.rest = [p.copy() for p in rest]
        self.x = [p.copy() for p in rest]
        self.v = [Vector((0,0,0)) for _ in rest]
        self.inv_mass = [1.0 for _ in rest]
        
        # Voronoi Logic
        # Generate Seeds in UV space (0..1)
        voronoi_points = []
        for _ in range(VORONOI_SEEDS):
            vp = Vector((rnd.random(), rnd.random(), 0))
            voronoi_points.append(vp)
            
        self.edges = edges
        self.hinges = []
        
        # Check Hinges against Voronoi
        vertex_uvs = uv_layer 
        count_creases = 0
        
        # Pre-calculate vertex regions
        vert_region = []
        for uv in vertex_uvs:
            best_dist = 100.0
            best_r = -1
            for r, vp in enumerate(voronoi_points):
                # 2D Dist
                d = ((uv.x - vp.x)**2 + (uv.y - vp.y)**2)**0.5
                if d < best_dist:
                    best_dist = d
                    best_r = r
            vert_region.append(best_r)
            
        for (i,j,k,l) in hinges:
            # If vertices i and j belong to DIFFERENT regions, this edge is a boundary.
            r_i = vert_region[i]
            r_j = vert_region[j]
            
            is_crease = (r_i != r_j)
            
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
            
        print(f"Generated Voronoi Pattern with {VORONOI_SEEDS} seeds.")
        print(f"Marked {count_creases} boundary limit hinges.")
        
        self.anchors = []

    def step(self, dt):
        x_prev = [p.copy() for p in self.x]
        for i in range(len(self.x)):
            f = Vector((0,0, -0.2)) # Gravity
            self.v[i] = (1.0 - 0.1) * (self.v[i] + dt*f) 
            self.x[i] += dt * self.v[i]
            
        for _ in range(10): 
            for (a, b, l0) in self.edges:
                diff = self.x[b] - self.x[a]
                if diff.length < 1e-12: continue
                corr = (diff.length - l0) * 1.0 * diff.normalized() 
                self.x[a] += 0.5 * corr
                self.x[b] -= 0.5 * corr
            for h in self.hinges:
                i, j, k, l = h['indices']
                target = h['angle']
                stiff = h['stiffness']
                e = self.x[j] - self.x[i]
                if e.length < 1e-12: continue
                axis = e.normalized()
                current = _dihedral_angle(self.x[i], self.x[j], self.x[k], self.x[l])
                diff = current - target
                corr = -stiff * diff
                self.x[k] = _rotate_around_axis(self.x[k], self.x[i], axis, 0.5*corr)
                self.x[l] = _rotate_around_axis(self.x[l], self.x[i], axis, -0.5*corr)
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
    floor = bpy.context.object
    bpy.ops.object.modifier_add(type='COLLISION')
    floor.collision.damping = 0.8
    floor.collision.cloth_friction = 0.8 # Fixed API
    
    # 2. Paper
    obj = _build_triangulated_grid("Paper", 32.0/42.0, NX, NY)
    mesh = obj.data
    
    # UVs
    vertex_uvs = [Vector((0,0)) for _ in mesh.vertices]
    uv_layer = mesh.uv_layers.active.data
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop = mesh.loops[loop_idx]
            uv = uv_layer[loop_idx].uv
            vertex_uvs[loop.vertex_index] = uv
            
    rest, edges, hinges = _mesh_topology(mesh)
    
    sim = PaperXPBD(rest, edges, hinges, vertex_uvs, SEED)
    
    # Run
    scene.frame_start = 1
    scene.frame_end = 42
    dt = 1.0/24.0/5.0
    for f in range(1, 43):
        scene.frame_set(f)
        for _ in range(5):
            sim.step(dt)
        for i, v in enumerate(mesh.vertices):
            v.co = sim.x[i]
        mesh.update()
        
    bpy.ops.object.shade_smooth()
    
    # Light/Cam
    bpy.ops.object.light_add(type='AREA', location=(0,0,5))
    bpy.ops.object.camera_add(location=(0, -2.5, 2.5))
    cam = bpy.context.object
    bpy.ops.object.constraint_add(type='TRACK_TO')
    const = cam.constraints["Track To"]
    const.target = obj
    const.track_axis = 'TRACK_NEGATIVE_Z'
    const.up_axis = 'UP_Y'
    scene.camera = cam
    
    out_path = os.path.abspath("data/benchmarks/method_F_voronoi.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()
