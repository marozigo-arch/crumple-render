
# This is a focused version of simulate_paper_xpbd.py for benchmarking Frame 42.

import bpy
import bmesh
import os
import sys
import math
import random
from mathutils import Vector, Matrix

# EXPERIMENT 13 PARAMETERS (Hardcoded for benchmark)
SEED = 8282795
NX = 32
NY = 42
N_CREASES = 300
CREASE_ANGLE_DEG = 140.0
STRETCH_STIFFNESS = 1.0
HINGE_STIFFNESS = 0.85
UNFOLD_MAX = 0.8
UNFOLD_POWER = 1.3
ATTRACT0 = 10.0
ATTRACT_TAU = 0.02
CREASE_TAU = 0.3
PRE_ROLL = 60
SUBSTEPS = 5
ITERS = 20
DAMPING = 0.2
GRAVITY = -0.15
START_FRAME = 33
TARGET_FRAME = 42

# --- CORE XPBD CLASSES ---
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
    return obj

def _mesh_topology(mesh):
    rest = [v.co.copy() for v in mesh.vertices]
    edge_map = {}
    for e in mesh.edges:
        a, b = e.vertices[0], e.vertices[1]
        k = (a, b) if a < b else (b, a)
        if k not in edge_map:
            l0 = (rest[k[1]] - rest[k[0]]).length
            edge_map[k] = l0
    edges = [(a, b, l0) for (a, b), l0 in edge_map.items()]
    
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
    def __init__(self, rest, edges, hinges, seed):
        rnd = random.Random(seed)
        self.rest = [p.copy() for p in rest]
        self.x = [p.copy() for p in rest]
        self.v = [Vector((0,0,0)) for _ in rest]
        self.inv_mass = [1.0 for _ in rest]
        for i in range(len(self.x)):
            self.x[i].y += rnd.uniform(-0.01, 0.01)
            self.x[i].z += rnd.uniform(-0.01, 0.01)
        self.edges = edges
        crease_hinges = hinges[:]
        rnd.shuffle(crease_hinges)
        self.creases = []
        for (i,j,k,l) in crease_hinges[:N_CREASES]:
            theta0 = rnd.uniform(-math.radians(CREASE_ANGLE_DEG), math.radians(CREASE_ANGLE_DEG))
            self.creases.append((i,j,k,l,theta0))
        
        # Unfold Anchors
        xs = [p.x for p in self.rest]
        zs = [p.z for p in self.rest]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        mid_x = 0.5*(min_x+max_x)
        mid_z = 0.5*(min_z+max_z)
        targets = [Vector((min_x,0,min_z)), Vector((min_x,0,max_z)), 
                   Vector((max_x,0,min_z)), Vector((max_x,0,max_z)),
                   Vector((mid_x,0,min_z)), Vector((mid_x,0,max_z)),
                   Vector((min_x,0,mid_z)), Vector((max_x,0,mid_z))]
        self.anchors = []
        for t in targets:
            best_i = 0
            best_d = 1e18
            for i, p in enumerate(self.rest):
                d = (p-t).length_squared
                if d < best_d: best_d, best_i = d, i
            self.anchors.append(best_i)
        self.anchors = sorted(list(set(self.anchors)))
        self.attract_strength = 0.0

    def step(self, dt, crease_scale, unfold_alpha):
        x_prev = [p.copy() for p in self.x]
        # Integrate
        for i in range(len(self.x)):
            f = Vector((0,0, GRAVITY))
            if self.attract_strength > 0:
                f += -self.attract_strength * self.x[i] # Pull to center
            self.v[i] = (1.0 - DAMPING) * (self.v[i] + dt*f)
            self.x[i] += dt * self.v[i]
            
        # Project
        for _ in range(ITERS):
            # Creases
            for (i, j, k, l, theta0) in self.creases:
                e = self.x[j] - self.x[i]
                if e.length < 1e-12: continue
                axis = e.normalized()
                theta = _dihedral_angle(self.x[i], self.x[j], self.x[k], self.x[l])
                diff = theta - (theta0 * crease_scale)
                corr = -HINGE_STIFFNESS * diff
                self.x[k] = _rotate_around_axis(self.x[k], self.x[i], axis, 0.5*corr)
                self.x[l] = _rotate_around_axis(self.x[l], self.x[i], axis, -0.5*corr)
            # Edges
            for (a, b, l0) in self.edges:
                diff = self.x[b] - self.x[a]
                if diff.length < 1e-12: continue
                corr = (diff.length - l0) * STRETCH_STIFFNESS * diff.normalized()
                self.x[a] += 0.5 * corr
                self.x[b] -= 0.5 * corr
            # Unfold
            if unfold_alpha > 0:
                for idx in self.anchors:
                    self.x[idx] = self.x[idx].lerp(self.rest[idx], unfold_alpha)
                    
        # Update V
        for i in range(len(self.x)):
            self.v[i] = (self.x[i] - x_prev[i]) / dt

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Camera
    bpy.ops.object.camera_add(location=(0, -2.5, 2.5))
    cam = bpy.context.object
    bpy.ops.object.constraint_add(type='TRACK_TO')
    const = cam.constraints["Track To"]
    const.track_axis = 'TRACK_NEGATIVE_Z'
    const.up_axis = 'UP_Y'
    scene.camera = cam
    bpy.ops.object.empty_add()
    const.target = bpy.context.object
    
    # Light
    bpy.ops.object.light_add(type='AREA', location=(0,0,5))
    
    # Mesh
    obj = _build_triangulated_grid("Paper", 32.0/42.0, NX, NY)
    mesh = obj.data
    rest, edges, hinges = _mesh_topology(mesh)
    
    sim = PaperXPBD(rest, edges, hinges, SEED)
    
    dt = 1.0/24.0/SUBSTEPS
    
    # Pre-roll
    for _ in range(PRE_ROLL):
        sim.step(dt, 1.0, 0.0)
        
    # Run to Frame 42 (TARGET)
    for f in range(START_FRAME, TARGET_FRAME + 1):
        u = (f - START_FRAME) / (45 - START_FRAME)
        unfold_alpha = UNFOLD_MAX * (u**UNFOLD_POWER)
        crease_scale = math.exp(-u/CREASE_TAU)
        sim.attract_strength = ATTRACT0 * math.exp(-u/ATTRACT_TAU)
        
        for _ in range(SUBSTEPS):
            sim.step(dt, crease_scale, unfold_alpha)
            
    # Update Mesh
    for i, v in enumerate(mesh.vertices):
        v.co = sim.x[i]
    mesh.update()
    bpy.ops.object.shade_smooth()
    
    # Render
    out_path = os.path.abspath("data/benchmarks/method_C_xpbd.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {out_path}")

if __name__ == "__main__":
    main()
