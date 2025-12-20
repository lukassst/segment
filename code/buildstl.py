#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate STL files from vessel point cloud data.
Creates tubular meshes for lumen and vessel wall.
"""

import numpy as np
from stl import mesh
from scipy.interpolate import CubicSpline
import os
import glob


def parse_contours(filepath):
    """Parse txt file and extract lumen and vessel wall contours."""
    lumen_contours = []
    vessel_contours = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('# Contour index:'):
            group = None
            num_points = 0
            
            while i < len(lines) and not lines[i].strip().startswith('# Number of points:'):
                if lines[i].strip().startswith('# group:'):
                    group = lines[i].split(':')[1].strip()
                i += 1
            
            if i < len(lines) and lines[i].strip().startswith('# Number of points:'):
                num_points = int(lines[i].split(':')[1].strip())
                i += 1
                
                points = []
                for _ in range(num_points):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) == 3:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        i += 1
                
                if len(points) > 0:
                    if group == 'Lumen':
                        lumen_contours.append(np.array(points))
                    elif group == 'VesselWall':
                        vessel_contours.append(np.array(points))
        else:
            i += 1
    
    return lumen_contours, vessel_contours


def create_tube_mesh(contours):
    """Create a tubular mesh from a list of contours (cross-sections)."""
    if len(contours) < 2:
        return None
    
    faces = []
    
    for i in range(len(contours) - 1):
        c1 = contours[i]
        c2 = contours[i + 1]
        
        n1 = len(c1)
        n2 = len(c2)
        
        for j in range(min(n1, n2)):
            j_next = (j + 1) % min(n1, n2)
            
            p1 = i * max(n1, n2) + j
            p2 = i * max(n1, n2) + j_next
            p3 = (i + 1) * max(n1, n2) + j
            p4 = (i + 1) * max(n1, n2) + j_next
            
            faces.append([p1, p3, p2])
            faces.append([p2, p3, p4])
    
    all_points = []
    for contour in contours:
        all_points.extend(contour)
    
    vertices = np.array(all_points)
    faces = np.array(faces)
    
    if len(faces) == 0:
        return None
    
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for idx, face in enumerate(faces):
        for j in range(3):
            if face[j] < len(vertices):
                stl_mesh.vectors[idx][j] = vertices[face[j]]
    
    return stl_mesh


def create_tube_mesh_simple(contours, capped=True):
    """
    Create a simple tubular mesh by connecting consecutive rings.
    Optionally caps the ends to create a solid volume.
    """
    if len(contours) < 2:
        return None
    
    faces = []
    all_vertices = []
    vertex_offset = 0
    
    for i in range(len(contours) - 1):
        ring1 = contours[i]
        ring2 = contours[i + 1]
        
        n1 = len(ring1)
        n2 = len(ring2)
        n = min(n1, n2)
        
        for j in range(n):
            j_next = (j + 1) % n
            
            v0 = vertex_offset + j
            v1 = vertex_offset + j_next
            v2 = vertex_offset + n1 + j
            v3 = vertex_offset + n1 + j_next
            
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
        
        if i == 0:
            all_vertices.extend(ring1)
        all_vertices.extend(ring2)
        vertex_offset += n1
    
    if capped:
        # Start Cap
        c0 = contours[0]
        n0 = len(c0)
        c0_center = np.mean(c0, axis=0)
        c0_center_idx = len(all_vertices)
        all_vertices.append(c0_center)
        
        # Connect center to first ring (indices 0 to n0-1)
        # Orientation: Center -> Next -> Current (Points "backwards")
        for j in range(n0):
            j_next = (j + 1) % n0
            faces.append([c0_center_idx, j_next, j])
            
        # End Cap
        c_last = contours[-1]
        n_last = len(c_last)
        c_last_center = np.mean(c_last, axis=0)
        c_last_center_idx = len(all_vertices)
        all_vertices.append(c_last_center)
        
        # Last ring start index is (total_vertices_before_caps) - n_last
        # But we just added one vertex (c0_center), so it is len(all_vertices) - 1 - n_last
        last_ring_start = len(all_vertices) - 1 - n_last
        
        # Connect center to last ring
        # Orientation: Center -> Current -> Next (Points "forwards")
        for j in range(n_last):
            j_next = (j + 1) % n_last
            v_curr = last_ring_start + j
            v_next = last_ring_start + j_next
            faces.append([c_last_center_idx, v_curr, v_next])

    vertices = np.array(all_vertices)
    faces = np.array(faces)
    
    if len(faces) == 0:
        return None
    
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for idx, face in enumerate(faces):
        for j in range(3):
            if face[j] < len(vertices):
                stl_mesh.vectors[idx][j] = vertices[face[j]]
    
    return stl_mesh


def resample_polygon(polygon, n_points):
    """
    Resample a 3D polygon (closed loop) to have exactly n_points.
    Uses Cubic Spline interpolation for smoother results suitable for CFD.
    """
    # Close the loop for correct perimeter calculation
    closed = np.vstack((polygon, polygon[0]))
    
    # Calculate cumulative distance along the path
    diffs = np.diff(closed, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    perimeter = cum_dists[-1]
    
    # Use CubicSpline with periodic boundary condition
    # We interpolate x, y, z as functions of distance
    cs_x = CubicSpline(cum_dists, closed[:, 0], bc_type='periodic')
    cs_y = CubicSpline(cum_dists, closed[:, 1], bc_type='periodic')
    cs_z = CubicSpline(cum_dists, closed[:, 2], bc_type='periodic')
    
    # Generate new evenly spaced distances
    new_dists = np.linspace(0, perimeter, n_points + 1)[:-1]
    
    # Evaluate splines
    new_x = cs_x(new_dists)
    new_y = cs_y(new_dists)
    new_z = cs_z(new_dists)
    
    return np.column_stack((new_x, new_y, new_z))


def align_contours(contours):
    """
    Rotational alignment of contours to prevent twisting.
    Aligns each ring's start point to be closest to the previous ring's start point.
    """
    if not contours:
        return []
        
    aligned = [contours[0]]
    for i in range(len(contours) - 1):
        prev = aligned[i]
        curr = contours[i+1]
        
        # Find index in 'curr' that is closest to prev[0]
        # We assume prev and curr have same number of points (after resampling)
        distances = np.linalg.norm(curr - prev[0], axis=1)
        start_idx = np.argmin(distances)
        
        # Shift the ring so it starts at the closest point
        aligned.append(np.roll(curr, -start_idx, axis=0))
    return aligned


def process_file(filepath, output_dir):
    """Process a single txt file and generate STL files."""
    print(f"Processing: {os.path.basename(filepath)}")
    
    lumen_contours, vessel_contours = parse_contours(filepath)
    
    print(f"  Found {len(lumen_contours)} lumen contours")
    print(f"  Found {len(vessel_contours)} vessel wall contours")
    
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    if len(lumen_contours) >= 2:
        # Resample all contours to the max number of points found to preserve detail
        max_points = max(len(c) for c in lumen_contours) if lumen_contours else 50
        # Ensure at least a reasonable minimum resolution
        target_points = max(max_points, 50)
        
        # 1. Resample (Cubic Spline)
        lumen_resampled = [resample_polygon(c, target_points) for c in lumen_contours]
        
        # 2. Align (Prevent Twisting)
        lumen_aligned = align_contours(lumen_resampled)
        
        # 3. Mesh & Cap
        lumen_mesh = create_tube_mesh_simple(lumen_aligned, capped=True)
        if lumen_mesh is not None:
            output_path = os.path.join(output_dir, f"{base_name}_lumen.stl")
            lumen_mesh.save(output_path)
            print(f"  Saved: {os.path.basename(output_path)}")
    
    if len(vessel_contours) >= 2:
        # Resample vessel contours
        max_points = max(len(c) for c in vessel_contours) if vessel_contours else 50
        target_points = max(max_points, 50)
        
        # 1. Resample (Cubic Spline)
        vessel_resampled = [resample_polygon(c, target_points) for c in vessel_contours]
        
        # 2. Align (Prevent Twisting)
        vessel_aligned = align_contours(vessel_resampled)
        
        # 3. Mesh & Cap
        vessel_mesh = create_tube_mesh_simple(vessel_aligned, capped=True)
        if vessel_mesh is not None:
            output_path = os.path.join(output_dir, f"{base_name}_vessel.stl")
            vessel_mesh.save(output_path)
            print(f"  Saved: {os.path.basename(output_path)}")


def main():
    data_dir = r"C:\Users\lukass\Desktop\personal\github\flow\data"
    
    txt_files = [
        "01-BER-0088_20181010_Session_Session 16_04_2021 14_46 _ecrf_d1.txt",
        "01-BER-0088_20181010_Session_Session 16_04_2021 14_46 _ecrf_lad.txt",
        "01-BER-0088_20181010_Session_Session 16_04_2021 14_46 _ecrf_lcx.txt",
        "01-BER-0088_20181010_Session_Session 16_04_2021 14_46 _ecrf_rca.txt"
    ]
    
    for txt_file in txt_files:
        filepath = os.path.join(data_dir, txt_file)
        if os.path.exists(filepath):
            process_file(filepath, data_dir)
        else:
            print(f"File not found: {txt_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
