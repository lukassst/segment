#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert MEDIS contour export files to GIfTI mesh format.

Handles varying point counts per contour via cubic spline resampling.
Produces separate files for Lumen (inner) and VesselWall (outer).
Creates water-tight meshes with end caps for CFD compatibility.

Usage:
    python medis_to_gii.py <medis_txt> <nifti_cta> <output_dir>
    
Output naming: {patient_id}_{vessel}_{inner|outer}.gii
"""

import numpy as np
from scipy.interpolate import CubicSpline
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray, GiftiMetaData
import os
import sys
import argparse


def parse_medis_contours(filepath: str) -> tuple:
    """
    Parse MEDIS-format contour file with varying point counts.
    
    Returns:
        contours: {'Lumen': [np.array, ...], 'VesselWall': [np.array, ...]}
        metadata: {'vessel_name': 'lad', 'patient_id': '...', ...}
    """
    contours = {'Lumen': [], 'VesselWall': []}
    metadata = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print(f"  Debug: Read {len(lines)} lines from file")
    
    i = 0
    contour_count = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse metadata headers
        if line.startswith('# vessel_name'):
            metadata['vessel_name'] = line.split(':')[1].strip()
            i += 1
        elif line.startswith('# patient_id'):
            metadata['patient_id'] = line.split(':')[1].strip()
            i += 1
        elif line.startswith('# study_description'):
            metadata['study_description'] = line.split(':')[1].strip()
            i += 1
        elif line.startswith('# Contour index:'):
            contour_count += 1
            if contour_count % 10 == 0:
                print(f"  Debug: Parsing contour {contour_count} at line {i}")
            
            # Start of a new contour block
            group = None
            num_points = 0
            
            # Read contour metadata until we find Number of points
            while i < len(lines) and not lines[i].strip().startswith('# Number of points:'):
                current_line = lines[i].strip()
                if current_line.startswith('# group:'):
                    group = current_line.split(':')[1].strip()
                
                # Safety break if we hit another contour index unexpectedly
                if current_line.startswith('# Contour index:') and group is not None:
                     break
                
                i += 1
            
            # Parse number of points
            if i < len(lines) and lines[i].strip().startswith('# Number of points:'):
                try:
                    num_points = int(lines[i].split(':')[1].strip())
                except (ValueError, IndexError):
                    print(f"  Warning: Could not parse number of points at line {i}")
                    num_points = 0
                
                i += 1
                
                # Read the actual points
                points = []
                for _ in range(num_points):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 3:
                            try:
                                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            except ValueError:
                                pass
                        i += 1
                
                # Store the contour if valid
                if len(points) > 2 and group in contours:
                    contours[group].append(np.array(points))
        else:
            i += 1
    
    print(f"  Debug: Finished parsing. Found {len(contours['Lumen'])} Lumen, {len(contours['VesselWall'])} VesselWall")
    return contours, metadata


def resample_polygon(polygon: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resample a 3D polygon (closed loop) to exactly n_points.
    Uses Cubic Spline interpolation for smooth results suitable for CFD.
    """
    if len(polygon) < 3:
        return polygon
        
    # Close the loop for correct perimeter calculation
    closed = np.vstack((polygon, polygon[0]))
    
    # Calculate cumulative distance along the path
    diffs = np.diff(closed, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    perimeter = cum_dists[-1]
    
    if perimeter < 1e-10:
        return polygon
    
    # Use CubicSpline with periodic boundary condition
    cs_x = CubicSpline(cum_dists, closed[:, 0], bc_type='periodic')
    cs_y = CubicSpline(cum_dists, closed[:, 1], bc_type='periodic')
    cs_z = CubicSpline(cum_dists, closed[:, 2], bc_type='periodic')
    
    # Generate new evenly spaced distances
    new_dists = np.linspace(0, perimeter, n_points + 1)[:-1]
    
    # Evaluate splines
    return np.column_stack((cs_x(new_dists), cs_y(new_dists), cs_z(new_dists)))


def align_contours(contours: list) -> list:
    """
    Rotational alignment of contours to prevent twisting.
    Aligns each ring's start point to be closest to the previous ring's start point.
    """
    if not contours:
        return []
        
    aligned = [contours[0]]
    for i in range(len(contours) - 1):
        prev = aligned[i]
        curr = contours[i + 1]
        
        # Find index in 'curr' that is closest to prev[0]
        distances = np.linalg.norm(curr - prev[0], axis=1)
        start_idx = np.argmin(distances)
        
        # Shift the ring so it starts at the closest point
        aligned.append(np.roll(curr, -start_idx, axis=0))
    return aligned


def create_tube_mesh(contours: list, capped: bool = True) -> tuple:
    """
    Create a triangulated tube mesh from contour rings.
    Optionally caps ends for water-tight solid.
    
    Args:
        contours: List of resampled, aligned contour arrays (same point count)
        capped: Whether to cap the ends
    
    Returns:
        vertices: Nx3 float32 array
        faces: Mx3 int32 array (triangle indices)
    """
    if len(contours) < 2:
        return None, None
    
    n_points = len(contours[0])
    n_rings = len(contours)
    
    # Stack all vertices
    all_vertices = []
    for ring in contours:
        all_vertices.extend(ring)
    
    faces = []
    
    # Create tube faces (quads split into triangles)
    for r in range(n_rings - 1):
        for p in range(n_points):
            p_next = (p + 1) % n_points
            
            # Indices in the flat vertex list
            v0 = r * n_points + p
            v1 = r * n_points + p_next
            v2 = (r + 1) * n_points + p
            v3 = (r + 1) * n_points + p_next
            
            # Two triangles per quad (CCW winding)
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    
    if capped:
        # Start cap (first ring)
        c0 = contours[0]
        c0_center = np.mean(c0, axis=0)
        c0_center_idx = len(all_vertices)
        all_vertices.append(c0_center)
        
        # Fan triangles pointing inward (reversed winding)
        for p in range(n_points):
            p_next = (p + 1) % n_points
            faces.append([c0_center_idx, p_next, p])
        
        # End cap (last ring)
        c_last = contours[-1]
        c_last_center = np.mean(c_last, axis=0)
        c_last_center_idx = len(all_vertices)
        all_vertices.append(c_last_center)
        
        # Last ring starts at index (n_rings - 1) * n_points
        last_ring_start = (n_rings - 1) * n_points
        
        # Fan triangles pointing outward
        for p in range(n_points):
            p_next = (p + 1) % n_points
            v_curr = last_ring_start + p
            v_next = last_ring_start + p_next
            faces.append([c_last_center_idx, v_curr, v_next])
    
    vertices = np.array(all_vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    
    return vertices, faces


def contours_to_gifti(
    contours: list,
    metadata: dict,
    group_name: str,
    target_points: int = 64,
    capped: bool = True
) -> GiftiImage:
    """
    Convert contour rings to GIfTI mesh.
    
    Args:
        contours: List of contour arrays (each may have different point counts)
        metadata: Vessel metadata
        group_name: 'Lumen' or 'VesselWall'
        target_points: Target points per ring after resampling
        capped: Whether to cap ends for water-tight mesh
    
    Returns:
        GiftiImage object
    """
    if not contours or len(contours) < 2:
        return None
    
    # Resample all contours to uniform point count
    resampled = [resample_polygon(c, target_points) for c in contours]
    
    # Align contours to prevent twisting
    aligned = align_contours(resampled)
    
    # Create mesh
    vertices, faces = create_tube_mesh(aligned, capped=capped)
    
    if vertices is None or faces is None:
        return None
    
    # Create GIfTI data arrays
    vertex_array = GiftiDataArray(
        data=vertices,
        intent='NIFTI_INTENT_POINTSET',
        datatype='NIFTI_TYPE_FLOAT32'
    )
    
    face_array = GiftiDataArray(
        data=faces,
        intent='NIFTI_INTENT_TRIANGLE',
        datatype='NIFTI_TYPE_INT32'
    )
    
    # Embed metadata
    gii_metadata = GiftiMetaData.from_dict({
        'VesselName': metadata.get('vessel_name', 'Unknown'),
        'PatientID': metadata.get('patient_id', 'Unknown'),
        'Group': group_name,
        'NumRings': str(len(aligned)),
        'PointsPerRing': str(target_points),
        'Capped': str(capped),
        'ConvertedBy': 'medis_to_gii.py'
    })
    
    # Create GIfTI image
    gii = GiftiImage(darrays=[vertex_array, face_array], meta=gii_metadata)
    
    return gii


def convert_medis_to_gii(
    medis_txt_path: str,
    nifti_cta_path: str,
    output_dir: str,
    target_points: int = 64,
    capped: bool = True
) -> list:
    """
    Convert MEDIS contour file to GIfTI mesh files.
    
    Args:
        medis_txt_path: Path to MEDIS contour .txt file
        nifti_cta_path: Path to CTA NIfTI file (for metadata reference)
        output_dir: Directory for output files
        target_points: Points per ring after resampling
        capped: Whether to cap ends for water-tight meshes
    
    Returns:
        List of created file paths
    """
    print(f"Parsing: {os.path.basename(medis_txt_path)}")
    
    # Parse MEDIS file
    contours, metadata = parse_medis_contours(medis_txt_path)
    
    # Extract naming components
    patient_id = metadata.get('patient_id', 'Unknown')
    vessel_name = metadata.get('vessel_name', 'Unknown').upper()
    
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    
    # Process Lumen (inner)
    if contours['Lumen'] and len(contours['Lumen']) >= 2:
        print(f"  Found {len(contours['Lumen'])} Lumen contours")
        point_counts = [len(c) for c in contours['Lumen']]
        print(f"  Point counts range: {min(point_counts)} - {max(point_counts)}")
        
        gii = contours_to_gifti(
            contours['Lumen'], metadata, 'Lumen', target_points, capped
        )
        
        if gii:
            output_path = os.path.join(output_dir, f"{patient_id}_{vessel_name}_inner.gii")
            nib.save(gii, output_path)
            output_files.append(output_path)
            
            n_verts = len(gii.darrays[0].data)
            n_faces = len(gii.darrays[1].data)
            print(f"  ✓ Saved: {os.path.basename(output_path)} ({n_verts} vertices, {n_faces} faces)")
    
    # Process VesselWall (outer)
    if contours['VesselWall'] and len(contours['VesselWall']) >= 2:
        print(f"  Found {len(contours['VesselWall'])} VesselWall contours")
        point_counts = [len(c) for c in contours['VesselWall']]
        print(f"  Point counts range: {min(point_counts)} - {max(point_counts)}")
        
        gii = contours_to_gifti(
            contours['VesselWall'], metadata, 'VesselWall', target_points, capped
        )
        
        if gii:
            output_path = os.path.join(output_dir, f"{patient_id}_{vessel_name}_outer.gii")
            nib.save(gii, output_path)
            output_files.append(output_path)
            
            n_verts = len(gii.darrays[0].data)
            n_faces = len(gii.darrays[1].data)
            print(f"  ✓ Saved: {os.path.basename(output_path)} ({n_verts} vertices, {n_faces} faces)")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert MEDIS contour export to GIfTI mesh'
    )
    parser.add_argument('medis_txt', help='Path to MEDIS contour .txt file')
    parser.add_argument('nifti_cta', help='Path to CTA NIfTI file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--points', type=int, default=64, 
                        help='Target points per ring (default: 64)')
    parser.add_argument('--no-caps', action='store_true',
                        help='Do not cap mesh ends (hollow tube)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.medis_txt):
        print(f"Error: MEDIS file not found: {args.medis_txt}")
        sys.exit(1)
    
    output_files = convert_medis_to_gii(
        args.medis_txt,
        args.nifti_cta,
        args.output_dir,
        args.points,
        capped=not args.no_caps
    )
    
    print(f"\nCreated {len(output_files)} GIfTI file(s)")


if __name__ == "__main__":
    main()
