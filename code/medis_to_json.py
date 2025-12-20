#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert MEDIS contour export files to Niivue Connectome JSON point cloud format.

Handles varying point counts per contour via cubic spline resampling.
Produces separate files for Lumen (inner) and VesselWall (outer).

Usage:
    python medis_to_json.py <medis_txt> <nifti_cta> <output_dir>
    
Output naming: {patient_id}_{vessel}_{inner|outer}.json
"""

import numpy as np
from scipy.interpolate import CubicSpline
import json
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
            # We must ensure we don't go past the end of the file OR the start of a new contour (safety check)
            while i < len(lines) and not lines[i].strip().startswith('# Number of points:'):
                current_line = lines[i].strip()
                if current_line.startswith('# group:'):
                    group = current_line.split(':')[1].strip()
                
                # Safety break if we hit another contour index unexpectedly
                if current_line.startswith('# Contour index:') and group is not None:
                     # This shouldn't happen if logic is correct, but good for safety
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


def contours_to_niivue_json(
    contours: list,
    metadata: dict,
    group_name: str,
    target_points: int = 64
) -> dict:
    """
    Convert contour rings to Niivue Connectome JSON format.
    
    Args:
        contours: List of contour arrays (each may have different point counts)
        metadata: Vessel metadata
        group_name: 'Lumen' or 'VesselWall'
        target_points: Target points per ring after resampling
    
    Returns:
        Niivue connectome JSON structure
    """
    if not contours:
        return None
    
    # Resample all contours to uniform point count
    resampled = [resample_polygon(c, target_points) for c in contours]
    
    # Align contours to prevent twisting
    aligned = align_contours(resampled)
    
    # Flatten all points
    all_points = np.vstack(aligned)
    
    # Build nodes
    nodes = {
        "x": all_points[:, 0].tolist(),
        "y": all_points[:, 1].tolist(),
        "z": all_points[:, 2].tolist(),
        "colorValue": [1.0] * len(all_points),
        "sizeValue": [1.0] * len(all_points)
    }
    
    # Create edges: connect points within each ring and between rings
    edges = {"first": [], "second": []}
    num_rings = len(aligned)
    
    for r in range(num_rings):
        offset = r * target_points
        
        # Intra-ring edges (close the ring)
        for i in range(target_points):
            p1 = offset + i
            p2 = offset + ((i + 1) % target_points)
            edges["first"].append(p1)
            edges["second"].append(p2)
        
        # Inter-ring edges (connect to next ring, every 4th point)
        if r < num_rings - 1:
            for i in range(0, target_points, 4):
                p1 = offset + i
                p2 = offset + target_points + i
                edges["first"].append(p1)
                edges["second"].append(p2)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "vessel_type": group_name,
            "vessel_name": metadata.get('vessel_name', 'Unknown'),
            "patient_id": metadata.get('patient_id', 'Unknown'),
            "num_rings": num_rings,
            "points_per_ring": target_points,
            "source": "MEDIS Export",
            "converter": "medis_to_json.py"
        }
    }


def convert_medis_to_json(
    medis_txt_path: str,
    nifti_cta_path: str,
    output_dir: str,
    target_points: int = 64
) -> list:
    """
    Convert MEDIS contour file to Niivue JSON point cloud files.
    
    Args:
        medis_txt_path: Path to MEDIS contour .txt file
        nifti_cta_path: Path to CTA NIfTI file (for metadata reference)
        output_dir: Directory for output files
        target_points: Points per ring after resampling
    
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
    if contours['Lumen']:
        print(f"  Found {len(contours['Lumen'])} Lumen contours")
        point_counts = [len(c) for c in contours['Lumen']]
        print(f"  Point counts range: {min(point_counts)} - {max(point_counts)}")
        
        json_data = contours_to_niivue_json(
            contours['Lumen'], metadata, 'Lumen', target_points
        )
        
        if json_data:
            output_path = os.path.join(output_dir, f"{patient_id}_{vessel_name}_inner.json")
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            output_files.append(output_path)
            print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    # Process VesselWall (outer)
    if contours['VesselWall']:
        print(f"  Found {len(contours['VesselWall'])} VesselWall contours")
        point_counts = [len(c) for c in contours['VesselWall']]
        print(f"  Point counts range: {min(point_counts)} - {max(point_counts)}")
        
        json_data = contours_to_niivue_json(
            contours['VesselWall'], metadata, 'VesselWall', target_points
        )
        
        if json_data:
            output_path = os.path.join(output_dir, f"{patient_id}_{vessel_name}_outer.json")
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            output_files.append(output_path)
            print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert MEDIS contour export to Niivue JSON point cloud'
    )
    parser.add_argument('medis_txt', help='Path to MEDIS contour .txt file')
    parser.add_argument('nifti_cta', help='Path to CTA NIfTI file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--points', type=int, default=64, 
                        help='Target points per ring (default: 64)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.medis_txt):
        print(f"Error: MEDIS file not found: {args.medis_txt}")
        sys.exit(1)
    
    output_files = convert_medis_to_json(
        args.medis_txt,
        args.nifti_cta,
        args.output_dir,
        args.points
    )
    
    print(f"\nCreated {len(output_files)} JSON file(s)")


if __name__ == "__main__":
    main()
