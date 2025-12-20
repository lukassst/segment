#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform mesh (GIfTI) and point cloud (JSON) coordinates to CPR space.

Takes existing mesh/point files in world coordinates and transforms them
to the straightened CPR coordinate system for overlay visualization.

Usage:
    python transform_to_cpr.py <frame_json> <input_gii_or_json> <output_file>
    
    # Or batch transform all files for a vessel:
    python transform_to_cpr.py --batch <frame_json> <output_dir>

Output:
    {original_name}_cpr.gii   - Mesh in CPR coordinates
    {original_name}_cpr.json  - Point cloud in CPR coordinates
"""

import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path


def load_frame(frame_path: str) -> dict:
    """Load CPR frame data from JSON file."""
    with open(frame_path, 'r') as f:
        data = json.load(f)
    
    return {
        'centerline': np.array(data['centerline']),
        'tangents': np.array(data['tangents']),
        'normals': np.array(data['normals']),
        'binormals': np.array(data['binormals']),
        'n_points': data['n_points']
    }


def world_to_cpr(world_points: np.ndarray, frame: dict) -> np.ndarray:
    """
    Transform world coordinates to CPR (s, u, v) coordinates.
    
    Args:
        world_points: Nx3 array of [x, y, z] world coordinates
        frame: Frame data from load_frame()
    
    Returns:
        cpr_points: Nx3 array of [u, v, s] CPR coordinates
                    u = offset in normal direction
                    v = offset in binormal direction  
                    s = index along centerline (as float for interpolation)
    """
    centerline = frame['centerline']
    normals = frame['normals']
    binormals = frame['binormals']
    
    n_points = len(world_points)
    cpr_points = np.zeros((n_points, 3), dtype=np.float32)
    
    for i, world_pt in enumerate(world_points):
        # Find nearest centerline point
        distances = np.linalg.norm(centerline - world_pt, axis=1)
        s_idx = np.argmin(distances)
        
        # Refine with projection onto line segment for sub-index precision
        if s_idx > 0 and s_idx < len(centerline) - 1:
            # Check both adjacent segments
            best_s = float(s_idx)
            best_dist = distances[s_idx]
            
            for seg_start in [s_idx - 1, s_idx]:
                if seg_start < 0 or seg_start >= len(centerline) - 1:
                    continue
                A = centerline[seg_start]
                B = centerline[seg_start + 1]
                AB = B - A
                AP = world_pt - A
                
                # Project onto segment
                t = np.clip(np.dot(AP, AB) / (np.dot(AB, AB) + 1e-10), 0, 1)
                proj = A + t * AB
                dist = np.linalg.norm(world_pt - proj)
                
                if dist < best_dist:
                    best_dist = dist
                    best_s = seg_start + t
            
            s_idx = int(round(best_s))
            s_idx = np.clip(s_idx, 0, len(centerline) - 1)
        
        # Project displacement onto local frame
        P = centerline[s_idx]
        D = world_pt - P
        u = np.dot(D, normals[s_idx])
        v = np.dot(D, binormals[s_idx])
        
        # Store as (u, v, s) - this maps to (x, y, z) in CPR volume
        cpr_points[i] = [u, v, float(s_idx)]
    
    return cpr_points


def transform_gifti_to_cpr(gii_path: str, frame: dict, output_path: str, 
                           cpr_spacing: float = 0.25, cross_section_size: int = 120):
    """
    Transform GIfTI mesh vertices to CPR coordinates and save.
    
    Args:
        gii_path: Path to input GIfTI file
        frame: CPR frame data
        output_path: Path for output GIfTI file
        cpr_spacing: CPR pixel spacing in mm
        cross_section_size: Size of CPR cross-section in pixels
    """
    import nibabel as nib
    from nibabel.gifti import GiftiImage, GiftiDataArray
    
    # Load original mesh
    gii = nib.load(gii_path)
    
    # Find vertex array
    vertices = None
    faces = None
    for darray in gii.darrays:
        if darray.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
            vertices = darray.data.copy()
        elif darray.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
            faces = darray.data
    
    if vertices is None:
        raise ValueError(f"No vertex data found in {gii_path}")
    
    print(f"  Transforming {len(vertices)} vertices...")
    
    # Transform to CPR coordinates
    cpr_coords = world_to_cpr(vertices, frame)
    
    # Convert to voxel coordinates (centered in cross-section)
    half_size = cross_section_size // 2
    cpr_voxels = np.zeros_like(cpr_coords)
    cpr_voxels[:, 0] = cpr_coords[:, 0] / cpr_spacing + half_size  # u -> x
    cpr_voxels[:, 1] = cpr_coords[:, 1] / cpr_spacing + half_size  # v -> y
    cpr_voxels[:, 2] = cpr_coords[:, 2]  # s -> z (already in slice units)
    
    # Create new GIfTI with transformed vertices
    new_gii = GiftiImage()
    
    # Add transformed vertices
    vertex_darray = GiftiDataArray(
        data=cpr_voxels.astype(np.float32),
        intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET'],
        datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32']
    )
    new_gii.add_gifti_data_array(vertex_darray)
    
    # Add faces if present
    if faces is not None:
        face_darray = GiftiDataArray(
            data=faces,
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_INT32']
        )
        new_gii.add_gifti_data_array(face_darray)
    
    # Copy metadata
    new_gii.meta = gii.meta
    
    # Save
    nib.save(new_gii, output_path)
    print(f"  Saved: {output_path}")


def transform_json_to_cpr(json_path: str, frame: dict, output_path: str,
                          cpr_spacing: float = 0.25, cross_section_size: int = 120):
    """
    Transform JSON point cloud to CPR coordinates and save.
    
    Args:
        json_path: Path to input JSON file (Niivue connectome format)
        frame: CPR frame data
        output_path: Path for output JSON file
        cpr_spacing: CPR pixel spacing in mm
        cross_section_size: Size of CPR cross-section in pixels
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract world coordinates
    nodes = data['nodes']
    world_points = np.column_stack([nodes['x'], nodes['y'], nodes['z']])
    
    print(f"  Transforming {len(world_points)} points...")
    
    # Transform to CPR coordinates
    cpr_coords = world_to_cpr(world_points, frame)
    
    # Convert to voxel coordinates
    half_size = cross_section_size // 2
    cpr_voxels = np.zeros_like(cpr_coords)
    cpr_voxels[:, 0] = cpr_coords[:, 0] / cpr_spacing + half_size
    cpr_voxels[:, 1] = cpr_coords[:, 1] / cpr_spacing + half_size
    cpr_voxels[:, 2] = cpr_coords[:, 2]
    
    # Update nodes with CPR coordinates
    data['nodes']['x'] = cpr_voxels[:, 0].tolist()
    data['nodes']['y'] = cpr_voxels[:, 1].tolist()
    data['nodes']['z'] = cpr_voxels[:, 2].tolist()
    
    # Add CPR metadata
    if 'metadata' not in data:
        data['metadata'] = {}
    data['metadata']['coordinate_system'] = 'CPR'
    data['metadata']['cpr_spacing'] = cpr_spacing
    data['metadata']['cross_section_size'] = cross_section_size
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved: {output_path}")


def batch_transform(frame_path: str, data_dir: str, output_dir: str = None):
    """
    Batch transform all matching GII and JSON files to CPR coordinates.
    
    Matches files based on vessel name in frame filename.
    """
    if output_dir is None:
        output_dir = data_dir
    
    # Extract vessel name from frame filename
    # e.g., "01-BER-0088_LAD_cpr_frame.json" -> "LAD"
    frame_name = os.path.basename(frame_path)
    parts = frame_name.replace('_cpr_frame.json', '').split('_')
    vessel = parts[-1] if len(parts) > 1 else None
    patient = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
    
    print(f"\nBatch transform for {patient} - {vessel}")
    print(f"Frame: {frame_path}")
    
    # Load frame
    frame = load_frame(frame_path)
    print(f"Centerline points: {frame['n_points']}")
    
    # Find matching files
    data_path = Path(data_dir)
    
    # GIfTI files
    gii_patterns = [
        f"{patient}_{vessel}_inner.gii",
        f"{patient}_{vessel}_outer.gii",
    ]
    
    # JSON files
    json_patterns = [
        f"{patient}_{vessel}_inner.json",
        f"{patient}_{vessel}_outer.json",
    ]
    
    output_files = []
    
    # Transform GIfTI files
    for pattern in gii_patterns:
        gii_path = data_path / pattern
        if gii_path.exists():
            output_name = pattern.replace('.gii', '_cpr.gii')
            output_path = Path(output_dir) / output_name
            print(f"\nTransforming: {pattern}")
            transform_gifti_to_cpr(str(gii_path), frame, str(output_path))
            output_files.append(str(output_path))
    
    # Transform JSON files
    for pattern in json_patterns:
        json_path = data_path / pattern
        if json_path.exists():
            output_name = pattern.replace('.json', '_cpr.json')
            output_path = Path(output_dir) / output_name
            print(f"\nTransforming: {pattern}")
            transform_json_to_cpr(str(json_path), frame, str(output_path))
            output_files.append(str(output_path))
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Transform mesh/point cloud to CPR coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform single file:
  python transform_to_cpr.py frame.json mesh.gii mesh_cpr.gii
  
  # Batch transform all files for a vessel:
  python transform_to_cpr.py --batch 01-BER-0088_LAD_cpr_frame.json ../data/
        """
    )
    parser.add_argument('--batch', action='store_true',
                        help='Batch transform all matching files')
    parser.add_argument('frame_json', help='CPR frame JSON file')
    parser.add_argument('input_or_dir', help='Input file or directory (for batch)')
    parser.add_argument('output', nargs='?', help='Output file (not needed for batch)')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_transform(args.frame_json, args.input_or_dir)
    else:
        if not args.output:
            parser.error("Output file required for single file transform")
        
        frame = load_frame(args.frame_json)
        
        if args.input_or_dir.endswith('.gii'):
            transform_gifti_to_cpr(args.input_or_dir, frame, args.output)
        elif args.input_or_dir.endswith('.json'):
            transform_json_to_cpr(args.input_or_dir, frame, args.output)
        else:
            print(f"Unknown file type: {args.input_or_dir}")
            sys.exit(1)


if __name__ == "__main__":
    main()
