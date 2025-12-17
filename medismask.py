import sys,os,time
import argparse, json
import numpy as np 
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

 

def log_image_metadata(image, label):
    """Prints the basic geometry and pixel information of a SimpleITK image."""
    size = image.GetSize()
    dim = len(size)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    pixel_id = image.GetPixelID()
    print(f"{label} metadata:")
    print(f"  Dimension: {dim}D")
    print(f"  Size: {size}")
    print(f"  Spacing: {spacing}")
    print(f"  Origin: {origin}")
    print(f"  Direction: {direction}")
    print(f"  Pixel ID: {pixel_id}")
    print(f"  Pixel type: {image.GetPixelIDTypeAsString()}")


def load_dicom_series(dicom_folder_path):
    """
    Load a 3D DICOM series from a folder using GetGDCMSeriesFileNames.
    
    Parameters:
    dicom_folder_path (str): Path to the folder containing DICOM files
    
    Returns:
    sitk.Image: SimpleITK image object
    """
    if not os.path.isdir(dicom_folder_path):
        raise FileNotFoundError(f"Directory not found: {dicom_folder_path}")
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    
    if not dicom_names:
        raise ValueError(f"No DICOM files found in: {dicom_folder_path}")
        
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    return image

def load_contours_from_txt(txt_file_path, lesions_only=False):
    """
    Load lumen and vessel wall contours from a text file.
    
    Parameters:
    txt_file_path (str): Path to the contour text file
    lesions_only (bool): If True, only load slices with lesion_names annotation
    
    Returns:
    tuple: (lumen_dict, vessel_wall_dict, lesion_info_dict) where keys are slice distances,
           values are coordinate arrays, and lesion_info_dict maps slice_distance to lesion_name
    """
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"Contour file not found: {txt_file_path}")
    
    lumen = {}
    vessel_wall = {}
    lesion_info = {}
    
    with open(txt_file_path, 'r') as fp:
        bits = fp.read().split("\n\n")
    
    for string in bits[1:]:
        if "SliceDistance:" not in string:
            continue
        
        # Parse lesion_names if present
        lesion_name = None
        if "# lesion_names:" in string:
            lesion_line = [line for line in string.split("\n") if "# lesion_names:" in line]
            if lesion_line:
                lesion_name = lesion_line[0].split("# lesion_names:")[1].strip()
        
        # Skip this slice if lesions_only is True and no lesion is present
        if lesions_only and lesion_name is None:
            continue
            
        slice_distance = float(string.split("SliceDistance: ")[-1].split("\n")[0])
        
        # Store lesion info if present
        if lesion_name:
            lesion_info[slice_distance] = lesion_name
        
        if "Lumen" in string:
            coords = [np.fromstring(element, sep=' ') for element in string.split("\n", 2)[2].split("\n") if element != '' and "#" not in element]
            if coords:
                lumen[slice_distance] = np.vstack(coords)
                
        if "Wall" in string:
            coords = [np.fromstring(element, sep=' ') for element in string.split("\n", 2)[2].split("\n") if element != '' and "#" not in element]
            if coords:
                vessel_wall[slice_distance] = np.vstack(coords)
    
    return lumen, vessel_wall, lesion_info

def match_lumen_wall(lumen, vessel_wall):
    """
    Match the number of points between lumen and vessel wall contours.
    
    Parameters:
    lumen (np.array): Lumen contour coordinates for one slice
    vessel_wall (np.array): Vessel wall contour coordinates for one slice
    
    Returns:
    np.array: Matched vessel wall coordinates
    """
    diff = vessel_wall.shape[0] - lumen.shape[0]
    if diff > 0:
        q = vessel_wall.shape[0] // diff
        to_delete = [i * q - 1 for i in range(1, diff + 1)]
        vessel_wall_matched = np.delete(vessel_wall, to_delete, 0)
    elif diff == 0:
        vessel_wall_matched = vessel_wall
    else:
        vessel_wall_matched = np.concatenate([vessel_wall] + [vessel_wall[-2:-1, :] for i in range(-diff)])
    
    return vessel_wall_matched

def polygon_to_mask_2d(points_2d, shape):
    """
    Rasterize a 2D polygon into a binary mask using scanline filling.
    
    Parameters:
    points_2d (np.array): Nx2 array of (row, col) polygon vertices
    shape (tuple): (height, width) of output mask
    
    Returns:
    np.array: 2D binary mask with polygon filled
    """
    if len(points_2d) < 3:
        return np.zeros(shape, dtype=np.uint8)
    
    # Use OpenCV for fast polygon filling (much faster than matplotlib)
    try:
        import cv2
        mask = np.zeros(shape, dtype=np.uint8)
        # OpenCV expects points as [(x,y), ...] where x=col, y=row
        points_xy = points_2d[:, [1, 0]].astype(np.int32)  # Convert (row,col) to (col,row)
        cv2.fillPoly(mask, [points_xy], color=1)
        return mask
    except ImportError:
        # Fallback to matplotlib if OpenCV not available
        from matplotlib.path import Path
        
        points_xy = points_2d[:, [1, 0]]  # Swap to (x, y)
        rows, cols = np.mgrid[:shape[0], :shape[1]]
        points_grid = np.column_stack((cols.ravel(), rows.ravel()))
        path = Path(points_xy)
        mask = path.contains_points(points_grid)
        mask = mask.reshape(shape)
        return mask.astype(np.uint8)


def create_plaque_mask_radial(lumen_dict, vessel_wall_dict, ct_image, n_points=10):
    """
    Create a plaque mask using radial interpolation method.
    
    This approach interpolates points between lumen and wall contours along radial lines,
    paints them into the volume, then applies morphological closing to fill gaps.
    
    Method: RADIAL_INTERPOLATION
    - Matches lumen/wall point counts
    - Assumes radial correspondence between contours
    - Interpolates n_points+1 layers between lumen and wall
    - Point painting with morphological closing
    
    Parameters:
    lumen_dict (dict): Dictionary of lumen contours (slice_distance -> physical coords)
    vessel_wall_dict (dict): Dictionary of vessel wall contours  
    ct_image (sitk.Image): CT image for metadata reference
    n_points (int): Number of radial interpolation points between lumen and wall
    
    Returns:
    sitk.Image: Binary mask image
    """
    size = ct_image.GetSize()
    origin = ct_image.GetOrigin()
    spacing = ct_image.GetSpacing()
    direction = ct_image.GetDirection()
    
    # Create mask with same geometry as CT
    mask = sitk.Image(size[0], size[1], size[2], sitk.sitkInt16)
    mask.SetSpacing(spacing)
    mask.SetOrigin(origin)
    mask.SetDirection(direction)
    
    # Get common slices
    common_slices = sorted(set(lumen_dict.keys()) & set(vessel_wall_dict.keys()))
    
    if not common_slices:
        print("  WARNING: No common slices between lumen and wall contours!")
        return sitk.Cast(mask, sitk.sitkUInt8)
    
    print(f"  Processing {len(common_slices)} common slices...")
    
    points_painted = 0
    points_skipped = 0
    
    # Collect all interpolated points
    all_points = []
    
    for slice_distance in common_slices:
        lumen_coords = lumen_dict[slice_distance]
        wall_coords = vessel_wall_dict[slice_distance]
        
        if lumen_coords.shape[0] < 3 or wall_coords.shape[0] < 3:
            continue
        
        # Match point counts between lumen and wall
        wall_matched = match_lumen_wall(lumen_coords, wall_coords)
        
        # Interpolate radially between lumen and wall (including both endpoints)
        # This creates n_points+1 layers: lumen, n_points-1 intermediate, wall
        for i in range(n_points + 1):
            interpolated = lumen_coords + (i / n_points) * (wall_matched - lumen_coords)
            all_points.append(interpolated)
    
    if not all_points:
        print("  WARNING: No valid contour points found!")
        return sitk.Cast(mask, sitk.sitkUInt8)
    
    # Concatenate all points
    points = np.concatenate(all_points, axis=0)
    print(f"  Total interpolated points: {points.shape[0]}")
    
    # Paint points into mask
    for i in range(points.shape[0]):
        try:
            idx = mask.TransformPhysicalPointToIndex(tuple(points[i, :]))
            mask[idx] = 1
            points_painted += 1
        except RuntimeError:
            points_skipped += 1
    
    print(f"  Painted {points_painted} points, skipped {points_skipped} out-of-bounds")
    
    # Apply morphological closing to fill gaps
    mask_closed = sitk.BinaryMorphologicalClosing(mask, (1, 1, 1), sitk.sitkBall)
    mask_closed = sitk.Cast(mask_closed, sitk.sitkUInt8)
    
    return mask_closed


def create_plaque_mask_polygon(lumen_dict, vessel_wall_dict, ct_image):
    """
    Create a plaque mask using polygon filling method.
    
    This approach treats lumen and wall as independent closed polygons,
    fills them slice-by-slice, then computes the annular ring via set subtraction.
    
    Method: POLYGON_FILLING
    - No point matching required
    - No radial correspondence assumption
    - Exact topology via scanline rasterization
    - Ring = Wall ∩ ¬Lumen (set-theoretic definition)
    
    Parameters:
    lumen_dict (dict): Dictionary of lumen contours (slice_distance -> physical coords)
    vessel_wall_dict (dict): Dictionary of vessel wall contours
    ct_image (sitk.Image): CT image for metadata reference
    
    Returns:
    sitk.Image: Binary mask image
    """
    size = ct_image.GetSize()
    origin = ct_image.GetOrigin()
    spacing = ct_image.GetSpacing()
    direction = ct_image.GetDirection()
    
    # Get common slices
    common_slices = sorted(set(lumen_dict.keys()) & set(vessel_wall_dict.keys()))
    
    if not common_slices:
        print("  WARNING: No common slices between lumen and wall contours!")
        mask = sitk.Image(size[0], size[1], size[2], sitk.sitkUInt8)
        mask.CopyInformation(ct_image)
        return mask
    
    print(f"  Processing {len(common_slices)} common slices with polygon filling...")
    
    # Build a mapping from z-index to 2D mask
    z_index_to_mask = {}
    slices_processed = 0
    slices_skipped = 0
    skip_reasons = {'vertex_count': 0, 'transform_error': 0, 'z_mismatch': 0}
    
    # Process each contour pair
    for slice_distance in common_slices:
        lumen_coords = lumen_dict[slice_distance]
        wall_coords = vessel_wall_dict[slice_distance]
        
        # Validate polygon vertex count
        if lumen_coords.shape[0] < 3 or wall_coords.shape[0] < 3:
            slices_skipped += 1
            skip_reasons['vertex_count'] += 1
            continue
        
        # Transform physical coordinates to image indices for the complete contours
        # Keep them as complete polygons, don't fragment by z-index
        try:
            lumen_indices = np.array([ct_image.TransformPhysicalPointToIndex(tuple(pt)) for pt in lumen_coords])
            wall_indices = np.array([ct_image.TransformPhysicalPointToIndex(tuple(pt)) for pt in wall_coords])
        except RuntimeError:
            slices_skipped += 1
            skip_reasons['transform_error'] += 1
            continue
        
        # Get the range of z-slices this contour spans
        lumen_z_min, lumen_z_max = lumen_indices[:, 2].min(), lumen_indices[:, 2].max()
        wall_z_min, wall_z_max = wall_indices[:, 2].min(), wall_indices[:, 2].max()
        
        # Use the overlapping z-range
        z_min = max(lumen_z_min, wall_z_min)
        z_max = min(lumen_z_max, wall_z_max)
        
        if z_max < z_min:
            slices_skipped += 1
            skip_reasons['z_mismatch'] += 1
            continue
        
        # Paint this contour across all z-slices it spans
        # Use 2D projection (y, x) coordinates for each slice
        lumen_2d = lumen_indices[:, [1, 0]]  # (y, x)
        wall_2d = wall_indices[:, [1, 0]]  # (y, x)
        
        if len(lumen_2d) < 3 or len(wall_2d) < 3:
            slices_skipped += 1
            skip_reasons['vertex_count'] += 1
            continue
        
        # Create 2D polygon masks for this contour's projection
        slice_shape = (size[1], size[0])  # (height, width) = (y, x)
        wall_mask_2d = polygon_to_mask_2d(wall_2d, slice_shape)
        lumen_mask_2d = polygon_to_mask_2d(lumen_2d, slice_shape)
        
        # Fill interior holes to ensure solid regions
        wall_mask_2d = binary_fill_holes(wall_mask_2d).astype(np.uint8)
        lumen_mask_2d = binary_fill_holes(lumen_mask_2d).astype(np.uint8)
        
        # Compute ring: wall AND NOT lumen
        ring_mask_2d = wall_mask_2d & (~lumen_mask_2d)
        
        # Paint this 2D ring across all z-slices in the contour's range
        for z in range(z_min, z_max + 1):
            if z not in z_index_to_mask:
                z_index_to_mask[z] = ring_mask_2d.copy()
            else:
                z_index_to_mask[z] = z_index_to_mask[z] | ring_mask_2d
        
        slices_processed += 1
    
    print(f"  Processed {slices_processed} contours, skipped {slices_skipped}")
    print(f"  Generated masks for {len(z_index_to_mask)} z-slices")
    if slices_skipped > 0:
        print(f"  Skip reasons: {skip_reasons}")
    
    # Convert ITK indices back to physical coordinates, then paint like radial method
    # This avoids coordinate system confusion with negative z-indices
    all_physical_points = []
    
    for z_idx, ring_2d in z_index_to_mask.items():
        yx_coords = np.argwhere(ring_2d > 0)
        for y, x in yx_coords:
            # Convert ITK index (x, y, z) to physical point
            try:
                phys_pt = ct_image.TransformIndexToPhysicalPoint((int(x), int(y), int(z_idx)))
                all_physical_points.append(phys_pt)
            except:
                pass
    
    print(f"  Painting {len(all_physical_points)} voxels into mask...")
    
    # Paint physical points into mask
    mask = sitk.Image(size[0], size[1], size[2], sitk.sitkInt16)
    mask.SetSpacing(spacing)
    mask.SetOrigin(origin)
    mask.SetDirection(direction)
    
    painted = 0
    for phys_pt in all_physical_points:
        try:
            idx = mask.TransformPhysicalPointToIndex(phys_pt)
            mask[idx] = 1
            painted += 1
        except RuntimeError:
            pass
    
    print(f"  Successfully painted {painted} voxels")
    
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    return mask


def resample_mask_to_reference(mask_image, reference_image):
    """Resample mask to match reference image geometry using nearest neighbor."""
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetReferenceImage(reference_image)
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    resample_filter.SetDefaultPixelValue(0)
    resample_filter.SetTransform(sitk.Transform())
    resampled_mask = resample_filter.Execute(mask_image)
    return resampled_mask

def create_nifti_from_ct_and_contours(ct_image, contour_files, output_path, method='polygon', n_points=10, lesions_only=False):
    """
    Main function to create a NIfTI mask from a preloaded CT image and contour files.
    
    Parameters:
    ct_image (sitk.Image): CT image providing the target geometry (ideally 3D)
    contour_files (Sequence[str]): Paths to contour text files
    output_path (str): Output path for the NIfTI file
    method (str): Mask creation method - 'polygon' (default) or 'radial'
    n_points (int): Number of interpolation points (only used for radial method)
    lesions_only (bool): If True, only create mask for slices with lesion annotations
    """
    if isinstance(contour_files, str):
        contour_files = [contour_files]
    
    # Validate method
    if method not in ['polygon', 'radial']:
        raise ValueError(f"Invalid method '{method}'. Choose 'polygon' or 'radial'.")

    print("Using provided CT image for mask geometry...")
    log_image_metadata(ct_image, "Target geometry")
    print(f"\n*** METHOD: {method.upper()} ***")
    
    if lesions_only:
        print("*** LESIONS ONLY MODE: Creating mask only for annotated lesion regions ***\n")

    combined_mask = None
    total_lesion_slices = 0

    for idx, contour_file in enumerate(contour_files, start=1):
        print(f"\nProcessing contour file {idx}/{len(contour_files)}: {os.path.basename(contour_file)}")
        lumen_dict, vessel_wall_dict, lesion_info = load_contours_from_txt(contour_file, lesions_only=lesions_only)
        print(f"  Found {len(lumen_dict)} lumen slices and {len(vessel_wall_dict)} wall slices")
        
        if lesions_only:
            unique_lesions = set(lesion_info.values())
            print(f"  Lesion slices: {len(lesion_info)}, Unique lesions: {len(unique_lesions)}")
            if unique_lesions:
                print(f"  Lesion IDs: {', '.join(sorted(unique_lesions))}")
            total_lesion_slices += len(lesion_info)

        print(f"  Creating plaque mask using {method} method...")
        
        # Dispatch to appropriate method
        if method == 'polygon':
            partial_mask = create_plaque_mask_polygon(lumen_dict, vessel_wall_dict, ct_image)
        else:  # radial
            partial_mask = create_plaque_mask_radial(lumen_dict, vessel_wall_dict, ct_image, n_points)

        combined_mask = partial_mask if combined_mask is None else sitk.Or(combined_mask, partial_mask)

    if combined_mask is None:
        raise ValueError("No contours produced a mask. Check contour file contents.")

    mask_image = sitk.Cast(combined_mask, sitk.sitkUInt8)

    print("Resampling mask to target geometry...")
    mask_image = resample_mask_to_reference(mask_image, ct_image)

    log_image_metadata(mask_image, "Mask")
    
    if lesions_only:
        print(f"\nTotal lesion slices across all files: {total_lesion_slices}")
    
    print(f"Saving mask to {output_path}")
    sitk.WriteImage(mask_image, output_path)
    print("Done!")

def load_ct_series_to_nifti(dicom_folder, output_path):
    """Loads DICOM series, logs metadata, converts to 3D if necessary, and saves as NIfTI."""
    print("Loading CT DICOM series for export...")
    ct_image = load_dicom_series(dicom_folder)
    log_image_metadata(ct_image, "Raw CT")

    exported_image = ct_image
    if len(ct_image.GetSize()) == 4:
        print("Raw CT is 4D; squeezing to 3D before saving")
        array = sitk.GetArrayFromImage(ct_image)
        array = np.squeeze(array)
        exported_image = sitk.GetImageFromArray(array)
        exported_image = sitk.Cast(exported_image, ct_image.GetPixelID())
        spacing = ct_image.GetSpacing()
        origin = ct_image.GetOrigin()
        direction = ct_image.GetDirection()
        exported_image.SetSpacing(spacing[:3])
        exported_image.SetOrigin(origin[:3])
        exported_image.SetDirection([
            direction[0], direction[1], direction[2],
            direction[4], direction[5], direction[6],
            direction[8], direction[9], direction[10],
        ])

    log_image_metadata(exported_image, "Exported CT volume")
    sitk.WriteImage(exported_image, output_path)
    print(f"Saved CT volume to {output_path}")
    return exported_image


def run_plaque_pipeline(lesions_only=False, method='polygon', compare_methods=False):
    """
    Run the plaque mask creation pipeline.
    
    Parameters:
    lesions_only (bool): If True, create mask only for lesion-annotated regions.
                        If False, create mask for all vessel wall regions.
    method (str): Mask creation method - 'polygon' (default) or 'radial'
    compare_methods (bool): If True, run both methods and compare results
    """
    plaque_dir = r"C:\Users\lukass\Downloads\plaque"
    #plaque_dir = r"C:\Users\steff\Downloads\plaque"
    
   
    dicom_folder = os.path.join(plaque_dir, "1.2.392.200036.9116.2.6.1.3268.2047366984.1465512903.187230")
    contour_files = [
        os.path.join(plaque_dir, "04-PRA-0025_20160610_Session_Session 23_06_2021 10_49 Eser_lad.txt"),
        os.path.join(plaque_dir, "04-PRA-0025_20160610_Session_Session 23_06_2021 10_49 Eser_lcx.txt"),
        os.path.join(plaque_dir, "04-PRA-0025_20160610_Session_Session 23_06_2021 10_49 Eser_om1.txt"),
        os.path.join(plaque_dir, "04-PRA-0025_20160610_Session_Session 23_06_2021 10_49 Eser_rca.txt"),
    ]
    ct_output_path = os.path.join(plaque_dir, "04-PRA-0025_19.nii.gz")
    
    ct_image = load_ct_series_to_nifti(dicom_folder, ct_output_path)

    if compare_methods:
        print("\n" + "="*80)
        print("COMPARISON MODE: Running both polygon and radial methods")
        print("="*80 + "\n")
        
        methods_to_run = ['polygon', 'radial']
        results = {}
        
        for method_name in methods_to_run:
            print(f"\n{'='*80}")
            print(f"Running {method_name.upper()} method...")
            print(f"{'='*80}\n")
            
            # Choose output filename
            suffix = f"_{method_name}"
            if lesions_only:
                suffix += "_lesions_only"
            mask_output_path = os.path.join(plaque_dir, f"plaque_mask{suffix}.nii.gz")
            
            try:
                create_nifti_from_ct_and_contours(
                    ct_image,
                    contour_files,
                    mask_output_path,
                    method=method_name,
                    lesions_only=lesions_only,
                )
                
                # Load and analyze the mask
                mask = sitk.ReadImage(mask_output_path)
                mask_array = sitk.GetArrayFromImage(mask)
                n_voxels = np.sum(mask_array > 0)
                
                results[method_name] = {
                    'path': mask_output_path,
                    'n_voxels': n_voxels,
                    'mask_array': mask_array
                }
                
                print(f" {method_name.upper()} method completed: {mask_output_path}")
                print(f"  Non-zero voxels: {n_voxels:,}")
                
            except Exception as e:
                print(f" Error with {method_name} method: {e}")
                import traceback
                traceback.print_exc()
        
        # Compare results
        if len(results) == 2:
            print(f"\n{'='*80}")
            print("COMPARISON RESULTS")
            print(f"{'='*80}\n")
            
            poly_voxels = results['polygon']['n_voxels']
            radial_voxels = results['radial']['n_voxels']
            
            print(f"Polygon method:  {poly_voxels:,} voxels")
            print(f"Radial method:   {radial_voxels:,} voxels")
            print(f"Difference:      {abs(poly_voxels - radial_voxels):,} voxels ({100*abs(poly_voxels - radial_voxels)/max(poly_voxels, radial_voxels):.2f}%)")
            
            # Compute overlap metrics
            poly_array = results['polygon']['mask_array']
            radial_array = results['radial']['mask_array']
            
            intersection = np.sum((poly_array > 0) & (radial_array > 0))
            union = np.sum((poly_array > 0) | (radial_array > 0))
            dice = 2 * intersection / (poly_voxels + radial_voxels) if (poly_voxels + radial_voxels) > 0 else 0
            iou = intersection / union if union > 0 else 0
            
            print(f"\nOverlap metrics:")
            print(f"  Intersection:    {intersection:,} voxels")
            print(f"  Union:           {union:,} voxels")
            print(f"  Dice coefficient: {dice:.4f}")
            print(f"  IoU (Jaccard):    {iou:.4f}")
            
            # Voxels unique to each method
            only_polygon = np.sum((poly_array > 0) & (radial_array == 0))
            only_radial = np.sum((poly_array == 0) & (radial_array > 0))
            
            print(f"\nUnique voxels:")
            print(f"  Only in polygon: {only_polygon:,} voxels")
            print(f"  Only in radial:  {only_radial:,} voxels")
            
    else:
        # Single method run
        suffix = f"_{method}"
        if lesions_only:
            suffix += "_lesions_only"
        mask_output_path = os.path.join(plaque_dir, f"plaque_mask{suffix}.nii.gz")

        try:
            create_nifti_from_ct_and_contours(
                ct_image,
                contour_files,
                mask_output_path,
                method=method,
                lesions_only=lesions_only,
            )
            print(f"\nSuccessfully created mask: {mask_output_path}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Create plaque masks from coronary artery contours')
    parser.add_argument('--lesions-only', action='store_true', 
                       help='Create mask only for lesion-annotated regions (default: create mask for all vessel wall)')
    parser.add_argument('--method', choices=['polygon', 'radial'], default='polygon',
                       help='Mask creation method (default: polygon)')
    parser.add_argument('--compare-methods', action='store_true',
                       help='Run both polygon and radial methods and compare results')
    args = parser.parse_args()
    
    run_plaque_pipeline(
        lesions_only=args.lesions_only,
        method=args.method,
        compare_methods=args.compare_methods
    )


if __name__ == '__main__':
    main()