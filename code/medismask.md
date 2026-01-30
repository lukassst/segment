# MEDIS Mask Operations - Comprehensive Documentation

## Overview

The `medismask.py` script provides a comprehensive pipeline for converting MEDIS contour export files into binary masks suitable for medical imaging analysis. It bridges the gap between clinical contour annotations and quantitative imaging research by transforming 2D vessel contours into 3D volumetric masks.

## Core Functionality

### Primary Purpose
- Convert MEDIS TXT contour exports → Binary NIfTI masks
- Preserve exact CT geometry for spatial accuracy
- Support both full vessel wall and lesion-specific analysis
- Provide two complementary mask creation methods

### Input Data
- **MEDIS TXT files** - Contour coordinates with slice distance metadata
- **DICOM CT series** - Reference geometry for mask creation
- **Lesion annotations** - Optional lesion-specific processing

### Output Products
- **Binary NIfTI masks** - Analysis-ready volumetric data
- **Metadata logs** - Complete processing audit trail
- **Comparison metrics** - Method evaluation statistics

## Method Architecture

### 1. Data Loading Pipeline

#### DICOM Series Processing
```python
load_dicom_series(dicom_folder_path)
```
- Uses SimpleITK's `GetGDCMSeriesFileNames()` for proper DICOM ordering
- Handles multi-frame DICOM series automatically
- Preserves all spatial metadata (spacing, origin, direction)

#### CT Volume Export
```python
load_ct_series_to_nifti(dicom_folder, output_path)
```
- Converts DICOM → NIfTI with geometry preservation
- Handles 4D→3D dimension reduction (squeezing)
- Maintains coordinate system integrity

#### Contour File Parsing
```python
load_contours_from_txt(txt_file_path, lesions_only=False)
```
- Parses MEDIS TXT format with slice distance organization
- Extracts lumen and vessel wall coordinates separately
- Handles lesion annotations for targeted analysis
- Supports varying point counts per contour

### 2. Mask Creation Methods

#### Method A: Polygon Filling (`create_plaque_mask_polygon()`)

**Algorithmic Approach:**
- Treats lumen and wall as independent closed polygons
- Uses scanline rasterization for exact polygon filling
- Computes annular ring via set operations: `Wall ∩ ¬Lumen`

**Technical Process:**
1. Transform physical coordinates → image indices
2. Create 2D polygon masks for each slice
3. Apply binary fill holes for solid regions
4. Compute ring mask using boolean operations
5. Paint across z-slices with proper coordinate mapping

**Advantages:**
- No point matching required between contours
- Exact topology preservation
- Handles complex vessel geometries
- Robust to point count variations

#### Method B: Radial Interpolation (`create_plaque_mask_radial()`)

**Algorithmic Approach:**
- Assumes radial correspondence between lumen and wall
- Interpolates points between contours along radial lines
- Uses morphological operations for gap filling

**Technical Process:**
1. Match point counts between lumen and wall contours
2. Interpolate `n_points` layers radially between contours
3. Paint interpolated points into 3D volume
4. Apply morphological closing to fill gaps

**Advantages:**
- Smooth vessel wall representation
- Good for regular vessel morphologies
- Configurable interpolation density
- Computationally efficient

### 3. Geometry and Coordinate Systems

#### Coordinate Transformations
- **Physical → Index**: `TransformPhysicalPointToIndex()`
- **Index → Physical**: `TransformIndexToPhysicalPoint()`
- **Multi-slice projection**: Handles contours spanning multiple z-slices

#### Geometry Preservation
```python
log_image_metadata(image, label)
```
- Tracks all spatial parameters (size, spacing, origin, direction)
- Ensures mask alignment with original CT volume
- Provides complete audit trail for reproducibility

#### Resampling Operations
```python
resample_mask_to_reference(mask_image, reference_image)
```
- Nearest-neighbor interpolation to preserve binary nature
- Maintains exact alignment with reference geometry
- Handles different spatial resolutions

### 4. Advanced Features

#### Lesion-Specific Processing
- **`lesions_only` mode**: Processes only annotated lesion regions
- Extracts lesion names from MEDIS comment fields
- Enables targeted plaque analysis
- Reduces processing time for focused studies

#### Method Comparison Framework
```python
compare_methods=True
```
- Runs both polygon and radial methods simultaneously
- Computes overlap metrics (Dice coefficient, IoU)
- Provides voxel-wise difference analysis
- Supports method validation studies

#### Point Matching Algorithm
```python
match_lumen_wall(lumen, vessel_wall)
```
- Intelligent point count alignment
- Handles both excess and deficit cases
- Preserves contour topology during matching

## Implementation Details

### Dependencies
- **SimpleITK**: DICOM/NIfTI handling, image operations
- **NumPy**: Array operations, coordinate transformations
- **SciPy**: Binary fill holes, morphological operations
- **OpenCV** (optional): Fast polygon filling with fallback

### Performance Optimizations
- **OpenCV acceleration**: Fast polygon filling when available
- **Vectorized operations**: NumPy-based coordinate transformations
- **Memory efficient**: Streaming processing for large volumes
- **Parallel processing**: Ready for multi-threading implementation

### Error Handling
- **Robust coordinate validation**: Handles out-of-bounds transformations
- **Vertex count checking**: Validates polygon integrity
- **Detailed logging**: Comprehensive skip reason tracking
- **Graceful degradation**: Fallback methods for edge cases

## Usage Patterns

### Basic Mask Creation
```python
# Full vessel wall mask using polygon method
create_nifti_from_ct_and_contours(
    ct_image, contour_files, output_path, 
    method='polygon', lesions_only=False
)
```

### Lesion-Specific Analysis
```python
# Only lesion-annotated regions
create_nifti_from_ct_and_contours(
    ct_image, contour_files, output_path,
    method='radial', lesions_only=True, n_points=15
)
```

### Method Comparison
```python
run_plaque_pipeline(
    lesions_only=False, 
    method='polygon', 
    compare_methods=True
)
```

## Clinical Applications

### Research Use Cases
- **Plaque quantification**: Volume measurements of vessel wall tissue
- **Lesion tracking**: Longitudinal analysis of specific plaque regions
- **Morphometry studies**: Vessel geometry and remodeling analysis
- **Treatment planning**: Pre-procedural vessel assessment

### Integration Opportunities
- **CFD preprocessing**: Generate vessel geometries for flow simulation
- **AI training**: Create ground truth masks for machine learning
- **Population studies**: Batch processing for large cohorts
- **Clinical trials**: Standardized mask generation for multi-center studies

## Quality Assurance

### Validation Features
- **Method comparison**: Built-in validation between approaches
- **Metadata tracking**: Complete processing audit trail
- **Error reporting**: Detailed skip reason statistics
- **Geometry verification**: Spatial accuracy confirmation

### Reproducibility
- **Deterministic processing**: Same inputs → same outputs
- **Parameter logging**: All settings recorded in output
- **Version tracking**: Method selection documented
- **Cross-platform compatibility**: Works on Windows/Linux/Mac

## Future Development

### Potential Enhancements
- **GPU acceleration**: CUDA-based processing for large volumes
- **Advanced interpolation**: Spline-based contour interpolation
- **Multi-modal support**: Integration with PET/MRI data
- **Real-time processing**: Live contour-to-mask conversion

### Extension Points
- **Custom contour formats**: Adaptable parser for different vendors
- **Advanced morphometrics**: Curvature, thickness measurements
- **Machine learning integration**: Automated quality assessment
- **Cloud deployment**: Scalable processing infrastructure

---

## Technical Summary

The `medismask.py` script represents a comprehensive solution for converting clinical MEDIS contour annotations into research-ready binary masks. By providing two complementary methods, robust geometry handling, and extensive validation features, it bridges the gap between clinical workflow and quantitative imaging research while maintaining the spatial accuracy required for medical applications.
