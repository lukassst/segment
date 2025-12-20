# 🫀 Interactive Vessel Segmentation Tool (Classical Methods)

**Project:** Complete Interactive Vessel Segmentation Using Classical Computer Vision  
**Method:** Fast Marching Method (FMM) + Polar Dynamic Programming + Multi-Planar Reformation  
**Goal:** Full vessel segmentation (centerline + lumen + outer wall) from 2 user clicks  
**Constraint:** Standard Python libraries (numpy, scipy, skimage, SimpleITK)  
**Future:** Browser/WASM compatible (Pyodide + NiiVue)

---

## 📋 Project Overview

### What This Tool Does

**Interactive Vessel Segmenter** is a **complete segmentation pipeline** that extracts detailed coronary vessel geometry from CTA volumes using only **two user-placed points** (start + end).

**Outputs:**
1. **Refined centerline** - Accurate vessel path (sub-voxel precision)
2. **Lumen boundary** - Inner vessel wall (blood pool surface)
3. **Outer wall boundary** - Outer vessel boundary (including plaque)
4. **Cross-sectional slices** - Perpendicular vessel sections at every point
5. **3D meshes** - STL export for visualization or CFD simulation

**Key Distinction:** This is **NOT** just centerline extraction - the centerline is one component of a complete vessel segmentation workflow.

### Clinical Context
**CT-FFR (CT-based Fractional Flow Reserve)** and plaque analysis require:
- **Precise vessel lumen boundary** for blood flow simulation (CFD)
- **Outer vessel wall boundary** for plaque volume quantification
- **Accurate centerline** for computational fluid dynamics mesh generation
- **Cross-sectional measurements** for stenosis severity and remodeling index

### Interactive Workflow

**User interaction (2 clicks):**
```
1. USER: Place start point (e.g., ostium of LAD)
2. USER: Place end point (e.g., distal LAD)
3. TOOL: Automatic processing (~20-60s)
   ├─ Extract optimal centerline (FMM)
   ├─ Segment lumen and outer wall at each point (Polar DP)
   └─ Refine centerline using lumen centroids
4. OUTPUT: Complete vessel geometry ready for CFD or analysis
```

### Algorithmic Approach (Classical CV)

This tool uses **classical computer vision algorithms** (no AI/deep learning):

**Phase 1: Preprocessing**
- **Vesselness filter (Frangi):** Enhance tubular structures using Hessian eigenvalues
- **Cost map:** Invert vesselness for path optimization

**Phase 2: Initial Centerline (FMM)**
- **Fast Marching Method:** Propagate wave from start point → global optimal path
- **Gradient descent backtracking:** Follow arrival time gradient to extract path

**Phase 3: Vessel Wall Segmentation**
- **Multi-Planar Reformation (MPR):** Extract perpendicular cross-sections
- **Polar transform:** Convert Cartesian → polar coordinates (unwrap vessel)
- **Dynamic Programming:** Graph search for min-cost lumen/wall boundaries
- **Constraint enforcement:** Ensure outer wall > lumen at every angle

**Phase 4: Centerline Refinement**
- **Centroid computation:** Geometric center of segmented lumen
- **Path update:** Replace FMM path with lumen centroids (corrects "corner cutting")
- **Spline smoothing:** Remove jitter for smooth final centerline

### Advantages Over AI Methods

- ✅ **Deterministic:** Same input → same output (reproducible for clinical validation)
- ✅ **No training data required:** Works out-of-the-box on any scanner/protocol
- ✅ **Mathematically guaranteed optimality:** FMM finds global minimum path (no local minima)
- ✅ **Interpretable:** Every step is mathematically defined (no black box)
- ✅ **Browser-compatible:** NumPy/SciPy can run in WASM (Pyodide)
- ✅ **Open-source:** No proprietary algorithms or licensing (academic/commercial use)
- ✅ **Fast:** ~20-60s per vessel (no GPU required)

### Comparison: Interactive Tool vs AI Segmentation

| Aspect | This Tool (Classical) | SAM3D Platform (AI) |
|--------|----------------------|---------------------|
| **User input** | 2 points (start + end) | 1-3 points anywhere on vessel |
| **Method** | FMM + Polar DP | SAM-Med3D foundation model |
| **Training** | None required | Pre-trained on 143K masks |
| **Speed** | ~20-60s (CPU) | <2s (with GPU + cache) |
| **Reproducibility** | 100% deterministic | Varies slightly with prompts |
| **Deployment** | Desktop/WASM | Hospital server (GPU cluster) |
| **Output** | Centerline + lumen + outer wall | Segmentation mask (any structure) |
| **Best for** | CT-FFR, precise vessel geometry | General anatomical segmentation |

---

## 🏗️ Architecture

### Python Stack (Prototype)

```
ct-ffr-pipeline/
├── preprocessing/
│   ├── load_volume.py          # NIfTI loading (SimpleITK/nibabel)
│   ├── resample.py             # Isotropic resampling
│   └── vesselness_filter.py    # Frangi filter (Hessian eigenvalues)
├── centerline/
│   ├── fmm_path.py              # Fast Marching Method
│   ├── gradient_descent.py      # Backtracking from arrival map
│   └── refinement.py            # Centroid-based correction
├── segmentation/
│   ├── mpr_extraction.py        # Multi-planar reformation
│   ├── polar_transform.py       # Cartesian → polar coordinates
│   └── dynamic_programming.py   # Graph search for vessel boundaries
├── utils/
│   ├── interpolation.py         # Trilinear/tricubic
│   └── geometry.py              # 3D math utilities
└── main.py                      # End-to-end pipeline
```

### Technology Stack

**Immediate (Python Prototype):**
- **Loading:** `SimpleITK` or `nibabel`
- **Math/Arrays:** `numpy`
- **Image Processing:** `scipy.ndimage`, `skimage`
- **FMM:** `scikit-fmm` or `SimpleITK.FastMarchingImageFilter`

**Future (Browser/WASM):**
- **Compute Engine:** Pyodide (Python in WASM) or WebAssembly (C++)
- **Rendering:** NiiVue (WebGL) for volume visualization
- **Bottleneck:** Frangi filter (3D convolution) → implement as GLSL shader

---

## 📐 Algorithm Overview: 4-Phase Pipeline

```
Phase 1: DATA INGESTION & PREPROCESSING
         ├─ Load CTA volume (NIfTI)
         ├─ Resample to isotropic voxels (optional)
         ├─ Compute vesselness map (Frangi filter)
         └─ Build cost map (inverted vesselness)

Phase 2: INITIAL CENTERLINE (FMM)
         ├─ User places start + end points
         ├─ Fast Marching: propagate wave from start
         ├─ Arrival map: time to reach each voxel
         └─ Gradient descent: backtrack to find path

Phase 3: VESSEL WALL SEGMENTATION
         ├─ Resample centerline (equidistant points)
         ├─ Extract cross-sections (MPR)
         ├─ Polar transform (Cartesian → r,θ)
         ├─ Dynamic programming: find lumen + outer wall
         └─ Transform back to 3D coordinates

Phase 4: CENTERLINE REFINEMENT
         ├─ Compute lumen centroids from segmentation
         ├─ Replace initial centerline with centroids
         ├─ Smooth with 3D spline
         └─ Output: Refined centerline + vessel walls
```

---

## 🔬 Phase 1: Data Ingestion & Preprocessing

### Goal
Prepare CTA volume for fast traversal and path finding.

### Step 1.1: Load Volume

```python
import nibabel as nib
import SimpleITK as sitk
import numpy as np

def load_nifti(filepath: str) -> tuple:
    """
    Load NIfTI volume and return as numpy array.
    
    Returns:
        volume: 3D numpy array (float32)
        affine: 4×4 affine matrix (voxel → physical space)
        spacing: [sx, sy, sz] voxel spacing in mm
    """
    # Option 1: nibabel
    img = nib.load(filepath)
    volume = img.get_fdata().astype(np.float32)
    affine = img.affine
    spacing = img.header.get_zooms()[:3]
    
    # Option 2: SimpleITK (alternative)
    # sitk_img = sitk.ReadImage(filepath)
    # volume = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    # spacing = sitk_img.GetSpacing()
    # affine = ... (compute from direction + origin + spacing)
    
    return volume, affine, spacing
```

### Step 1.2: Resample to Isotropic (Optional)

**Why?** If voxels are highly anisotropic (e.g., 0.3×0.3×2.0 mm), the FMM wave propagates unevenly. Resampling to isotropic (~0.5mm) ensures uniform propagation.

```python
from scipy.ndimage import zoom

def resample_isotropic(
    volume: np.ndarray,
    spacing: tuple,
    target_spacing: float = 0.5
) -> tuple:
    """
    Resample volume to isotropic voxels.
    
    Args:
        volume: 3D array
        spacing: [sx, sy, sz] current voxel size
        target_spacing: Desired isotropic spacing
    
    Returns:
        resampled_volume: 3D array
        new_spacing: [target, target, target]
    """
    zoom_factors = [s / target_spacing for s in spacing]
    resampled = zoom(volume, zoom_factors, order=1)  # Trilinear
    new_spacing = (target_spacing, target_spacing, target_spacing)
    return resampled, new_spacing
```

### Step 1.3: Vesselness Filter (Frangi)

**Mathematical Foundation:**

Frangi filter analyzes Hessian matrix eigenvalues at each voxel:
- **Eigenvalues:** λ₁ ≤ λ₂ ≤ λ₃
- **Tube-like structure:** λ₁ ≈ λ₂ < 0 (negative curvature in 2 directions), λ₃ ≈ 0

**Vesselness measure:**
```
V(x) = 0                                    if λ₂ > 0 or λ₃ > 0
     = (1 - exp(-Rᵦ²/2β²)) × exp(-S²/2c²) × (1 - exp(-Rₐ²/2α²))

where:
  Rᵦ = |λ₁| / √(|λ₂| × |λ₃|)    # Blob-like vs tube-like
  Rₐ = |λ₂| / |λ₃|               # Deviation from perfect tube
  S = √(λ₁² + λ₂² + λ₃²)         # Frobenius norm (structure strength)
```

**Implementation:**

```python
from skimage.filters import frangi, hessian

def compute_vesselness_frangi(
    volume: np.ndarray,
    sigmas: list = [1.0, 2.0, 3.0],  # Multi-scale (vessel radii in voxels)
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0
) -> np.ndarray:
    """
    Compute vesselness using Frangi filter.
    
    Args:
        volume: 3D CTA volume
        sigmas: Scales to detect (vessel radii in voxels)
        alpha, beta, gamma: Frangi parameters
    
    Returns:
        vesselness: 3D array (0=background, 1=vessel)
    """
    vesselness = frangi(
        volume,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False  # Vessels are bright in CTA
    )
    return vesselness

# Alternative: Direct Hessian computation
def compute_vesselness_hessian(volume: np.ndarray, sigma: float = 2.0):
    """Manual Hessian eigenvalue computation."""
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    
    H = hessian_matrix(volume, sigma=sigma)
    eigvals = hessian_matrix_eigvals(H)  # Returns sorted [λ₁, λ₂, λ₃]
    
    # Frangi vesselness (simplified)
    l1, l2, l3 = eigvals
    vesselness = np.zeros_like(volume)
    mask = (l2 < 0) & (l3 < 0)  # Tube-like
    vesselness[mask] = np.abs(l2[mask]) * np.abs(l3[mask])
    
    return vesselness
```

**Browser Note:** Frangi filter is computationally expensive. For browser deployment:
- **Pre-compute** vesselness map on server
- **OR** implement as WebGL/GLSL shader (GPU-accelerated)

### Step 1.4: Cost Map Construction

```python
def build_cost_map(vesselness: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Invert vesselness to create cost map for FMM.
    
    High vesselness → Low cost (fast travel)
    Low vesselness → High cost (slow travel)
    
    Args:
        vesselness: 3D vesselness map [0, 1]
        alpha: Scaling factor
    
    Returns:
        cost: 3D cost map (higher = slower propagation)
    """
    cost = np.exp(-alpha * vesselness)
    return cost
```

---

## 🚀 Phase 2: Initial Centerline Extraction (FMM)

### Theoretical Background: Fast Marching Method

**Problem:** Find the path between two points that minimizes "travel time" through the volume.

**Analogy:** 
- Vessels = highways (low cost, fast travel)
- Background = rough terrain (high cost, slow travel)

**FMM Algorithm:**
1. **Initialization:** Place "wave front" at start point
2. **Propagation:** Wave expands outwards
   - Travels fast inside vessels (low cost)
   - Travels slow outside vessels (high cost)
3. **Output:** "Arrival time" map T(x) = time to reach voxel x
4. **Path extraction:** Follow gradient ∇T backwards from end point

**Key Properties:**
- ✅ **Global optimum:** Guaranteed to find shortest path (no local minima)
- ✅ **Efficient:** O(N log N) complexity
- ✅ **Robust:** Doesn't get "stuck" like greedy algorithms

### Mathematical Formulation

**Eikonal Equation:**
```
||∇T(x)|| = F(x)

where:
  T(x) = Arrival time at point x
  F(x) = 1 / Cost(x) = Speed function
```

FMM solves this equation numerically using upwind finite differences.

### Implementation: FMM with scikit-fmm

```python
import skfmm  # scikit-fmm

def fmm_extract_path(
    cost_map: np.ndarray,
    start_point: tuple,  # (x, y, z) in voxel coordinates
    end_point: tuple
) -> np.ndarray:
    """
    Extract path using Fast Marching Method.
    
    Args:
        cost_map: 3D cost map (higher = slower)
        start_point: Starting voxel (x, y, z)
        end_point: Ending voxel (x, y, z)
    
    Returns:
        path: Nx3 array of voxel coordinates along centerline
    """
    # Step 1: Create mask (1 = compute, -1 = start point)
    phi = np.ones_like(cost_map)
    phi[start_point] = -1  # Seed
    
    # Step 2: Compute speed (inverse of cost)
    speed = 1.0 / (cost_map + 1e-10)  # Avoid division by zero
    
    # Step 3: Run Fast Marching
    arrival_time = skfmm.travel_time(phi, speed)
    
    # Step 4: Gradient descent (backtracking)
    path = gradient_descent_backtrack(arrival_time, end_point, start_point)
    
    return path
```

### Gradient Descent Backtracking

```python
from scipy.ndimage import sobel

def gradient_descent_backtrack(
    arrival_time: np.ndarray,
    start: tuple,
    end: tuple,
    step_size: float = 0.5
) -> np.ndarray:
    """
    Backtrack from end to start by following -∇T.
    
    Args:
        arrival_time: T(x) from FMM
        start: Ending position (backtrack FROM here)
        end: Starting position (backtrack TO here)
        step_size: Step size in voxels
    
    Returns:
        path: Nx3 array of positions
    """
    # Compute gradient of arrival time
    grad_x = sobel(arrival_time, axis=0)
    grad_y = sobel(arrival_time, axis=1)
    grad_z = sobel(arrival_time, axis=2)
    
    path = [start]
    current = np.array(start, dtype=float)
    
    max_steps = 10000  # Safety limit
    for _ in range(max_steps):
        # Interpolate gradient at current position
        ix, iy, iz = current.astype(int)
        if not (0 <= ix < arrival_time.shape[0] and
                0 <= iy < arrival_time.shape[1] and
                0 <= iz < arrival_time.shape[2]):
            break
        
        gx = grad_x[ix, iy, iz]
        gy = grad_y[ix, iy, iz]
        gz = grad_z[ix, iy, iz]
        
        # Normalize gradient
        grad_norm = np.sqrt(gx**2 + gy**2 + gz**2)
        if grad_norm < 1e-6:
            break
        
        gx /= grad_norm
        gy /= grad_norm
        gz /= grad_norm
        
        # Step in -∇T direction (downhill)
        current -= step_size * np.array([gx, gy, gz])
        path.append(current.copy())
        
        # Check if reached end
        if np.linalg.norm(current - np.array(end)) < 1.0:
            break
    
    return np.array(path)
```

### Alternative: SimpleITK FastMarchingImageFilter

```python
import SimpleITK as sitk

def fmm_extract_path_sitk(
    cost_map: np.ndarray,
    start_point: tuple,
    end_point: tuple
) -> np.ndarray:
    """SimpleITK-based FMM (alternative to scikit-fmm)."""
    # Convert to SimpleITK image
    sitk_img = sitk.GetImageFromArray(cost_map)
    
    # Create speed image (1 / cost)
    speed_img = sitk.Reciprocal(sitk_img + 1e-10)
    
    # Fast Marching filter
    fm = sitk.FastMarchingImageFilter()
    fm.AddTrialPoint(start_point[::-1])  # ITK uses z,y,x order
    fm.SetStoppingValue(1000.0)
    
    arrival_time_img = fm.Execute(speed_img)
    arrival_time = sitk.GetArrayFromImage(arrival_time_img)
    
    # Backtrack (same as above)
    path = gradient_descent_backtrack(arrival_time, end_point, start_point)
    return path
```

---

## 🔬 Phase 3: Vessel Wall Segmentation

### Goal
For every point on the centerline, segment:
1. **Lumen boundary** (inner wall, blood pool)
2. **Outer wall boundary** (includes plaque)

### Strategy: Polar Transform + Dynamic Programming

For your requirement—segmenting both the inner (lumen) and outer (adventitia) vessel walls along a centerline, specifically handling calcified and soft plaque, without commercial tools—the **Polar Transform + Graph Search (Dynamic Programming)** method is the industry standard "best practice" for robust implementation.

This approach transforms the difficult problem of "finding a circle in 2D" into the much easier problem of "finding a line in a graph."

#### The Concept
Instead of trying to fit a circle to the vessel in the original image slice (which gets messy with plaque), we "unwrap" the vessel image around the centerline.
*   **In Cartesian space:** The vessel walls are closed loops.
*   **In Polar space:** The vessel walls become roughly horizontal lines.
*   **The Goal:** Find the optimal path (line) from the left side of the image to the right.

---

#### Algorithm Overview

**1. Multi-Planar Reformation (MPR)**
*   **Goal:** Isolate the cross-section.
*   **Input:** 3D Volume + Centerline point + Tangent vector.
*   **Action:** Extract a 2D plane perpendicular to the centerline at each point.
*   **Result:** A stack of 2D images where the vessel is (roughly) a circle in the center.

**2. The Polar Transform ("Unwrapping")**
*   **Goal:** Linearize the problem.
*   **Action:** Sample the 2D MPR slice using polar coordinates $(r, \theta)$.
    *   $\theta$ (x-axis): 0 to 360 degrees.
    *   $r$ (y-axis): Distance from center (e.g., 0 to 5mm).
*   **Result:** A rectangular image.
    *   Top of image = Center of vessel.
    *   Bottom of image = Periphery.
    *   **Lumen boundary:** A wavy horizontal line near the top.
    *   **Outer wall:** A wavy horizontal line further down.

**3. Gradient & Cost Function Calculation**
*   **Goal:** define where the "walls" likely are.
You need two different cost maps: one for the Lumen, one for the Outer Wall.

*   **Lumen Cost Map:** Look for the transition from **Bright** (Contrast) $\rightarrow$ **Darker** (Plaque/Wall).
    *   *Simple Gradient:* Directional derivative along the radius.
    *   *Calcification Handling:* If a pixel is "Calcium Bright" (>600 HU), the gradient is misleading. You must add a "penalty" term to prevent the boundary from cutting *through* the calcium. The lumen boundary should wrap *around* the calcium (calcium is usually in the wall).

*   **Outer Wall Cost Map:** Look for the transition from **Gray** (Media/Adventitia) $\rightarrow$ **Dark** (Epicardial Fat).
    *   This gradient is weaker and noisier. It relies heavily on the "smoothness" constraint from the graph search.

**4. Optimal Path Search (Dynamic Programming)**
*   **Goal:** Find the best lines.
You can use **Dijkstra's Algorithm** or a simple **Viterbi (Dynamic Programming)** solver.

*   **Nodes:** Every pixel in the polar image.
*   **Edges:** Connect pixel $(x, y)$ to $(x+1, y)$, $(x+1, y-1)$, and $(x+1, y+1)$.
*   **Weights:** Based on the Cost Map (Low cost = high gradient).
*   **Constraint 1 (Smoothness):** The path cannot jump too many pixels in $r$ (radius) for a single step in $\theta$ (angle).
*   **Constraint 2 (Closed Loop):** The $r$ value at $\theta=0$ must match the $r$ value at $\theta=360$.
*   **Constraint 3 (Nested Surfaces):** The Outer Wall must always have a radius $r_{outer} \ge r_{lumen}$.

#### Why This is the "Best Option"

1.  **Global Optimum:** Unlike "Snake" or "Level Set" algorithms, which iterate and can get stuck on local noise (like a spot of calcium), Dynamic Programming guarantees finding the mathematically *best* path through the slice based on your cost function.
2.  **Plaque Robustness:**
    *   **Soft Plaque:** It appears as a gray zone between the bright lumen and the outer wall. By searching for two distinct surfaces (Inner and Outer), you explicitly capture the plaque burden (area between lines).
    *   **Calcified Plaque:** By converting to polar coordinates, the calcium "bloom" becomes a distinct high-intensity block. You can easily adjust your cost function to "ride the edge" of the high-intensity block rather than getting confused by it.
3.  **Feasibility:** This does not require complex solvers. It relies on basic image resampling and standard graph search, both available in `scipy`, `scikit-image`, and `networkx`.

#### Python Toolkit Strategy

*   **`scipy.ndimage.map_coordinates`**: For creating the MPR and Polar transforms (fast interpolation).
*   **`skimage.filters.sobel`**: For calculating gradients (edges).
*   **`numpy`**: For building the cost accumulation matrix (the Dynamic Programming step).

---

### Step 3.1: Resample Centerline

```python
from scipy.interpolate import splprep, splev

def resample_centerline(
    path: np.ndarray,
    spacing: float = 0.5  # mm
) -> np.ndarray:
    """
    Resample path to equidistant points.
    
    Args:
        path: Nx3 centerline path
        spacing: Desired spacing between points (mm)
    
    Returns:
        resampled_path: Mx3 equidistant points
    """
    # Fit B-spline
    tck, u = splprep(path.T, s=0, k=3)  # Cubic spline
    
    # Compute total arc length
    u_fine = np.linspace(0, 1, 1000)
    path_fine = np.array(splev(u_fine, tck)).T
    dists = np.linalg.norm(np.diff(path_fine, axis=0), axis=1)
    total_length = np.sum(dists)
    
    # Resample at fixed spacing
    num_points = int(total_length / spacing) + 1
    u_resample = np.linspace(0, 1, num_points)
    resampled_path = np.array(splev(u_resample, tck)).T
    
    return resampled_path
```

### Step 3.2: Multi-Planar Reformation (MPR)

Extract cross-sectional slices perpendicular to centerline.

```python
from scipy.ndimage import map_coordinates

def extract_cross_section(
    volume: np.ndarray,
    center_point: np.ndarray,  # [x, y, z]
    tangent: np.ndarray,        # Direction vector
    size: int = 64,             # Pixels
    spacing: float = 0.5        # mm per pixel
) -> np.ndarray:
    """
    Extract perpendicular cross-section at centerline point.
    
    Returns:
        cross_section: 2D image (size × size)
    """
    # Compute orthogonal basis
    normal, binormal = get_perpendicular_vectors(tangent)
    
    # Create sampling grid
    u_coords = np.linspace(-size/2, size/2, size) * spacing
    v_coords = np.linspace(-size/2, size/2, size) * spacing
    U, V = np.meshgrid(u_coords, v_coords)
    
    # 3D positions to sample
    positions = (
        center_point[0] + U * normal[0] + V * binormal[0],
        center_point[1] + U * normal[1] + V * binormal[1],
        center_point[2] + U * normal[2] + V * binormal[2]
    )
    
    # Interpolate (tricubic for smoothness)
    cross_section = map_coordinates(volume, positions, order=3)
    
    return cross_section

def get_perpendicular_vectors(tangent: np.ndarray) -> tuple:
    """Get two orthogonal vectors perpendicular to tangent."""
    # Choose axis least aligned with tangent
    if abs(tangent[0]) < abs(tangent[1]) and abs(tangent[0]) < abs(tangent[2]):
        axis = np.array([1, 0, 0])
    elif abs(tangent[1]) < abs(tangent[2]):
        axis = np.array([0, 1, 0])
    else:
        axis = np.array([0, 0, 1])
    
    # Cross products
    normal = np.cross(tangent, axis)
    normal /= np.linalg.norm(normal)
    
    binormal = np.cross(tangent, normal)
    binormal /= np.linalg.norm(binormal)
    
    return normal, binormal
```

### Step 3.3: Polar Transform

Convert Cartesian cross-section to polar coordinates.

```python
import cv2

def cartesian_to_polar(
    cross_section: np.ndarray,
    num_radii: int = 32,
    num_angles: int = 360
) -> np.ndarray:
    """
    Convert cross-section to polar coordinates (unwrap).
    
    Args:
        cross_section: 2D Cartesian image (size × size)
        num_radii: Number of radius samples
        num_angles: Number of angle samples (0-360°)
    
    Returns:
        polar_image: 2D polar image (num_radii × num_angles)
    """
    center = (cross_section.shape[0] // 2, cross_section.shape[1] // 2)
    max_radius = cross_section.shape[0] // 2
    
    # cv2.linearPolar (if available)
    polar = cv2.linearPolar(
        cross_section,
        center,
        max_radius,
        cv2.WARP_FILL_OUTLIERS
    )
    
    # Resample to desired resolution
    polar_resized = cv2.resize(polar, (num_angles, num_radii))
    
    return polar_resized
```

### Step 3.4: Dynamic Programming for Boundary Detection

**Cost Function Design:**

**Lumen Cost:**
```
C_lumen(r, θ) = w₁ × (1 - Gradient(r,θ)) + w₂ × (1 - Intensity(r,θ))
```
- Want high gradient (edge) and high intensity (contrast-filled lumen)

**Outer Wall Cost:**
```
C_outer(r, θ) = w₃ × (1 - Gradient(r,θ)) + w₄ × Distance_from_lumen(r,θ)
```

**Implementation:**

```python
def segment_boundaries_dp(
    polar_image: np.ndarray,
    gradient_weight: float = 0.7,
    intensity_weight: float = 0.3
) -> tuple:
    """
    Segment lumen and outer wall using dynamic programming.
    
    Args:
        polar_image: (num_radii × num_angles)
    
    Returns:
        lumen_boundary: Array of radii (one per angle)
        outer_boundary: Array of radii (one per angle)
    """
    num_radii, num_angles = polar_image.shape
    
    # Compute gradient magnitude
    grad_r = np.diff(polar_image, axis=0, append=polar_image[-1:, :])
    gradient_mag = np.abs(grad_r)
    
    # Normalize intensity
    intensity_norm = polar_image / (np.max(polar_image) + 1e-6)
    
    # Cost for lumen (prefer bright + high gradient)
    cost_lumen = (
        gradient_weight * (1 - gradient_mag) +
        intensity_weight * (1 - intensity_norm)
    )
    
    # Dynamic programming: Find min-cost path
    lumen_boundary = dp_min_path(cost_lumen)
    
    # Outer wall: start search from lumen + offset
    cost_outer = gradient_mag.copy()  # Simplified: just use gradient
    outer_boundary = dp_min_path_constrained(
        cost_outer,
        min_radius=lumen_boundary + 2  # At least 2 pixels beyond lumen
    )
    
    return lumen_boundary, outer_boundary

def dp_min_path(cost: np.ndarray) -> np.ndarray:
    """
    Find minimum cost path from θ=0 to θ=360 (Viterbi algorithm).
    
    Args:
        cost: (num_radii × num_angles)
    
    Returns:
        path: Array of radii (one per angle)
    """
    num_radii, num_angles = cost.shape
    
    # Cumulative cost matrix
    dp = np.full((num_radii, num_angles), np.inf)
    dp[:, 0] = cost[:, 0]  # Initialize first column
    
    # Forward pass
    for theta in range(1, num_angles):
        for r in range(num_radii):
            # Allow transition from r-1, r, r+1 (smoothness constraint)
            candidates = []
            for r_prev in [r-1, r, r+1]:
                if 0 <= r_prev < num_radii:
                    candidates.append(dp[r_prev, theta-1])
            
            dp[r, theta] = cost[r, theta] + min(candidates)
    
    # Backtrack to find path
    path = np.zeros(num_angles, dtype=int)
    path[-1] = np.argmin(dp[:, -1])
    
    for theta in range(num_angles - 2, -1, -1):
        r_curr = path[theta + 1]
        candidates = [r_curr - 1, r_curr, r_curr + 1]
        candidates = [r for r in candidates if 0 <= r < num_radii]
        
        path[theta] = min(candidates, key=lambda r: dp[r, theta])
    
    return path
```

### Step 3.5: Transform Back to 3D

Convert polar boundaries back to Cartesian 3D coordinates.

```python
def polar_to_cartesian_3d(
    lumen_boundary: np.ndarray,  # Radii per angle
    outer_boundary: np.ndarray,
    center_point: np.ndarray,
    normal: np.ndarray,
    binormal: np.ndarray,
    spacing: float = 0.5
) -> tuple:
    """
    Convert polar boundaries to 3D Cartesian coordinates.
    
    Returns:
        lumen_points: Nx3 array
        outer_points: Nx3 array
    """
    num_angles = len(lumen_boundary)
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    lumen_points = []
    outer_points = []
    
    for theta, r_lumen, r_outer in zip(angles, lumen_boundary, outer_boundary):
        # Polar to Cartesian in 2D plane
        u = r_lumen * spacing * np.cos(theta)
        v = r_lumen * spacing * np.sin(theta)
        
        # 3D position
        p_lumen = center_point + u * normal + v * binormal
        lumen_points.append(p_lumen)
        
        # Same for outer wall
        u_outer = r_outer * spacing * np.cos(theta)
        v_outer = r_outer * spacing * np.sin(theta)
        p_outer = center_point + u_outer * normal + v_outer * binormal
        outer_points.append(p_outer)
    
    return np.array(lumen_points), np.array(outer_points)
```

---

## 🎯 Phase 4: Centerline Refinement

### Problem
FMM centerlines tend to "cut corners" on curved vessels (take inside track like a race car).

### Solution
Use segmented lumen boundaries to compute accurate centroids.

```python
def refine_centerline(
    initial_path: np.ndarray,
    lumen_boundaries: list  # List of Nx3 lumen points per cross-section
) -> np.ndarray:
    """
    Refine centerline using lumen centroids.
    
    Args:
        initial_path: Initial FMM path (Mx3)
        lumen_boundaries: List of M lumen contours (each Nx3)
    
    Returns:
        refined_path: Mx3 refined centerline
    """
    centroids = []
    
    for lumen_points in lumen_boundaries:
        # Compute geometric centroid
        centroid = np.mean(lumen_points, axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Smooth with spline
    from scipy.interpolate import splprep, splev
    tck, u = splprep(centroids.T, s=1.0, k=3)  # Small smoothing
    
    u_fine = np.linspace(0, 1, len(centroids))
    refined_path = np.array(splev(u_fine, tck)).T
    
    return refined_path
```

---

## 🚀 End-to-End Pipeline

```python
# main.py
import numpy as np

def ct_ffr_pipeline(
    nifti_path: str,
    start_point: tuple,  # User click 1 (voxel coords)
    end_point: tuple     # User click 2 (voxel coords)
) -> dict:
    """
    Complete CT-FFR centerline and vessel wall segmentation.
    
    Returns:
        {
            'refined_centerline': Nx3 array,
            'lumen_boundaries': List of Nx3 arrays,
            'outer_boundaries': List of Nx3 arrays
        }
    """
    print("Phase 1: Loading and preprocessing...")
    volume, affine, spacing = load_nifti(nifti_path)
    volume_iso, spacing_iso = resample_isotropic(volume, spacing)
    
    print("Phase 1: Computing vesselness (Frangi filter)...")
    vesselness = compute_vesselness_frangi(volume_iso, sigmas=[1, 2, 3])
    cost_map = build_cost_map(vesselness, alpha=1.0)
    
    print("Phase 2: Extracting initial centerline (FMM)...")
    initial_path = fmm_extract_path(cost_map, start_point, end_point)
    resampled_path = resample_centerline(initial_path, spacing=0.5)
    
    print("Phase 3: Segmenting vessel walls...")
    lumen_boundaries = []
    outer_boundaries = []
    
    for i in range(len(resampled_path)):
        center = resampled_path[i]
        tangent = compute_tangent(resampled_path, i)
        
        # Extract cross-section
        cross_section = extract_cross_section(
            volume_iso, center, tangent, size=64, spacing=0.5
        )
        
        # Polar transform
        polar = cartesian_to_polar(cross_section, num_radii=32, num_angles=360)
        
        # Dynamic programming
        lumen_r, outer_r = segment_boundaries_dp(polar)
        
        # Back to 3D
        normal, binormal = get_perpendicular_vectors(tangent)
        lumen_pts, outer_pts = polar_to_cartesian_3d(
            lumen_r, outer_r, center, normal, binormal, spacing=0.5
        )
        
        lumen_boundaries.append(lumen_pts)
        outer_boundaries.append(outer_pts)
    
    print("Phase 4: Refining centerline...")
    refined_centerline = refine_centerline(resampled_path, lumen_boundaries)
    
    print("Done!")
    return {
        'refined_centerline': refined_centerline,
        'lumen_boundaries': lumen_boundaries,
        'outer_boundaries': outer_boundaries
    }

def compute_tangent(path: np.ndarray, i: int) -> np.ndarray:
    """Compute tangent at point i using central difference."""
    if i == 0:
        tangent = path[1] - path[0]
    elif i == len(path) - 1:
        tangent = path[i] - path[i-1]
    else:
        tangent = path[i+1] - path[i-1]
    
    return tangent / np.linalg.norm(tangent)
```

---

## 📊 Comparison: FMM vs Dijkstra vs Deep Learning

| **Algorithm** | **Pros** | **Cons** | **Best Use Case** |
|---------------|----------|----------|-------------------|
| **Fast Marching (FMM)** | ✅ **Best overall.** Sub-pixel accuracy, smooth paths, guarantees global optimum. | Requires defining a good speed function (Frangi). | **Interactive 2-point tracking** (our use case) |
| **Dijkstra / A*** | Easiest to implement. Works on standard graphs (pixels = nodes). | Discrete (blocky path), requires more memory for large 3D volumes. | Quick prototype without complex math libraries. |
| **Deep Learning (CNN)** | Can be more robust to noise or calcifications (hard plaque). | Overkill for this task. Requires massive training data and GPU. | Fully automated extraction (no user clicks required). |

---

## 🌐 Browser / WASM Feasibility

### Strategy 1: Pyodide (Python in WASM)

**What works:**
- ✅ `numpy`, `scipy` run in Pyodide
- ✅ `scikit-fmm` can be compiled to WASM
- ✅ Volume rendering via NiiVue (WebGL)

**Bottleneck:**
- ⚠️ Frangi filter (3D convolution) is slow in WASM

**Solution:**
- Pre-compute vesselness map on server
- OR implement Frangi as GLSL shader (GPU-accelerated)

### Strategy 2: WebAssembly (C++ Port)

**Port to C++:**
- Use ITK (Insight Segmentation and Registration Toolkit)
- Compile to WASM with Emscripten
- All algorithms available in ITK

**Example:**
```cpp
// C++ ITK implementation
#include "itkFastMarchingImageFilter.h"
#include "itkHessianRecursiveGaussianImageFilter.h"

// Compile with:
// emcc pipeline.cpp -I/path/to/ITK -o pipeline.wasm
```

---

## 🎯 Implementation Advice

### Don't Implement FMM from Scratch

Use existing libraries:
- **Python:** `scikit-fmm` (pure C extension, fast)
- **Python:** `SimpleITK.FastMarchingImageFilter` (ITK wrapper)
- **C++:** ITK directly (`itk::FastMarchingImageFilter`)
- **MATLAB:** `msfm` or built-in `imsegfmm`

### For VMTK Users

**VMTK (Vascular Modeling Toolkit)** has ready-to-use pipeline:
```bash
vmtkcenterlines -ifile input.vti \
                 -seedselector pointlist \
                 -sourcepoints 10 20 30 \
                 -targetpoints 50 60 70 \
                 -ofile centerlines.vtp
```

This implements the same FMM + refinement workflow.

---

## 📚 Related Documentation

- **MEDIS Viewer:** `medis-viewer.md` - Visualization platform for MEDIS contours
- **SAM3D Platform:** `sam3d.md` - AI-driven segmentation (different approach)
- **Funding Proposal:** `proposal.md` - DFG Koselleck grant

---

## 🔬 Mathematical References

**Fast Marching Method:**
- Sethian, J.A. (1996). "A fast marching level set method for monotonically advancing fronts." PNAS 93(4): 1591-1595.
- Kimmel, R., Sethian, J.A. (1998). "Computing geodesic paths on manifolds." PNAS 95(15): 8431-8435.

**Frangi Vesselness:**
- Frangi, A.F., et al. (1998). "Multiscale vessel enhancement filtering." MICCAI 1998.

**Polar Dynamic Programming:**
- Sun, Y., et al. (2003). "Automated 3-D segmentation of lungs with lung cancer in CT data using a novel robust active shape model approach." IEEE TMI 2012.

---

## ⚙️ Performance Targets

**Python Prototype:**
- Vesselness filter: 10-30s (depends on volume size)
- FMM propagation: 2-5s
- Single cross-section: <100ms
- Full segmentation (200 slices): ~20-60s

**Optimized (C++ / WASM):**
- Vesselness filter: 1-3s (with GPU shader: <500ms)
- FMM propagation: <1s
- Full pipeline: <10s

---

## 🗂️ Phase 5: 3D Mesh Export & Visualization

### The Coordinate System Problem

When exporting vessel geometry for web visualization (Niivue), **format choice is critical** for spatial alignment.

**Your segmentation data contains absolute scanner coordinates** (e.g., $z \approx 1967$ mm). The export format must preserve this spatial reference.

### Format Comparison: STL vs GIfTI vs MZ3

| **Format** | **Coordinates** | **Metadata** | **Niivue Support** | **Use Case** |
|------------|-----------------|--------------|--------------------|--------------|
| **STL** | ❌ No header for world coords | ❌ None | ⚠️ May float "beside" volume | Quick prototyping only |
| **GIfTI (.gii)** | ✅ Preserves RAS/LPS | ✅ XML metadata | ✅ Native alignment | **Recommended for medical** |
| **MZ3** | ✅ Binary coords | ✅ Layer attributes | ✅ High performance | Large meshes, color mapping |

### Why STL is Problematic for Medical Imaging

```
Problem: STL files store raw vertex coordinates without:
  1. Origin reference (where is 0,0,0?)
  2. Coordinate system (RAS vs LPS vs scanner coords)
  3. Affine transformation matrix

Result: Mesh "floats" next to the CTA instead of overlaying correctly.
```

### Why GIfTI is the Correct Choice

**GIfTI (Geometry format for GIFTI)** is the "NIfTI for surfaces":

1. **Coordinate Preservation:** Stores vertices in world coordinates (same as your CTA NIfTI)
2. **Metadata Support:** Embeds StudyInstanceUID, VesselName, etc. in XML header
3. **Niivue Native:** Automatically aligns with loaded NIfTI volumes
4. **Dual Array Structure:**
   - `NIFTI_INTENT_POINTSET`: Vertex positions (Nx3 float32)
   - `NIFTI_INTENT_TRIANGLE`: Face indices (Mx3 int32)

### The Lofting Algorithm: Contours → Tube Mesh

**Problem:** Segmentation output is a series of 2D contours (closed rings of points per slice).

**Solution:** "Lofting" connects adjacent contours into a continuous 3D surface.

```
Slice n:     ●─●─●─●─●  (ring of points)
             │╲│╲│╲│╲│  ← triangles connect to next slice
Slice n+1:   ●─●─●─●─●  (ring of points)
```

**Triangulation Logic:**
```
For each contour pair (slice n, slice n+1):
  For each point index i:
    p1 = slice[n][i]
    p2 = slice[n][(i+1) % points_per_contour]  # wrap around
    p3 = slice[n+1][i]
    p4 = slice[n+1][(i+1) % points_per_contour]
    
    # Two triangles form a quad
    face1 = [p1, p2, p3]  # CCW winding for correct normals
    face2 = [p2, p4, p3]
```

**⚠️ Face Winding Order:** Counter-clockwise (CCW) winding is critical. WebGL uses backface culling—wrong winding makes triangles invisible from one side.

---

## 🐍 Phase 6: Python Conversion Pipeline (TXT → GIfTI)

### Overview

This pipeline converts MEDIS-format contour text files into GIfTI meshes for Niivue visualization.

**Dependencies:**
```bash
pip install nibabel numpy SimpleITK scipy
```

### ⚠️ Critical: Handling Varying Point Counts Per Contour

**Problem:** MEDIS export files contain cross-section rings with **varying numbers of points**. For example:
- Contour 0 (Lumen): 50 points
- Contour 1 (VesselWall): 49 points  
- Contour 2 (Lumen): 55 points

**Why This Matters:**
1. **Naive meshing fails:** If Ring A has 50 points and Ring B has 49, connecting them 1:1 drops data
2. **Twist artifacts:** Arbitrary start points per ring cause mesh "twisting"
3. **Non-watertight meshes:** Gaps appear where point counts differ

**Solution (Implemented in `code/buildstl.py` and conversion scripts):**

1. **Resample all contours** to a uniform point count using **Cubic Spline interpolation**
2. **Align start points** between consecutive rings (closest-neighbor matching)
3. **Cap ends** to create water-tight solids for CFD

```python
from scipy.interpolate import CubicSpline

def resample_polygon(polygon: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resample a 3D polygon (closed loop) to exactly n_points.
    Uses Cubic Spline interpolation for smooth results (CFD-grade).
    """
    closed = np.vstack((polygon, polygon[0]))  # Close the loop
    
    # Cumulative arc length
    diffs = np.diff(closed, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    perimeter = cum_dists[-1]
    
    # Cubic spline with periodic boundary
    cs_x = CubicSpline(cum_dists, closed[:, 0], bc_type='periodic')
    cs_y = CubicSpline(cum_dists, closed[:, 1], bc_type='periodic')
    cs_z = CubicSpline(cum_dists, closed[:, 2], bc_type='periodic')
    
    new_dists = np.linspace(0, perimeter, n_points + 1)[:-1]
    return np.column_stack((cs_x(new_dists), cs_y(new_dists), cs_z(new_dists)))


def align_contours(contours: list) -> list:
    """
    Rotational alignment to prevent twisting.
    Each ring's start point aligns to closest neighbor in previous ring.
    """
    if not contours:
        return []
    aligned = [contours[0]]
    for i in range(len(contours) - 1):
        prev = aligned[i]
        curr = contours[i + 1]
        distances = np.linalg.norm(curr - prev[0], axis=1)
        start_idx = np.argmin(distances)
        aligned.append(np.roll(curr, -start_idx, axis=0))
    return aligned
```

**File Naming Convention:**
```
{patient_id}_{vessel}_{inner|outer}.{gii|json}

Examples:
- 01-BER-0088_LAD_inner.gii   (Lumen mesh)
- 01-BER-0088_LAD_outer.gii   (VesselWall mesh)
- 01-BER-0088_RCA_inner.json  (Lumen point cloud)
```

### Complete Conversion Script

```python
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray, GiftiMetaData
import SimpleITK as sitk
import os
import re

def parse_medis_contour_file(filepath: str) -> tuple:
    """
    Parse MEDIS-format contour file with Lumen and VesselWall groups.
    
    Returns:
        segments: {'Lumen': [...], 'VesselWall': [...]}
        metadata: {'vessel_name': 'lad', 'patient_id': '...', ...}
    """
    segments = {'Lumen': [], 'VesselWall': []}
    metadata = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    current_group = None
    points_expected = 0
    points_collected = 0
    
    for line in lines:
        line = line.strip()
        
        # Parse metadata headers
        if line.startswith('# vessel_name'):
            metadata['vessel_name'] = line.split(':')[1].strip()
        elif line.startswith('# patient_id'):
            metadata['patient_id'] = line.split(':')[1].strip()
        elif line.startswith('# study_description'):
            metadata['study_description'] = line.split(':')[1].strip()
        elif line.startswith('# group:'):
            current_group = line.split(':')[1].strip()
            points_collected = 0
        elif line.startswith('# Number of points:'):
            points_expected = int(line.split(':')[1].strip())
        elif line.startswith('# SliceDistance:'):
            metadata['slice_distance'] = float(line.split(':')[1].strip())
        
        # Parse coordinate data
        elif line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    point = [float(parts[0]), float(parts[1]), float(parts[2])]
                    if current_group in segments:
                        segments[current_group].append(point)
                        points_collected += 1
                except ValueError:
                    continue
    
    return segments, metadata


def create_lofted_mesh(points: list, points_per_contour: int = 40) -> tuple:
    """
    Create triangulated mesh from stacked contours via lofting.
    
    Args:
        points: Flat list of 3D points (all contours concatenated)
        points_per_contour: Number of points per ring (from file metadata)
    
    Returns:
        vertices: Nx3 float32 array
        faces: Mx3 int32 array (triangle indices)
    """
    if len(points) < points_per_contour * 2:
        raise ValueError(f"Need at least 2 contours, got {len(points)} points")
    
    num_contours = len(points) // points_per_contour
    vertices = np.array(points, dtype=np.float32)
    faces = []
    
    for c in range(num_contours - 1):
        for p in range(points_per_contour):
            # Current contour indices
            p1 = c * points_per_contour + p
            p2 = c * points_per_contour + (p + 1) % points_per_contour
            
            # Next contour indices
            p3 = (c + 1) * points_per_contour + p
            p4 = (c + 1) * points_per_contour + (p + 1) % points_per_contour
            
            # Two triangles per quad (CCW winding for correct normals)
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
    
    return vertices, np.array(faces, dtype=np.int32)


def convert_contours_to_gifti(
    txt_path: str,
    output_dir: str,
    nifti_reference_path: str = None,
    points_per_contour: int = 40
) -> list:
    """
    Convert MEDIS contour file to GIfTI mesh files.
    
    Creates separate .gii files for Lumen and VesselWall.
    
    Args:
        txt_path: Path to MEDIS contour .txt file
        output_dir: Directory for output .gii files
        nifti_reference_path: Optional CTA volume for StudyInstanceUID
        points_per_contour: Points per ring (check file header)
    
    Returns:
        List of created .gii file paths
    """
    # Parse input file
    segments, txt_meta = parse_medis_contour_file(txt_path)
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    
    # Get reference metadata if available
    study_uid = "Unknown"
    if nifti_reference_path and os.path.exists(nifti_reference_path):
        try:
            ref_img = sitk.ReadImage(nifti_reference_path)
            if ref_img.HasMetaDataKey("0020|000d"):
                study_uid = ref_img.GetMetaData("0020|000d")
        except Exception as e:
            print(f"Warning: Could not read NIfTI metadata: {e}")
    
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    
    for group_name, points in segments.items():
        if not points or len(points) < points_per_contour * 2:
            print(f"Skipping {group_name}: insufficient points")
            continue
        
        try:
            vertices, faces = create_lofted_mesh(points, points_per_contour)
        except ValueError as e:
            print(f"Skipping {group_name}: {e}")
            continue
        
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
        
        # Embed metadata in GIfTI header
        gii_metadata = GiftiMetaData({
            'StudyInstanceUID': study_uid,
            'VesselName': txt_meta.get('vessel_name', 'Unknown'),
            'PatientID': txt_meta.get('patient_id', 'Unknown'),
            'Group': group_name,
            'SliceDistance': str(txt_meta.get('slice_distance', 0.25)),
            'PointsPerContour': str(points_per_contour),
            'ConvertedBy': 'VesselSegmenter-Pipeline'
        })
        
        # Create and save GIfTI image
        gii = GiftiImage(darrays=[vertex_array, face_array], meta=gii_metadata)
        
        output_path = os.path.join(output_dir, f"{base_name}_{group_name}.gii")
        nib.save(gii, output_path)
        output_files.append(output_path)
        print(f"✓ Exported: {output_path} ({len(vertices)} vertices, {len(faces)} faces)")
    
    return output_files


# Usage Example
if __name__ == "__main__":
    convert_contours_to_gifti(
        txt_path='01-BER-0088_ecrf_lad.txt',
        output_dir='./dist/meshes',
        nifti_reference_path='cta_volume.nii.gz',
        points_per_contour=40
    )
```

### Coordinate System Conventions

**Critical:** Ensure your contour coordinates match the CTA's coordinate system:

| **System** | **X** | **Y** | **Z** | **Used By** |
|------------|-------|-------|-------|-------------|
| **RAS** | Right→Left | Anterior→Posterior | Inferior→Superior | NIfTI, Niivue |
| **LPS** | Left→Right | Posterior→Anterior | Inferior→Superior | DICOM, ITK |
| **Scanner** | Varies | Varies | Table position | Raw DICOM |

**If mesh appears mirrored or offset:** Check if coordinate transformation is needed:
```python
# LPS to RAS conversion (flip X and Y)
vertices_ras = vertices.copy()
vertices_ras[:, 0] *= -1  # Flip X
vertices_ras[:, 1] *= -1  # Flip Y
```

---

## 🔀 Phase 7: Dual Representation Strategy (Mesh + JSON)

### Architectural Principle: Single Source of Truth, Multiple Representations

For professional medical visualization, maintain **two complementary representations**:

| **Representation** | **Format** | **Purpose** | **Precision** |
|--------------------|------------|-------------|---------------|
| **Geometric Mesh** | GIfTI (.gii) | Smooth 3D surface visualization | Interpolated (lofted) |
| **Point Cloud** | JSON (Connectome) | Raw measurement data | Ground truth |

### Why Both Are Needed

1. **Analytical vs Visual Precision:**
   - Mesh: Beautiful for 3D rendering, but lofting interpolates between measurements
   - JSON: Shows exact measurement points without interpolation

2. **Debugging:**
   - If mesh looks "twisted", JSON helps identify if error is in raw data or triangulation

3. **Interactive Features:**
   - Click individual points to show HU values
   - Measure diameters between nodes
   - Dynamic coloring based on stenosis grade

4. **Performance:**
   - JSON loads instantly (show skeleton while heavy mesh loads)
   - Can render thousands of points with per-point styling

### Niivue Connectome JSON Format

Niivue uses a specific JSON structure for point/line visualization:

```json
{
  "nodes": {
    "x": [6.15, 6.10, 6.03, ...],
    "y": [-0.09, -0.22, -0.35, ...],
    "z": [1974.59, 1974.60, 1974.61, ...],
    "colorValue": [1, 1, 1, ...],
    "sizeValue": [1, 1, 1, ...]
  },
  "edges": {
    "first": [0, 1, 2, ...],
    "second": [1, 2, 3, ...]
  },
  "metadata": {
    "vessel_type": "Lumen",
    "source": "01-BER-0088_lad.txt"
  }
}
```

### Python Script: TXT → Niivue JSON

```python
import json
import os
import numpy as np

def convert_contours_to_niivue_json(
    txt_path: str,
    output_json: str,
    points_per_ring: int = 40
) -> None:
    """
    Convert MEDIS contours to Niivue Connectome JSON format.
    
    Creates nodes (points) and edges (lines connecting ring points).
    """
    # Parse points (reuse parse function from above)
    points = []
    metadata = {}
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    current_group = None
    for line in lines:
        line = line.strip()
        if line.startswith('# vessel_name'):
            metadata['vessel_name'] = line.split(':')[1].strip()
        elif line.startswith('# group:'):
            current_group = line.split(':')[1].strip()
        elif line.startswith('# Number of points:'):
            points_per_ring = int(line.split(':')[1].strip())
        elif line and not line.startswith('#') and current_group == 'Lumen':
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    if not points:
        raise ValueError("No Lumen points found in file")
    
    # Build Niivue Connectome structure
    nodes = {
        "x": [p[0] for p in points],
        "y": [p[1] for p in points],
        "z": [p[2] for p in points],
        "colorValue": [1.0] * len(points),  # Uniform color
        "sizeValue": [1.0] * len(points)    # Uniform size
    }
    
    # Create edges to form closed rings
    edges = {"first": [], "second": []}
    num_rings = len(points) // points_per_ring
    
    for r in range(num_rings):
        offset = r * points_per_ring
        for i in range(points_per_ring):
            p1 = offset + i
            p2 = offset + ((i + 1) % points_per_ring)  # Wrap to close ring
            edges["first"].append(p1)
            edges["second"].append(p2)
    
    # Optionally connect rings along vessel axis
    for r in range(num_rings - 1):
        for i in range(0, points_per_ring, 4):  # Every 4th point
            p1 = r * points_per_ring + i
            p2 = (r + 1) * points_per_ring + i
            edges["first"].append(p1)
            edges["second"].append(p2)
    
    data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "vessel_type": "Lumen",
            "vessel_name": metadata.get('vessel_name', 'Unknown'),
            "source": os.path.basename(txt_path),
            "num_rings": num_rings,
            "points_per_ring": points_per_ring
        }
    }
    
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ JSON point cloud created: {output_json}")
    print(f"  Nodes: {len(points)}, Edges: {len(edges['first'])}")


# Usage
if __name__ == "__main__":
    convert_contours_to_niivue_json(
        '01-BER-0088_lad.txt',
        'lad_points.json',
        points_per_ring=40
    )
```

### Visual Layering Strategy

In Niivue, combine both representations:

```
Layer 1: CTA Volume (base)           - 100% opacity
Layer 2: GIfTI Mesh (vessel wall)    - 30% opacity, gray
Layer 3: GIfTI Mesh (lumen)          - 50% opacity, red  
Layer 4: JSON Points (measurements)  - 100% opacity, bright dots
```

This shows smooth surface AND exact measurement locations simultaneously.

---

## 📐 Phase 8: Straightened Curved Planar Reconstruction (sCPR)

### Conceptual Overview: The "Garden Hose" Analogy

Imagine a **tangled garden hose** (the coronary artery). To inspect for cracks (stenosis, calcifications), looking at standard axial CT slices is difficult—the vessel curves in and out of every slice.

**sCPR mathematically "pulls the hose straight":**
- Transform from global Cartesian $(x, y, z)$ to vessel-centric $(s, u, v)$
- $s$ = distance along centerline ("length of hose")
- $(u, v)$ = cross-sectional coordinates ("looking down the hose")

**Result:** A new 3D volume where clinicians scroll along the vessel from ostium to distal end as if it were a straight pipe.

### Mathematical Foundation

#### A. Local Coordinate Frame at Each Centerline Point

For each centerline point $P_i$, we need an orthonormal basis $(\vec{T}_i, \vec{N}_i, \vec{B}_i)$:

| **Vector** | **Name** | **Direction** |
|------------|----------|---------------|
| $\vec{T}$ | Tangent | Along vessel ("forward") |
| $\vec{N}$ | Normal | Perpendicular ("up" in cross-section) |
| $\vec{B}$ | Binormal | Perpendicular ("right" in cross-section) |

#### B. Calculating the Tangent Vector $\vec{T}$

Use **central difference** for smooth tangent estimation:

$$\vec{T}_i = \text{normalize}(P_{i+1} - P_{i-1})$$

**Boundary conditions:**
- First point ($i=0$): Forward difference $\vec{T}_0 = \text{normalize}(P_1 - P_0)$
- Last point ($i=N-1$): Backward difference $\vec{T}_{N-1} = \text{normalize}(P_{N-1} - P_{N-2})$

```python
def compute_tangent_vectors(centerline: np.ndarray) -> np.ndarray:
    """
    Compute tangent vectors using central differences.
    
    Args:
        centerline: Nx3 array of centerline points
    
    Returns:
        tangents: Nx3 array of unit tangent vectors
    """
    n = len(centerline)
    tangents = np.zeros_like(centerline)
    
    # Central difference for interior points
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    
    # Forward difference for first point
    tangents[0] = centerline[1] - centerline[0]
    
    # Backward difference for last point
    tangents[-1] = centerline[-1] - centerline[-2]
    
    # Normalize
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-10)
    
    return tangents
```

#### C. The Twist Problem: Frenet-Serret vs Rotation Minimizing Frame

**Problem with Frenet-Serret frame:**
The standard Frenet frame uses curvature to define $\vec{N}$. At inflection points (where curvature changes sign), the frame **flips 180°**, causing spiral artifacts.

**Solution: Rotation Minimizing Frame (RMF) / Bishop Frame**

The RMF propagates the normal vector smoothly by projecting it onto each successive perpendicular plane:

$$\vec{N}_i = \text{normalize}\left(\vec{N}_{i-1} - (\vec{N}_{i-1} \cdot \vec{T}_i)\vec{T}_i\right)$$

$$\vec{B}_i = \vec{T}_i \times \vec{N}_i$$

```python
def compute_rotation_minimizing_frame(tangents: np.ndarray) -> tuple:
    """
    Compute Rotation Minimizing Frame (Bishop Frame).
    
    Avoids the "twist" problem of Frenet-Serret frames.
    
    Args:
        tangents: Nx3 unit tangent vectors
    
    Returns:
        normals: Nx3 unit normal vectors
        binormals: Nx3 unit binormal vectors
    """
    n = len(tangents)
    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)
    
    # Initialize: pick arbitrary vector perpendicular to first tangent
    T0 = tangents[0]
    
    # Choose axis least aligned with tangent
    if abs(T0[0]) <= abs(T0[1]) and abs(T0[0]) <= abs(T0[2]):
        ref = np.array([1.0, 0.0, 0.0])
    elif abs(T0[1]) <= abs(T0[2]):
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])
    
    # Initial normal via cross product
    normals[0] = np.cross(T0, ref)
    normals[0] /= np.linalg.norm(normals[0])
    binormals[0] = np.cross(T0, normals[0])
    
    # Propagate frame along centerline
    for i in range(1, n):
        Ti = tangents[i]
        Ni_prev = normals[i - 1]
        
        # Project previous normal onto plane perpendicular to current tangent
        Ni = Ni_prev - np.dot(Ni_prev, Ti) * Ti
        norm = np.linalg.norm(Ni)
        
        if norm < 1e-10:
            # Degenerate case: use previous normal
            Ni = Ni_prev
        else:
            Ni = Ni / norm
        
        normals[i] = Ni
        binormals[i] = np.cross(Ti, Ni)
    
    return normals, binormals
```

#### D. The Resampling Mapping

For pixel $(u, v)$ in cross-section $i$, the corresponding world coordinate is:

$$W_{i,u,v} = P_i + (u \cdot \Delta_{res} \cdot \vec{N}_i) + (v \cdot \Delta_{res} \cdot \vec{B}_i)$$

**Parameters:**
- $P_i$: Centerline point (center of cross-section, $u=0, v=0$)
- $\Delta_{res}$: Pixel resolution (typically 0.1–0.2 mm)
- Slice spacing: `SliceDistance` from your data (e.g., 0.25 mm)

```python
def compute_scpr_coordinates(
    centerline: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    slice_size: int = 256,
    pixel_resolution: float = 0.1  # mm per pixel
) -> np.ndarray:
    """
    Compute world coordinates for sCPR volume.
    
    Args:
        centerline: Nx3 centerline points
        normals: Nx3 normal vectors (from RMF)
        binormals: Nx3 binormal vectors (from RMF)
        slice_size: Cross-section size in pixels (e.g., 256x256)
        pixel_resolution: Physical size of each pixel in mm
    
    Returns:
        world_coords: (N, slice_size, slice_size, 3) array of world coordinates
    """
    n_slices = len(centerline)
    half_size = slice_size // 2
    
    # Create UV grid (centered at 0)
    u_coords = np.arange(-half_size, half_size) * pixel_resolution
    v_coords = np.arange(-half_size, half_size) * pixel_resolution
    U, V = np.meshgrid(u_coords, v_coords, indexing='xy')
    
    # Output array: (n_slices, height, width, 3)
    world_coords = np.zeros((n_slices, slice_size, slice_size, 3), dtype=np.float32)
    
    for i in range(n_slices):
        P = centerline[i]
        N = normals[i]
        B = binormals[i]
        
        # W = P + u*N + v*B
        for axis in range(3):
            world_coords[i, :, :, axis] = (
                P[axis] + 
                U * N[axis] + 
                V * B[axis]
            )
    
    return world_coords
```

### Complete sCPR Pipeline

```python
from scipy.ndimage import map_coordinates

def create_straightened_cpr(
    volume: np.ndarray,
    spacing: tuple,
    centerline: np.ndarray,
    slice_size: int = 256,
    pixel_resolution: float = 0.1
) -> np.ndarray:
    """
    Create Straightened Curved Planar Reconstruction.
    
    Args:
        volume: 3D CTA volume
        spacing: (sx, sy, sz) voxel spacing in mm
        centerline: Nx3 centerline in world coordinates
        slice_size: Output cross-section size
        pixel_resolution: Output pixel size in mm
    
    Returns:
        scpr_volume: (slice_size, slice_size, N) straightened volume
    """
    # Step 1: Compute frame
    tangents = compute_tangent_vectors(centerline)
    normals, binormals = compute_rotation_minimizing_frame(tangents)
    
    # Step 2: Compute world coordinates for each pixel
    world_coords = compute_scpr_coordinates(
        centerline, normals, binormals,
        slice_size, pixel_resolution
    )
    
    # Step 3: Convert world coords to voxel coords
    voxel_coords = world_coords.copy()
    for axis in range(3):
        voxel_coords[..., axis] /= spacing[axis]
    
    # Step 4: Resample via trilinear interpolation
    n_slices = len(centerline)
    scpr_volume = np.zeros((slice_size, slice_size, n_slices), dtype=np.float32)
    
    for i in range(n_slices):
        coords = [
            voxel_coords[i, :, :, 0].ravel(),
            voxel_coords[i, :, :, 1].ravel(),
            voxel_coords[i, :, :, 2].ravel()
        ]
        
        sampled = map_coordinates(
            volume, coords,
            order=1,  # Trilinear
            mode='constant',
            cval=0
        )
        
        scpr_volume[:, :, i] = sampled.reshape(slice_size, slice_size)
    
    return scpr_volume
```

### Slice Spacing vs Thickness

**Question:** "Is slice thickness = 2 × SliceDistance?"

**Answer:** No. In a resampled volume:
- **Slice Spacing (Z-dimension):** Equals `SliceDistance` (e.g., 0.25 mm)
- **Voxel Thickness:** In digital volumes, voxels are point samples (infinitely thin)
- **Slab MIP:** If doing Maximum Intensity Projection slabs, use 0.5–1.0 mm thickness

For standard sCPR, set output NIfTI `pixdim[3] = SliceDistance`.

### Orthogonal Slicing: Longitudinal Reconstruction

Once the sCPR volume is generated, you are **not limited to cross-sectional views**. The straightened volume is a standard 3D rectilinear grid that supports orthogonal slicing along any axis.

#### Understanding the sCPR Volume Axes

| **Axis** | **Name** | **Physical Meaning** | **Typical Size** |
|----------|----------|---------------------|------------------|
| **S** | Longitudinal | Distance along vessel centerline | N slices × SliceDistance |
| **U** | Horizontal | Cross-sectional width | slice_size × pixel_resolution |
| **V** | Vertical | Cross-sectional height | slice_size × pixel_resolution |

#### Three Clinical Views from sCPR

| **View** | **Plane** | **What You See** | **Clinical Use** |
|----------|-----------|------------------|------------------|
| **Cross-Sectional** | $uv$ | "Donut" view of vessel | Lumen area, eccentric plaque detection |
| **Longitudinal** | $su$ or $sv$ | Vessel "cut in half" lengthwise | Lesion length, diameter step-downs |
| **MIP Projection** | $s$-projection | All calcifications in one image | Calcium scoring along vessel |

#### The Longitudinal View: Clinical Gold Standard

The **longitudinal view** ($su$ or $sv$ plane) is often the most clinically valuable:

```
Cross-Sectional (uv):        Longitudinal (su):
                             
    ┌─────────┐              S ↑  ════════════════
    │  ○○○○○  │                │  ║  Lumen      ║
    │ ○     ○ │                │  ║    ████     ║ ← Stenosis
    │ ○  ●  ○ │  ← Lumen       │  ║  Lumen      ║
    │ ○     ○ │                │  ════════════════
    │  ○○○○○  │                └──────────────────→ U
    └─────────┘                    
    
    Single slice view          Entire vessel length
```

**Advantages:**
- Vessel appears as a **straight pipe** (even if original was tortuous)
- **Centerline centering** ($u=0, v=0$) keeps vessel in frame center
- Easy to measure **stenosis length** in mm
- Visualize **plaque tapering** over distance

**Example:** With `SliceDistance = 0.25 mm` and 50 centerline points:
- Longitudinal view spans: $50 \times 0.25 = 12.5$ mm of vessel
- Resolution along S-axis: 0.25 mm/pixel

#### Niivue Integration for Multiplanar Views

```typescript
// Load sCPR volume and display all three orthogonal views
async function displaySCPRMultiplanar(nv: Niivue, scprVolume: NVImage): Promise<void> {
  await nv.addVolume(scprVolume);
  
  // Enable multiplanar view (shows XY, XZ, YZ simultaneously)
  // In sCPR context: XY = uv (cross-section), XZ = su, YZ = sv (longitudinal)
  nv.setSliceType(nv.sliceTypeMultiplanar);
  
  // Optional: Set crosshair to vessel center
  nv.scene.crosshairPos = [0.5, 0.5, 0.5];  // Center of volume
}
```

### Coordinate Transformations: Forward and Inverse

The sCPR system requires **bidirectional mapping** between coordinate systems:
- **Forward:** $(s, u, v) \to (x, y, z)$ — Used for volume resampling
- **Inverse:** $(x, y, z) \to (s, u, v)$ — Used for mapping findings back to sCPR

#### Forward Transformation: $(s, u, v) \to (x, y, z)$

This is the **Resampling Mapping** used to construct the straightened volume.

**Mathematical Form:**
$$\vec{W}(x, y, z) = P(s) + u \cdot \vec{N}(s) + v \cdot \vec{B}(s)$$

**Matrix Form** (for a specific slice at $s$):
$$\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} N_x & B_x \\ N_y & B_y \\ N_z & B_z \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} + \begin{bmatrix} P_x(s) \\ P_y(s) \\ P_z(s) \end{bmatrix}$$

**Note:** The tangent $\vec{T}(s)$ is not used in position calculation because $u, v$ exist strictly within the plane perpendicular to $\vec{T}$.

```python
def forward_transform(
    s_index: int,
    u: float,
    v: float,
    centerline: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray
) -> np.ndarray:
    """
    Transform from sCPR coordinates to world coordinates.
    
    Args:
        s_index: Index along centerline (discrete s coordinate)
        u: Lateral offset in normal direction (mm)
        v: Lateral offset in binormal direction (mm)
        centerline: Nx3 centerline points
        normals: Nx3 normal vectors
        binormals: Nx3 binormal vectors
    
    Returns:
        world_coord: [x, y, z] in world space
    """
    P = centerline[s_index]
    N = normals[s_index]
    B = binormals[s_index]
    
    return P + u * N + v * B
```

#### Inverse Transformation: $(x, y, z) \to (s, u, v)$

This is the **Projection Mapping** used to locate CTA findings in the straightened view.

**Step A: Find $s$ (Longitudinal Coordinate)**

The $s$ coordinate minimizes the distance to the centerline:
$$s = \text{argmin}_s \|\vec{W} - P(s)\|$$

For discrete centerlines, find the nearest point index. For sub-voxel precision, project onto the line segment between adjacent points.

**Step B: Find $u$ and $v$ (Cross-Sectional Coordinates)**

Compute the displacement vector and project onto local axes:
$$\vec{D} = \vec{W} - P(s)$$
$$u = \vec{D} \cdot \vec{N}(s)$$
$$v = \vec{D} \cdot \vec{B}(s)$$

```python
def inverse_transform(
    world_point: np.ndarray,
    centerline: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    slice_distance: float = 0.25
) -> tuple:
    """
    Transform from world coordinates to sCPR coordinates.
    
    Args:
        world_point: [x, y, z] in world space
        centerline: Nx3 centerline points
        normals: Nx3 normal vectors
        binormals: Nx3 binormal vectors
        slice_distance: Physical distance between slices (mm)
    
    Returns:
        s: Longitudinal coordinate (mm along vessel)
        u: Cross-sectional coordinate (mm)
        v: Cross-sectional coordinate (mm)
        s_index: Nearest centerline index
    """
    # Step A: Find nearest centerline point
    distances = np.linalg.norm(centerline - world_point, axis=1)
    s_index = np.argmin(distances)
    
    # Optional: Refine with projection onto line segment
    s_index_refined, t = refine_projection(
        world_point, centerline, s_index
    )
    
    # Step B: Project displacement onto local frame
    P = centerline[s_index]
    N = normals[s_index]
    B = binormals[s_index]
    
    D = world_point - P
    u = np.dot(D, N)
    v = np.dot(D, B)
    
    # Convert index to physical distance
    s = s_index * slice_distance
    
    return s, u, v, s_index


def refine_projection(
    point: np.ndarray,
    centerline: np.ndarray,
    nearest_idx: int
) -> tuple:
    """
    Refine s coordinate by projecting onto line segment.
    
    Returns:
        refined_idx: Float index (e.g., 5.3 means between points 5 and 6)
        t: Interpolation parameter [0, 1]
    """
    n = len(centerline)
    
    # Check both adjacent segments
    best_dist = float('inf')
    best_idx = nearest_idx
    best_t = 0.0
    
    for idx in [nearest_idx - 1, nearest_idx]:
        if idx < 0 or idx >= n - 1:
            continue
        
        A = centerline[idx]
        B = centerline[idx + 1]
        AB = B - A
        AP = point - A
        
        # Project onto segment
        t = np.clip(np.dot(AP, AB) / (np.dot(AB, AB) + 1e-10), 0, 1)
        proj = A + t * AB
        dist = np.linalg.norm(point - proj)
        
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
            best_t = t
    
    return best_idx + best_t, best_t
```

#### Transformation Summary Table

| **Direction** | **Operation** | **Mathematical Tool** | **Use Case** |
|---------------|---------------|----------------------|--------------|
| $(s, u, v) \to (x, y, z)$ | Vector Addition | Linear combination of basis vectors | Volume resampling |
| $(x, y, z) \to s$ | Optimization | Nearest neighbor / distance minimization | Find slice index |
| $(x, y, z) \to (u, v)$ | Projection | Dot product with $\vec{N}, \vec{B}$ | Find position in slice |

#### Research Application: Plaque Burden Mapping

The inverse transformation enables **quantitative plaque analysis**:

```python
def map_plaque_to_scpr(
    plaque_mask: np.ndarray,
    volume_spacing: tuple,
    centerline: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    slice_distance: float = 0.25
) -> np.ndarray:
    """
    Map plaque segmentation mask to sCPR coordinates.
    
    Returns plaque burden per mm of vessel length.
    """
    # Find all plaque voxels
    plaque_coords = np.argwhere(plaque_mask > 0)
    
    # Convert to world coordinates
    world_coords = plaque_coords * np.array(volume_spacing)
    
    # Map each voxel to sCPR space
    n_slices = len(centerline)
    plaque_per_slice = np.zeros(n_slices)
    
    for world_point in world_coords:
        s, u, v, s_idx = inverse_transform(
            world_point, centerline, normals, binormals, slice_distance
        )
        
        # Accumulate plaque area at this slice
        voxel_area = volume_spacing[0] * volume_spacing[1]  # mm²
        plaque_per_slice[s_idx] += voxel_area
    
    # Convert to plaque burden per mm of vessel
    plaque_burden_per_mm = plaque_per_slice / slice_distance
    
    return plaque_burden_per_mm
```

**Output:** A 1D array showing plaque burden (mm²/mm) along the vessel length, enabling:
- Identification of **maximum plaque burden location**
- Measurement of **total plaque volume**
- Correlation with **FFR measurements**

### Frame Consistency: Why RMF Matters for Transformations

**Critical:** The inverse transformation only works correctly if the $\{\vec{N}, \vec{B}\}$ basis is **consistent** along the vessel.

| **Frame Type** | **Behavior** | **Effect on Inverse Transform** |
|----------------|--------------|--------------------------------|
| **RMF (Bishop)** | Smooth propagation | ✅ Stable $u, v$ coordinates |
| **Frenet-Serret** | Flips at inflection points | ❌ $u, v$ "spin" around vessel |

With Frenet-Serret, a straight plaque would appear to **spiral** around the vessel in the sCPR view. The RMF ensures anatomical features maintain consistent orientation.

---

## 🌐 Phase 9: TypeScript / Niivue / Vite Integration

### Project Structure

```
vessel-viewer/
├── public/
│   ├── volumes/
│   │   └── cta.nii.gz           # CTA volume
│   ├── meshes/
│   │   ├── lad_Lumen.gii        # Lumen surface
│   │   └── lad_VesselWall.gii   # Wall surface
│   └── points/
│       └── lad_points.json      # Connectome data
├── src/
│   ├── main.ts
│   └── vessel-loader.ts
├── index.html
├── package.json
└── vite.config.ts
```

### Loading GIfTI Meshes in Niivue

```typescript
import { Niivue } from '@niivue/niivue';

async function initializeViewer(): Promise<void> {
  const nv = new Niivue({
    backColor: [0.1, 0.1, 0.1, 1],
    show3Dcrosshair: true,
  });
  
  await nv.attachToCanvas(document.getElementById('gl-canvas') as HTMLCanvasElement);
  
  // Load CTA volume as base
  await nv.loadVolumes([
    { url: '/volumes/cta.nii.gz' }
  ]);
  
  // Load vessel meshes as overlays
  const meshList = [
    {
      url: '/meshes/lad_VesselWall.gii',
      rgba255: [200, 200, 200, 100],  // Semi-transparent gray
      opacity: 0.3,
      visible: true,
      name: 'Vessel Wall'
    },
    {
      url: '/meshes/lad_Lumen.gii',
      rgba255: [255, 50, 50, 255],    // Bright red
      opacity: 0.8,
      visible: true,
      name: 'Lumen'
    }
  ];
  
  await nv.loadMeshes(meshList);
  
  // Apply shader for better vessel visualization
  if (nv.meshes.length > 0) {
    nv.setMeshShader(nv.meshes[0].id, 'Matcap');
    nv.setMeshShader(nv.meshes[1].id, 'Matcap');
  }
  
  // Set 3D rendering mode
  nv.setSliceType(nv.sliceTypeRender);
}

initializeViewer();
```

### Loading Point Cloud (Connectome JSON)

```typescript
async function loadPointCloud(nv: Niivue, jsonUrl: string): Promise<void> {
  // Niivue loads connectome JSON as a special mesh type
  await nv.loadConnectome(jsonUrl);
  
  // Configure point/line appearance
  nv.opts.connectome = {
    nodeScale: 0.5,      // Point size
    edgeScale: 0.2,      // Line thickness
    nodeColormap: 'warm',
    edgeColormap: 'warm'
  };
  
  nv.updateGLVolume();
}
```

### View Switcher: Mesh vs Points

```typescript
type ViewMode = 'mesh' | 'points' | 'both';

function setViewMode(nv: Niivue, mode: ViewMode): void {
  const hasMeshes = nv.meshes.length > 0;
  
  switch (mode) {
    case 'mesh':
      // Show meshes, hide connectome
      nv.meshes.forEach(m => m.visible = true);
      nv.opts.connectome.nodeScale = 0;
      break;
      
    case 'points':
      // Hide meshes, show connectome
      nv.meshes.forEach(m => m.visible = false);
      nv.opts.connectome.nodeScale = 0.5;
      break;
      
    case 'both':
      // Show mesh at low opacity + points
      nv.meshes.forEach(m => {
        m.visible = true;
        m.opacity = 0.3;
      });
      nv.opts.connectome.nodeScale = 0.3;
      break;
  }
  
  nv.updateGLVolume();
}
```

### Loading sCPR Volume from Memory

```typescript
import { NVImage } from '@niivue/niivue';

async function loadStraightenedVolume(
  nv: Niivue,
  scprBuffer: Float32Array,
  dims: [number, number, number],
  pixdims: [number, number, number]
): Promise<void> {
  // Create NVImage from raw buffer
  const scprImage = new NVImage(
    scprBuffer.buffer,
    'straightened_vessel',
    'gray',  // colormap
    1.0,     // opacity
    0,       // frame
    'linear' // interpolation
  );
  
  // Set dimensions
  scprImage.dims = [1, dims[0], dims[1], dims[2], 1, 1, 1, 1];
  scprImage.pixDims = [1, pixdims[0], pixdims[1], pixdims[2], 1, 1, 1, 1];
  
  await nv.addVolume(scprImage);
  
  // Switch to multiplanar view for scrolling through vessel
  nv.setSliceType(nv.sliceTypeMultiplanar);
}
```

### Netlify Deployment Configuration

**`netlify.toml`:**
```toml
[build]
  command = "npm run build"
  publish = "dist"

[[headers]]
  for = "/*.gii"
  [headers.values]
    Content-Type = "application/octet-stream"
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/*.nii.gz"
  [headers.values]
    Content-Type = "application/gzip"
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/*.json"
  [headers.values]
    Content-Type = "application/json"
    Cache-Control = "public, max-age=86400"
```

**Vite Configuration (`vite.config.ts`):**
```typescript
import { defineConfig } from 'vite';

export default defineConfig({
  assetsInclude: ['**/*.gii', '**/*.nii', '**/*.nii.gz'],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    }
  }
});
```

---

## 📄 Appendix A: Sample Data Format Reference

### MEDIS Contour File Structure

```
# <GENERAL_INFO>
# study_description : DISCHARGE
# series_description : HALF 720ms 0.92s CTA/HALF DISCHARGE 720ms AIDR 3D STD CTA
# patient_date_of_birth : 
# patient_id : 01-BER-0088
# patient_sex : F
# file_id : 8028
# patient_name : 01-BER-0088
# base_name : 01-BER-0088_20181010_Session_..._ecrf.dcm<27243150:4348990>
# vessel_name : rca

# <CONTOUR_INFO>
# Contour index: 0
# group: Lumen                    ← Lumen or VesselWall
# segment_name: Ostium
# segment_type: SEGMENT_TYPE_EXCLUDED
# segment_ahalabel: -1
# SliceDistance: 0.25             ← Distance between contours (mm)
# Number of points: 40            ← Points per contour ring
6.152891159057617 -0.09696578979492188 1974.5980224609375
6.102381706237793 -0.2288517951965332 1974.6019287109375
...
```

### Key Fields

| **Field** | **Description** | **Usage** |
|-----------|-----------------|----------|
| `vessel_name` | Coronary artery (lad, lcx, rca) | File naming, metadata |
| `group` | Lumen or VesselWall | Separate mesh generation |
| `SliceDistance` | Spacing between contours | Z-spacing for sCPR |
| `Number of points` | Points per ring | `points_per_contour` parameter |
| Coordinates | X Y Z in world space | Direct vertex positions |

### Coordinate Interpretation

- **X, Y:** Position in axial plane (mm from scanner origin)
- **Z:** Table position / slice location (e.g., 1974.59 mm)
- **Reference:** Same coordinate system as source DICOM/NIfTI

---

## 📚 Related Documentation

- **MEDIS Viewer:** `medis-viewer.md` - Visualization platform for MEDIS contours
- **SAM3D Platform:** `sam3d.md` - AI-driven segmentation (different approach)
- **Funding Proposal:** `proposal.md` - DFG Koselleck grant

---

## 🔬 Mathematical References

**Fast Marching Method:**
- Sethian, J.A. (1996). "A fast marching level set method for monotonically advancing fronts." PNAS 93(4): 1591-1595.
- Kimmel, R., Sethian, J.A. (1998). "Computing geodesic paths on manifolds." PNAS 95(15): 8431-8435.

**Frangi Vesselness:**
- Frangi, A.F., et al. (1998). "Multiscale vessel enhancement filtering." MICCAI 1998.

**Polar Dynamic Programming:**
- Sun, Y., et al. (2003). "Automated 3-D segmentation of lungs with lung cancer in CT data using a novel robust active shape model approach." IEEE TMI 2012.

**Rotation Minimizing Frames:**
- Wang, W., et al. (2008). "Computation of rotation minimizing frames." ACM TOG 27(1).

**Curved Planar Reformation:**
- Kanitsar, A., et al. (2002). "CPR - Curved planar reformation." IEEE Visualization 2002.

---

## ⚙️ Performance Targets

**Python Prototype:**
- Vesselness filter: 10-30s (depends on volume size)
- FMM propagation: 2-5s
- Single cross-section: <100ms
- Full segmentation (200 slices): ~20-60s
- GIfTI conversion: <1s per vessel
- sCPR generation: 5-15s

**Optimized (C++ / WASM):**
- Vesselness filter: 1-3s (with GPU shader: <500ms)
- FMM propagation: <1s
- Full pipeline: <10s
- sCPR with WebGL: <2s

**Web (Niivue/Vite):**
- Initial load (CTA + meshes): 2-5s
- Mesh toggle: <50ms
- View rotation: 60fps

---

**Document Version:** 2.1  
**Last Updated:** 2025-12-20  
**Project:** Classical Centerline Pipeline (FMM-based) + Web Visualization
