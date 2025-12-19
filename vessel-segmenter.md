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

**Document Version:** 1.0  
**Last Updated:** 2025-12-18  
**Project:** Classical Centerline Pipeline (FMM-based)
