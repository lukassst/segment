---
title: "CT-FFR Centerline and Vessel Wall Segmentation – Implementation Plan"
author: "Max Karlsen"
date: "December 2025"
bibliography: ../draft/references.bib
documentclass: article
geometry:
  - top=2.5cm
  - bottom=1.5cm
  - left=2.5cm
  - right=2.0cm
fontsize: 11pt
linestretch: 1.2
numbersections: false
header-includes: |
  \usepackage{booktabs}
  \usepackage{fontspec}
  \usepackage{amsmath}
  \usepackage{amssymb}
  \setmainfont{Arial}
  \setmonofont{Courier New}
  \renewcommand{\maketitle}{}
---

# CT-FFR Pipeline Implementation Plan
**Objective:** From a 3D NIfTI CTA volume and two user points, extract a refined centerline and segment inner/outer vessel walls.
**Constraint:** Use standard Python libraries (`numpy`, `scipy`, `skimage`, `SimpleITK`). Future-proof for browser/niivue (WASM).

---

## Phase 1: Data Ingestion & Preprocessing
**Goal:** Load the volume and prepare it for fast traversal.

1.  **Load Volume:**
    *   Use `nibabel` or `SimpleITK` to load `.nii.gz`.
    *   Convert to `numpy` float32 array.
    *   **Resample (Optional):** If voxels are highly anisotropic (e.g., $0.3 \times 0.3 \times 2.0$ mm), resample to isotropic (~0.5mm) using `scipy.ndimage.zoom` or ITK to ensure the FMM wave propagates evenly.

2.  **Vesselness Filter (Frangi):**
    *   Compute Hessian matrix eigenvalues $(\lambda_1, \lambda_2, \lambda_3)$ at each voxel.
    *   Use `skimage.filters.hessian` or `itk.Hessian3DToVesselnessMeasure`.
    *   **Browser Note:** This is computationally expensive ($O(N)$ per voxel). For browser, we will later use a pre-computed map or a WebGL shader-based Frangi filter.

3.  **Speed/Cost Map Construction:**
    *   Invert Vesselness: $Cost(x) = \exp(-\alpha \cdot \text{Vesselness}(x))$.
    *   High vesselness $\rightarrow$ Low cost.

---

## Phase 2: Initial Centerline Extraction (FMM)
**Goal:** Connect Start ($P_{start}$) and End ($P_{end}$) points with a global minimum path.

1.  **Fast Marching Method:**
    *   Use `scikit-fmm` (pure C extension, compilable to WASM) or `SimpleITK.FastMarchingImageFilter`.
    *   Seed wave at $P_{start}$.
    *   Propagate until $P_{end}$ is reached.
    *   Output: `ArrivalMap` (Time $T$ to reach every pixel).

2.  **Gradient Descent (Backtracking):**
    *   Start at $P_{end}$.
    *   Step backwards along $-\nabla T$ (gradient of arrival time).
    *   Stop at $P_{start}$.
    *   **Result:** `Centerline_v1` (List of 3D coordinates).

---

## Phase 3: Vessel Wall Segmentation (Polar + DP)
**Goal:** For every point on `Centerline_v1`, segment Lumen and Outer Wall.

1.  **Resample Path:** Spline interpolation (`scipy.interpolate`) to equidistance (e.g., every 0.5mm).
2.  **Multi-Planar Reformation (MPR):**
    *   Compute tangent vector $\vec{t}$ at each path point.
    *   Define normal vectors $\vec{u}, \vec{v}$ orthogonal to $\vec{t}$.
    *   Extract $64 \times 64$ pixel cross-section slice using `scipy.ndimage.map_coordinates` (tricubic interpolation).

3.  **Polar Transform:**
    *   Unwrap the $64 \times 64$ slice into a $32 \times 360$ (Radius $\times$ Angle) image.
    *   Use `cv2.linearPolar` or manual `map_coordinates` mapping.

4.  **Cost Function Definition:**
    *   **Lumen Cost:** Gradient magnitude + Intensity (Bright).
        *   $C_{lumen} = w_1 \cdot (1 - \text{Gradient}) + w_2 \cdot (1 - \text{Intensity})$.
        *   *Calcification Handling:* If Intensity > 600 HU, penalize "cutting through" it; prefer wrapping around.
    *   **Outer Wall Cost:** Gradient magnitude (weaker) + Distance from Lumen.

5.  **Dynamic Programming (Graph Search):**
    *   Build a graph where columns are angles $\theta$ and rows are radii $r$.
    *   Find min-cost path from $\theta=0$ to $\theta=360$.
    *   **Constraint:** $r_{lumen}(\theta) < r_{outer}(\theta)$.
    *   **Solver:** Simple Viterbi algorithm in pure `numpy` (numba-optimized if possible).

6.  **Transform Back:** Convert polar $(r, \theta)$ paths back to Cartesian $(x,y)$ on the slice, then to $(x,y,z)$ in the volume.

---

## Phase 4: Centerline Refinement
**Goal:** Correct the "corner cutting" of FMM using the segmented lumen.

1.  **Compute Centroids:**
    *   For each segmented slice, calculate the **Geometric Centroid** of the lumen mask.
2.  **Update Path:**
    *   Replace `Centerline_v1` points with these new Centroids.
3.  **Smoothing:**
    *   Fit a 3D Spline through the new centroids to remove jitter.
    *   **Result:** `Centerline_Refined`.

---

## Technical Stack & Browser Feasibility

### Python Prototype (Immediate)
*   **Loading:** `SimpleITK` / `nibabel`
*   **Math/Arrays:** `numpy`
*   **Image Proc:** `scipy.ndimage`, `skimage`
*   **FMM:** `scikit-fmm` (or `SimpleITK`)

### Browser / Niivue Transition (Future)
*   **Volume Rendering:** `niivue` (WebGL).
*   **Compute Engine:** **Pyodide** (Python in WASM) or **WebAssembly (C++)**.
    *   `numpy` and `scipy` run in Pyodide.
    *   `scikit-fmm` can be compiled to WASM.
    *   **Bottleneck:** The Frangi filter (3D convolution).
    *   **Solution:** Implement Frangi as a GLSL Shader in Niivue or use a reduced-resolution map computed in a Web Worker.

### "No-Commercial" Validation
*   This pipeline uses **zero** proprietary algorithms.
*   MPR + Polar + DP is the standard "Academic/Open" alternative to commercial "Model-Based" segmentation.
*   FMM is the standard "Academic/Open" alternative to commercial "Centerline Probing."
