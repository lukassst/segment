# 🩺 MEDIS Viewer Platform

**Project:** Interactive Visualization Platform for MEDIS Coronary Contour Data  
**Data Input:** MEDIS TXT files (lumen + vessel wall contours from QCA)  
**CTA Volume:** NIfTI format coronary CT angiography  
**Core Technology:** NiiVue (WebGL2) + Direct Mesh Construction (Client-Side)

---

## 📋 Project Overview

### Clinical Context
**MEDIS TXT files** contain expert-traced coronary artery contours from quantitative coronary angiography (QCA):
- **Lumen contours:** Inner vessel boundary (blood pool)
- **Vessel wall contours:** Outer vessel boundary (including plaque)
- **Format:** Stacked rings of 3D points (x, y, z in mm) with known connectivity
- **Spacing:** Typically 0.25-0.5 mm between contour planes

### Visualization Goals
Display multiple coordinated views of coronary vessel geometry:

1. **Original CTA Volume** - 3D rendering with overlays
2. **Cross-Sections** - Perpendicular slices along centerline (interactive)
3. **3D Mesh** - Lumen + vessel wall surfaces (from contour rings)
4. **STL Export** - For 3D printing or external analysis
5. **Straightened MPR** - Curved reformation (vessel "unfolded" to straight view)
6. **Overlay Views** - CTA + mesh combined visualization

### Key Features
- ✅ **Client-side mesh generation:** Parse TXT → NVMesh in 50-100ms (no backend needed)
- ✅ **Real-time interaction:** Slider-controlled position/rotation/zoom
- ✅ **Quad-view layout:** CTA, cross-section, straightened MPR, 3D mesh
- ✅ **General rotation viewer:** Free 3D rotation with mouse drag (Rodrigues formula)
- ✅ **Export capabilities:** STL, MZ3, PLY formats

---

## 🏗️ Architecture

### Frontend Stack (TypeScript + Vite + NiiVue)

```
medis-viewer-frontend/
├── src/
│   ├── main.ts                      # Application entry
│   ├── components/
│   │   ├── ViewerQuad.ts            # Quad-view layout manager
│   │   ├── CTAViewer.ts             # CTA volume display (Panel 1)
│   │   ├── CrossSectionViewer.ts    # Cross-section viewer (Panel 2)
│   │   ├── StraightenedMPRViewer.ts # Straightened MPR (Panel 3)
│   │   ├── MeshViewer3D.ts          # 3D mesh display (Panel 4)
│   │   └── GeneralRotationViewer.ts # Free rotation viewer
│   ├── services/
│   │   ├── medisParser.ts           # Parse MEDIS TXT files
│   │   ├── medisMeshDirect.ts       # Direct NVMesh construction
│   │   ├── centerlineExtractor.ts   # Compute centerline from contours
│   │   ├── crossSectionExtractor.ts # Extract perpendicular slices
│   │   ├── straightenedMPR.ts       # Create straightened volume
│   │   └── meshExporter.ts          # Export STL/MZ3/PLY
│   ├── math/
│   │   ├── frenetFrame.ts           # TNB frame computation
│   │   ├── rotation.ts              # Rodrigues, Euler angles
│   │   └── interpolation.ts         # Trilinear interpolation
│   └── types/
│       ├── medis.ts                 # MEDIS contour types
│       └── geometry.ts              # 3D math types
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### Data Flow

```
1. USER LOADS FILES
   ├─ CTA volume (.nii.gz) → NiiVue
   └─ MEDIS TXT file → medisParser.ts

2. PARSE MEDIS TXT (medisParser.ts)
   ├─ Extract lumen contours (N rings, ~50 points each)
   ├─ Extract vessel wall contours
   └─ Metadata: SliceDistance, coordinate system

3. COMPUTE CENTERLINE (centerlineExtractor.ts)
   ├─ Centroids of lumen contours → N points
   ├─ Tangent vectors (finite differences)
   └─ Frenet-Serret frame → Normal + Binormal vectors

4. GENERATE MESHES (medisMeshDirect.ts)
   ├─ Connect contour rings → triangles
   ├─ Create NVMesh (lumen + vessel wall)
   └─ 50-100ms client-side (no backend)

5. EXTRACT CROSS-SECTIONS (crossSectionExtractor.ts)
   ├─ At each centerline point
   ├─ Perpendicular plane (64×64 pixels)
   └─ Trilinear interpolation from CTA

6. BUILD STRAIGHTENED MPR (straightenedMPR.ts)
   ├─ Stack all cross-sections → [64×64×N] volume
   └─ Save as NIfTI for visualization

7. DISPLAY QUAD-VIEW
   ├─ Panel 1: CTA + centerline overlay
   ├─ Panel 2: Current cross-section (interactive slider)
   ├─ Panel 3: Straightened MPR
   └─ Panel 4: 3D mesh (lumen + vessel wall)

8. USER INTERACTION
   ├─ Slider: Move along centerline
   ├─ Rotation: Adjust viewing angle
   ├─ Zoom: Focus on region of interest
   └─ Export: Save mesh as STL/MZ3
```

---

## 📐 MEDIS TXT Format

### File Structure

```
# General information
# Date: 2021-04-16
# Patient ID: 01-BER-0088
# Series: LAD

# Contour index: 0
# group: Lumen
# SliceDistance: 0.25
# Number of points: 50
17.518342971801758 16.819992065429688 1967.189453125
17.93111801147461 16.195648193359375 1967.219482421875
18.367334365844727 15.595882415771484 1967.2694091796875
...

# Contour index: 1
# group: VesselWall
# SliceDistance: 0.25
# Number of points: 49
18.509117126464844 15.870161056518555 1966.8944091796875
18.950408935546875 15.255702018737793 1966.9444091796875
...

# Contour index: 2
# group: Lumen
# SliceDistance: 0.50
# Number of points: 48
...
```

### Key Properties
- **Coordinate system:** Physical space (mm), typically RAS or LPS
- **Lumen contours:** Inner wall boundary (~50 points per ring)
- **VesselWall contours:** Outer wall boundary (~49 points per ring)
- **SliceDistance:** Spacing along vessel (mm) - varies by contour
- **Point order:** Counter-clockwise or clockwise (consistent within file)

---

## 🛤️ Straightened MPR (Curved Reformation) - Complete Algorithm

### Overview: Mathematical Foundation

**Straightened MPR** (also called **Curved Planar Reformation - CPR**) is a visualization technique that "unfolds" a curved vessel into a straight view, enabling easier assessment of stenosis, plaque, and vessel wall abnormalities along the entire length.

**Three Key Components:**
1. **Centerline extraction** from MEDIS contour point clouds
2. **Orthogonal cross-section extraction** at each centerline point
3. **Volume reconstruction** by stacking cross-sections into straightened 3D volume

**Mathematical Approach:** Frenet-Serret Frame (TNB frame)
- **T** (Tangent): Direction of vessel at each point
- **N** (Normal): Principal curvature direction
- **B** (Binormal): T × N, completes right-handed coordinate system

---

## 📐 Algorithm 1: Centerline Extraction from MEDIS TXT

### Algorithm 1.1: Extract Centerline Points

**Method: Centroid of Lumen Contours**

```typescript
// services/centerlineExtractor.ts
import type { MedisContour } from '../types/medis';

export interface CenterlineData {
  points: Float32Array;       // Nx3 array [x, y, z]
  tangents: Float32Array;     // Nx3 unit vectors
  normals: Float32Array;      // Nx3 unit vectors
  binormals: Float32Array;    // Nx3 unit vectors
  sliceDistances: number[];   // Distance along vessel
}

export function extractCenterline(
  lumenContours: MedisContour[]
): CenterlineData {
  const N = lumenContours.length;
  const points = new Float32Array(N * 3);
  const sliceDistances: number[] = [];
  
  // Step 1: Compute centroids
  for (let i = 0; i < N; i++) {
    const contour = lumenContours[i];
    const pts = contour.points; // Mx3 array
    const M = pts.length / 3;
    
    let cx = 0, cy = 0, cz = 0;
    for (let j = 0; j < M; j++) {
      cx += pts[j * 3 + 0];
      cy += pts[j * 3 + 1];
      cz += pts[j * 3 + 2];
    }
    
    points[i * 3 + 0] = cx / M;
    points[i * 3 + 1] = cy / M;
    points[i * 3 + 2] = cz / M;
    
    sliceDistances.push(contour.sliceDistance);
  }
  
  // Step 2: Compute tangent vectors
  const tangents = computeTangents(points, N);
  
  // Step 3: Compute normal and binormal vectors (Frenet-Serret)
  const { normals, binormals } = computeFrenetFrame(points, tangents, N);
  
  return { points, tangents, normals, binormals, sliceDistances };
}
```

### Algorithm 1.2: Compute Tangent Vectors

```typescript
function computeTangents(
  points: Float32Array,
  N: number
): Float32Array {
  const tangents = new Float32Array(N * 3);
  
  for (let i = 0; i < N; i++) {
    let tx, ty, tz;
    
    if (i === 0) {
      // Forward difference at start
      tx = points[3] - points[0];
      ty = points[4] - points[1];
      tz = points[5] - points[2];
    } else if (i === N - 1) {
      // Backward difference at end
      tx = points[i * 3 + 0] - points[(i - 1) * 3 + 0];
      ty = points[i * 3 + 1] - points[(i - 1) * 3 + 1];
      tz = points[i * 3 + 2] - points[(i - 1) * 3 + 2];
    } else {
      // Central difference (more accurate)
      tx = points[(i + 1) * 3 + 0] - points[(i - 1) * 3 + 0];
      ty = points[(i + 1) * 3 + 1] - points[(i - 1) * 3 + 1];
      tz = points[(i + 1) * 3 + 2] - points[(i - 1) * 3 + 2];
    }
    
    // Normalize
    const len = Math.sqrt(tx * tx + ty * ty + tz * tz);
    tangents[i * 3 + 0] = tx / len;
    tangents[i * 3 + 1] = ty / len;
    tangents[i * 3 + 2] = tz / len;
  }
  
  return tangents;
}
```

### Algorithm 1.3: Frenet-Serret Frame

```typescript
function computeFrenetFrame(
  points: Float32Array,
  tangents: Float32Array,
  N: number
): { normals: Float32Array; binormals: Float32Array } {
  const normals = new Float32Array(N * 3);
  const binormals = new Float32Array(N * 3);
  const epsilon = 1e-6;
  
  for (let i = 0; i < N; i++) {
    const T = [
      tangents[i * 3 + 0],
      tangents[i * 3 + 1],
      tangents[i * 3 + 2]
    ];
    
    // Compute curvature vector dT/ds
    let dTx, dTy, dTz;
    if (i === 0) {
      dTx = tangents[3] - tangents[0];
      dTy = tangents[4] - tangents[1];
      dTz = tangents[5] - tangents[2];
    } else if (i === N - 1) {
      dTx = tangents[i * 3 + 0] - tangents[(i - 1) * 3 + 0];
      dTy = tangents[i * 3 + 1] - tangents[(i - 1) * 3 + 1];
      dTz = tangents[i * 3 + 2] - tangents[(i - 1) * 3 + 2];
    } else {
      dTx = tangents[(i + 1) * 3 + 0] - tangents[(i - 1) * 3 + 0];
      dTy = tangents[(i + 1) * 3 + 1] - tangents[(i - 1) * 3 + 1];
      dTz = tangents[(i + 1) * 3 + 2] - tangents[(i - 1) * 3 + 2];
    }
    
    const curvatureMag = Math.sqrt(dTx * dTx + dTy * dTy + dTz * dTz);
    
    let N_vec: [number, number, number];
    if (curvatureMag > epsilon) {
      // Normal is normalized curvature direction
      N_vec = [dTx / curvatureMag, dTy / curvatureMag, dTz / curvatureMag];
    } else {
      // Straight segment: choose arbitrary perpendicular
      N_vec = perpendicularTo(T as [number, number, number]);
    }
    
    // Binormal via cross product
    const B_vec = crossProduct(T as [number, number, number], N_vec);
    const B_normalized = normalize(B_vec);
    
    normals[i * 3 + 0] = N_vec[0];
    normals[i * 3 + 1] = N_vec[1];
    normals[i * 3 + 2] = N_vec[2];
    
    binormals[i * 3 + 0] = B_normalized[0];
    binormals[i * 3 + 1] = B_normalized[1];
    binormals[i * 3 + 2] = B_normalized[2];
  }
  
  return { normals, binormals };
}

// Helper: perpendicular vector to v
function perpendicularTo(v: [number, number, number]): [number, number, number] {
  const [vx, vy, vz] = v;
  const absX = Math.abs(vx);
  const absY = Math.abs(vy);
  const absZ = Math.abs(vz);
  
  let axis: [number, number, number];
  if (absX < absY && absX < absZ) {
    axis = [1, 0, 0];
  } else if (absY < absZ) {
    axis = [0, 1, 0];
  } else {
    axis = [0, 0, 1];
  }
  
  const perp = crossProduct(v, axis);
  return normalize(perp);
}

function crossProduct(
  a: [number, number, number],
  b: [number, number, number]
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function normalize(v: [number, number, number]): [number, number, number] {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return [v[0] / len, v[1] / len, v[2] / len];
}
```

---

## 🔬 Algorithm 2: Cross-Section Extraction

### Goal
Extract perpendicular 2D cross-sectional images from CTA volume at each centerline point.

### Algorithm 2.1: Sample Cross-Section Grid

```typescript
// services/crossSectionExtractor.ts
import { Niivue } from '@niivue/niivue';

export interface CrossSectionParams {
  size: [number, number];      // [width, height] in pixels (e.g., [64, 64])
  spacing: [number, number];   // [du, dv] in mm (e.g., [0.5, 0.5])
}

export function extractCrossSection(
  nv: Niivue,
  volumeIndex: number,
  centerlinePoint: [number, number, number],  // [px, py, pz] in mm
  normal: [number, number, number],           // N vector
  binormal: [number, number, number],         // B vector
  tangent: [number, number, number],          // T vector
  params: CrossSectionParams
): Float32Array {
  const [W, H] = params.size;
  const [du, dv] = params.spacing;
  const crossSection = new Float32Array(W * H);
  
  const volume = nv.volumes[volumeIndex];
  const ctaData = volume.img;  // 3D volume data
  const affine = volume.matRAS; // Affine matrix
  
  // Compute inverse affine (physical → voxel)
  const affineInv = invertAffine(affine);
  
  for (let iu = 0; iu < W; iu++) {
    for (let iv = 0; iv < H; iv++) {
      // Physical offset from centerline
      const u = (iu - W / 2) * du;
      const v = (iv - H / 2) * dv;
      
      // 3D physical position
      const Qx = centerlinePoint[0] + u * normal[0] + v * binormal[0];
      const Qy = centerlinePoint[1] + u * normal[1] + v * binormal[1];
      const Qz = centerlinePoint[2] + u * normal[2] + v * binormal[2];
      
      // Transform to voxel coordinates
      const [vx, vy, vz] = applyAffine(affineInv, [Qx, Qy, Qz]);
      
      // Trilinear interpolation
      const intensity = trilinearInterpolate(ctaData, vx, vy, vz, volume.dimsRAS);
      
      crossSection[iu + iv * W] = intensity;
    }
  }
  
  return crossSection;
}
```

### Algorithm 2.2: Trilinear Interpolation

```typescript
function trilinearInterpolate(
  volume: Uint8Array | Int16Array | Float32Array,
  x: number,
  y: number,
  z: number,
  dims: [number, number, number]
): number {
  const [Nx, Ny, Nz] = dims;
  
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const z1 = z0 + 1;
  
  // Bounds check
  if (x0 < 0 || x1 >= Nx || y0 < 0 || y1 >= Ny || z0 < 0 || z1 >= Nz) {
    return 0; // Background
  }
  
  const fx = x - x0;
  const fy = y - y0;
  const fz = z - z0;
  
  // Sample 8 corners
  const idx = (x: number, y: number, z: number) => x + y * Nx + z * Nx * Ny;
  
  const c000 = volume[idx(x0, y0, z0)];
  const c001 = volume[idx(x0, y0, z1)];
  const c010 = volume[idx(x0, y1, z0)];
  const c011 = volume[idx(x0, y1, z1)];
  const c100 = volume[idx(x1, y0, z0)];
  const c101 = volume[idx(x1, y0, z1)];
  const c110 = volume[idx(x1, y1, z0)];
  const c111 = volume[idx(x1, y1, z1)];
  
  // Trilinear interpolation
  const c00 = c000 * (1 - fx) + c100 * fx;
  const c01 = c001 * (1 - fx) + c101 * fx;
  const c10 = c010 * (1 - fx) + c110 * fx;
  const c11 = c011 * (1 - fx) + c111 * fx;
  
  const c0 = c00 * (1 - fy) + c10 * fy;
  const c1 = c01 * (1 - fy) + c11 * fy;
  
  return c0 * (1 - fz) + c1 * fz;
}

function invertAffine(affine: number[][]): number[][] {
  // 4×4 matrix inversion (standard algorithm)
  // Implementation omitted for brevity
  // Use library like gl-matrix or implement Gauss-Jordan
  return computeInverse4x4(affine);
}

function applyAffine(
  affine: number[][],
  point: [number, number, number]
): [number, number, number] {
  const [x, y, z] = point;
  return [
    affine[0][0] * x + affine[0][1] * y + affine[0][2] * z + affine[0][3],
    affine[1][0] * x + affine[1][1] * y + affine[1][2] * z + affine[1][3],
    affine[2][0] * x + affine[2][1] * y + affine[2][2] * z + affine[2][3]
  ];
}
```

---

## 🎞️ Algorithm 3: Straightened MPR Volume Construction

### Algorithm 3.1: Build Straightened Volume

```typescript
// services/straightenedMPR.ts
import type { CenterlineData } from './centerlineExtractor';
import type { CrossSectionParams } from './crossSectionExtractor';

export function createStraightenedMPR(
  nv: Niivue,
  volumeIndex: number,
  centerlineData: CenterlineData,
  params: CrossSectionParams
): { volume: Float32Array; dims: [number, number, number] } {
  const [W, H] = params.size;
  const N = centerlineData.points.length / 3;
  
  const straightenedVolume = new Float32Array(W * H * N);
  
  for (let d = 0; d < N; d++) {
    const P: [number, number, number] = [
      centerlineData.points[d * 3 + 0],
      centerlineData.points[d * 3 + 1],
      centerlineData.points[d * 3 + 2]
    ];
    const N_vec: [number, number, number] = [
      centerlineData.normals[d * 3 + 0],
      centerlineData.normals[d * 3 + 1],
      centerlineData.normals[d * 3 + 2]
    ];
    const B_vec: [number, number, number] = [
      centerlineData.binormals[d * 3 + 0],
      centerlineData.binormals[d * 3 + 1],
      centerlineData.binormals[d * 3 + 2]
    ];
    const T_vec: [number, number, number] = [
      centerlineData.tangents[d * 3 + 0],
      centerlineData.tangents[d * 3 + 1],
      centerlineData.tangents[d * 3 + 2]
    ];
    
    const crossSection = extractCrossSection(
      nv, volumeIndex, P, N_vec, B_vec, T_vec, params
    );
    
    // Copy cross-section into straightened volume
    straightenedVolume.set(crossSection, d * W * H);
  }
  
  return {
    volume: straightenedVolume,
    dims: [W, H, N]
  };
}
```

---

## 🎨 Direct Mesh Construction (Client-Side)

### Overview

**Key Insight:** MEDIS contours are stacked rings with **known topology** (ring N connects to ring N+1). We can build the mesh directly in the browser without backend processing.

**Performance Comparison:**
- Backend (buildstl.py → STL → network → NiiVue): **500-2000ms**
- Client-side (parse TXT → NVMesh): **50-100ms** ⚡

### Algorithm: Ring-to-Ring Connectivity

```typescript
// services/medisMeshDirect.ts
import { Niivue, NVMesh } from '@niivue/niivue';
import type { MedisContour } from '../types/medis';

export function medisContoursToMesh(
  contours: MedisContour[],
  meshType: 'lumen' | 'vessel',
  nv: Niivue
): NVMesh {
  // Filter contours by type
  const filtered = contours.filter(c => 
    (meshType === 'lumen' && c.group === 'Lumen') ||
    (meshType === 'vessel' && c.group === 'VesselWall')
  );
  
  const vertices: number[] = [];
  const triangles: number[] = [];
  let vertexOffset = 0;
  
  // Add all vertices
  for (const contour of filtered) {
    vertices.push(...contour.points);
  }
  
  // Connect rings
  for (let ringIdx = 0; ringIdx < filtered.length - 1; ringIdx++) {
    const ring0 = filtered[ringIdx];
    const ring1 = filtered[ringIdx + 1];
    const M0 = ring0.points.length / 3;
    const M1 = ring1.points.length / 3;
    
    // Offset for this ring's vertices
    const offset0 = vertexOffset;
    const offset1 = offset0 + M0;
    
    // Create quad strip (2 triangles per quad)
    for (let i = 0; i < Math.min(M0, M1); i++) {
      const i0_curr = offset0 + i;
      const i0_next = offset0 + ((i + 1) % M0);
      const i1_curr = offset1 + i;
      const i1_next = offset1 + ((i + 1) % M1);
      
      // Triangle 1: [i0_curr, i1_curr, i0_next]
      triangles.push(i0_curr, i1_curr, i0_next);
      
      // Triangle 2: [i0_next, i1_curr, i1_next]
      triangles.push(i0_next, i1_curr, i1_next);
    }
    
    vertexOffset += M0;
  }
  
  // Create NVMesh
  const nvMesh = nv.createMeshFromVertices(
    new Float32Array(vertices),
    new Uint32Array(triangles)
  );
  
  // Set color
  if (meshType === 'lumen') {
    nvMesh.rgba255 = [255, 0, 0, 200]; // Red, semi-transparent
  } else {
    nvMesh.rgba255 = [0, 0, 255, 100]; // Blue, more transparent
  }
  
  return nvMesh;
}
```

---

## 🎮 Interactive UI Controls

### Quad-View Layout

```
+------------------+------------------+
| Panel 1: CTA     | Panel 2: Cross-  |
| + Centerline     | Section (64×64)  |
| Overlay          | Interactive      |
+------------------+------------------+
| Panel 3:         | Panel 4: 3D Mesh |
| Straightened MPR | (Lumen + Vessel) |
| (Sagittal View)  | Rotation/Zoom    |
+------------------+------------------+
```

### Control Panel

**Position Controls:**
```
[Slider] Centerline Position: 0 ─────●─────────── 200
         → Moves along vessel
         → Updates Panel 2 and Panel 3

[Input] Jump to Slice: [___] (0-200)
        → Direct navigation
```

**Cross-Section Controls:**
```
[Slider] Rotation Angle: 0° ──●── 360°
         → Rotates cross-section around tangent axis
         → Updates viewing orientation

[Slider] Zoom: 0.5× ──●── 3.0×
         → Magnifies cross-section
```

**Straightened MPR Controls:**
```
[Slider] Viewing Angle: 0° ──●── 360°
         → Rotates MPR volume around long axis
         → For multi-planar views

[Checkbox] Show Centerline Overlay
           → Draws centerline on CTA (Panel 1)
```

**Mesh Controls:**
```
[Checkbox] Show Lumen Mesh (Red)
[Checkbox] Show Vessel Wall Mesh (Blue)
[Slider] Mesh Opacity: 0% ──●── 100%

[Button] Export STL
[Button] Export MZ3
[Button] Export PLY
```

---

## 🔄 General Rotation Viewer

### Mathematical Foundation

**Rodrigues' Rotation Formula:** Rotate vector **v** by angle θ around axis **n**

```
v_rot = v*cos(θ) + (n × v)*sin(θ) + n*(n·v)*(1 - cos(θ))
```

**Rotation Matrix Form:**
```typescript
function matrixFromAxisAngle(
  axis: [number, number, number],
  theta: number
): number[][] {
  const [nx, ny, nz] = normalize(axis);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const C = 1 - c;
  
  return [
    [c + nx*nx*C,     nx*ny*C - nz*s,  nx*nz*C + ny*s],
    [ny*nx*C + nz*s,  c + ny*ny*C,     ny*nz*C - nx*s],
    [nz*nx*C - ny*s,  nz*ny*C + nx*s,  c + nz*nz*C]
  ];
}
```

### Mouse Drag Rotation

```typescript
// components/GeneralRotationViewer.ts
interface RotationState {
  R: number[][];           // Current 3×3 rotation matrix
  eulerXYZ: [number, number, number]; // Euler angles (degrees)
  center: [number, number, number];   // Rotation center
}

function handleMouseDrag(
  ev: MouseEvent,
  lastPos: [number, number],
  currentPos: [number, number],
  state: RotationState
): void {
  const canvasCenter = [canvas.width / 2, canvas.height / 2];
  
  // Compute rotation angle from mouse arc
  const angle = angleBetweenPoints(lastPos, canvasCenter, currentPos);
  
  // Rotation axis = perpendicular to canvas (screen normal)
  const screenNormal = [0, 0, 1];
  
  // Build incremental rotation matrix
  const R_increment = matrixFromAxisAngle(screenNormal, angle);
  
  // Apply to current rotation: R_new = R_increment * R_old
  state.R = matrixMultiply(R_increment, state.R);
  
  // Update Euler angles for display
  state.eulerXYZ = eulerFromMatrix(state.R);
  
  // Redraw all views with new rotation
  updateAllViews(state);
}

function angleBetweenPoints(
  p1: [number, number],
  center: [number, number],
  p2: [number, number]
): number {
  const v1 = [p1[0] - center[0], p1[1] - center[1]];
  const v2 = [p2[0] - center[0], p2[1] - center[1]];
  
  const dot = v1[0]*v2[0] + v1[1]*v2[1];
  const cross = v1[0]*v2[1] - v1[1]*v2[0];
  
  return Math.atan2(cross, dot);
}
```

---

## 📤 Export Capabilities

### STL Export

```typescript
// services/meshExporter.ts
export function exportSTL(
  mesh: NVMesh,
  filename: string
): void {
  const vertices = mesh.pts;   // Float32Array (Nx3)
  const triangles = mesh.tris; // Uint32Array (Mx3)
  
  // STL binary format
  const header = new Uint8Array(80); // Header (80 bytes)
  const numTriangles = triangles.length / 3;
  
  const buffer = new ArrayBuffer(84 + numTriangles * 50);
  const view = new DataView(buffer);
  
  // Write header
  view.setUint32(80, numTriangles, true);
  
  let offset = 84;
  for (let i = 0; i < numTriangles; i++) {
    const i0 = triangles[i * 3 + 0];
    const i1 = triangles[i * 3 + 1];
    const i2 = triangles[i * 3 + 2];
    
    const v0 = [vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]];
    const v1 = [vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]];
    const v2 = [vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]];
    
    // Compute normal
    const normal = computeTriangleNormal(v0, v1, v2);
    
    // Write normal (12 bytes)
    view.setFloat32(offset + 0, normal[0], true);
    view.setFloat32(offset + 4, normal[1], true);
    view.setFloat32(offset + 8, normal[2], true);
    
    // Write vertices (36 bytes)
    view.setFloat32(offset + 12, v0[0], true);
    view.setFloat32(offset + 16, v0[1], true);
    view.setFloat32(offset + 20, v0[2], true);
    view.setFloat32(offset + 24, v1[0], true);
    view.setFloat32(offset + 28, v1[1], true);
    view.setFloat32(offset + 32, v1[2], true);
    view.setFloat32(offset + 36, v2[0], true);
    view.setFloat32(offset + 40, v2[1], true);
    view.setFloat32(offset + 44, v2[2], true);
    
    // Attribute byte count (2 bytes, unused)
    view.setUint16(offset + 48, 0, true);
    
    offset += 50;
  }
  
  // Download
  const blob = new Blob([buffer], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

---

## 🚀 Performance Targets

### Client-Side Operations
- **MEDIS TXT parsing:** <50ms (typical file: 200 contours × 50 points)
- **Centerline extraction:** <100ms (200-point centerline)
- **Mesh generation:** 50-100ms (direct NVMesh construction)
- **Single cross-section:** <10ms (real-time slider interaction)
- **Straightened MPR:** <2s for full volume (64×64×200)

### Memory Requirements
- **CTA volume:** ~200 MB (512×512×300 × 2 bytes)
- **MEDIS contours:** ~2 MB (200 rings × 50 points × 3 coords × 8 bytes)
- **Straightened volume:** ~3 MB (64×64×200 × 4 bytes)
- **Meshes:** ~5 MB (lumen + vessel, ~50K triangles each)

---

## � Centerline Construction from MEDIS Files

### Understanding SliceDistance

The `SliceDistance` parameter in MEDIS TXT files represents the **cumulative arc length** along the vessel centerline from the origin (ostium), measured in millimeters:

```
# Contour index: 0
# SliceDistance: 0.25    ← 0.25 mm from ostium

# Contour index: 2  
# SliceDistance: 0.75    ← 0.75 mm from ostium (NOT 0.5 mm after previous!)
```

**Key Insight:** SliceDistance values are **NOT uniform intervals**. The spacing between consecutive contours varies (e.g., 0.25, 0.75, 1.25... means 0.5mm gaps, but could also be 0.25, 0.50, 0.75...).

### Centerline Point = Centroid (Center of Mass)

For each contour ring, the centerline point is the **geometric centroid** of the lumen contour:

```python
def compute_centerline_point(lumen_contour: np.ndarray) -> np.ndarray:
    """
    Centerline point = center of mass of lumen contour.
    
    Args:
        lumen_contour: Nx3 array of (x, y, z) points on lumen boundary
    
    Returns:
        centerline_point: [cx, cy, cz] centroid coordinates
    """
    return np.mean(lumen_contour, axis=0)
```

**Why Centroid?**
- Represents geometric center of blood pool cross-section
- Natural reference for perpendicular plane orientation
- Used by MEDIS internally for SliceDistance computation
- Robust to asymmetric vessel shapes (eccentric plaque)

### Complete Centerline Extraction

```python
def extract_centerline_from_medis(contours: dict) -> dict:
    """
    Extract centerline from parsed MEDIS contours.
    
    Args:
        contours: {'Lumen': [np.array, ...], 'VesselWall': [np.array, ...]}
    
    Returns:
        {
            'points': Nx3 array of centerline points,
            'slice_distances': N array of arc lengths (mm),
            'tangents': Nx3 array of unit tangent vectors
        }
    """
    lumen_contours = contours['Lumen']
    
    # Compute centroids
    centerline_points = np.array([
        np.mean(contour, axis=0) for contour in lumen_contours
    ])
    
    # SliceDistance values (parse from file or compute from point spacing)
    # Note: These are cumulative arc lengths, not inter-slice gaps
    
    # Compute tangent vectors via finite differences
    tangents = compute_tangents(centerline_points)
    
    return {
        'points': centerline_points,
        'tangents': tangents
    }
```

---

## 📐 CPR Slice Thickness Considerations

### The Slice Thickness Question

When constructing a **Curved Planar Reformation (CPR)** or straightened MPR volume, we must decide how much of the original CTA volume each "slice" should represent.

**Options:**

1. **Zero thickness (single plane):** Sample exactly at the centerline point
2. **Symmetric slab:** Sample ±half_thickness around centerline
3. **Forward/backward slab:** Sample in direction of vessel progression

### Recommended Approach: Symmetric Slab with SliceDistance-Based Thickness

```
Slice i covers: [centerline[i] - T/2 * tangent, centerline[i] + T/2 * tangent]

where:
  T = slice thickness (typically = inter-slice spacing)
  tangent = unit vector along vessel direction
```

**Thickness Calculation:**

```python
def compute_slice_thickness(slice_distances: list, index: int) -> float:
    """
    Compute slice thickness based on neighboring SliceDistance values.
    
    Uses half the distance to previous + half to next neighbor.
    This ensures no gaps or overlaps between adjacent slices.
    """
    n = len(slice_distances)
    
    if index == 0:
        # First slice: use distance to next
        return slice_distances[1] - slice_distances[0]
    elif index == n - 1:
        # Last slice: use distance from previous
        return slice_distances[n-1] - slice_distances[n-2]
    else:
        # Interior: half-half from neighbors
        d_prev = slice_distances[index] - slice_distances[index - 1]
        d_next = slice_distances[index + 1] - slice_distances[index]
        return (d_prev + d_next) / 2
```

### CPR Volume Construction

```python
def construct_cpr_volume(
    cta_volume: np.ndarray,
    centerline_points: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    slice_distances: list,
    cross_section_size: int = 64,
    pixel_spacing: float = 0.5  # mm
) -> np.ndarray:
    """
    Construct straightened CPR volume.
    
    Output dimensions: [cross_section_size, cross_section_size, N_slices]
    
    Each slice is sampled perpendicular to the vessel tangent.
    Slice thickness is automatically determined from SliceDistance spacing.
    """
    N = len(centerline_points)
    cpr_volume = np.zeros((cross_section_size, cross_section_size, N))
    
    for i in range(N):
        # Get slice thickness (for potential MIP or averaging)
        thickness = compute_slice_thickness(slice_distances, i)
        
        # Sample cross-section at this centerline point
        # For thin slices, single-plane sampling is sufficient
        # For thick slices, consider averaging multiple sub-samples
        cross_section = sample_perpendicular_plane(
            cta_volume,
            center=centerline_points[i],
            normal=normals[i],
            binormal=binormals[i],
            size=cross_section_size,
            spacing=pixel_spacing
        )
        
        cpr_volume[:, :, i] = cross_section
    
    return cpr_volume
```

### Practical Notes for CPR

- **Typical MEDIS spacing:** 0.25-0.5 mm between contours
- **CTA voxel size:** ~0.3-0.5 mm isotropic
- **Recommendation:** Use single-plane sampling when inter-slice spacing ≤ CTA voxel size
- **For thicker slabs:** Use Maximum Intensity Projection (MIP) or averaging

---

## 🔄 MEDIS File Conversion Pipeline

### Available Converters (in `code/` directory)

| Script | Input | Output | Purpose |
|--------|-------|--------|----------|
| `medis_to_gii.py` | MEDIS TXT | GIfTI mesh (.gii) | Water-tight meshes for Niivue overlay |
| `medis_to_json.py` | MEDIS TXT | JSON point cloud | Niivue connectome visualization |
| `medis_to_cpr.py` | MEDIS TXT + NIfTI | CPR volumes (.nii.gz) | Straightened vessel views |
| `medis_stl.py` | MEDIS TXT | STL mesh | 3D printing, external tools |
| `run_conversion.py` | - | All formats | Batch convert example files |

### Conversion Quality Standards

**GIfTI Generation (`medis_to_gii.py`):**
- ✅ Cubic spline resampling to uniform 64 points/ring
- ✅ Rotational alignment prevents mesh twisting
- ✅ Water-tight end caps for CFD compatibility
- ✅ Embedded metadata (PatientID, VesselName, etc.)
- ✅ Correct GIfTI intent codes (POINTSET + TRIANGLE)

**JSON Generation (`medis_to_json.py`):**
- ✅ Niivue connectome format with nodes + edges
- ✅ Intra-ring edges (closed contours)
- ✅ Inter-ring edges (every 4th point for wireframe)
- ✅ Uniform resampling for consistent visualization

**STL Generation (`medis_stl.py`):**
- ✅ Binary STL format
- ✅ Proper face winding (CCW for correct normals)
- ✅ Capped ends option for solid meshes
- ⚠️ No coordinate system metadata (use GIfTI for medical alignment)

**CPR Generation (`medis_to_cpr.py`):**
- ✅ Cross-sectional CPR: Straightened 3D volume (scroll through vessel)
- ✅ Longitudinal CPR: 2D planes along vessel axis (SU and SV cuts)
- ✅ Centerline extracted as lumen contour centroids
- ✅ Rotation Minimizing Frame (RMF) prevents twist artifacts
- ✅ Natural spacing: resampled to match CTA voxel resolution
- ✅ Frame JSON exported for mesh/point coordinate transformation

### Running Conversions

```bash
cd code/
python run_conversion.py  # Converts all example files in data/

# Or individual conversion:
python medis_to_gii.py ../data/medis_file.txt ../data/cta.nii.gz ../data/
python medis_to_cpr.py ../data/medis_file.txt ../data/cta.nii.gz ../data/
```

### CPR Output Files

For each vessel, `medis_to_cpr.py` generates three files:

| File | Axes | Description |
|------|------|-------------|
| `{patient}_{vessel}_cpr_cross.nii.gz` | U × V × S (120×120×N) | Cross-sectional volume: scroll S to move along vessel |
| `{patient}_{vessel}_cpr_long.nii.gz` | S × U × V (N×120×120) | Longitudinal volume: scroll V to rotate around vessel |
| `{patient}_{vessel}_cpr_frame.json` | - | Centerline + RMF for coordinate transform |

**Cross-sectional CPR (U×V×S = 120×120×N):**
- **U-axis (dim 0):** Normal direction (horizontal in cross-section)
- **V-axis (dim 1):** Binormal direction (vertical in cross-section)  
- **S-axis (dim 2):** Along vessel centerline (scroll to move through vessel)
- Each S-slice shows a perpendicular "donut" view centered on lumen

**Longitudinal CPR (S×U×V = N×120×120):**
- **S-axis (dim 0):** Along vessel centerline
- **U-axis (dim 1):** Normal direction (lateral offset)
- **V-axis (dim 2):** Binormal direction (depth offset)
- **Scrolling V:** Pans the longitudinal cut from front to back (center slice = mid-vessel)
- **Rotating:** Since this is a full 3D volume, use the **viewer's rotation tools** to spin the view 360° around the centerline

**Frame JSON:** Contains centerline points and RMF vectors for coordinate transformation.

### Transformed Mesh/Point Cloud Files

Use `transform_to_cpr.py` to convert mesh and point cloud files to CPR coordinates:

```bash
# Batch transform all files for a vessel:
python transform_to_cpr.py --batch ../data/01-BER-0088_LAD_cpr_frame.json ../data/
```

| Original File | Transformed File | Description |
|---------------|------------------|-------------|
| `{patient}_{vessel}_inner.gii` | `{patient}_{vessel}_inner_cpr.gii` | Lumen mesh in CPR coords |
| `{patient}_{vessel}_outer.gii` | `{patient}_{vessel}_outer_cpr.gii` | Vessel wall mesh in CPR coords |
| `{patient}_{vessel}_inner.json` | `{patient}_{vessel}_inner_cpr.json` | Lumen points in CPR coords |
| `{patient}_{vessel}_outer.json` | `{patient}_{vessel}_outer_cpr.json` | Vessel wall points in CPR coords |

**Viewing in Niivue:**

1. **Load Volumes:**
   - Cross-sectional: `{vessel}_cpr_cross.nii.gz`
   - Longitudinal: `{vessel}_cpr_long.nii.gz`

2. **Load Overlays (Optional):**
   - Mesh: `{vessel}_inner_cpr.gii`
   - Points: `{vessel}_inner_cpr.json`

3. **Interact:**
   - **Cross-sectional:** Scroll S-axis to fly through vessel.
   - **Longitudinal:** Scroll V-axis to pan depth. **Rotate** to see different angles.

### 🎮 Implementing Rotation in NiiVue

Since NiiVue uses WebGL, rotation is hardware-accelerated and smooth. You can programmatically set the viewing angle for the Longitudinal CPR:

```javascript
// 1. Load volume
await nv.loadVolumes([{url: "./data/01-BER-0088_LAD_cpr_long.nii.gz"}]);

// 2. Set to 3D Rendering mode
nv.setSliceType(nv.sliceTypeRender);

// 3. Set "Hero" angle (e.g., standard longitudinal view)
nv.scene.azimuth = 120;   // Horizontal rotation (0-360)
nv.scene.elevation = 15;  // Vertical tilt (-90 to 90)

// 4. Update scene
nv.drawScene();
```

**Pro-Tip for finding angles:**
1. Rotate manually in the browser until perfect.
2. Open console (F12) and check `nv.scene.azimuth` and `nv.scene.elevation`.
3. Copy values to your code.

### Coordinate System Details

CPR coordinates are in **voxel units** centered on the cross-section:

| Axis | Range | Meaning |
|------|-------|----------|
| X (U) | 0-119 | Normal direction, center = 60 |
| Y (V) | 0-119 | Binormal direction, center = 60 |
| Z (S) | 0 to N-1 | Slice index along centerline |

**World → CPR Transform (`transform_to_cpr.py`):**

```python
def world_to_cpr(world_point, frame_data):
    # 1. Find nearest centerline point (gives S index)
    # 2. Project displacement onto local frame:
    #    u = dot(world_point - centerline[s], normal[s])
    #    v = dot(world_point - centerline[s], binormal[s])
    # 3. Convert to voxel coords:
    #    X = u / spacing + center
    #    Y = v / spacing + center
    #    Z = s
```

The `transform_to_cpr.py` script handles this automatically for GIfTI and JSON files.

---

## 📚 Related Documentation

- **SAM3D Platform:** `sam3d.md` - AI-driven segmentation (different project)
- **Vessel Segmenter:** `vessel-segmenter.md` - Classical FMM-based segmentation pipeline
- **Funding Proposal:** `proposal.md` - DFG Koselleck grant
- **Code Implementation:** `code/medis_stl.py` - Python reference for STL mesh generation

---

## 🎯 Implementation Priorities

### Phase 1: Core Visualization (Essential)
1. ✅ MEDIS TXT parser
2. ✅ Direct mesh construction (client-side)
3. ✅ CTA volume display with NiiVue
4. ✅ Mesh overlay (lumen + vessel wall)

### Phase 2: Advanced Views (Important)
1. ✅ Centerline extraction (lumen centroids + RMF)
2. ✅ Cross-section extraction (perpendicular slices)
3. ✅ Straightened CPR construction (`medis_to_cpr.py`)
4. ⏳ Quad-view layout (viewer integration)

### Phase 3: Interactivity (Enhancement)
1. ⏳ Slider controls (position, rotation, zoom)
2. ⏳ General rotation viewer
3. ⏳ Export capabilities (STL, MZ3, PLY)
4. ⏳ Real-time performance optimization

---

**Document Version:** 2.1  
**Last Updated:** 2025-12-20  
**Project:** MEDIS Viewer Platform
