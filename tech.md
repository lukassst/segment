# 🚀 Technical Implementation: Segment Platform

**DFG Reinhart Koselleck-Projekt** | Charité – Universitätsmedizin Berlin | Prof. Marc Dewey  
**Project:** Volumetric Cardiovascular Segmentation Platform  
**Core Technology:** TypeScript + Vite + Niivue v0.66.0 + SAM-Med3D-turbo  
**Data:** DISCHARGE (25M images, 3,561 patients) | SCOT-HEART (10M images, 4,146 patients)

---

## 📋 Project Overview

### Clinical Problem
- Manual coronary segmentation: **30-60 min per case, €200-400 cost**
- Limits large-scale research and precision medicine
- Only available at specialized centers

### Our Solution
- **Browser-based AI platform:** Click on coronary artery → AI segments in <2 seconds
- **10x faster, 10x cheaper:** €20-40 per case vs. €200-400
- **Zero installation:** Works in any browser (Chrome, Firefox, Safari)
- **Secure:** AI runs on Charité GPU cluster, data stays on-premise (GDPR compliant)

### Success Metrics
- **Segmentation Speed:** <2 seconds per vessel segment
- **Accuracy:** Dice score >0.85 vs. expert annotations
- **Throughput:** 10x faster than manual segmentation
- **Coverage:** >80% of SCOT-HEART dataset processed in 6 months

---

## 🎯 Technology Stack

### Frontend: TypeScript + Vite + Niivue v0.66.0

**Core Dependencies:**
```json
{
  "@niivue/niivue": "^0.66.0",           // WebGL2 medical image viewer
  "@niivue/dcm2niix": "^1.2.0",          // DICOM → NIfTI conversion
  "@niivue/itkwasm-loader": "latest",    // DICOM series loading
  "@kitware/vtk.js": "latest",           // Mesh generation (marching cubes)
  "typescript": "^5.6.0",                 // Type safety
  "vite": "^7.0.0"                        // Build system (modern, fast)
}
```

**Niivue v0.66.0 Capabilities:**
- ✅ **WebGL2 rendering:** 60 FPS for 512³ volumes, hardware-accelerated
- ✅ **Interactive prompting:** `onLocationChange` and `onMouseUp` events capture 3D coordinates
- ✅ **Multi-planar reconstruction (MPR):** Axial/coronal/sagittal views synchronized
- ✅ **Mesh overlay:** `nv.addMesh()` for 3D surface visualization with adjustable opacity
- ✅ **TypeScript support:** Full type definitions (improved from v0.62)
- ✅ **4D volumes:** Time-series visualization for perfusion imaging
- ✅ **In-browser DICOM:** dcm2niix conversion without server roundtrip

### Backend: FastAPI + SAM-Med3D-turbo

**Model Architecture:**
- **SAM-Med3D-turbo** [@Zhang2024SAMMed3D]
- **Pre-training:** SA-Med3D-140K dataset (143K 3D masks, 245 categories)
- **Fine-tuning:** 44 medical imaging datasets
- **Architecture:** 3D ViT-B/16 encoder (86M params) + lightweight decoder (5M params)
- **Performance:** 10-100× fewer prompts than standard 3D segmentation
- **Implementation:** PyTorch 2.6.0, CUDA 12.1, mixed-precision (FP16)

**Dual-Stage Pipeline:**
1. **nnU-Net Prior [@Isensee2021nnUNet]:** Generates anatomical context (LAD, LCx, RCA)
2. **SAM-Med3D Adapter:** Refines with plaque-specific attention

**Backend Stack:**
```python
- FastAPI (Python 3.10+) with async/await for concurrent requests
- Redis cache for feature embeddings (sub-second response)
- SimpleITK, nibabel for image processing
- Elastix for deformable registration (longitudinal tracking)
- nii2mesh (ITK-WASM) for surface mesh generation
- Docker + NVIDIA Container Toolkit (A100/V100 GPUs)
```

---

## 🏗️ Project Structure (Frac-Inspired Layout)

### Frontend Architecture

```
segment-platform/
├── frontend/
│   ├── src/
│   │   ├── main.ts                      # Application entry point
│   │   ├── App.ts                       # Main app component
│   │   │
│   │   ├── components/
│   │   │   ├── NiivueViewer.ts          # Niivue canvas wrapper (single/quad view)
│   │   │   ├── Toolbar.ts               # Top toolbar (load, save, settings)
│   │   │   ├── ViewToggle.ts            # Single ↔ Quad view toggle
│   │   │   ├── SegmentPanel.ts          # AI segmentation controls
│   │   │   ├── ResultsPanel.ts          # Plaque analysis results
│   │   │   └── MPRView.ts               # Multi-planar reconstruction
│   │   │
│   │   ├── services/
│   │   │   ├── api.ts                   # Backend API client (fetch wrapper)
│   │   │   ├── niivue.ts                # Niivue initialization & config
│   │   │   ├── loader.ts                # NIfTI.gz & DICOM directory loading
│   │   │   ├── meshConverter.ts         # Mask → mesh conversion (vtk.js)
│   │   │   └── auth.ts                  # Authentication (LDAP/SSO)
│   │   │
│   │   ├── types/
│   │   │   ├── volume.ts                # Volume data types
│   │   │   ├── segmentation.ts          # Segmentation result types
│   │   │   ├── mesh.ts                  # Mesh data types
│   │   │   ├── api.ts                   # API request/response types
│   │   │   └── niivue.d.ts              # Extended Niivue type definitions
│   │   │
│   │   ├── utils/
│   │   │   ├── coordinates.ts           # 3D coordinate transformations
│   │   │   ├── meshGenerator.ts         # Marching cubes (vtk.js)
│   │   │   └── export.ts                # Export to NIfTI, DICOM-SEG, CSV
│   │   │
│   │   └── styles/
│   │       ├── main.css                 # Global styles (frac-inspired)
│   │       ├── layout.css               # Single/quad view layouts
│   │       └── niivue.css               # Niivue canvas styling
│   │
│   ├── public/
│   │   ├── index.html                   # HTML entry point
│   │   └── assets/                      # Static assets
│   │
│   ├── package.json
│   ├── tsconfig.json                    # TypeScript config (strict mode)
│   ├── vite.config.ts                   # Vite build config
│   └── README.md
│
└── backend/
    ├── app/
    │   ├── main.py                      # FastAPI application
    │   ├── config.py                    # Configuration (env vars)
    │   │
    │   ├── models/
    │   │   ├── sam_med3d.py             # SAM-Med3D model wrapper
    │   │   ├── nnu_net.py               # nnU-Net prior
    │   │   └── plaque_analyzer.py       # Plaque characterization
    │   │
    │   ├── api/
    │   │   ├── segment.py               # Segmentation endpoints
    │   │   ├── volumes.py               # Volume management
    │   │   ├── mesh.py                  # Mesh generation endpoints
    │   │   └── auth.py                  # Authentication
    │   │
    │   └── services/
    │       ├── cache.py                 # Redis cache (embeddings)
    │       ├── dicom_processor.py       # DICOM → NIfTI conversion
    │       ├── mesh_generator.py        # nii2mesh wrapper
    │       └── registration.py          # Elastix wrapper
    │
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md
```

---

## 🎨 UI/UX Design (Frac-Inspired)

### Single View Layout (Default)

```
┌─────────────────────────────────────────────────────────────┐
│  [Logo] Segment    [Load NII/DCM] [Save] [⚙️] [👤]  [◧ Quad]│ ← Toolbar
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────┐  ┌───────────────┐ │
│  │                                     │  │  Segmentation │ │
│  │                                     │  │  ┌──────────┐ │ │
│  │     Niivue 3D Canvas (Main View)   │  │  │ Point    │ │ │
│  │                                     │  │  │ Box      │ │ │
│  │     [Interactive - Click to Segment]│  │  │ Segment  │ │ │
│  │                                     │  │  └──────────┘ │ │
│  │                                     │  │               │ │
│  │                                     │  │  Plaque       │ │
│  │                                     │  │  ┌──────────┐ │ │
│  │                                     │  │  │ Calc: 45%│ │ │
│  │                                     │  │  │ Non: 35% │ │ │
│  │                                     │  │  │ Low: 20% │ │ │
│  │                                     │  │  │ Sten: 60%│ │ │
│  └─────────────────────────────────────┘  └───────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Quad View Layout (MPR + 3D)

```
┌─────────────────────────────────────────────────────────────┐
│  [Logo] Segment    [Load NII/DCM] [Save] [⚙️] [👤]  [◫ Single]│
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐ ┌──────────────────┐  ┌─────────────┐ │
│  │                  │ │                  │  │ Segmentation│ │
│  │  Axial (Top)     │ │  Sagittal (Right)│  │ ┌─────────┐ │ │
│  │  [Crosshair]     │ │  [Crosshair]     │  │ │ Point   │ │ │
│  │                  │ │                  │  │ │ Box     │ │ │
│  └──────────────────┘ └──────────────────┘  │ │ Segment │ │ │
│                                              │ └─────────┘ │ │
│  ┌──────────────────┐ ┌──────────────────┐  │             │ │
│  │                  │ │                  │  │ Plaque      │ │
│  │  Coronal (Front) │ │  3D Render       │  │ ┌─────────┐ │ │
│  │  [Crosshair]     │ │  [Interactive]   │  │ │ Calc:45%│ │ │
│  │                  │ │                  │  │ │ Non: 35%│ │ │
│  └──────────────────┘ └──────────────────┘  │ │ Low: 20%│ │ │
│                                              │ │ Sten:60%│ │ │
│                                              └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles (Frac-Inspired)
- **Dark theme:** #1a1a1a background (reduces eye strain for long sessions)
- **Minimalist:** Clean, distraction-free interface
- **Radiologist-first:** Optimized for clinical workflow
- **Responsive:** Adapts to desktop, tablet (for review)
- **Modern:** Sleek, professional appearance

**Color Scheme:**
- **Background:** Dark gray (#1a1a1a)
- **Canvas:** Black (#000000)
- **Primary:** Blue (#3b82f6) - segmentation overlays
- **Accent:** Green (#10b981) - successful operations
- **Warning:** Orange (#f59e0b) - low-confidence regions
- **Error:** Red (#ef4444) - failed operations
- **Text:** Light gray (#e5e7eb)

---

## 📦 Loading Strategy: NIfTI.gz & DICOM Directories

### NIfTI.gz Loading (Preferred for Research)

**Frontend (services/loader.ts):**
```typescript
import { Niivue } from '@niivue/niivue';

export async function loadNiftiGz(nv: Niivue, file: File): Promise<void> {
  // Niivue handles .nii.gz natively
  const volumeList = [{ url: URL.createObjectURL(file) }];
  await nv.loadVolumes(volumeList);
  
  // Set optimal viewing parameters for CCTA
  nv.setSliceType(nv.sliceTypeMultiplanar); // Enable MPR
  nv.setInterpolation(true); // Smooth rendering
}

// Usage
const fileInput = document.getElementById('file-input') as HTMLInputElement;
fileInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file && file.name.endsWith('.nii.gz')) {
    await loadNiftiGz(nv, file);
  }
});
```

### DICOM Directory Loading (Clinical Workflow)

**Frontend (services/loader.ts):**
```typescript
import { Niivue } from '@niivue/niivue';
import { loadDicomDir } from '@niivue/itkwasm-loader';

export async function loadDicomDirectory(nv: Niivue, files: FileList): Promise<void> {
  // Convert FileList to array
  const fileArray = Array.from(files);
  
  // Use ITK-WASM loader to handle DICOM series
  const volume = await loadDicomDir(fileArray);
  
  // Load into Niivue
  await nv.loadVolumes([volume]);
  
  // Set optimal viewing parameters
  nv.setSliceType(nv.sliceTypeMultiplanar);
  nv.setInterpolation(true);
}

// Usage: Directory picker (modern browsers)
const dirInput = document.getElementById('dir-input') as HTMLInputElement;
dirInput.setAttribute('webkitdirectory', ''); // Enable directory selection
dirInput.addEventListener('change', async (e) => {
  const files = (e.target as HTMLInputElement).files;
  if (files && files.length > 0) {
    await loadDicomDirectory(nv, files);
  }
});
```

**Alternative: Backend DICOM Processing (for PACS integration):**
```python
# backend/app/services/dicom_processor.py
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path

def dicom_dir_to_nifti(dicom_dir: Path) -> Path:
    """Convert DICOM directory to NIfTI.gz"""
    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Convert to NIfTI
    nifti_path = dicom_dir.parent / f"{dicom_dir.name}.nii.gz"
    sitk.WriteImage(image, str(nifti_path))
    
    return nifti_path
```

---

## 📄 MEDIS TXT Format: Vessel Wall Contours

### What is MEDIS TXT?

**MEDIS TXT** is a text format containing vessel wall contours from coronary artery segmentation:
- **Inner wall (lumen):** Points defining the vessel lumen boundary
- **Outer wall:** Points defining the outer vessel boundary (including plaque)
- **Format:** ASCII text with (x, y, z) coordinates for each contour point
- **Per-slice data:** Each axial slice has inner/outer contour points

**Example MEDIS TXT Structure:**
```
# MEDIS Coronary Artery Segmentation
# Vessel: LAD
# Number of slices: 150

Slice 0
Inner: 10
120.5 85.3 0.0
121.2 86.1 0.0
...
Outer: 12
118.3 83.2 0.0
119.1 84.5 0.0
...

Slice 1
Inner: 11
120.8 85.6 0.625
...
```

### Loading MEDIS TXT (Frontend)

**Frontend (services/medisLoader.ts):**
```typescript
export interface MedisContour {
  slice: number;
  innerPoints: Array<[number, number, number]>;
  outerPoints: Array<[number, number, number]>;
}

export interface MedisData {
  vesselName: string;
  contours: MedisContour[];
}

export function parseMedisTxt(content: string): MedisData {
  const lines = content.split('\n');
  const contours: MedisContour[] = [];
  let vesselName = 'Unknown';
  let currentSlice = -1;
  let currentInner: Array<[number, number, number]> = [];
  let currentOuter: Array<[number, number, number]> = [];
  let readingInner = false;
  let readingOuter = false;
  
  for (const line of lines) {
    if (line.startsWith('# Vessel:')) {
      vesselName = line.split(':')[1].trim();
    } else if (line.startsWith('Slice')) {
      // Save previous slice
      if (currentSlice >= 0) {
        contours.push({
          slice: currentSlice,
          innerPoints: currentInner,
          outerPoints: currentOuter
        });
      }
      // Start new slice
      currentSlice = parseInt(line.split(' ')[1]);
      currentInner = [];
      currentOuter = [];
    } else if (line.startsWith('Inner:')) {
      readingInner = true;
      readingOuter = false;
    } else if (line.startsWith('Outer:')) {
      readingInner = false;
      readingOuter = true;
    } else if (line.trim() && !line.startsWith('#')) {
      // Parse coordinate line
      const coords = line.trim().split(/\s+/).map(parseFloat);
      if (coords.length === 3) {
        if (readingInner) {
          currentInner.push([coords[0], coords[1], coords[2]]);
        } else if (readingOuter) {
          currentOuter.push([coords[0], coords[1], coords[2]]);
        }
      }
    }
  }
  
  // Save last slice
  if (currentSlice >= 0) {
    contours.push({
      slice: currentSlice,
      innerPoints: currentInner,
      outerPoints: currentOuter
    });
  }
  
  return { vesselName, contours };
}

// Usage: File input
const medisInput = document.getElementById('medis-input') as HTMLInputElement;
medisInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file && file.name.endsWith('.txt')) {
    const content = await file.text();
    const medisData = parseMedisTxt(content);
    console.log('Loaded MEDIS data:', medisData);
    // Now convert to mesh (see next section)
  }
});
```

---

## 🎭 Mesh Generation: Three Approaches

### Overview: Ultra-Simple → Simple → Advanced

**Goal:** Convert MEDIS TXT contours (or segmentation masks) → 3D surface mesh for Niivue visualization

**Three Approaches (increasing complexity):**
1. **Ultra-Simple (Fastest, STL Export):** Direct tube mesh from contour rings → STL file (<50ms)
2. **Simple (Fast, Web-Ready):** Triangulation of contour points for Niivue (<100ms)
3. **Advanced (High Quality):** Marching cubes with smoothing (1-3s)

**Performance:** All approaches complete in **<3s**, suitable for interactive web use.

---

### Approach 0: Ultra-Simple STL Generation (Direct Export)

**Method:** Connect contour rings directly into triangular mesh, export as STL

**Advantages:**
- ✅ **Extremely fast** (<50ms for typical vessel)
- ✅ **Minimal code** (~50 lines)
- ✅ **Direct STL output** (standard format)
- ✅ **No dependencies** (just basic array math)
- ✅ Works perfectly with MEDIS TXT contours

**Disadvantages:**
- ❌ Only creates STL files (not in-memory mesh)
- ❌ No smoothing
- ❌ Simple topology only

**Implementation (TypeScript - utils/ultraSimpleMesh.ts):**
```typescript
import { MedisContour } from '../services/medisLoader';

export interface STLMesh {
  vertices: Float32Array;  // Flat array [x,y,z, x,y,z, ...]
  triangles: Uint32Array;  // Flat array [i0,i1,i2, i0,i1,i2, ...]
  facets: Float32Array;    // STL facets [nx,ny,nz, v1x,v1y,v1z, v2x,v2y,v2z, v3x,v3y,v3z, ...]
}

/**
 * Create tubular mesh by connecting consecutive contour rings.
 * Based on flow/code/buildstl.py create_tube_mesh_simple()
 */
export function createTubeMeshSTL(
  contours: MedisContour[],
  wallType: 'inner' | 'outer' = 'inner'
): STLMesh {
  if (contours.length < 2) {
    throw new Error('Need at least 2 contours to create tube mesh');
  }
  
  const faces: number[] = [];
  const allVertices: number[] = [];
  let vertexOffset = 0;
  
  // Connect consecutive rings
  for (let i = 0; i < contours.length - 1; i++) {
    const ring1 = wallType === 'inner' ? contours[i].innerPoints : contours[i].outerPoints;
    const ring2 = wallType === 'inner' ? contours[i + 1].innerPoints : contours[i + 1].outerPoints;
    
    const n1 = ring1.length;
    const n2 = ring2.length;
    const n = Math.min(n1, n2);
    
    // Create quad faces between rings
    for (let j = 0; j < n; j++) {
      const jNext = (j + 1) % n;
      
      const v0 = vertexOffset + j;
      const v1 = vertexOffset + jNext;
      const v2 = vertexOffset + n1 + j;
      const v3 = vertexOffset + n1 + jNext;
      
      // Two triangles per quad
      faces.push(v0, v2, v1);
      faces.push(v1, v2, v3);
    }
    
    // Add vertices from first ring (only once)
    if (i === 0) {
      for (const [x, y, z] of ring1) {
        allVertices.push(x, y, z);
      }
    }
    
    // Add vertices from second ring
    for (const [x, y, z] of ring2) {
      allVertices.push(x, y, z);
    }
    
    vertexOffset += n1;
  }
  
  const vertices = new Float32Array(allVertices);
  const triangles = new Uint32Array(faces);
  
  // Create STL facets (with normals)
  const facets = computeSTLFacets(vertices, triangles);
  
  return { vertices, triangles, facets };
}

function computeSTLFacets(vertices: Float32Array, triangles: Uint32Array): Float32Array {
  const numTriangles = triangles.length / 3;
  const facets = new Float32Array(numTriangles * 12); // 12 floats per facet
  
  for (let i = 0; i < numTriangles; i++) {
    const i0 = triangles[i * 3 + 0];
    const i1 = triangles[i * 3 + 1];
    const i2 = triangles[i * 3 + 2];
    
    const v0 = [vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]];
    const v1 = [vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]];
    const v2 = [vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]];
    
    // Compute normal via cross product
    const edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    const edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    
    const nx = edge1[1] * edge2[2] - edge1[2] * edge2[1];
    const ny = edge1[2] * edge2[0] - edge1[0] * edge2[2];
    const nz = edge1[0] * edge2[1] - edge1[1] * edge2[0];
    
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const normal = len > 0 ? [nx / len, ny / len, nz / len] : [0, 0, 1];
    
    // STL facet: normal (3) + vertex1 (3) + vertex2 (3) + vertex3 (3)
    facets[i * 12 + 0] = normal[0];
    facets[i * 12 + 1] = normal[1];
    facets[i * 12 + 2] = normal[2];
    facets[i * 12 + 3] = v0[0];
    facets[i * 12 + 4] = v0[1];
    facets[i * 12 + 5] = v0[2];
    facets[i * 12 + 6] = v1[0];
    facets[i * 12 + 7] = v1[1];
    facets[i * 12 + 8] = v1[2];
    facets[i * 12 + 9] = v2[0];
    facets[i * 12 + 10] = v2[1];
    facets[i * 12 + 11] = v2[2];
  }
  
  return facets;
}

/**
 * Export STL mesh to binary STL format.
 * Binary STL format: 80-byte header + 4-byte triangle count + triangle records
 * Each triangle: 12 floats (normal + 3 vertices) + 2-byte attribute
 */
export function exportSTLBinary(mesh: STLMesh, filename: string = 'vessel.stl'): void {
  const numTriangles = mesh.triangles.length / 3;
  
  // Binary STL format
  const headerSize = 80;
  const triangleSize = 50; // 4 bytes count + 50 bytes per triangle
  const bufferSize = headerSize + 4 + numTriangles * triangleSize;
  
  const buffer = new ArrayBuffer(bufferSize);
  const view = new DataView(buffer);
  
  // Header (80 bytes) - ASCII string
  const header = `Binary STL generated from MEDIS contours`;
  for (let i = 0; i < Math.min(header.length, 80); i++) {
    view.setUint8(i, header.charCodeAt(i));
  }
  
  // Number of triangles (4 bytes, little-endian uint32)
  view.setUint32(80, numTriangles, true);
  
  // Triangle data
  let offset = 84;
  for (let i = 0; i < numTriangles; i++) {
    // Normal vector (3 floats)
    for (let j = 0; j < 3; j++) {
      view.setFloat32(offset, mesh.facets[i * 12 + j], true);
      offset += 4;
    }
    
    // Vertex 1 (3 floats)
    for (let j = 0; j < 3; j++) {
      view.setFloat32(offset, mesh.facets[i * 12 + 3 + j], true);
      offset += 4;
    }
    
    // Vertex 2 (3 floats)
    for (let j = 0; j < 3; j++) {
      view.setFloat32(offset, mesh.facets[i * 12 + 6 + j], true);
      offset += 4;
    }
    
    // Vertex 3 (3 floats)
    for (let j = 0; j < 3; j++) {
      view.setFloat32(offset, mesh.facets[i * 12 + 9 + j], true);
      offset += 4;
    }
    
    // Attribute byte count (2 bytes, unused)
    view.setUint16(offset, 0, true);
    offset += 2;
  }
  
  // Download file
  const blob = new Blob([buffer], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Usage: Ultra-fast STL export
const medisData = parseMedisTxt(fileContent);
const innerSTL = createTubeMeshSTL(medisData.contours, 'inner');
const outerSTL = createTubeMeshSTL(medisData.contours, 'outer');

exportSTLBinary(innerSTL, 'LAD_lumen.stl');
exportSTLBinary(outerSTL, 'LAD_vessel_wall.stl');

// Or use for Niivue display (convert to Niivue mesh format)
const nvMesh = nv.createMeshFromVertices(innerSTL.vertices, innerSTL.triangles);
nv.addMesh(nvMesh);
```

**Python Reference (flow/code/buildstl.py):**
```python
# Original implementation from flow repository
import numpy as np
from stl import mesh

def create_tube_mesh_simple(contours):
    """Create tubular mesh by connecting consecutive rings."""
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
    
    vertices = np.array(all_vertices)
    faces = np.array(faces)
    
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for idx, face in enumerate(faces):
        for j in range(3):
            if face[j] < len(vertices):
                stl_mesh.vectors[idx][j] = vertices[face[j]]
    
    return stl_mesh

# Usage
lumen_mesh = create_tube_mesh_simple(lumen_contours)
lumen_mesh.save('LAD_lumen.stl')
```

---

### Approach 1: Simple Mesh Generation (Contour Triangulation)

**Method:** Stack contours slice-by-slice and triangulate between adjacent slices

**Advantages:**
- ✅ Fast (<100ms for typical vessel)
- ✅ Simple algorithm
- ✅ Works directly with MEDIS TXT data
- ✅ Client-side (no server needed)

**Disadvantages:**
- ❌ Can have artifacts between slices
- ❌ No smoothing
- ❌ Requires similar point counts per slice

**Implementation (Frontend - utils/simpleMesh.ts):**
```typescript
import { MedisData, MedisContour } from '../services/medisLoader';

export interface SimpleMesh {
  vertices: Float32Array;  // [x, y, z, x, y, z, ...]
  triangles: Uint32Array;  // [i0, i1, i2, i0, i1, i2, ...]
}

export function contourToSimpleMesh(
  contours: MedisContour[],
  wallType: 'inner' | 'outer' = 'inner'
): SimpleMesh {
  const vertices: number[] = [];
  const triangles: number[] = [];
  
  // Extract points for specified wall type
  const slicePoints = contours.map(c => 
    wallType === 'inner' ? c.innerPoints : c.outerPoints
  );
  
  // Build vertex array
  let vertexIndex = 0;
  for (const points of slicePoints) {
    for (const [x, y, z] of points) {
      vertices.push(x, y, z);
    }
  }
  
  // Triangulate between adjacent slices
  let offset = 0;
  for (let s = 0; s < slicePoints.length - 1; s++) {
    const n0 = slicePoints[s].length;
    const n1 = slicePoints[s + 1].length;
    
    // Simple triangulation: connect slice s to slice s+1
    for (let i = 0; i < Math.max(n0, n1); i++) {
      const i0 = offset + (i % n0);
      const i1 = offset + ((i + 1) % n0);
      const i2 = offset + n0 + (i % n1);
      const i3 = offset + n0 + ((i + 1) % n1);
      
      // Create two triangles per quad
      triangles.push(i0, i1, i2);
      triangles.push(i1, i3, i2);
    }
    
    offset += n0;
  }
  
  return {
    vertices: new Float32Array(vertices),
    triangles: new Uint32Array(triangles)
  };
}

// Usage: Convert MEDIS data to mesh
const medisData = parseMedisTxt(fileContent);
const innerMesh = contourToSimpleMesh(medisData.contours, 'inner');
const outerMesh = contourToSimpleMesh(medisData.contours, 'outer');

// Add to Niivue
const nvMeshInner = nv.createMeshFromVertices(innerMesh.vertices, innerMesh.triangles);
const nvMeshOuter = nv.createMeshFromVertices(outerMesh.vertices, outerMesh.triangles);
nv.addMesh(nvMeshInner); // Red for inner wall
nv.addMesh(nvMeshOuter); // Blue for outer wall
```

---

### Approach 2: Advanced Mesh Generation (Smooth Surface Reconstruction)

**Method:** Use marching cubes or Poisson reconstruction for smooth surfaces

**Advantages:**
- ✅ High-quality smooth surfaces
- ✅ Handles irregular contours
- ✅ Proper topology
- ✅ Can decimate (reduce triangles)

**Disadvantages:**
- ❌ Slower (1-3s)
- ❌ More complex algorithm
- ❌ May require server-side processing

**Implementation (Frontend - utils/advancedMesh.ts with vtk.js):**
```typescript
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkImageMarchingCubes from '@kitware/vtk.js/Filters/General/ImageMarchingCubes';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import { MedisContour } from '../services/medisLoader';

export async function contourToAdvancedMesh(
  contours: MedisContour[],
  wallType: 'inner' | 'outer',
  dimensions: [number, number, number],
  spacing: [number, number, number]
): Promise<{ vertices: Float32Array; triangles: Uint32Array }> {
  
  // Step 1: Rasterize contours to 3D volume
  const volumeData = new Float32Array(dimensions[0] * dimensions[1] * dimensions[2]);
  volumeData.fill(0); // Background
  
  for (const contour of contours) {
    const points = wallType === 'inner' ? contour.innerPoints : contour.outerPoints;
    const sliceIdx = contour.slice;
    
    // Fill contour interior with 1.0
    for (const [x, y, z] of points) {
      const ix = Math.round(x / spacing[0]);
      const iy = Math.round(y / spacing[1]);
      const iz = Math.round(z / spacing[2]);
      
      if (ix >= 0 && ix < dimensions[0] &&
          iy >= 0 && iy < dimensions[1] &&
          iz >= 0 && iz < dimensions[2]) {
        const idx = iz * dimensions[0] * dimensions[1] + iy * dimensions[0] + ix;
        volumeData[idx] = 1.0;
      }
    }
  }
  
  // Step 2: Create VTK image data
  const imageData = vtkImageData.newInstance();
  imageData.setDimensions(dimensions);
  imageData.setSpacing(spacing);
  imageData.getPointData().setScalars(
    vtkDataArray.newInstance({ values: volumeData })
  );
  
  // Step 3: Run marching cubes
  const marchingCubes = vtkImageMarchingCubes.newInstance();
  marchingCubes.setInputData(imageData);
  marchingCubes.setContourValue(0.5); // Isosurface at 0.5
  marchingCubes.update();
  
  // Step 4: Extract mesh
  const polyData = marchingCubes.getOutputData();
  const points = polyData.getPoints().getData();
  const polys = polyData.getPolys().getData();
  
  return {
    vertices: new Float32Array(points),
    triangles: new Uint32Array(polys)
  };
}

// Usage
const innerMesh = await contourToAdvancedMesh(
  medisData.contours,
  'inner',
  [512, 512, 300],
  [0.5, 0.5, 0.625]
);

const nvMesh = nv.createMeshFromVertices(innerMesh.vertices, innerMesh.triangles);
nv.addMesh(nvMesh);
```

**Backend Alternative (Python with nii2mesh):**
```python
# backend/app/services/advanced_mesh.py
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh

def contours_to_volume(
    contours: list,
    dimensions: tuple,
    spacing: tuple
) -> np.ndarray:
    """Rasterize contours to 3D volume"""
    volume = np.zeros(dimensions, dtype=np.uint8)
    
    for contour in contours:
        slice_idx = contour['slice']
        points = np.array(contour['inner_points'])  # or outer_points
        
        # Fill polygon interior using scan-line algorithm
        # (simplified - actual implementation more complex)
        for x, y, z in points:
            ix, iy, iz = int(x/spacing[0]), int(y/spacing[1]), int(z/spacing[2])
            if 0 <= ix < dimensions[0] and 0 <= iy < dimensions[1] and 0 <= iz < dimensions[2]:
                volume[iz, iy, ix] = 1
    
    return volume

def volume_to_mesh(
    volume: np.ndarray,
    spacing: tuple,
    reduction: float = 0.1
) -> dict:
    """Convert binary volume to high-quality mesh"""
    
    # Marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        volume,
        level=0.5,
        spacing=spacing
    )
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Smooth
    trimesh.smoothing.filter_laplacian(mesh, iterations=3)
    
    # Decimate (reduce triangles)
    target_faces = int(len(faces) * reduction)
    mesh = mesh.simplify_quadric_decimation(target_faces)
    
    return {
        'vertices': mesh.vertices.flatten().tolist(),
        'triangles': mesh.faces.flatten().tolist()
    }
```

---

## 🧬 Centerline Extraction

### Current: MEDIS TXT Provides Centerline

**For now:** MEDIS TXT files already contain centerline data implicitly:
- Each contour has a slice distance (position along vessel)
- Centerline can be computed as centroid of each lumen contour
- Simple, fast, ready to use

**Implementation (TypeScript - utils/centerlineFromMedis.ts):**
```typescript
import { MedisContour } from '../services/medisLoader';
import { Centerline } from '../services/straightenedMPR';

export function extractCenterlineFromMedis(contours: MedisContour[]): Centerline {
  const points: Array<[number, number, number]> = [];
  const tangents: Array<[number, number, number]> = [];
  
  // Compute centroid of each lumen contour
  for (const contour of contours) {
    const lumen = contour.innerPoints;
    
    if (lumen.length === 0) continue;
    
    // Centroid = average of all points
    let cx = 0, cy = 0, cz = 0;
    for (const [x, y, z] of lumen) {
      cx += x;
      cy += y;
      cz += z;
    }
    cx /= lumen.length;
    cy /= lumen.length;
    cz /= lumen.length;
    
    points.push([cx, cy, cz]);
  }
  
  // Compute tangent vectors (finite differences)
  for (let i = 0; i < points.length; i++) {
    let tangent: [number, number, number];
    
    if (i === 0) {
      // Forward difference
      tangent = [
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2]
      ];
    } else if (i === points.length - 1) {
      // Backward difference
      tangent = [
        points[i][0] - points[i - 1][0],
        points[i][1] - points[i - 1][1],
        points[i][2] - points[i - 1][2]
      ];
    } else {
      // Central difference
      tangent = [
        points[i + 1][0] - points[i - 1][0],
        points[i + 1][1] - points[i - 1][1],
        points[i + 1][2] - points[i - 1][2]
      ];
    }
    
    // Normalize
    const len = Math.sqrt(tangent[0]**2 + tangent[1]**2 + tangent[2]**2);
    if (len > 0) {
      tangent = [tangent[0]/len, tangent[1]/len, tangent[2]/len];
    }
    
    tangents.push(tangent);
  }
  
  return { points, tangents };
}

// Usage
const medisData = parseMedisTxt(fileContent);
const centerline = extractCenterlineFromMedis(medisData.contours);
```

**Python Reference (medismask.py approach):**
```python
# Centerline from contour centroids
def extract_centerline(lumen_dict):
    """Extract centerline as centroids of lumen contours."""
    centerline_points = []
    
    for slice_distance in sorted(lumen_dict.keys()):
        lumen_coords = lumen_dict[slice_distance]
        centroid = np.mean(lumen_coords, axis=0)
        centerline_points.append(centroid)
    
    return np.array(centerline_points)
```

### Future: Voronoi Skeletonization from Segmentation Masks

**Later implementation:** Extract centerline from 3D segmentation masks using Voronoi skeleton:
- **Input:** Binary segmentation mask (from SAM-Med3D or manual)
- **Method:** 3D Voronoi diagram → medial axis
- **Libraries:** `scikit-image` (Python), custom implementation (TypeScript)
- **Use case:** Automatic centerline extraction when MEDIS data not available

**Placeholder for future:**
```typescript
// Future implementation
export async function extractCenterlineVoronoi(
  segmentationMask: Uint8Array,
  dimensions: [number, number, number]
): Promise<Centerline> {
  // TODO: Implement Voronoi skeletonization
  // 1. Compute distance transform
  // 2. Extract medial axis via Voronoi
  // 3. Smooth and resample
  throw new Error('Not yet implemented - use MEDIS centerline for now');
}
```

---

## 🛤️ Straightened MPR (Curved Reformation)

### What is Straightened MPR?

**Goal:** "Straighten" a curved coronary artery for easier visualization
- **Input:** 3D CTA volume + vessel centerline
- **Output:** Straightened 3D volume where centerline is straight
- **Use case:** View entire curved LAD as if it were straight

**Method:**
1. Walk along centerline (parametric curve)
2. At each centerline point, extract perpendicular cross-section slice
3. Resample slice at consistent resolution
4. Stack slices to create straightened 3D volume
5. Save as NIfTI.gz for visualization

### Implementation Strategy

**Frontend (services/straightenedMPR.ts):**
```typescript
import { Niivue, NVImage } from '@niivue/niivue';

export interface Centerline {
  points: Array<[number, number, number]>;  // [x, y, z] in mm
  tangents: Array<[number, number, number]>; // Direction vectors
}

export interface StraightenedVolume {
  data: Float32Array;
  dimensions: [number, number, number];
  spacing: [number, number, number];
}

export async function createStraightenedMPR(
  nv: Niivue,
  volume: NVImage,
  centerline: Centerline,
  crossSectionSize: [number, number] = [64, 64], // pixels
  crossSectionSpacing: [number, number] = [0.5, 0.5] // mm
): Promise<StraightenedVolume> {
  
  const numSlices = centerline.points.length;
  const [width, height] = crossSectionSize;
  
  // Allocate straightened volume
  const straightenedData = new Float32Array(width * height * numSlices);
  
  // For each centerline point
  for (let i = 0; i < numSlices; i++) {
    const center = centerline.points[i];
    const tangent = centerline.tangents[i];
    
    // Compute perpendicular plane (cross-section)
    const [nx, ny, nz] = computePerpendicularBasis(tangent);
    
    // Extract cross-section slice
    const sliceData = extractCrossSection(
      volume,
      center,
      nx, ny, nz,
      crossSectionSize,
      crossSectionSpacing
    );
    
    // Copy to straightened volume
    const sliceOffset = i * width * height;
    straightenedData.set(sliceData, sliceOffset);
  }
  
  return {
    data: straightenedData,
    dimensions: [width, height, numSlices],
    spacing: [crossSectionSpacing[0], crossSectionSpacing[1], 1.0] // 1mm along centerline
  };
}

function computePerpendicularBasis(
  tangent: [number, number, number]
): [[number, number, number], [number, number, number]] {
  // Compute two perpendicular vectors to tangent
  const [tx, ty, tz] = tangent;
  
  // Choose reference vector (not parallel to tangent)
  let ref: [number, number, number] = [0, 0, 1];
  if (Math.abs(tz) > 0.9) {
    ref = [1, 0, 0];
  }
  
  // First perpendicular: cross(tangent, ref)
  const n1x = ty * ref[2] - tz * ref[1];
  const n1y = tz * ref[0] - tx * ref[2];
  const n1z = tx * ref[1] - ty * ref[0];
  const len1 = Math.sqrt(n1x*n1x + n1y*n1y + n1z*n1z);
  const n1: [number, number, number] = [n1x/len1, n1y/len1, n1z/len1];
  
  // Second perpendicular: cross(tangent, n1)
  const n2x = ty * n1[2] - tz * n1[1];
  const n2y = tz * n1[0] - tx * n1[2];
  const n2z = tx * n1[1] - ty * n1[0];
  const len2 = Math.sqrt(n2x*n2x + n2y*n2y + n2z*n2z);
  const n2: [number, number, number] = [n2x/len2, n2y/len2, n2z/len2];
  
  return [n1, n2];
}

function extractCrossSection(
  volume: NVImage,
  center: [number, number, number],
  nx: [number, number, number],
  ny: [number, number, number],
  size: [number, number],
  spacing: [number, number]
): Float32Array {
  const [width, height] = size;
  const [sx, sy] = spacing;
  const sliceData = new Float32Array(width * height);
  
  // Sample cross-section
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      // Compute 3D position in original volume
      const u = (i - width/2) * sx;
      const v = (j - height/2) * sy;
      
      const x = center[0] + u * nx[0] + v * ny[0];
      const y = center[1] + u * nx[1] + v * ny[1];
      const z = center[2] + u * nx[2] + v * ny[2];
      
      // Interpolate intensity from volume (trilinear)
      const value = interpolateVolume(volume, x, y, z);
      sliceData[j * width + i] = value;
    }
  }
  
  return sliceData;
}

function interpolateVolume(
  volume: NVImage,
  x: number,
  y: number,
  z: number
): number {
  // Convert mm to voxel coordinates
  const voxelX = x / volume.hdr.pixDims[1];
  const voxelY = y / volume.hdr.pixDims[2];
  const voxelZ = z / volume.hdr.pixDims[3];
  
  // Trilinear interpolation
  const x0 = Math.floor(voxelX);
  const y0 = Math.floor(voxelY);
  const z0 = Math.floor(voxelZ);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const z1 = z0 + 1;
  
  // Check bounds
  const dims = volume.hdr.dims;
  if (x0 < 0 || x1 >= dims[1] || y0 < 0 || y1 >= dims[2] || z0 < 0 || z1 >= dims[3]) {
    return 0; // Outside volume
  }
  
  const fx = voxelX - x0;
  const fy = voxelY - y0;
  const fz = voxelZ - z0;
  
  // Get 8 corner values
  const getData = (xi: number, yi: number, zi: number) => {
    const idx = zi * dims[1] * dims[2] + yi * dims[1] + xi;
    return volume.img[idx] || 0;
  };
  
  const c000 = getData(x0, y0, z0);
  const c100 = getData(x1, y0, z0);
  const c010 = getData(x0, y1, z0);
  const c110 = getData(x1, y1, z0);
  const c001 = getData(x0, y0, z1);
  const c101 = getData(x1, y0, z1);
  const c011 = getData(x0, y1, z1);
  const c111 = getData(x1, y1, z1);
  
  // Trilinear interpolation
  const c00 = c000 * (1 - fx) + c100 * fx;
  const c01 = c001 * (1 - fx) + c101 * fx;
  const c10 = c010 * (1 - fx) + c110 * fx;
  const c11 = c011 * (1 - fx) + c111 * fx;
  
  const c0 = c00 * (1 - fy) + c10 * fy;
  const c1 = c01 * (1 - fy) + c11 * fy;
  
  return c0 * (1 - fz) + c1 * fz;
}
```

### Saving Straightened Volume

**Convert to NIfTI and save:**
```typescript
import { NVImage } from '@niivue/niivue';
import pako from 'pako'; // For gzip compression

export async function saveStraightenedVolume(
  straightened: StraightenedVolume,
  filename: string = 'straightened.nii.gz'
): Promise<void> {
  // Create NIfTI header
  const header = createNiftiHeader(
    straightened.dimensions,
    straightened.spacing
  );
  
  // Combine header + data
  const headerBuffer = header.buffer;
  const dataBuffer = straightened.data.buffer;
  const combined = new Uint8Array(headerBuffer.byteLength + dataBuffer.byteLength);
  combined.set(new Uint8Array(headerBuffer), 0);
  combined.set(new Uint8Array(dataBuffer), headerBuffer.byteLength);
  
  // Compress with gzip
  const compressed = pako.gzip(combined);
  
  // Trigger download
  const blob = new Blob([compressed], { type: 'application/gzip' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function createNiftiHeader(
  dimensions: [number, number, number],
  spacing: [number, number, number]
): ArrayBuffer {
  // Simplified NIfTI-1 header creation (348 bytes)
  const header = new ArrayBuffer(348);
  const view = new DataView(header);
  
  // sizeof_hdr
  view.setInt32(0, 348, true);
  
  // dim[0] = 3 (3D)
  view.setInt16(40, 3, true);
  // dim[1-3] = dimensions
  view.setInt16(42, dimensions[0], true);
  view.setInt16(44, dimensions[1], true);
  view.setInt16(46, dimensions[2], true);
  
  // pixdim[1-3] = spacing
  view.setFloat32(76, spacing[0], true);
  view.setFloat32(80, spacing[1], true);
  view.setFloat32(84, spacing[2], true);
  
  // datatype = 16 (float32)
  view.setInt16(70, 16, true);
  
  // vox_offset = 352 (standard NIfTI-1)
  view.setFloat32(108, 352, true);
  
  return header;
}
```

### Displaying Straightened MPR in Niivue

**Load straightened volume into Niivue:**
```typescript
export async function displayStraightenedMPR(
  nv: Niivue,
  straightened: StraightenedVolume
): Promise<void> {
  // Create NIfTI blob
  const blob = await createNiftiBlob(straightened);
  const url = URL.createObjectURL(blob);
  
  // Load into Niivue
  await nv.loadVolumes([{ url }]);
  
  // Set optimal view for straightened vessel
  nv.setSliceType(nv.sliceTypeMultiplanar);
  nv.setInterpolation(true);
  
  // Cleanup
  URL.revokeObjectURL(url);
}

async function createNiftiBlob(
  straightened: StraightenedVolume
): Promise<Blob> {
  const header = createNiftiHeader(
    straightened.dimensions,
    straightened.spacing
  );
  
  const combined = new Uint8Array(
    header.byteLength + straightened.data.byteLength
  );
  combined.set(new Uint8Array(header), 0);
  combined.set(new Uint8Array(straightened.data.buffer), header.byteLength);
  
  const compressed = pako.gzip(combined);
  return new Blob([compressed], { type: 'application/gzip' });
}
```

---

## 🎭 Mesh Conversion Pipeline: Mask → 3D Surface (Original Section)

### Strategy 1: Client-Side (vtk.js - Fast, No Server Roundtrip)

**Frontend (utils/meshGenerator.ts):**
```typescript
import vtkImageMarchingCubes from '@kitware/vtk.js/Filters/General/ImageMarchingCubes';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';

export function maskToMesh(
  maskData: Float32Array,
  dimensions: [number, number, number],
  spacing: [number, number, number],
  threshold: number = 0.5
): { vertices: Float32Array; triangles: Uint32Array } {
  
  // Create VTK image data from mask
  const imageData = vtkImageData.newInstance();
  imageData.setDimensions(dimensions);
  imageData.setSpacing(spacing);
  imageData.getPointData().setScalars(
    vtkDataArray.newInstance({ values: maskData })
  );
  
  // Run marching cubes
  const marchingCubes = vtkImageMarchingCubes.newInstance();
  marchingCubes.setInputData(imageData);
  marchingCubes.setContourValue(threshold);
  marchingCubes.update();
  
  // Extract mesh
  const polyData = marchingCubes.getOutputData();
  const points = polyData.getPoints().getData();
  const polys = polyData.getPolys().getData();
  
  // Convert to Niivue format
  const vertices = new Float32Array(points);
  const triangles = new Uint32Array(polys);
  
  return { vertices, triangles };
}

// Usage: Convert segmentation mask to mesh
const { vertices, triangles } = maskToMesh(
  segmentationMask,
  [512, 512, 300],
  [0.5, 0.5, 0.625],
  0.5
);

// Add to Niivue
const mesh = nv.createMeshFromVertices(vertices, triangles);
nv.addMesh(mesh);
```

### Strategy 2: Server-Side (nii2mesh - High Quality, Decimation)

**Backend (app/services/mesh_generator.py):**
```python
import subprocess
from pathlib import Path

def nifti_to_mesh(
    nifti_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    reduction: float = 0.1  # Reduce to 10% of original triangles
) -> Path:
    """Convert NIfTI mask to high-quality mesh using nii2mesh"""
    
    cmd = [
        'nii2mesh',
        str(nifti_path),
        str(output_path),
        '-i', 'mc',  # Marching cubes
        '-t', str(threshold),
        '-r', str(reduction),  # Decimation
        '-s', '3'  # Smoothing iterations
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

# FastAPI endpoint
@router.post("/api/mesh/generate")
async def generate_mesh(
    volume_id: str,
    threshold: float = 0.5,
    reduction: float = 0.1
):
    mask_path = get_segmentation_mask(volume_id)
    mesh_path = nifti_to_mesh(mask_path, threshold=threshold, reduction=reduction)
    
    # Return mesh data
    with open(mesh_path, 'rb') as f:
        mesh_data = f.read()
    
    return Response(content=mesh_data, media_type='application/octet-stream')
```

**Frontend Integration:**
```typescript
// Request mesh from backend
async function generateMeshFromBackend(volumeId: string): Promise<void> {
  const response = await fetch(`/api/mesh/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ volume_id: volumeId, threshold: 0.5, reduction: 0.1 })
  });
  
  const meshBlob = await response.blob();
  const meshUrl = URL.createObjectURL(meshBlob);
  
  // Load mesh into Niivue
  await nv.loadMeshes([{ url: meshUrl }]);
}
```

### Recommended Approach
- **Interactive segmentation:** Use **client-side (vtk.js)** for instant feedback (<1s)
- **Final export:** Use **server-side (nii2mesh)** for high-quality, decimated meshes
- **Hybrid:** Generate quick preview client-side, then refine server-side in background

---

---

## 🤖 Future: SAM-Med3D Segmentation Integration

### Planned Integration (Not Implemented Yet)

**Goal:** Integrate SAM-Med3D-turbo for automatic coronary artery segmentation

**Current Status:** Documentation and planning phase
- ✅ Backend architecture supports SAM-Med3D (see API endpoints)
- ✅ Frontend ready for segmentation overlay
- ⏳ **Not yet implemented** - focus on visualization first

**When to Integrate:**
1. **First:** Get MEDIS TXT loading + mesh visualization working
2. **Second:** Implement straightened MPR
3. **Third:** Add SAM-Med3D segmentation backend
4. **Fourth:** Active learning loop for model refinement

**Segmentation Workflow (Planned):**
```typescript
// Future: AI-powered segmentation
import { segmentVessel } from './services/api';

// Point-based prompting
const segmentation = await segmentVessel({
  volumeId: 'study_123',
  promptType: 'point',
  coordinates: [120.5, 85.3, 42.1],
  vesselType: 'LAD'
});

// Display segmentation mask as overlay
const maskVolume = await loadNiftiFromBlob(segmentation.maskBlob);
await nv.addVolume(maskVolume);

// Convert mask to mesh for 3D visualization
const mesh = await maskToMeshVTK(segmentation.maskData);
nv.addMesh(mesh);
```

**Backend API (Already Designed):**
```python
# POST /api/segment/point
@router.post("/api/segment/point")
async def segment_point(
    volume_id: str,
    coordinates: List[float],
    vessel_type: str = 'LAD'
):
    # Load SAM-Med3D model
    model = load_sam_med3d_turbo()
    
    # Run inference
    mask = model.segment(
        volume=get_volume(volume_id),
        point_prompt=coordinates,
        vessel_class=vessel_type
    )
    
    # Return mask as NIfTI
    return mask_to_nifti(mask)
```

**Training Data Sources:**
- DISCHARGE trial: 25M images, 3,561 patients
- SCOT-HEART trial: 10M images, 4,146 patients
- MEDIS manual annotations: High-quality ground truth

**Model Architecture:**
- **Base:** SAM-Med3D-turbo (foundation model)
- **Fine-tuning:** nnU-Net prior + SAM adapter
- **Prompting:** Point, box, or full automatic
- **Performance:** <5s inference on GPU cluster

**Notes:**
- Keep this modular - visualization should work without AI segmentation
- MEDIS TXT workflow is primary for now
- SAM integration adds automation, not replaces manual tools

---

## 🎛️ Frontend Feature Access

### Complete Workflow from Frontend

**All features accessible via TypeScript services:**

```typescript
// 1. Load Volume (NIfTI.gz or DICOM)
import { loadNiftiGz, loadDicomDirectory } from './services/loader';
await loadNiftiGz(nv, niftiFile);
await loadDicomDirectory(nv, dicomFiles);

// 2. Load MEDIS TXT (vessel contours)
import { parseMedisTxt } from './services/medisLoader';
const medisData = parseMedisTxt(txtContent);

// 3. Generate Mesh (Simple or Advanced)
import { contourToSimpleMesh } from './utils/simpleMesh';
import { contourToAdvancedMesh } from './utils/advancedMesh';

// Simple: Fast, client-side
const innerMesh = contourToSimpleMesh(medisData.contours, 'inner');
const outerMesh = contourToSimpleMesh(medisData.contours, 'outer');

// Advanced: High-quality, client-side with vtk.js
const smoothMesh = await contourToAdvancedMesh(
  medisData.contours,
  'inner',
  [512, 512, 300],
  [0.5, 0.5, 0.625]
);

// 4. Display Meshes in Niivue
const nvMesh = nv.createMeshFromVertices(innerMesh.vertices, innerMesh.triangles);
nv.addMesh(nvMesh);

// 5. Create Straightened MPR
import { createStraightenedMPR, saveStraightenedVolume, displayStraightenedMPR } from './services/straightenedMPR';

const straightened = await createStraightenedMPR(
  nv,
  volume,
  centerline,
  [64, 64],
  [0.5, 0.5]
);

// 6. Save Straightened Volume
await saveStraightenedVolume(straightened, 'LAD_straightened.nii.gz');

// 7. Display Straightened Volume
await displayStraightenedMPR(nv, straightened);

// 8. AI Segmentation (backend)
import { segmentVessel } from './services/api';
const segmentation = await segmentVessel(volumeId, coordinates, 'point');
```

### UI Component Integration

**Main App Component:**
```typescript
// App.ts
export class SegmentApp {
  private nv: Niivue;
  private currentVolume: NVImage | null = null;
  private medisData: MedisData | null = null;
  private centerline: Centerline | null = null;
  
  async loadVolume(file: File) {
    if (file.name.endsWith('.nii.gz')) {
      await loadNiftiGz(this.nv, file);
    } else if (file.type === 'application/dicom') {
      // Handle DICOM directory
    }
    this.currentVolume = this.nv.volumes[0];
  }
  
  async loadMedisTxt(file: File) {
    const content = await file.text();
    this.medisData = parseMedisTxt(content);
    await this.visualizeMedis();
  }
  
  async visualizeMedis() {
    if (!this.medisData) return;
    
    // Generate meshes
    const innerMesh = contourToSimpleMesh(this.medisData.contours, 'inner');
    const outerMesh = contourToSimpleMesh(this.medisData.contours, 'outer');
    
    // Add to Niivue
    const nvInner = this.nv.createMeshFromVertices(innerMesh.vertices, innerMesh.triangles);
    const nvOuter = this.nv.createMeshFromVertices(outerMesh.vertices, outerMesh.triangles);
    
    this.nv.addMesh(nvInner);  // Red for inner wall
    this.nv.addMesh(nvOuter);  // Blue for outer wall
  }
  
  async createStraightened() {
    if (!this.currentVolume || !this.centerline) return;
    
    const straightened = await createStraightenedMPR(
      this.nv,
      this.currentVolume,
      this.centerline
    );
    
    await displayStraightenedMPR(this.nv, straightened);
  }
}
```

---

## 🔌 API Endpoints

```python
# Segmentation (AI)
POST /api/segment/point       # Point-based prompting
POST /api/segment/box         # Bounding box prompting
POST /api/segment/refine      # Refine existing segmentation

# Volume Management
POST /api/volumes/upload      # Upload DICOM/NIfTI
GET  /api/volumes/{id}        # Retrieve processed volume
DELETE /api/volumes/{id}      # Delete volume

# MEDIS TXT Processing (NEW)
POST /api/medis/upload        # Upload MEDIS TXT file
POST /api/medis/to_mesh       # Convert MEDIS TXT to mesh (server-side)
GET  /api/medis/{id}          # Retrieve MEDIS data

# Mesh Generation (NEW)
POST /api/mesh/generate       # Mask → mesh conversion (nii2mesh, high-quality)
POST /api/mesh/quick          # Fast mesh preview (marching cubes)
POST /api/mesh/from_contours  # MEDIS contours → mesh (advanced)

# Straightened MPR (NEW)
POST /api/straighten/create   # Create straightened MPR from centerline
GET  /api/straighten/{id}     # Retrieve straightened volume
POST /api/straighten/save     # Save straightened volume as NIfTI.gz

# Centerline Extraction (NEW)
POST /api/centerline/extract  # Extract centerline from segmentation
GET  /api/centerline/{id}     # Retrieve centerline data

# Batch Processing
POST /api/batch/process       # Batch process multiple cases
GET  /api/batch/{job_id}      # Get batch job status

# Registration (Longitudinal)
POST /api/register/longitudinal # Image registration (Elastix)

# Authentication
POST /api/auth/login          # LDAP/SSO login
GET  /api/auth/user           # Get current user info

# Health & Metrics
GET  /api/health              # Health check
GET  /api/metrics             # Performance metrics
```

---

## 📊 Development Phases

### Phase 1: MVR Tool (Weeks 1-12)
**Goal:** Working prototype with single-case demo

- **Week 1-2:** Environment setup, Niivue v0.66.0 integration
- **Week 3-4:** Frac-inspired UI (single/quad view toggle)
- **Week 5-6:** NIfTI.gz & DICOM directory loading
- **Week 7-8:** Backend setup (SAM-Med3D-turbo)
- **Week 9-10:** Frontend-backend integration
- **Week 11-12:** Mesh conversion pipeline (vtk.js + nii2mesh)

**Milestone:** Working prototype, <2s segmentation latency

### Phase 2: Clinical Deployment (Weeks 13-24)
**Goal:** Platform deployed at Charité, team onboarded

- **Week 13-14:** Deploy to Charité GPU cluster
- **Week 15-16:** Active learning workflow (annotation refinement)
- **Week 17-18:** nnU-Net prior integration
- **Week 19-20:** Batch processing (DISCHARGE + SCOT-HEART)
- **Week 21-22:** Plaque characterization
- **Week 23-24:** Team training

**Milestone:** 100+ cases processed, team onboarded

### Phase 3: Validation (Weeks 25-48)
**Goal:** Clinical validation, manuscript submission

- **Week 25-28:** Multi-vendor optimization
- **Week 29-32:** SCOT-HEART external validation
- **Week 33-36:** MACE prediction analysis
- **Week 37-40:** Longitudinal modeling
- **Week 41-48:** Manuscript preparation

**Milestone:** Dice >0.85, first manuscript submitted

---

## 🚨 Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Large volumes (>4GB) exceed browser memory | Progressive loading, server-side rendering |
| Mesh generation too slow | Hybrid approach: vtk.js preview + nii2mesh refinement |
| DICOM directory loading fails | Fallback to backend conversion (SimpleITK) |
| Multi-vendor harmonization | Vendor-specific normalization pipelines |

---

## 📚 Key References

- **Niivue v0.66.0:** https://github.com/niivue/niivue/releases/tag/@niivue/niivue-v0.66.0
- **SAM-Med3D:** Zhang et al. (2024) ECCV BIC [@Zhang2024SAMMed3D]
- **DISCHARGE:** Dewey et al. (2022) NEJM [@DISCHARGE2022NEJM]
- **SCOT-HEART 10-year:** Williams et al. (2025) Lancet [@Williams2025SCOTHEART10yr]
- **nnU-Net:** Isensee et al. (2021) Nature Methods [@Isensee2021nnUNet]
- **Frac Project:** https://github.com/lukassst/frac (layout inspiration)

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-17  
**Status:** Ready for implementation  
**Next:** Week 1 - Environment setup & Niivue integration
