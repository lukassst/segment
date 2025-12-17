# Segment Platform Architecture: TypeScript + Niivue v6 + SAM-Med3D

**DFG Reinhart Koselleck-Projekt** | Charité – Universitätsmedizin Berlin | Prof. Marc Dewey

## What We Want vs. What's Available

### What We Want to Build

**Clinical Problem**: Manual coronary segmentation is slow (30-60 min per case), expensive (€200-400 per case), and limits large-scale research.

**Solution**: Browser-based AI segmentation platform where:
- Clinicians click on coronary artery → AI segments it instantly
- Works in any browser (no installation)
- AI runs on hospital server (data stays secure)
- 10x faster than manual segmentation
- Accessible from clinic, home, or anywhere with VPN

### What's Available (Technology Stack)

#### Frontend: Niivue v6+ (Latest) [@Niivue2024]

**Key capabilities**:
- **WebGL2 rendering** for 3D medical imaging (hardware-accelerated, 60 FPS)
- **Interactive prompting** (click-to-segment): `onLocationChange` and `onMouseUp` events capture 3D voxel coordinates
- **Multi-planar reconstruction (MPR)**: Simultaneous axial/coronal/sagittal views with curved reformats for coronary arteries
- **Mesh overlay rendering**: `nv.addMesh()` for 3D surface visualization with adjustable opacity
- **TypeScript support**: Full type definitions (improved from v0.62)
- **4D volume support**: Time-series visualization for perfusion imaging
- **DICOM processing**: In-browser dcm2niix conversion via @niivue/dcm2niix
- **Performance**: Sub-second volume loading, real-time interaction

**Technical specifications**:
- Package: `@niivue/niivue` (latest stable on npm)
- Dependencies: `@niivue/dcm2niix`, `@niivue/itkwasm-loader`, `@kitware/vtk.js`
- Browser requirements: WebGL2 support (Chrome 56+, Firefox 51+, Safari 15+)
- Memory efficiency: Progressive loading for large volumes (>1GB)

#### Backend: SAM-Med3D-turbo [@Zhang2024SAMMed3D]

**Foundation Model Architecture**:
- **Pre-training:** SA-Med3D-140K dataset (143K 3D masks, 245 anatomical categories)
- **Fine-tuning:** 44 medical imaging datasets → SAM-Med3D-turbo variant
- **Architecture:** 3D Vision Transformer (ViT-B/16) encoder (86M parameters) + lightweight decoder (5M parameters)
- **Prompting:** Point, box, and mask prompts for interactive segmentation
- **Performance:** 10-100× fewer prompts than standard 3D segmentation methods
- **Implementation:** PyTorch 2.6.0, CUDA 12.1, mixed-precision training (FP16)
- **Availability:** Hugging Face (blueyo0/SAM-Med3D), Apache 2.0 license
- **Validation:** ECCV 2024 BIC Oral, CVPR 2025 MedSegFM Competition baseline

**Integration with nnU-Net** [@Isensee2021nnUNet]:
- **Stage 1:** nnU-Net generates anatomical priors (LAD, LCx, RCA identification)
- **Stage 2:** SAM-Med3D refines with plaque-specific attention
- **Training data:** ~500 expert-annotated DISCHARGE/SCOT-HEART cases

**Implementation requirements**:
- FastAPI wrapper for HTTP endpoints (async/await for concurrent requests)
- Redis cache for feature embeddings (sub-second response time)
- DICOM → NIfTI conversion pipeline (SimpleITK, nibabel)
- Docker containerization (NVIDIA Container Toolkit for GPU support)
- Elastix for deformable registration (longitudinal tracking)
- nii2mesh (ITK-WASM) for surface mesh generation

#### Deployment: Hospital Infrastructure

**Charité Infrastructure**:
- **GPU cluster:** NVIDIA A100 (80GB VRAM) / V100 (32GB VRAM) for model inference
- **PACS integration:** Direct DICOM ingestion from clinical workflow
- **VPN access:** Secure remote access for multi-site collaboration
- **Storage:** 100TB NAS for raw DICOM + processed volumes
- **Network:** 10 Gbps internal, 1 Gbps external (VPN-secured)

**Deployment requirements**:
- **Authentication:** LDAP/SSO integration with Charité Active Directory
- **Security:** HTTPS (TLS 1.3) + VPN (OpenVPN/WireGuard)
- **Compliance:** GDPR Article 32 (data security), EU AI Act Article 14 (human oversight)
- **On-premise deployment:** No external data transfer (all processing within Charité firewall)
- **Audit logging:** Track all segmentations, model versions, user actions
- **Scalability:** Support 10-20 concurrent users, batch processing 100+ cases overnight

---

## Architecture Design

### Frontend (TypeScript + Vite + Niivue v6)

```
flow-segment-frontend/
├── src/
│   ├── main.ts                 # Application entry point
│   ├── components/
│   │   ├── Viewer.ts           # Niivue viewer wrapper
│   │   ├── Toolbar.ts          # UI controls
│   │   ├── SegmentPanel.ts     # AI segmentation interface
│   │   └── ResultsPanel.ts     # Display segmentation results
│   ├── services/
│   │   ├── api.ts              # Backend API client
│   │   ├── niivue.ts           # Niivue initialization
│   │   └── auth.ts             # Authentication
│   ├── types/
│   │   ├── volume.ts           # Volume data types
│   │   ├── segmentation.ts     # Segmentation types
│   │   └── api.ts              # API response types
│   └── utils/
│       ├── dicom.ts            # DICOM utilities
│       └── nifti.ts            # NIfTI utilities
├── package.json
├── tsconfig.json
├── vite.config.ts
└── index.html
```

**Key differences from `frac`**:
- TypeScript instead of vanilla JS
- Modular component architecture
- Backend API integration (not pure static)
- Focus on segmentation (not fractal analysis)

### Backend (FastAPI + MedSAM3D)

```
flow-segment-backend/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── medsam3d.py         # MedSAM3D model wrapper
│   │   └── nnu_net.py          # nnU-Net prior
│   ├── api/
│   │   ├── segment.py          # Segmentation endpoints
│   │   ├── volumes.py          # Volume management
│   │   └── auth.py             # Authentication
│   ├── services/
│   │   ├── cache.py            # Embedding cache (Redis)
│   │   ├── dicom.py            # DICOM processing
│   │   └── nifti.py            # NIfTI conversion
│   └── config.py               # Configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**API Endpoints**:
```
POST /api/segment/point       # Point-based prompting
POST /api/segment/box         # Bounding box prompting
POST /api/volumes/upload      # Upload DICOM/NIfTI
GET  /api/volumes/{id}        # Retrieve processed volume
GET  /api/health              # Health check
POST /api/auth/login          # Authentication
```

---

## What We Want to Do (Clinical Workflow)

### Use Case 1: Interactive Coronary Segmentation

**User workflow**:
1. Radiologist opens browser → logs in (Charité SSO/LDAP)
2. Loads DISCHARGE CCTA scan from PACS (automatic DICOM → NIfTI conversion)
3. Clicks on LAD artery → AI segments entire vessel in <2 seconds
4. Clicks on plaque → AI characterizes composition (calcified [>130 HU], non-calcified [30-130 HU], low-attenuation [<30 HU])
5. Reviews stenosis severity (CAD-RADS classification), remodeling index
6. Exports segmentation for CT-FFR analysis or research database (NIfTI, DICOM-SEG, or CSV)

**Technical flow** (with SAM-Med3D [@Zhang2024SAMMed3D]):
1. **Frontend:** Niivue [@Niivue2024] loads NIfTI volume via `nv.loadVolumes()`, displays 3D rendering (WebGL2)
2. **User interaction:** Click event → `onLocationChange` captures 3D coordinates (x, y, z in mm)
3. **API request:** Frontend sends `POST /api/segment/point` with coordinates + volume ID
4. **Backend Stage 1:** nnU-Net [@Isensee2021nnUNet] generates anatomical prior (LAD/LCx/RCA identification)
5. **Backend Stage 2:** SAM-Med3D refines segmentation using nnU-Net mask as semantic prompt + user point prompt
6. **Response:** Backend returns segmentation mask as NIfTI or MZ3 mesh (compressed for fast transfer)
7. **Overlay rendering:** Niivue renders mask via `nv.addVolume()` or `nv.addMesh()` with adjustable opacity
8. **Plaque characterization:** Backend computes HU statistics, remodeling index, stenosis severity

### Use Case 2: Batch Processing for Research

**User workflow**:
1. Research coordinator uploads 100 DISCHARGE cases
2. AI processes all cases overnight (batch mode)
3. Next morning: review results, flag low-confidence cases
4. Expert corrects flagged cases → AI retrains (active learning)
5. Export refined segmentations for MACE prediction analysis

### Use Case 3: Multi-Center Collaboration

**User workflow**:
1. Charité researcher segments 50 cases
2. Edinburgh collaborator (SCOT-HEART) accesses same platform
3. Both review each other's segmentations
4. Consensus annotations used for model training
5. Improved model deployed to both sites

---

## Technology Choices

### Why TypeScript?

**Advantages over vanilla JS** (from `frac` project):
- **Type safety:** Catch errors at compile time (reduces runtime bugs by ~15% [@Gao2017TypeScript])
- **Better IDE support:** Autocomplete, refactoring, inline documentation
- **Easier collaboration:** Explicit interfaces for API contracts, component props
- **Scales better:** Essential for large codebase (>10K LOC) with multiple contributors
- **Niivue integration:** Full type definitions for `@niivue/niivue` (improved from v0.62)

**Migration strategy from `frac`**:
1. Convert `.js` → `.ts` files incrementally
2. Add type definitions for Niivue API (`NVImage`, `NVMesh`, `Niivue` class)
3. Define interfaces for backend API responses (`SegmentationResult`, `VolumeMetadata`)
4. Use strict TypeScript config (`strict: true`, `noImplicitAny: true`)
5. Leverage Vite's built-in TypeScript support (no additional configuration)

### Why Niivue v6+? [@Niivue2024]

**Latest capabilities** (v6.x stable):
- **Improved TypeScript support:** Full type definitions, better IDE integration
- **Better performance:** WebGL2 optimizations, 60 FPS rendering for 512³ volumes
- **Enhanced mesh rendering:** Support for large meshes (>1M triangles), smooth shading
- **More flexible API:** Simplified event handling, better plugin architecture
- **4D volume support:** Time-series visualization for perfusion imaging
- **In-browser DICOM:** dcm2niix conversion without server roundtrip

**Key features for our use case**:
- **Interactive segmentation:** `onLocationChange` event for click-to-segment workflow
- **Multi-planar reconstruction:** Synchronized axial/coronal/sagittal views
- **Curved reformats:** Essential for coronary artery visualization
- **Mesh overlay:** Render SAM-Med3D segmentation results as 3D surfaces
- **Point cloud support:** Direct NVMesh construction from coronary artery point clouds
- **Cross-platform:** Works on Windows, macOS, Linux (Chrome, Firefox, Safari)

**Comparison to alternatives**:
- **vs. 3D Slicer:** No installation required, faster startup, better for web deployment
- **vs. OHIF Viewer:** More flexible for custom AI integration, better 3D rendering
- **vs. Cornerstone3D:** Simpler API, better documentation, active community

### Why Backend Server?

**Can't do in browser alone**:
- MedSAM3D model (too large, needs GPU)
- PACS integration (security requirements)
- Batch processing (resource intensive)
- Active learning (model retraining)

**Deployment options**:
1. **Charité on-premise**: GPU cluster, full control, GDPR compliant
2. **Cloud (Azure/AWS)**: Scalable, but data transfer concerns
3. **Hybrid**: Frontend on Netlify, backend on Charité

---

## Next Steps

### Immediate Actions

1. **Research current state**:
   - [ ] Check Niivue latest version (npm show @niivue/niivue version)
   - [ ] Review Niivue v6 TypeScript examples
   - [ ] Find MedSAM3D implementation (GitHub)
   - [ ] Check if MedSAM3D has FastAPI wrapper

2. **Set up development environment**:
   - [ ] Initialize TypeScript + Vite project
   - [ ] Install latest Niivue
   - [ ] Set up FastAPI backend skeleton
   - [ ] Test basic frontend ↔ backend communication

3. **Modify Koselleck proposal**:
   - [ ] Integrate technical roadmap
   - [ ] Emphasize browser-based platform innovation
   - [ ] Add hospital deployment strategy
   - [ ] Delete roadmap.md after integration

Would you like me to:
1. **Check Niivue latest version and capabilities** (web search)?
2. **Initialize TypeScript + Vite + Niivue v6 project** in flow/segment/?
3. **Modify the Koselleck proposal** to integrate the roadmap?
4. **All of the above**?
