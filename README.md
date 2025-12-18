# 🩺 Segment Platform - Project Documentation

**Multi-Project Repository for Medical Image Segmentation & Visualization**

---

## 📂 Repository Structure

```
segment/
├── README.md                    # This file - project overview
├── sam3d.md                     # Project 1: SAM-Med3D AI Platform
├── medis-viewer.md              # Project 2: MEDIS Viewer & Visualization
├── vessel-segmenter.md          # Project 3: Interactive Vessel Segmentation
├── proposal.md                  # Project 4: DFG Koselleck Funding Proposal
├── references.bib               # Shared bibliography
├── code/                        # Implementation scripts
│   ├── buildstl.py              # MEDIS → STL mesh generation
│   ├── medis.py                 # MEDIS TXT parser
│   └── medismask.py             # MEDIS mask operations
├── data/                        # Data files (18 items)
└── legacy/                      # Archived documentation (see legacy/README.md)
```

---

## 🎯 Four Independent Projects

### 1️⃣ **SAM3D AI Platform** (`sam3d.md`)

**Goal:** Interactive 3D medical image segmentation using foundation models

**Technology:**
- **Frontend:** TypeScript + Vite + NiiVue (WebGL2)
- **Backend:** FastAPI + SAM-Med3D-turbo (PyTorch)
- **Model:** nnU-Net + SAM-3D adapter

**Key Features:**
- Click-to-segment (point/box prompts)
- <2s response time (cached embeddings)
- Active learning loop
- Browser-based, zero installation

**Data:**
- DISCHARGE: 25M images, 3,561 patients
- SCOT-HEART: 10M images, 4,146 patients
- Prostate: 4M images

**Target:** Charité GPU cluster deployment (Prof. Marc Dewey)

---

### 2️⃣ **MEDIS Viewer** (`medis-viewer.md`)

**Goal:** Visualize MEDIS coronary contour data with CTA volumes

**Technology:**
- **Frontend:** TypeScript + NiiVue (client-side only)
- **Input:** MEDIS TXT files + NIfTI CTA volumes
- **Mesh:** Direct client-side construction (50-100ms)

**Key Features:**
- **Quad-view layout:**
  - Panel 1: CTA + centerline overlay
  - Panel 2: Cross-section (interactive slider)
  - Panel 3: Straightened MPR (curved reformation)
  - Panel 4: 3D mesh (lumen + vessel wall)
- **General rotation viewer** (Rodrigues formula, Euler angles)
- **Export:** STL, MZ3, PLY formats

**Algorithms:**
- Centerline extraction (Frenet-Serret frame)
- Cross-section extraction (trilinear interpolation)
- Straightened MPR (SMPR) volume construction
- Direct mesh from contour rings (no backend)

**Performance:** Real-time interaction, <2s straightened MPR

---

### 3️⃣ **Interactive Vessel Segmenter** (`vessel-segmenter.md`)

**Goal:** Complete vessel segmentation (centerline + lumen + outer wall) using classical algorithms (no AI)

**Technology:**
- **Language:** Python (numpy, scipy, skimage, SimpleITK)
- **Core Algorithm:** Fast Marching Method (FMM)
- **Future:** WASM-compatible (Pyodide)

**Pipeline (4 Phases):**
1. **Preprocessing:** Frangi vesselness filter → cost map
2. **Centerline:** FMM wave propagation → gradient descent backtracking
3. **Segmentation:** Polar transform + Dynamic Programming → lumen/outer wall
4. **Refinement:** Centroid-based centerline correction

**Advantages:**
- ✅ Deterministic (reproducible)
- ✅ No training data required
- ✅ Mathematically optimal (global minimum path)
- ✅ Browser-compatible (WASM)

**Use Case:** CT-FFR (fractional flow reserve), plaque quantification, CFD mesh generation

---

### 4️⃣ **DFG Koselleck Proposal** (`proposal.md`)

**Goal:** Strategic funding for Prof. Marc Dewey (Charité)

**Proposal:**
- **Title:** *Semantic Volumetrics: Foundation Models for Longitudinal Digital Twin in Coronary Artery Disease*
- **Funding:** DFG Reinhart Koselleck-Projekt
- **Duration:** 5 years
- **Budget:** €1.5-2.0M

**Innovation:**
- First browser-based foundation model platform for cardiovascular imaging
- Multi-domain: DISCHARGE + SCOT-HEART + Prostate (39M images total)
- Active learning loop for collaborative annotation

**Roadmap:**
- **Year 1:** Infrastructure + SAM-Med3D adapter
- **Year 2:** Multi-vendor optimization
- **Year 3:** External validation (SCOT-HEART)
- **Year 4:** Longitudinal modeling
- **Year 5:** Multi-center deployment + CE marking

---

## 🔬 Technical Details

### Technology Stack Comparison

| Aspect | SAM3D | MEDIS Viewer | Centerline Pipeline |
|--------|-------|--------------|---------------------|
| **Approach** | AI (Deep Learning) | Visualization | Classical Algorithms |
| **Frontend** | TypeScript + Vite + NiiVue | TypeScript + NiiVue | N/A (Python backend) |
| **Backend** | FastAPI + PyTorch | None (client-side) | Python scripts |
| **GPU** | Required (NVIDIA A100) | Not required | Not required |
| **Training Data** | 143K masks (SAM-Med3D) | Not required | Not required |
| **Speed** | <2s (with cache) | 50-100ms (mesh gen) | ~20-60s (full pipeline) |
| **Deployment** | Hospital server | Browser only | Desktop/server |

### Data Flow Examples

**SAM3D Workflow:**
```
User clicks on coronary → Coordinates sent to backend → 
SAM-Med3D inference → Segmentation mask returned → 
NiiVue overlay rendered (<2s total)
```

**MEDIS Viewer Workflow:**
```
Load MEDIS TXT + CTA NIfTI → Parse contours (client-side) → 
Build mesh directly in browser (50ms) → 
Compute centerline + SMPR → Display quad-view
```

**Centerline Pipeline Workflow:**
```
Load CTA NIfTI → Compute Frangi vesselness → 
User places 2 points → FMM path extraction → 
Extract cross-sections → Segment lumen/wall (DP) → 
Refine centerline → Output boundaries
```

---

## 📚 Documentation Map

### Quick Reference

| Question | Document | Section |
|----------|----------|---------|
| How does SAM-Med3D work? | `sam3d.md` | Foundation Model Approach |
| How to visualize MEDIS data? | `medis-viewer.md` | MEDIS TXT Format |
| What is straightened MPR? | `medis-viewer.md` | Algorithm 1-3 (SMPR) |
| How does FMM work? | `vessel-segmenter.md` | Phase 2 |
| What are the clinical use cases? | `sam3d.md` or `proposal.md` | Clinical Use Cases |
| How to export meshes? | `medis-viewer.md` | Export Capabilities |
| What's the funding plan? | `proposal.md` | Work Plan & Milestones |
| Where's the Python code? | `code/` folder | buildstl.py, medis.py |

### Cross-References

**MEDIS data appears in:**
- `medis-viewer.md` - Primary visualization documentation
- `sam3d.md` - Mentions MEDIS as input data source
- `code/buildstl.py` - Python implementation

**Centerline extraction appears in:**
- `medis-viewer.md` - Frenet-Serret frame approach (from MEDIS contours)
- `vessel-segmenter.md` - FMM approach (from CTA volume, interactive 2-point tool)
- Different methods for different input data!

**NiiVue integration appears in:**
- `sam3d.md` - For AI segmentation overlay
- `medis-viewer.md` - For mesh visualization + SMPR

---

## 🚀 Getting Started

### For AI Segmentation (SAM3D)
1. Read `sam3d.md`
2. Check `proposal.md` for context (DISCHARGE/SCOT-HEART data)
3. Backend: Set up FastAPI + SAM-Med3D-turbo
4. Frontend: Initialize TypeScript + Vite + NiiVue

### For MEDIS Visualization
1. Read `medis-viewer.md`
2. Review `code/buildstl.py` for Python reference
3. Frontend only: TypeScript + NiiVue
4. Implement: MEDIS parser → Direct mesh construction

### For Interactive Vessel Segmentation
1. Read `vessel-segmenter.md`
2. Install: `numpy`, `scipy`, `skimage`, `scikit-fmm`
3. Implement: 4-phase pipeline (preprocessing → FMM → segmentation → refinement)

### For Proposal Preparation
1. Read `proposal.md`
2. Review technical sections in `sam3d.md`
3. Customize for target institution/PI

---

## 📊 Project Status

| Project | Status | Priority | Next Steps |
|---------|--------|----------|------------|
| **SAM3D** | 📋 Documented | High | Backend setup, model download |
| **MEDIS Viewer** | 📋 Documented | Medium | Frontend implementation |
| **Vessel Segmenter** | 📋 Documented | Medium | Python prototype |
| **Proposal** | ✅ Complete | High | Review with Prof. Dewey |

---

## 🤝 Contributing

This is a research project documentation repository. Each project has:
- **Detailed algorithms** (pseudocode + math)
- **Implementation guidance** (technology stack)
- **Code references** (`code/` folder)
- **Cross-references** (related sections)

When adding new content:
1. Identify which project it belongs to
2. Update the appropriate `.md` file
3. Add cross-references if needed
4. Update this README if structure changes

---

## 📝 Notes

### Why Separate Documents?

Previously, everything was in `tech.md` (148KB), mixing:
- AI segmentation (SAM3D)
- MEDIS visualization
- Classical algorithms
- General rotation viewer

**Problem:** Hard to navigate, unclear which content applies to which project.

**Solution:** Split into focused documents by project type.

### What About `legacy/` Folder?

Contains old versions and duplicates. See `legacy/README.md` for migration details.

**Safe to delete after confirming migration completeness.**

---

## 📧 Contact

**Project:** Segment Platform  
**Institution:** Charité – Universitätsmedizin Berlin  
**Target PI:** Prof. Marc Dewey  
**Documentation Version:** 2.0 (Consolidated)  
**Last Updated:** 2025-12-18

---

**Quick Links:**
- [SAM3D AI Platform](sam3d.md)
- [MEDIS Viewer](medis-viewer.md)
- [Interactive Vessel Segmenter](vessel-segmenter.md)
- [DFG Koselleck Proposal](proposal.md)
- [Legacy Documentation](legacy/README.md)
