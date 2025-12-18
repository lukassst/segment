# Legacy Documentation Archive

**Status:** ⚠️ **CONTENT MIGRATED** - These files have been integrated into the main documentation.

**Migration Date:** 2025-12-18

---

## 📁 Files in This Folder

| File | Status | Migrated To | Notes |
|------|--------|-------------|-------|
| `plan.md` | ✅ Migrated | `../centerline-pipeline.md` | CT-FFR centerline extraction (FMM-based) |
| `fmm.md` | ✅ Migrated | `../centerline-pipeline.md` | Fast Marching Method theory |
| `medsam.md` | ✅ Migrated | `../sam3d.md` | SAM3D architecture overview |
| `architecture.md` | ✅ Migrated | `../sam3d.md` | Project structure diagrams |
| `roadmap.md` | ✅ Migrated | `../proposal.md` | Phase-by-phase implementation plan |
| `proposal.md` | ✅ Superseded | `../proposal.md` | Old proposal version (merge conflict resolved) |
| `plan.pdf` | 📄 Redundant | Delete | PDF version of plan.md |

---

## 🗂️ New Documentation Structure

The legacy content has been reorganized into **4 focused project documents**:

### 1. **`sam3d.md`** - AI Segmentation Platform
**Project:** SAM-Med3D foundation model for interactive 3D medical image segmentation

**Content from legacy:**
- `medsam.md` → SAM3D architecture
- `architecture.md` → Frontend/backend structure
- `roadmap.md` → Technical roadmap (partially)

**Covers:**
- SAM-Med3D foundation model architecture
- nnU-Net + SAM-3D adapter pipeline
- FastAPI backend + TypeScript frontend
- Browser-based segmentation workflow
- Active learning loop

---

### 2. **`medis-viewer.md`** - MEDIS Visualization Platform
**Project:** Interactive viewer for MEDIS coronary contour data

**New content (extracted from tech.md):**
- MEDIS TXT parsing and mesh generation
- Straightened MPR (Curved Reformation) algorithms
- Cross-section extraction with Frenet-Serret frames
- General rotation viewer (Rodrigues formula)
- Quad-view layout (CTA, cross-section, SMPR, 3D mesh)
- Client-side direct mesh construction (50-100ms)

**Covers:**
- MEDIS TXT format specification
- Centerline extraction from contours
- Multi-planar reformation (MPR)
- Straightened MPR (SMPR) volume construction
- Interactive UI controls
- STL/MZ3/PLY export

---

### 3. **`centerline-pipeline.md`** - Classical Centerline Extraction
**Project:** FMM-based vessel centerline and wall segmentation

**Content from legacy:**
- `plan.md` → Complete 4-phase pipeline
- `fmm.md` → Fast Marching Method theory

**Covers:**
- Vesselness filter (Frangi) for vessel enhancement
- Fast Marching Method (FMM) for optimal path extraction
- Polar transform + Dynamic Programming for wall segmentation
- Centroid-based centerline refinement
- Python/WASM compatibility
- No AI/deep learning (classical algorithms only)

---

### 4. **`proposal.md`** - DFG Koselleck Funding Proposal
**Project:** Strategic funding proposal for Prof. Marc Dewey

**Content from legacy:**
- `roadmap.md` → Detailed implementation phases, risk mitigation, success metrics
- `proposal.md` (old) → Superseded by current version

**Covers:**
- Executive summary and clinical impact
- DISCHARGE (25M) + SCOT-HEART (10M) + Prostate (4M) trials
- SAM-Med3D foundation model approach
- Browser-based platform architecture
- 5-year work plan with milestones
- Budget and team structure

---

## 🎯 Why the Reorganization?

**Problem:** Content was scattered across multiple overlapping files
- `tech.md` (148KB) mixed 3 different projects
- Legacy folder had duplicate/superseded versions
- Hard to navigate for specific project information

**Solution:** Separate documents for separate projects
- ✅ **sam3d.md** = AI-driven segmentation (PyTorch, deep learning)
- ✅ **medis-viewer.md** = MEDIS visualization (NiiVue, client-side)
- ✅ **centerline-pipeline.md** = Classical algorithms (FMM, no AI)
- ✅ **proposal.md** = Funding narrative

**Benefits:**
- Clear separation of concerns
- Each project has focused documentation
- Easier to maintain and update
- Better for different audiences (ML researchers vs CV engineers vs clinicians)

---

## 🚀 What to Use Going Forward

| If you need... | Use this document |
|----------------|-------------------|
| SAM-Med3D AI segmentation | `sam3d.md` |
| MEDIS contour visualization | `medis-viewer.md` |
| Interactive vessel segmentation (classical) | `vessel-segmenter.md` |
| Funding proposal narrative | `proposal.md` |
| Implementation code | `code/` folder |
| Citations | `references.bib` |

---

## 🗑️ Cleanup Actions

**Safe to delete:**
- `plan.pdf` (redundant binary)
- `proposal.md` (old version, superseded)
- Potentially entire `legacy/` folder (after confirming migration)

**Keep for reference:**
- This `README.md` (explains migration)
- Other legacy files (historical reference)

---

**Migration completed:** 2025-12-18  
**Migrated by:** Documentation consolidation project  
**New structure:** 4 focused project documents + code/ folder
