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

## 🧠 Foundation Model Approach: Unified Segmentation with MedSAM3D

### The Vision: Universal Medical Segmentation
**Goal:** A single foundation model that segments any anatomical structure across imaging modalities using minimal user interaction (1-3 clicks).

**Foundation Models in Medical Imaging:**
- **Traditional approach:** Task-specific models (one model per organ, per modality)
- **Foundation model approach:** Single large model pre-trained on diverse medical data → universal segmentation
- **Key advantage:** Generalization to new anatomies without retraining

**MedSAM3D** [@Ma2024MedSAM3D] represents this paradigm shift:
- **Pre-training:** 143K 3D segmentation masks across 245 anatomical categories
- **Architecture:** 3D Vision Transformer (ViT-B/16) with prompt-based decoder
- **Interactive segmentation:** User provides point/box prompts → model segments target structure
- **Zero-shot capability:** Works on unseen anatomies without fine-tuning

---

## 🏗️ Project Structure

### Frontend Architecture (TypeScript + Vite + Niivue)

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
│   │   ├── medisParser.ts      # MEDIS TXT parsing
│   │   ├── medisMeshDirect.ts  # Direct mesh construction
│   │   └── auth.ts             # Authentication
│   ├── types/
│   │   ├── volume.ts           # Volume data types
│   │   ├── segmentation.ts     # Segmentation types
│   │   ├── medis.ts            # MEDIS contour types
│   │   └── api.ts              # API response types
│   └── utils/
│       ├── dicom.ts            # DICOM utilities
│       └── nifti.ts            # NIfTI utilities
├── package.json
├── tsconfig.json
├── vite.config.ts
└── index.html
```

**Key Features:**
- **TypeScript throughout:** Type-safe, maintainable codebase
- **Modular architecture:** Separation of concerns (components, services, types, utils)
- **Backend API integration:** RESTful communication with FastAPI server
- **Focus on segmentation:** AI-driven interactive segmentation workflow

### Backend Architecture (FastAPI + SAM-Med3D)

```
flow-segment-backend/
├── app/
│   ├── main.py                 # FastAPI application entry
│   ├── models/
│   │   ├── medsam3d.py         # SAM-Med3D model wrapper
│   │   └── nnu_net.py          # nnU-Net prior (anatomical context)
│   ├── api/
│   │   ├── segment.py          # Segmentation endpoints
│   │   ├── volumes.py          # Volume management
│   │   ├── mesh.py             # Mesh generation (nii2mesh)
│   │   └── auth.py             # Authentication (LDAP/SSO)
│   ├── services/
│   │   ├── cache.py            # Embedding cache (Redis)
│   │   ├── dicom.py            # DICOM processing (SimpleITK)
│   │   ├── nifti.py            # NIfTI conversion
│   │   └── registration.py     # Image registration (Elastix)
│   └── config.py               # Configuration management
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**API Endpoints:**
```
POST /api/segment/point       # Point-based prompting (SAM-Med3D)
POST /api/segment/box         # Bounding box prompting
POST /api/volumes/upload      # Upload DICOM/NIfTI
GET  /api/volumes/{id}        # Retrieve processed volume
POST /api/mesh/from-mask      # Generate mesh from segmentation mask
GET  /api/health              # Health check
POST /api/auth/login          # Authentication
```

### Clinical Workflows

#### Workflow 1: Interactive Coronary Segmentation

**User Steps:**
1. Radiologist opens browser → logs in (Charité SSO)
2. Loads DISCHARGE CCTA scan from PACS
3. Clicks on LAD artery → AI segments entire vessel in 2 seconds
4. Clicks on plaque → AI characterizes composition (calcified, lipid, mixed)
5. Exports segmentation for CT-FFR analysis or research database

**Technical Flow:**
1. **Frontend:** Niivue loads NIfTI volume, displays 3D rendering
2. **User interaction:** Click → frontend captures 3D coordinates (mm space)
3. **API call:** `POST /api/segment/point` with coordinates + volume ID
4. **Backend:** SAM-Med3D generates segmentation mask (cached embeddings)
5. **Response:** Backend returns mask as NIfTI or mesh (MZ3/PLY format)
6. **Frontend:** Niivue overlays mask with adjustable opacity
7. **Real-time:** <2s end-to-end latency

#### Workflow 2: Batch Processing for Research

**User Steps:**
1. Research coordinator uploads 100 DISCHARGE cases
2. AI processes all cases overnight (batch mode)
3. Next morning: review results, flag low-confidence cases
4. Expert corrects flagged cases → AI retrains (active learning)
5. Export refined segmentations for MACE prediction analysis

**Technical Flow:**
1. **Batch upload:** `POST /api/volumes/upload` (multiple files)
2. **Queue processing:** Celery task queue + Redis
3. **Parallel execution:** Multi-GPU processing (4× NVIDIA A100)
4. **Quality control:** Auto-flag cases with Dice <0.75
5. **Active learning:** Corrected masks → fine-tune model weekly
6. **Export:** Bulk download as NIfTI + CSV metadata

#### Workflow 3: MEDIS TXT Visualization

**User Steps:**
1. Load CTA volume (NIfTI) in browser
2. Load MEDIS TXT file (LAD contour rings)
3. **Client-side mesh generation:** Parse TXT → construct NVMesh (50-100ms)
4. Overlay lumen + vessel wall meshes on CTA
5. Toggle between mesh views and straightened MPR

**Technical Flow:**
1. **Parse MEDIS TXT:** Extract lumen/vessel contour rings
2. **Direct mesh construction:** Connect rings → triangles (no backend needed)
3. **Coordinate validation:** Ensure mm space matches CTA header
4. **Render:** Add NVMesh to NiiVue viewer
5. **Optional:** Export STL via backend for 3D printing

---

## 🫀 Clinical Use Cases: Cardiac & Prostate CTA Segmentation

### Overview
Our platform targets **volumetric segmentation of cardiovascular and pelvic structures** from CT angiography (CTA):

| **Anatomy** | **Structure** | **Clinical Need** | **Difficulty** |
|-------------|---------------|-------------------|----------------|
| **Cardiac** | Myocardium | Quantify heart muscle mass, wall thickness, perfusion defects | ⭐⭐ Medium |
| **Cardiac** | Coronary arteries (inner wall/lumen) | Quantify stenosis, plaque burden, FFR-CT simulation | ⭐⭐⭐⭐⭐ Extreme |
| **Cardiac** | Coronary arteries (outer wall/vessel) | Quantify total plaque volume, vessel remodeling | ⭐⭐⭐⭐⭐ Extreme |
| **Pelvic** | Prostate | Quantify prostate volume, treatment planning | ⭐⭐ Medium |

**Why coronary arteries are the most challenging anatomy:**
1. **Motion artifacts:** Heart beats 60-80 times/min → blurring, ghosting despite ECG gating
2. **Low attenuation plaques:** Non-calcified/lipid-rich plaques (40-80 HU) barely visible vs. contrast (300-400 HU)
3. **Complex topology:** Bifurcations, overlapping vessels, crossing branches
4. **Small caliber:** 1.5-4mm diameter → partial volume effects at typical 0.5-0.625mm resolution
5. **Dual-wall segmentation:** Must segment both inner (lumen) and outer (vessel) boundaries simultaneously

---

## 🔬 Technical Challenges & Solutions

### Challenge 1: Coronary Artery Motion Artifacts
**Problem:**
- Despite ECG-gated acquisition (mid-diastole, 70-80% R-R interval), residual motion blurs vessel edges
- Severe cases: "ghost vessels" from arrhythmia or high heart rate (>70 bpm)
- Right coronary artery (RCA) most affected due to higher velocity

**Solution:**
```python
# Motion correction via temporal consistency
def motion_robust_segmentation(volume_4d, heart_rate):
    """
    Multi-phase coronary segmentation with motion compensation.
    
    Args:
        volume_4d: 4D volume [X, Y, Z, T] with multiple cardiac phases
        heart_rate: Patient heart rate (bpm)
    """
    if heart_rate > 70:
        # Use multi-phase reconstruction
        phases = extract_cardiac_phases(volume_4d, num_phases=10)
        
        # Segment each phase independently
        masks_per_phase = []
        for phase in phases:
            mask = medsam3d_segment(phase, prompt_type='point')
            masks_per_phase.append(mask)
        
        # Temporal median filter to remove motion ghosts
        mask_final = np.median(masks_per_phase, axis=0) > 0.5
        
        # Deformable registration to align phases
        mask_registered = elastix_align(masks_per_phase, reference=phases[5])
        
        return mask_registered
    else:
        # Single-phase segmentation sufficient
        return medsam3d_segment(volume_4d[:,:,:,0], prompt_type='point')
```

**Additional strategies:**
- **Pre-processing:** Edge-preserving denoising (bilateral filter) to reduce motion blur without losing vessel boundaries
- **Prompt placement:** Multiple prompts along vessel centerline (every 5-10mm) to guide segmentation through motion-corrupted regions
- **Quality control:** Automatic detection of motion artifacts via edge sharpness metrics → flag for manual review

---

### Challenge 2: Low Attenuation Plaque Detection
**Problem:**
- **Non-calcified plaque (NCP):** 40-80 HU, only 20-60 HU contrast vs. lumen (300-400 HU)
- **Lipid-rich plaque:** 20-50 HU, overlaps with myocardium (50-70 HU) → nearly invisible
- **Blooming artifacts:** Calcified plaque (>130 HU) causes beam hardening → obscures adjacent NCP

**Solution:**
```python
# Multi-scale feature extraction for low-contrast plaques
def detect_low_attenuation_plaque(volume, vessel_mask):
    """
    Enhanced plaque detection using multi-scale ViT features.
    
    Key insight: MedSAM3D's ViT encoder captures subtle texture patterns
    that distinguish plaque from lumen, even at low contrast.
    """
    # Step 1: Extract vessel region of interest (ROI)
    vessel_roi = volume * vessel_mask
    
    # Step 2: Contrast enhancement for low HU regions
    vessel_enhanced = adaptive_histogram_equalization(
        vessel_roi, 
        clip_limit=0.03,  # Preserve texture
        kernel_size=32    # Local enhancement
    )
    
    # Step 3: MedSAM3D feature extraction (multi-scale)
    features = medsam3d_encoder(vessel_enhanced)  # Shape: [B, 768, H/16, W/16, D/16]
    
    # Step 4: Plaque-specific attention head
    # Trained on SCOT-HEART annotations (n=4,146) to recognize:
    # - Napkin-ring sign (lipid core)
    # - Positive remodeling (vulnerable plaque)
    # - Low attenuation (<30 HU)
    plaque_probs = plaque_attention_decoder(
        features,
        prompt=inner_wall_mask,  # Use lumen as prior
        attn_weights='plaque_specific'
    )
    
    # Step 5: HU-based refinement
    # Rule: Plaque must be between lumen and outer wall
    plaque_mask = (plaque_probs > 0.5) & (vessel_roi < 200)  # Exclude calcified
    
    return plaque_mask
```

**Plaque characterization pipeline:**
```python
def characterize_plaque_components(volume, vessel_mask, lumen_mask):
    """
    Classify plaque into 3 categories based on HU thresholds.
    Validated against intravascular ultrasound (IVUS) gold standard.
    """
    # Extract vessel wall (between lumen and outer wall)
    vessel_wall = vessel_mask & ~lumen_mask
    hu_values = volume[vessel_wall]
    
    # Plaque component classification (SCOT-HEART criteria)
    calcified = (hu_values > 130)        # Dense calcium
    non_calcified = (hu_values >= 80) & (hu_values <= 130)  # Fibrous
    low_attenuation = (hu_values < 80)   # Lipid-rich (high risk)
    
    # Quantification
    total_volume = np.sum(vessel_wall) * voxel_volume_mm3
    calc_volume = np.sum(calcified) * voxel_volume_mm3
    noncalc_volume = np.sum(non_calcified) * voxel_volume_mm3
    lowatt_volume = np.sum(low_attenuation) * voxel_volume_mm3
    
    # Stenosis calculation
    stenosis_percent = (1 - np.mean(lumen_area) / np.mean(reference_area)) * 100
    
    return {
        'total_plaque_volume_mm3': total_volume,
        'calcified_percent': calc_volume / total_volume * 100,
        'non_calcified_percent': noncalc_volume / total_volume * 100,
        'low_attenuation_percent': lowatt_volume / total_volume * 100,
        'stenosis_percent': stenosis_percent,
        'high_risk_plaque': lowatt_volume > 0.04 * total_volume  # >4% LAP
    }
```

---

### Challenge 3: Dual-Wall Coronary Segmentation
**Problem:**
- Must segment **both inner wall (lumen)** and **outer wall (vessel)** to quantify plaque burden
- Spacing between walls: 0.5-3mm (only 1-5 voxels at 0.625mm resolution)
- Overlapping vessels and bifurcations make outer wall ambiguous

**Solution: Sequential Prompting Strategy**
```python
def dual_wall_coronary_segmentation(volume, user_click):
    """
    Two-stage segmentation: lumen first, then outer wall.
    
    Stage 1: Lumen (bright, high contrast → easy)
    Stage 2: Outer wall (low contrast → use lumen as prior)
    """
    # Stage 1: Segment lumen (inner wall)
    lumen_mask = medsam3d_segment(
        volume,
        prompt_point=user_click,
        prompt_type='point',
        target='high_intensity'  # Expects bright lumen (300-400 HU)
    )
    
    # Stage 2: Segment outer wall using lumen as prior
    # Key insight: Outer wall is always 0.5-3mm away from lumen
    outer_mask = medsam3d_segment(
        volume,
        prompt_point=user_click,
        prompt_type='point',
        prior_mask=lumen_mask,     # Context: lumen location
        expansion_range='0.5-3mm',  # Expected vessel wall thickness
        target='low_intensity'      # Expects darker plaque/vessel wall
    )
    
    # Post-processing: Ensure outer wall contains lumen
    outer_mask = morphological_closing(outer_mask, radius=2)
    outer_mask = outer_mask | lumen_mask  # Guarantee containment
    
    # Vessel wall = outer - inner
    vessel_wall_mask = outer_mask & ~lumen_mask
    
    return {
        'lumen': lumen_mask,
        'outer_wall': outer_mask,
        'vessel_wall': vessel_wall_mask
    }
```

**Alternative: Multi-class Segmentation**
```python
# Single-pass multi-class segmentation (faster but less accurate)
def multiclass_coronary_segment(volume, user_click):
    """
    Segment lumen and vessel wall simultaneously.
    Requires fine-tuning on DISCHARGE dataset (n=3,561).
    """
    output = medsam3d_segment(
        volume,
        prompt_point=user_click,
        num_classes=3,  # Background, Lumen, Vessel Wall
        class_weights=[0.1, 1.0, 0.8]  # Emphasize lumen (easier)
    )
    
    lumen_mask = output == 1
    vessel_wall_mask = output == 2
    outer_mask = (output == 1) | (output == 2)
    
    return {
        'lumen': lumen_mask,
        'outer_wall': outer_mask,
        'vessel_wall': vessel_wall_mask
    }
```

---

### Challenge 4: Myocardium Segmentation
**Problem:**
- Large structure (100-200mL) → computationally expensive
- Trabeculations and papillary muscles difficult to distinguish from myocardium
- Variable contrast enhancement (early vs. late arterial phase)

**Solution:**
```python
def segment_myocardium(volume, user_click):
    """
    Full myocardium segmentation using nnU-Net prior + MedSAM3D refinement.
    
    Strategy: Coarse-to-fine approach
    1. nnU-Net: Fast whole-heart segmentation (5s)
    2. MedSAM3D: Refine myocardial boundaries (2s)
    """
    # Stage 1: nnU-Net prior (pre-trained on SCOT-HEART)
    coarse_mask = nnunet_predict(
        volume,
        task='Task500_CardiacCTA',
        fold='all',
        checkpoint='best'
    )
    
    # Extract myocardium label (label=2 in our convention)
    myocardium_coarse = (coarse_mask == 2)
    
    # Stage 2: MedSAM3D boundary refinement
    myocardium_refined = medsam3d_segment(
        volume,
        prompt_mask=myocardium_coarse,  # Use nnU-Net as prior
        prompt_type='mask',
        refinement_mode=True,           # Only refine boundaries
        boundary_width=5                # Refine 5mm around edges
    )
    
    # Post-processing: Remove trabeculations (optional)
    myocardium_smooth = morphological_opening(
        myocardium_refined,
        radius=2  # Remove structures <2mm
    )
    
    return myocardium_smooth
```

---

### Challenge 5: Prostate Segmentation
**Problem:**
- Variable contrast (depends on timing of contrast injection)
- Prostate can be enlarged (benign prostatic hyperplasia, BPH) or irregular (cancer)
- Adjacent rectum and bladder can confuse segmentation

**Solution:**
```python
def segment_prostate(volume, user_click):
    """
    Prostate segmentation using MedSAM3D with pelvic anatomy priors.
    
    Relatively straightforward compared to coronaries:
    - Larger structure (20-80mL) → easier to segment
    - Less motion (no cardiac motion)
    - Higher contrast vs. surrounding fat
    """
    # MedSAM3D direct segmentation
    prostate_mask = medsam3d_segment(
        volume,
        prompt_point=user_click,
        prompt_type='point',
        target='moderate_intensity'  # Prostate: 40-60 HU
    )
    
    # Post-processing: Enforce anatomical constraints
    # 1. Prostate is above rectum, below bladder
    # 2. Typical volume: 20-80mL (flag if >100mL)
    prostate_volume = np.sum(prostate_mask) * voxel_volume_mm3
    
    if prostate_volume > 100_000:  # >100mL → likely includes rectum/bladder
        # Morphological refinement
        prostate_mask = keep_largest_component(prostate_mask)
        prostate_mask = morphological_opening(prostate_mask, radius=3)
    
    return prostate_mask
```

---

### Summary: Difficulty Ranking & Strategies

| **Anatomy** | **Key Challenge** | **Strategy** | **Processing Time** |
|-------------|-------------------|--------------|---------------------|
| **Prostate** | Variable contrast, adjacent organs | MedSAM3D direct + morphological refinement | <2s |
| **Myocardium** | Large structure, trabeculations | nnU-Net prior + MedSAM3D refinement | ~7s |
| **Coronary lumen** | Motion artifacts, small caliber | Multi-phase + multi-prompt MedSAM3D | ~3s |
| **Coronary outer wall** | Low contrast plaque, dual-wall | Sequential segmentation (lumen → outer) | ~5s |
| **Low-attenuation plaque** | Near-invisible (<80 HU) | Multi-scale ViT features + HU thresholds | ~2s |

**Overall coronary segmentation pipeline: ~10-15 seconds per vessel** (LAD, LCx, or RCA)

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

## 🎯 Vessel Geometry Creation: Complete Workflow

### Overview: From Segmentation to Mesh Visualization

**Multiple Pathways to Create Vessel Geometry:**

```
Segmentation Mask (NIfTI) ──┐
                             ├──> Marching Cubes ──> Mesh (PLY/VTK/GIfTI) ──> NiiVue Display
Point Cloud (from MEDIS) ────┤
                             ├──> Poisson Recon ───> Mesh (MZ3/OBJ) ────────> NiiVue Display
Contour Rings (MEDIS TXT) ───┘
                             └──> Direct Tube ────> Mesh (STL) ────────────> NiiVue Display
```

**Key Decision Factors:**
- **Speed:** Direct tube mesh (50ms) < Marching cubes (1-3s) < Poisson (3-10s)
- **Quality:** Poisson (smoothest) > Marching cubes (smooth) > Direct tube (faceted)
- **Use case:** Interactive preview (fast) vs. Final visualization (quality)

---

### Pathway 1: Segmentation Mask → Mesh (Marching Cubes)

**Best for:** AI-generated segmentation masks from SAM-Med3D or nnU-Net

**Workflow:**
1. **Input:** Binary segmentation mask (NIfTI format, 0=background, 1=vessel)
2. **Algorithm:** Marching cubes isosurface extraction at threshold 0.5
3. **Post-processing:** Laplacian smoothing + mesh decimation
4. **Output:** Triangular mesh (PLY, VTK, GIfTI, MZ3)

**Implementation Options:**

**Option A: Client-side with vtk.js (Fast Preview)**
```typescript
// Frontend: utils/marchingCubesMesh.ts
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkImageMarchingCubes from '@kitware/vtk.js/Filters/General/ImageMarchingCubes';
import vtkWindowedSincPolyDataFilter from '@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter';

export async function segmentationToMesh(
  maskData: Uint8Array,
  dimensions: [number, number, number],
  spacing: [number, number, number],
  smoothIterations: number = 15
): Promise<{ vertices: Float32Array; triangles: Uint32Array }> {
  
  // Create VTK image from mask
  const imageData = vtkImageData.newInstance();
  imageData.setDimensions(dimensions);
  imageData.setSpacing(spacing);
  imageData.getPointData().setScalars(
    vtkDataArray.newInstance({ values: maskData })
  );
  
  // Marching cubes
  const marchingCubes = vtkImageMarchingCubes.newInstance({
    contourValue: 0.5,
    computeNormals: true,
    mergePoints: true
  });
  marchingCubes.setInputData(imageData);
  
  // Smooth mesh (Windowed Sinc filter - better than Laplacian)
  const smoother = vtkWindowedSincPolyDataFilter.newInstance({
    numberOfIterations: smoothIterations,
    passBand: 0.1,
    nonManifoldSmoothing: true,
    normalizeCoordinates: true
  });
  smoother.setInputConnection(marchingCubes.getOutputPort());
  smoother.update();
  
  // Extract mesh data
  const polyData = smoother.getOutputData();
  const vertices = new Float32Array(polyData.getPoints().getData());
  const triangles = new Uint32Array(polyData.getPolys().getData());
  
  return { vertices, triangles };
}

// Usage: Convert SAM-Med3D output to mesh
const segMask = await fetchSegmentationMask('/api/segment', promptPoint);
const mesh = await segmentationToMesh(
  segMask.data,
  [512, 512, 300],
  [0.5, 0.5, 0.625],
  15  // Smooth iterations
);

// Display in NiiVue
const nvMesh = nv.createMeshFromVertices(mesh.vertices, mesh.triangles);
nv.addMesh(nvMesh);
```

**Option B: Server-side with nii2mesh (High Quality)**
```python
# Backend: app/services/nii2mesh_wrapper.py
import subprocess
import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np

def segmentation_to_mesh_nii2mesh(
    mask: np.ndarray,
    affine: np.ndarray,
    output_format: str = 'mz3',  # Fast, compact format
    pre_smooth: bool = True,
    reduction: float = 0.15,  # Keep 15% of triangles
    smooth_iterations: int = 10
) -> Path:
    """
    Convert segmentation mask to mesh using nii2mesh.
    
    Args:
        mask: Binary segmentation (0=background, 1=vessel)
        affine: NIfTI affine matrix
        output_format: 'mz3', 'ply', 'gii', 'obj', 'stl', 'vtk'
        pre_smooth: Gaussian blur before marching cubes
        reduction: Mesh simplification (0.15 = keep 15% triangles)
        smooth_iterations: Post-mesh smoothing iterations
    
    Returns:
        Path to output mesh file
    """
    # Save mask as temporary NIfTI
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_nii:
        nii_path = Path(tmp_nii.name)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), nii_path)
    
    # Output mesh path
    output_path = nii_path.with_suffix(f'.{output_format}')
    
    # Build nii2mesh command
    cmd = [
        'nii2mesh',
        str(nii_path),
        '-i', 'm',  # Medium intensity threshold (auto-detect)
        '-p', '1' if pre_smooth else '0',
        '-r', str(reduction),
        '-s', str(smooth_iterations),
        '-l', '1',  # Keep only largest component
        '-b', '1',  # Fill bubbles
        str(output_path)
    ]
    
    # Run nii2mesh
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Cleanup
    nii_path.unlink()
    
    return output_path

# FastAPI endpoint
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

@router.post('/api/mesh/from-segmentation')
async def create_mesh_from_segmentation(
    segmentation_id: str,
    format: str = 'mz3'
):
    """Generate mesh from segmentation mask."""
    
    # Load segmentation
    seg_nii = nib.load(f'/data/segmentations/{segmentation_id}.nii.gz')
    mask = seg_nii.get_fdata() > 0.5
    
    # Generate mesh
    mesh_path = segmentation_to_mesh_nii2mesh(
        mask,
        seg_nii.affine,
        output_format=format,
        reduction=0.15,
        smooth_iterations=10
    )
    
    return FileResponse(
        mesh_path,
        media_type='application/octet-stream',
        filename=f'{segmentation_id}.{format}'
    )
```

---

### Pathway 2: MEDIS Contours → Direct Mesh (Client-Side, Instant)

**Best for:** MEDIS TXT contours where topology is already known

**Discovery from frac project**: When triangles are already known (e.g., connecting consecutive contour rings), skip marching cubes entirely and **construct NVMesh directly in the browser**.

**Key Insight:** 
- MEDIS contours = stacked rings of points with known connectivity
- `buildstl.py` already computes this topology (connect ring N to ring N+1)
- **We can do this in TypeScript/browser instead of Python/backend**

**Performance:**
- Backend (buildstl.py → STL → network → NiiVue): **~500-2000ms**
- Client-side (parse TXT → NVMesh): **~50-100ms** ⚡

---

#### Implementation: Direct NVMesh Construction

**Algorithm (from frac notes + buildstl.py logic):**
```typescript
// Frontend: services/medisMeshDirect.ts
import { NVMesh } from '@niivue/niivue';
import type { MedisContour } from './medisParser';

export function medisContoursToMesh(
  contours: MedisContour[],  // From existing MEDIS parser
  meshType: 'lumen' | 'vessel',
  nv: Niivue
): NVMesh {
  /**
   * Direct mesh construction from MEDIS contour rings.
   * 
   * Topology: Connect ring N to ring N+1 with triangular facets
   * 
   *   Ring N:     p0 --- p1 --- p2 --- ... --- pM (closed loop)
   *               |  \    |  \    |              |
   *               |   \   |   \   |              |
   *   Ring N+1:   q0 --- q1 --- q2 --- ... --- qM
   * 
   * Each quad (p_i, p_{i+1}, q_i, q_{i+1}) → 2 triangles
   */
  
  // 1. Extract relevant contours (lumen or vessel wall)
  const rings = contours
    .filter(c => c.group === (meshType === 'lumen' ? 'Lumen' : 'VesselWall'))
    .sort((a, b) => a.sliceDistance - b.sliceDistance);
  
  if (rings.length < 2) {
    throw new Error('Need at least 2 contour rings to create mesh');
  }
  
  // 2. Build vertex array (all points from all rings)
  const totalPoints = rings.reduce((sum, r) => sum + r.points.length, 0);
  const pts = new Float32Array(totalPoints * 3);
  
  let vertexOffset = 0;
  const ringVertexOffsets: number[] = [];
  
  for (const ring of rings) {
    ringVertexOffsets.push(vertexOffset);
    
    for (const point of ring.points) {
      // Points are already in mm coordinates from MEDIS
      pts[vertexOffset * 3 + 0] = point.x;
      pts[vertexOffset * 3 + 1] = point.y;
      pts[vertexOffset * 3 + 2] = point.z;
      vertexOffset++;
    }
  }
  
  // 3. Build triangle index array (connect consecutive rings)
  const triangles: number[] = [];
  
  for (let ringIdx = 0; ringIdx < rings.length - 1; ringIdx++) {
    const ring0 = rings[ringIdx];
    const ring1 = rings[ringIdx + 1];
    const offset0 = ringVertexOffsets[ringIdx];
    const offset1 = ringVertexOffsets[ringIdx + 1];
    
    const n0 = ring0.points.length;
    const n1 = ring1.points.length;
    
    // Handle rings with different point counts (use simpler approach: min)
    const nPoints = Math.min(n0, n1);
    
    for (let i = 0; i < nPoints; i++) {
      const i_next = (i + 1) % nPoints;
      
      // Vertex indices
      const p0 = offset0 + i;
      const p1 = offset0 + i_next;
      const q0 = offset1 + i;
      const q1 = offset1 + i_next;
      
      // Two triangles per quad (CCW winding for outward normals)
      triangles.push(p0, q0, p1);  // Triangle 1
      triangles.push(p1, q0, q1);  // Triangle 2
    }
  }
  
  const tris = new Uint32Array(triangles);
  
  // 4. Create NVMesh
  const rgba = meshType === 'lumen' 
    ? new Uint8Array([255, 0, 0, 255])    // Red lumen
    : new Uint8Array([0, 100, 255, 255]); // Blue vessel wall
  
  const mesh = new NVMesh(
    pts,
    tris,
    `${meshType}-mesh`,
    rgba,
    meshType === 'lumen' ? 0.7 : 0.4,  // Lumen more opaque
    true,                               // visible
    nv.gl as WebGL2RenderingContext,
    null,  // connectome
    null, null, null,  // tractography
    false, // colorbarVisible
    `Coronary ${meshType}`
  );
  
  return mesh;
}

// Usage in app
export async function loadMedisWithMeshes(
  txtFile: File,
  nv: Niivue
): Promise<void> {
  // Parse MEDIS TXT (existing parser)
  const contours = await parseMedisTxt(txtFile);
  
  // Create meshes directly in browser
  const lumenMesh = medisContoursToMesh(contours, 'lumen', nv);
  const vesselMesh = medisContoursToMesh(contours, 'vessel', nv);
  
  // Add to NiiVue (instant overlay on CT)
  nv.addMesh(lumenMesh);
  nv.addMesh(vesselMesh);
  nv.drawScene?.();
  
  console.log('✅ Meshes rendered client-side in <100ms');
}
```

---

#### Coordinate System Handling

**Critical:** MEDIS points must be in same mm space as CT volume

**From your existing code** (`buildstl.py`), MEDIS points are already in mm (LPS or RAS):
```python
# buildstl.py already handles this correctly
points = [(float(x), float(y), float(z)) for x, y, z in point_coords]
# These are mm coordinates matching the DICOM/NIfTI header
```

**In TypeScript:** No conversion needed if MEDIS parser returns mm coordinates
```typescript
interface MedisPoint {
  x: number;  // mm, same coordinate system as CT
  y: number;
  z: number;
}
```

**Sanity Check (Debug Helper):**
```typescript
export function validateMeshCoordinates(
  mesh: NVMesh,
  ctVolume: NVImage
): boolean {
  // Extract mesh bounding box
  const pts = mesh.pts;
  const minMax = { 
    x: [Infinity, -Infinity], 
    y: [Infinity, -Infinity], 
    z: [Infinity, -Infinity] 
  };
  
  for (let i = 0; i < pts.length / 3; i++) {
    minMax.x[0] = Math.min(minMax.x[0], pts[i * 3 + 0]);
    minMax.x[1] = Math.max(minMax.x[1], pts[i * 3 + 0]);
    // ... same for y, z
  }
  
  // CT bounding box in mm
  const dims = ctVolume.hdr.dims;
  const pixDims = ctVolume.hdr.pixDims;
  const ctBBox = {
    x: [0, dims[1] * pixDims[1]],
    y: [0, dims[2] * pixDims[2]],
    z: [0, dims[3] * pixDims[3]]
  };
  
  // Mesh should be inside or near CT bounds
  const insideCT = 
    minMax.x[0] >= ctBBox.x[0] - 10 && minMax.x[1] <= ctBBox.x[1] + 10 &&
    minMax.y[0] >= ctBBox.y[0] - 10 && minMax.y[1] <= ctBBox.y[1] + 10 &&
    minMax.z[0] >= ctBBox.z[0] - 10 && minMax.z[1] <= ctBBox.z[1] + 10;
  
  if (!insideCT) {
    console.warn('⚠️ Mesh bounding box outside CT volume - coordinate mismatch?');
    console.log('Mesh bbox:', minMax);
    console.log('CT bbox:', ctBBox);
  }
  
  return insideCT;
}
```

---

#### Comparison: Backend vs Client-Side

| **Aspect** | **Backend (buildstl.py)** | **Client-Side (Direct NVMesh)** |
|------------|---------------------------|----------------------------------|
| **Speed** | 500-2000ms (network + I/O) | 50-100ms ⚡ |
| **Network** | Must upload TXT, download STL | Parse TXT locally, no upload |
| **Dependencies** | Python backend required | Pure TypeScript |
| **Quality** | Same (topology identical) | Same |
| **File size** | STL ~1-5 MB | No file transfer |
| **Use case** | Export meshes for external tools | Real-time interactive viewing |

**Recommendation:** 
- **Use client-side** for interactive visualization in NiiVue
- **Use backend** only if you need to export STL/OBJ for 3D printing, Meshlab, etc.

---

#### Alternative: Connectome Overlay for Centerline

**From frac notes:** For just the centerline (not full surface), use **connectome mesh**:

```typescript
export function medisContoursToConnectome(
  contours: MedisContour[],
  nv: Niivue
): void {
  // Extract centerline (centroids of lumen contours)
  const lumenRings = contours.filter(c => c.group === 'Lumen');
  
  const nodes = lumenRings.map((ring, i) => {
    const centroid = computeCentroid(ring.points);
    return {
      name: `c${i}`,
      x: centroid.x,
      y: centroid.y,
      z: centroid.z,
      colorValue: 1,
      sizeValue: 2
    };
  });
  
  const edges = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    edges.push({ first: i, second: i + 1, colorValue: 1 });
  }
  
  const connectome = {
    name: 'centerline',
    nodeColormap: 'warm',
    nodeColormapNegative: 'winter',
    nodeMinColor: 0,
    nodeMaxColor: 1,
    nodeScale: 3,
    edgeColormap: 'warm',
    edgeColormapNegative: 'winter',
    edgeMin: 0,
    edgeMax: 1,
    edgeScale: 2,
    legendLineThickness: 0,
    showLegend: false,
    nodes,
    edges
  };
  
  nv.loadConnectome(connectome);
  nv.drawScene?.();
}
```

**Use cases:**
- **Connectome**: Quick centerline visualization (minimal geometry)
- **Direct mesh**: Full vessel surface (lumen + wall)

---

### Pathway 3: Point Cloud → Mesh (Surface Reconstruction)

**Best for:** Sparse point clouds from external sources (not MEDIS)

**Three Algorithms:**

#### 2A: Ball Pivoting Algorithm (Fast, Good for Uniform Density)
```python
# Backend: app/services/point_cloud_mesh.py
import open3d as o3d
import numpy as np

def point_cloud_to_mesh_ball_pivoting(
    points: np.ndarray,
    normals: np.ndarray = None,
    radii: list = [0.5, 1.0, 2.0, 4.0]  # mm
) -> dict:
    """
    Ball pivoting algorithm for point cloud meshing.
    Fast but requires uniform point density.
    
    Args:
        points: Nx3 array of 3D points
        normals: Nx3 array of normal vectors (computed if None)
        radii: List of ball radii for multi-scale reconstruction
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals if not provided
    if normals is None:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.0, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Ball pivoting reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    
    # Post-process
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    return {
        'vertices': np.asarray(mesh.vertices).flatten().tolist(),
        'triangles': np.asarray(mesh.triangles).flatten().tolist()
    }
```

#### 2B: Poisson Surface Reconstruction (Smoothest, Best Quality)
```python
def point_cloud_to_mesh_poisson(
    points: np.ndarray,
    normals: np.ndarray = None,
    depth: int = 9,  # Octree depth (higher = more detail)
    density_threshold: float = 0.01  # Remove low-density regions
) -> dict:
    """
    Poisson surface reconstruction - produces smoothest meshes.
    Best for final high-quality visualization.
    
    Args:
        points: Nx3 array of 3D points
        normals: Nx3 array of normal vectors (required for Poisson)
        depth: Octree depth (8-10 typical, higher = more detail)
        density_threshold: Remove vertices with density < quantile
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals (critical for Poisson)
    if normals is None:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.0, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,  # Auto-compute
        scale=1.1,  # Slightly larger bounding box
        linear_fit=False
    )
    
    # Remove low-density vertices (extrapolated regions)
    densities = np.asarray(densities)
    density_threshold_value = np.quantile(densities, density_threshold)
    vertices_to_remove = densities < density_threshold_value
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return {
        'vertices': np.asarray(mesh.vertices).flatten().tolist(),
        'triangles': np.asarray(mesh.triangles).flatten().tolist()
    }
```

#### 2C: Alpha Shapes (Good for Non-Convex Shapes)
```python
def point_cloud_to_mesh_alpha_shape(
    points: np.ndarray,
    alpha: float = 0.03  # Smaller = tighter fit
) -> dict:
    """
    Alpha shape reconstruction - good for non-convex vessels.
    Faster than Poisson but less smooth.
    
    Args:
        points: Nx3 array of 3D points
        alpha: Alpha parameter (smaller = tighter fit to points)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Alpha shape
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    mesh.compute_vertex_normals()
    
    return {
        'vertices': np.asarray(mesh.vertices).flatten().tolist(),
        'triangles': np.asarray(mesh.triangles).flatten().tolist()
    }
```

---

### Pathway 3: Contour Rings → Mesh (Direct Tube)

**Best for:** MEDIS TXT contours (already implemented above in Approach 0)

**Advantages:**
- Fastest method (50ms)
- Preserves exact contour geometry
- No interpolation artifacts

**See earlier section:** "Approach 0: Ultra-Simple STL Generation"

---

## 🎨 NiiVue Mesh Format Support & Best Practices

### Supported Mesh Formats in NiiVue v0.66.0

**NiiVue natively supports 15+ mesh formats:**

| **Format** | **Extension** | **Type** | **Size** | **Speed** | **Recommended** |
|------------|---------------|----------|----------|-----------|------------------|
| **MZ3** | `.mz3` | Binary | ⭐⭐⭐⭐⭐ Smallest | ⭐⭐⭐⭐⭐ Fastest | ✅ **Best for web** |
| **PLY** | `.ply` | Binary | ⭐⭐⭐⭐ Small | ⭐⭐⭐⭐ Fast | ✅ **Good for web** |
| **GIfTI** | `.gii` | Base64 | ⭐⭐ Large | ⭐⭐ Slow | ❌ Neuroimaging only |
| **VTK** | `.vtk` | ASCII | ⭐⭐⭐ Medium | ⭐⭐⭐ Medium | ✅ **Good compatibility** |
| **OBJ** | `.obj` | ASCII | ⭐⭐ Large | ⭐⭐ Slow | ❌ 3D printing only |
| **STL** | `.stl` | Binary | ⭐ Huge | ⭐ Very slow | ❌ Avoid (no vertex reuse) |
| **FreeSurfer** | `.pial` | Binary | ⭐⭐⭐⭐ Small | ⭐⭐⭐⭐ Fast | ✅ Neuroimaging |
| **OFF** | `.off` | ASCII | ⭐⭐⭐ Medium | ⭐⭐⭐ Medium | ✅ Simple format |

**Recommendation for Web Application:**
1. **Primary:** MZ3 format (smallest, fastest loading)
2. **Fallback:** PLY binary (widely supported, fast)
3. **Avoid:** STL (3x larger), GIfTI (slow base64 decoding), OBJ (ASCII overhead)

---

### MZ3 Format: Best Choice for Web Viewing

**Why MZ3?**
- ✅ **Smallest file size:** 3-5x smaller than PLY, 10x smaller than STL
- ✅ **Fastest loading:** Binary format with efficient compression
- ✅ **Full feature support:** Vertices, faces, normals, colors, scalars
- ✅ **Native NiiVue support:** No conversion needed
- ✅ **Created by NiiVue author:** Optimized for medical imaging

**MZ3 Format Specification:**
```
MZ3 Binary Format:
- Magic number: 0x4D5A3301 ("MZ3" + version)
- Header: vertex count, face count, attribute flags
- Vertices: Float32Array (x,y,z per vertex)
- Faces: Uint32Array (i0,i1,i2 per triangle)
- Optional: Normals, colors, scalars (per-vertex data)
- Compression: Optional gzip compression
```

**Generate MZ3 with nii2mesh:**
```bash
# Command-line
nii2mesh input.nii.gz -i m -r 0.15 -s 10 output.mz3

# Python wrapper
from app.services.nii2mesh_wrapper import segmentation_to_mesh_nii2mesh

mesh_path = segmentation_to_mesh_nii2mesh(
    mask,
    affine,
    output_format='mz3',  # Smallest, fastest
    reduction=0.15,
    smooth_iterations=10
)
```

**Load MZ3 in NiiVue:**
```typescript
// Frontend: Load MZ3 mesh
import { Niivue } from '@niivue/niivue';

const nv = new Niivue();
await nv.attachToCanvas(document.getElementById('gl'));

// Load MZ3 mesh (fast!)
await nv.loadMeshes([
  { url: '/meshes/LAD_lumen.mz3', rgba255: [255, 0, 0, 200] },  // Red lumen
  { url: '/meshes/LAD_vessel.mz3', rgba255: [0, 0, 255, 100] }   // Blue vessel wall
]);

// Or load from API response
const response = await fetch('/api/mesh/LAD_lumen.mz3');
const blob = await response.blob();
const url = URL.createObjectURL(blob);
await nv.loadMeshes([{ url, rgba255: [255, 0, 0, 200] }]);
URL.revokeObjectURL(url);
```

---

### Mesh Optimization for Web Performance

**Target Metrics for Interactive Web Viewing:**
- **File size:** <5 MB per mesh (MZ3 compressed)
- **Triangle count:** 50K-200K triangles (balance quality/performance)
- **Loading time:** <500ms per mesh
- **Rendering:** 60 FPS on mid-range GPU

**Optimization Pipeline:**
```python
# Backend: app/services/mesh_optimizer.py
import trimesh
import numpy as np

def optimize_mesh_for_web(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = 100_000,
    smooth_iterations: int = 3
) -> dict:
    """
    Optimize mesh for web viewing:
    1. Decimate to target triangle count
    2. Smooth to reduce faceting
    3. Remove degenerate triangles
    4. Compute normals for lighting
    """
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 1. Decimate (reduce triangles)
    if len(faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
    
    # 2. Smooth (Laplacian)
    trimesh.smoothing.filter_laplacian(
        mesh,
        iterations=smooth_iterations,
        lamb=0.5  # Smoothing strength
    )
    
    # 3. Clean up
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    
    # 4. Compute normals
    mesh.fix_normals()  # Ensure consistent winding
    vertex_normals = mesh.vertex_normals
    
    return {
        'vertices': mesh.vertices.flatten().tolist(),
        'triangles': mesh.faces.flatten().tolist(),
        'normals': vertex_normals.flatten().tolist(),
        'stats': {
            'num_vertices': len(mesh.vertices),
            'num_triangles': len(mesh.faces),
            'bounds': mesh.bounds.tolist(),
            'volume_mm3': mesh.volume if mesh.is_watertight else None
        }
    }
```

---

### Complete Integration Example

**Scenario:** User clicks on coronary artery → AI segments → Display mesh in NiiVue

**Frontend (TypeScript):**
```typescript
// services/meshWorkflow.ts
import { Niivue } from '@niivue/niivue';
import { apiClient } from './api';

export async function segmentAndDisplayMesh(
  nv: Niivue,
  clickPoint: [number, number, number],
  vesselName: string
): Promise<void> {
  
  // Step 1: Request AI segmentation
  const segResponse = await apiClient.post('/api/segment', {
    point: clickPoint,
    vessel: vesselName
  });
  
  const segmentationId = segResponse.data.segmentation_id;
  
  // Step 2: Request mesh generation (MZ3 format)
  const meshResponse = await apiClient.post('/api/mesh/from-segmentation', {
    segmentation_id: segmentationId,
    format: 'mz3',  // Fastest for web
    reduction: 0.15,  // 15% of original triangles
    smooth: 10
  });
  
  // Step 3: Download mesh
  const meshBlob = await fetch(meshResponse.data.mesh_url).then(r => r.blob());
  const meshUrl = URL.createObjectURL(meshBlob);
  
  // Step 4: Load into NiiVue
  await nv.loadMeshes([{
    url: meshUrl,
    rgba255: [255, 0, 0, 200],  // Red, semi-transparent
    name: `${vesselName}_lumen`
  }]);
  
  // Cleanup
  URL.revokeObjectURL(meshUrl);
  
  console.log(`Mesh loaded: ${meshResponse.data.stats.num_triangles} triangles`);
}

// Usage
const nv = new Niivue();
await nv.attachToCanvas(document.getElementById('gl'));

// User clicks on LAD
const clickPoint: [number, number, number] = [120.5, 85.3, 42.1];
await segmentAndDisplayMesh(nv, clickPoint, 'LAD');
```

**Backend (Python FastAPI):**
```python
# app/api/mesh.py
from fastapi import APIRouter, BackgroundTasks
from app.services.sam_med3d import segment_vessel
from app.services.nii2mesh_wrapper import segmentation_to_mesh_nii2mesh
from app.services.mesh_optimizer import optimize_mesh_for_web
import nibabel as nib

router = APIRouter()

@router.post('/api/mesh/from-segmentation')
async def create_mesh(
    segmentation_id: str,
    format: str = 'mz3',
    reduction: float = 0.15,
    smooth: int = 10
):
    """Generate optimized mesh from segmentation."""
    
    # Load segmentation mask
    seg_path = f'/data/segmentations/{segmentation_id}.nii.gz'
    seg_nii = nib.load(seg_path)
    mask = seg_nii.get_fdata() > 0.5
    
    # Generate mesh with nii2mesh (high quality)
    mesh_path = segmentation_to_mesh_nii2mesh(
        mask,
        seg_nii.affine,
        output_format=format,
        reduction=reduction,
        smooth_iterations=smooth
    )
    
    # Get mesh stats
    import trimesh
    mesh = trimesh.load(mesh_path)
    
    return {
        'mesh_url': f'/static/meshes/{mesh_path.name}',
        'stats': {
            'num_vertices': len(mesh.vertices),
            'num_triangles': len(mesh.faces),
            'file_size_mb': mesh_path.stat().st_size / 1e6,
            'format': format
        }
    }
```

---

## 🚀 Recommended Workflow for Segment Platform

**For Interactive Web Application:**

1. **Fast Preview (Client-side):**
   - Use vtk.js marching cubes for immediate feedback (<1s)
   - Display low-poly mesh while high-quality mesh generates
   - Good for user interaction and validation

2. **High-Quality Export (Server-side):**
   - Use nii2mesh with MZ3 output for final visualization
   - Generate in background while user reviews preview
   - Swap preview mesh with high-quality mesh when ready

3. **Mesh Format Strategy:**
   - **Primary:** MZ3 (smallest, fastest)
   - **Export options:** PLY (3D software), STL (3D printing), VTK (analysis)
   - **Avoid:** GIfTI (unless neuroimaging), OBJ (unless required)

**Implementation Priority:**
1. ✅ **Week 1-2:** Implement vtk.js marching cubes (fast preview)
2. ✅ **Week 3-4:** Integrate nii2mesh backend (high quality)
3. ✅ **Week 5-6:** Add MZ3 format support (optimal web performance)
4. ✅ **Week 7-8:** Implement mesh optimization pipeline

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

### Input Data Format

**MEDIS TXT structure** (from `C:\Users\steff\Documents\GitHub\flow\data\*.txt`):
```
# Contour index: 0
# group: Lumen
# SliceDistance: 0.25
# Number of points: 50
17.518342971801758 16.819992065429688 1967.189453125
17.93111801147461 16.195648193359375 1967.219482421875
...

# Contour index: 1
# group: VesselWall
# SliceDistance: 0.25
# Number of points: 49
18.509117126464844 15.870161056518555 1966.8944091796875
...
```

**Key observations:**
- Each contour has fixed `SliceDistance` (spacing along vessel)
- `Lumen` contours define inner wall (blood pool)
- `VesselWall` contours define outer wall (including plaque)
- Points are in 3D physical space (mm): `[x, y, z]`

### Algorithm 1.1: Extract Centerline Points

**Method: Centroid of Lumen Contours**

```
Input:
  - lumen_contours: List of N lumen contours, each with M_i points
  
Output:
  - centerline_points: Array of N points [x, y, z]
  - slice_distances: Array of N distances along vessel

Algorithm:
  FOR each lumen_contour in lumen_contours:
    // Compute centroid (geometric center)
    centroid = [0, 0, 0]
    FOR each point in lumen_contour.points:
      centroid += point
    centroid /= len(lumen_contour.points)
    
    centerline_points.append(centroid)
    slice_distances.append(lumen_contour.SliceDistance)
  
  RETURN centerline_points, slice_distances
```

**Example:**
```
Contour 0 (50 lumen points) → Centroid: [15.2, 14.8, 1970.5]
Contour 1 (48 lumen points) → Centroid: [15.1, 14.9, 1971.0]
...
→ Centerline: N points spaced by SliceDistance (typically 0.25-0.5 mm)
```

### Algorithm 1.2: Compute Tangent Vectors (T)

**Method: Finite Differences**

```
Input:
  - centerline_points: Array of N points [x, y, z]
  
Output:
  - tangents: Array of N normalized direction vectors

Algorithm:
  tangents = []
  
  FOR i = 0 to N-1:
    IF i == 0:
      // Forward difference at start
      tangent = centerline_points[1] - centerline_points[0]
    ELSE IF i == N-1:
      // Backward difference at end
      tangent = centerline_points[i] - centerline_points[i-1]
    ELSE:
      // Central difference (more accurate)
      tangent = centerline_points[i+1] - centerline_points[i-1]
    
    // Normalize to unit vector
    length = ||tangent||
    tangent = tangent / length
    
    tangents.append(tangent)
  
  RETURN tangents
```

**Why central difference?**
- More accurate approximation of derivative
- Symmetric → reduces bias
- Standard in numerical differentiation

### Algorithm 1.3: Compute Normal and Binormal Vectors (N, B)

**Method: Frenet-Serret Frame Construction**

```
Input:
  - tangents: Array of N tangent vectors T
  
Output:
  - normals: Array of N normal vectors N
  - binormals: Array of N binormal vectors B

Algorithm:
  normals = []
  binormals = []
  
  FOR i = 0 to N-1:
    T = tangents[i]
    
    // Step 1: Compute curvature vector (dT/ds)
    IF i == 0:
      dT = tangents[1] - tangents[0]
    ELSE IF i == N-1:
      dT = tangents[i] - tangents[i-1]
    ELSE:
      dT = tangents[i+1] - tangents[i-1]
    
    // Step 2: Normal is normalized curvature direction
    curvature_magnitude = ||dT||
    IF curvature_magnitude > epsilon:
      N = dT / curvature_magnitude
    ELSE:
      // Straight segment: choose arbitrary perpendicular
      N = perpendicular_to(T)
    
    // Step 3: Binormal via cross product (right-handed system)
    B = cross_product(T, N)
    B = normalize(B)
    
    normals.append(N)
    binormals.append(B)
  
  RETURN normals, binormals

// Helper: Find arbitrary perpendicular vector
FUNCTION perpendicular_to(v):
  // Choose axis least aligned with v
  IF abs(v.x) < abs(v.y) AND abs(v.x) < abs(v.z):
    axis = [1, 0, 0]
  ELSE IF abs(v.y) < abs(v.z):
    axis = [0, 1, 0]
  ELSE:
    axis = [0, 0, 1]
  
  // Cross product gives perpendicular
  perp = cross_product(v, axis)
  RETURN normalize(perp)
```

**Frenet-Serret Frame Properties:**
- **Orthogonal**: T ⊥ N, T ⊥ B, N ⊥ B
- **Right-handed**: B = T × N
- **Unit vectors**: ||T|| = ||N|| = ||B|| = 1

---

## 🔬 Algorithm 2: Perpendicular Cross-Section Extraction

### Goal
Extract a 2D cross-sectional image perpendicular to the vessel centerline at each point.

### Mathematical Setup

**Local Coordinate System at point P:**
- **Origin**: Centerline point P = [px, py, pz]
- **u-axis**: Normal vector N (horizontal in cross-section)
- **v-axis**: Binormal vector B (vertical in cross-section)  
- **w-axis**: Tangent vector T (perpendicular to cross-section)

**Cross-section plane equation:**
```
Any point Q in the plane satisfies: dot(Q - P, T) = 0
Parametric form: Q(u, v) = P + u*N + v*B
  where u ∈ [-size_u/2, +size_u/2]
        v ∈ [-size_v/2, +size_v/2]
```

### Algorithm 2.1: Sample Cross-Section Grid

```
Input:
  - cta_volume: 3D volume data (NIfTI format)
  - cta_affine: 4×4 affine matrix (voxel → physical space)
  - centerline_point: [px, py, pz] in mm
  - normal: N vector (unit)
  - binormal: B vector (unit)
  - tangent: T vector (unit)
  - cross_section_size: [width, height] in pixels (e.g., [64, 64])
  - cross_section_spacing: [du, dv] in mm (e.g., [0.5, 0.5])
  
Output:
  - cross_section_image: 2D array [width × height]

Algorithm:
  P = centerline_point
  [W, H] = cross_section_size
  [du, dv] = cross_section_spacing
  
  cross_section_image = zeros(W, H)
  
  // Sample grid in (u,v) coordinates
  FOR iu = 0 to W-1:
    FOR iv = 0 to H-1:
      // Convert pixel indices to physical offsets
      u = (iu - W/2) * du  // Center at origin
      v = (iv - H/2) * dv
      
      // Compute 3D physical position
      Q_physical = P + u*N + v*B
      
      // Transform to voxel coordinates
      Q_voxel = world_to_voxel(Q_physical, cta_affine)
      
      // Interpolate intensity at Q_voxel
      intensity = trilinear_interpolate(cta_volume, Q_voxel)
      
      cross_section_image[iu, iv] = intensity
  
  RETURN cross_section_image
```

### Algorithm 2.2: Trilinear Interpolation

**Purpose:** Sample CTA volume at non-integer voxel coordinates

```
Input:
  - volume: 3D array [Nx × Ny × Nz]
  - position: [x, y, z] in voxel coordinates (can be fractional)
  
Output:
  - intensity: Interpolated value

Algorithm:
  [x, y, z] = position
  
  // Floor and fractional parts
  x0 = floor(x);  x1 = x0 + 1;  fx = x - x0
  y0 = floor(y);  y1 = y0 + 1;  fy = y - y0
  z0 = floor(z);  z1 = z0 + 1;  fz = z - z0
  
  // Bounds check (return 0 if out of bounds)
  IF x0 < 0 OR x1 >= Nx OR y0 < 0 OR y1 >= Ny OR z0 < 0 OR z1 >= Nz:
    RETURN 0  // Background value
  
  // Sample 8 corner voxels
  c000 = volume[x0, y0, z0]
  c001 = volume[x0, y0, z1]
  c010 = volume[x0, y1, z0]
  c011 = volume[x0, y1, z1]
  c100 = volume[x1, y0, z0]
  c101 = volume[x1, y0, z1]
  c110 = volume[x1, y1, z0]
  c111 = volume[x1, y1, z1]
  
  // Trilinear interpolation
  c00 = c000*(1-fx) + c100*fx
  c01 = c001*(1-fx) + c101*fx
  c10 = c010*(1-fx) + c110*fx
  c11 = c011*(1-fx) + c111*fx
  
  c0 = c00*(1-fy) + c10*fy
  c1 = c01*(1-fy) + c11*fy
  
  intensity = c0*(1-fz) + c1*fz
  
  RETURN intensity
```

---

## 🎞️ Algorithm 3: Straightened MPR Volume Construction

### Goal
Create a 3D volume where the vessel appears straight, with the centerline running along one axis.

### Volume Layout

```
Straightened Volume Dimensions: [W, H, D]
  - W: Width of cross-section (e.g., 64 pixels)
  - H: Height of cross-section (e.g., 64 pixels)
  - D: Depth = number of centerline points (e.g., 200 slices)

Voxel coordinates:
  - (u, v, d) where:
    * u ∈ [0, W-1]: Horizontal in cross-section
    * v ∈ [0, H-1]: Vertical in cross-section
    * d ∈ [0, D-1]: Along straightened centerline
```

### Algorithm 3.1: Build Straightened Volume

```
Input:
  - cta_volume: Original 3D CTA volume
  - cta_affine: Affine transformation matrix
  - centerline_points: Array of N points
  - normals: Array of N normal vectors
  - binormals: Array of N binormal vectors
  - tangents: Array of N tangent vectors
  - cross_section_size: [W, H] pixels
  - cross_section_spacing: [du, dv] mm
  
Output:
  - straightened_volume: 3D array [W × H × N]
  - straightened_affine: New affine matrix

Algorithm:
  [W, H] = cross_section_size
  N = len(centerline_points)
  
  straightened_volume = zeros(W, H, N)
  
  // Extract cross-section at each centerline point
  FOR d = 0 to N-1:
    P = centerline_points[d]
    N_vec = normals[d]
    B_vec = binormals[d]
    T_vec = tangents[d]
    
    cross_section = extract_cross_section(
      cta_volume, cta_affine,
      P, N_vec, B_vec, T_vec,
      cross_section_size,
      cross_section_spacing
    )
    
    straightened_volume[:, :, d] = cross_section
  
  // Construct affine matrix for straightened volume
  straightened_affine = build_straightened_affine(
    cross_section_spacing,
    slice_spacing  // Spacing along centerline
  )
  
  RETURN straightened_volume, straightened_affine
```

### Algorithm 3.2: Straightened Affine Matrix

**Purpose:** Define voxel-to-physical-space mapping for straightened volume

```
Algorithm:
  [du, dv] = cross_section_spacing
  ds = slice_spacing  // Typically mean distance between centerline points
  
  // Identity-based affine (straightened coordinate system)
  affine = [
    [du,  0,  0,  -W*du/2],  // u-axis (horizontal)
    [ 0, dv,  0,  -H*dv/2],  // v-axis (vertical)
    [ 0,  0, ds,          0],  // d-axis (along centerline)
    [ 0,  0,  0,          1]   // Homogeneous
  ]
  
  RETURN affine
```

**Note:** Origin at center of first cross-section, straightened along z-axis

---

## 🎮 Algorithm 4: Interactive Viewing Angle Controls

### Problem
Users need to adjust the viewing orientation of cross-sections and straightened MPR to:
1. **Rotate cross-section** around vessel axis (change N, B orientation)
2. **Tilt straightened MPR** to view from different angles
3. **Slide along centerline** to inspect specific locations

### Algorithm 4.1: Rotate Cross-Section Around Tangent

**Purpose:** Rotate N and B vectors around T by angle θ

```
Input:
  - normal: N vector (original)
  - binormal: B vector (original)
  - tangent: T vector (axis of rotation)
  - theta: Rotation angle in radians
  
Output:
  - normal_rotated: New N' vector
  - binormal_rotated: New B' vector

Algorithm (Rodrigues' rotation formula):
  // Rotate N around T by theta
  N' = N*cos(theta) + (T × N)*sin(theta) + T*(T·N)*(1 - cos(theta))
  
  // Rotate B around T by theta
  B' = B*cos(theta) + (T × B)*sin(theta) + T*(T·B)*(1 - cos(theta))
  
  // Simplification: Since N ⊥ T and B ⊥ T:
  //   T·N = 0, T·B = 0
  // So:
  N' = N*cos(theta) + (T × N)*sin(theta)
  B' = B*cos(theta) + (T × B)*sin(theta)
  
  // Alternative using rotation matrix:
  R = rotation_matrix_axis_angle(T, theta)
  N' = R * N
  B' = R * B
  
  RETURN N', B'

// Helper: Rotation matrix for axis-angle rotation
FUNCTION rotation_matrix_axis_angle(axis, theta):
  [x, y, z] = normalize(axis)
  c = cos(theta)
  s = sin(theta)
  t = 1 - c
  
  R = [
    [t*x*x + c,    t*x*y - z*s,  t*x*z + y*s],
    [t*x*y + z*s,  t*y*y + c,    t*y*z - x*s],
    [t*x*z - y*s,  t*y*z + x*s,  t*z*z + c  ]
  ]
  
  RETURN R
```

### Algorithm 4.2: UI Control Mapping

**Control Panel (Right Pane):**

```
Cross-Section Controls:
  [Slider] Centerline Position: 0 ──────●── N-1
           → Selects which cross-section to display
           → Updates position marker on centerline overlay
  
  [Slider] Rotation Angle: -180° ──●── +180°
           → Rotates N, B around T (vessel axis)
           → Recomputes cross-section on-the-fly
  
  [Slider] Zoom: 0.5× ──●── 4.0×
           → Adjusts cross_section_size (field of view)
  
  [Button] Reset View
           → theta = 0, zoom = 1.0

Straightened MPR Controls:
  [Slider] Viewing Angle: 0° ──●── 360°
           → Rotates MPR volume around long axis
           → For multi-planar views
  
  [Toggle] Curved vs. Straightened
           → Switch between CPR modes
  
  [Slider] Window/Level (HU)
           → Adjust contrast for plaque visibility

Mesh Display:
  [Checkbox] Show Lumen Mesh
  [Checkbox] Show Vessel Wall Mesh
  [Slider] Mesh Opacity: 0% ──●── 100%
```

---

## 🖼️ Algorithm 5: Quad-View Layout Integration

### Layout Design (Inspired by Frac 4-Panel)

```
┌─────────────────────────────────────────────────────────────┐
│  [Logo] Segment    [Load NII] [Load TXT] [⚙️] [👤] [◫ Layout]│
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐ ┌──────────────────┐  ┌─────────────┐ │
│  │                  │ │                  │  │  Controls   │ │
│  │  Panel 1:        │ │  Panel 2:        │  │             │ │
│  │  Original CTA    │ │  Cross-Section   │  │ ┌─────────┐ │ │
│  │  (Axial/MPR)     │ │  MPR (⊥ vessel)  │  │ │Position │ │ │
│  │                  │ │                  │  │ │ [Slider]│ │ │
│  │  + Centerline    │ │  + Current       │  │ │         │ │ │
│  │    overlay       │ │    position      │  │ │Rotation │ │ │
│  │                  │ │    marker        │  │ │ [Slider]│ │ │
│  └──────────────────┘ └──────────────────┘  │ │         │ │ │
│                                              │ │Zoom     │ │ │
│  ┌──────────────────┐ ┌──────────────────┐  │ │ [Slider]│ │ │
│  │                  │ │                  │  │ │         │ │ │
│  │  Panel 3:        │ │  Panel 4:        │  │ │[Reset]  │ │ │
│  │  Straightened    │ │  3D Mesh         │  │ └─────────┘ │ │
│  │  MPR (Vessel     │ │  Visualization   │  │             │ │
│  │  unfolded)       │ │                  │  │ Mesh        │ │
│  │                  │ │  + Lumen (red)   │  │ ┌─────────┐ │ │
│  │  + Plaque        │ │  + Vessel wall   │  │ │☑ Lumen  │ │ │
│  │    visible       │ │    (blue)        │  │ │☑ Vessel │ │ │
│  │                  │ │  + Rotatable     │  │ │Opacity  │ │ │
│  └──────────────────┘ └──────────────────┘  └─────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Panel-Specific Algorithms

**Panel 1: Original CTA + Centerline Overlay**
```
Display:
  - Load NIfTI CTA volume into NiiVue
  - Parse MEDIS TXT → extract centerline
  - Render centerline as 3D polyline overlay (yellow)
  - Allow user to click on centerline → jump to that position
```

**Panel 2: Cross-Section MPR**
```
Display:
  - Extract perpendicular cross-section at current position
  - Show as 2D image with vessel lumen in center
  - Overlay: Contour outlines (lumen=red, vessel=blue)
  - Updates in real-time when sliders change
```

**Panel 3: Straightened MPR**
```
Display:
  - Show straightened volume as 2D image (sagittal-like view)
  - Vertical axis = along centerline
  - Horizontal axis = cross-section width
  - Plaque and stenosis visible as intensity changes
```

**Panel 4: 3D Mesh Visualization**
```
Display:
  - Load STL meshes generated from buildstl.py
  - Lumen mesh (red, semi-transparent)
  - Vessel wall mesh (blue, more transparent)
  - Interactive rotation/zoom with mouse
```

---

## 🔄 Complete Workflow: Data Flow

### Step-by-Step Execution

```
1. USER LOADS DATA
   ├─ Load CTA volume (NIfTI): patient_001.nii.gz
   └─ Load MEDIS TXT: patient_001_lad.txt

2. PARSE MEDIS TXT
   ├─ Extract lumen contours (N rings of M points each)
   ├─ Extract vessel wall contours
   └─ Extract slice distances

3. COMPUTE CENTERLINE (Algorithm 1)
   ├─ Centroid of each lumen contour → N centerline points
   ├─ Finite differences → tangent vectors T
   └─ Frenet-Serret → normal N and binormal B vectors

4. GENERATE MESHES (Existing: buildstl.py)
   ├─ Lumen tube mesh (STL)
   └─ Vessel wall tube mesh (STL)

5. EXTRACT CROSS-SECTIONS (Algorithm 2)
   ├─ For each centerline point:
   │  ├─ Define perpendicular plane (N, B basis)
   │  ├─ Sample CTA volume on 64×64 grid
   │  └─ Store as 2D image
   └─ Current position controlled by slider

6. BUILD STRAIGHTENED MPR (Algorithm 3)
   ├─ Stack all cross-sections along z-axis
   ├─ Result: [64 × 64 × N] volume
   └─ Save as NIfTI for visualization

7. DISPLAY QUAD-VIEW (Algorithm 5)
   ├─ Panel 1: CTA + centerline overlay
   ├─ Panel 2: Current cross-section (interactive)
   ├─ Panel 3: Straightened MPR
   └─ Panel 4: 3D mesh (lumen + vessel wall)

8. USER INTERACTION (Algorithm 4)
   ├─ Slider: Move along centerline → update Panel 2
   ├─ Slider: Rotate cross-section → recompute Panel 2
   ├─ Slider: Viewing angle → rotate Panel 3
   └─ Toggle: Show/hide meshes in Panel 4
```

---

## 🎯 Implementation Priorities (No Coding Yet)

### Phase 1: Core Algorithms (Backend)
1. **Centerline extraction** from MEDIS TXT (Algorithm 1)
2. **Frenet-Serret frame** computation (Algorithm 1.3)
3. **Cross-section sampling** with trilinear interpolation (Algorithm 2)
4. **Straightened volume construction** (Algorithm 3)

### Phase 2: Visualization (Frontend)
1. **NiiVue integration** for CTA volume display
2. **Cross-section viewer** with real-time updates
3. **Straightened MPR viewer**
4. **STL mesh overlay** (already have meshes from buildstl.py)

### Phase 3: Interactivity (Frontend)
1. **Slider controls** for position/rotation/zoom
2. **Synchronized views** (click in Panel 1 → update Panel 2)
3. **Mouse interaction** for 3D mesh rotation
4. **Export functionality** (save straightened NIfTI, screenshots)

### Phase 4: Optimization
1. **GPU acceleration** for real-time cross-section extraction
2. **Caching** of computed cross-sections
3. **Progressive loading** for large datasets
4. **WebGL shaders** for fast interpolation

---

## 🔬 Mathematical Notes & Challenges

### Challenge 1: Handling Vessel Bifurcations
**Problem:** Centerline branches (e.g., LAD → diagonal branch)
**Solution:** 
- Detect branches in MEDIS contours (sudden topology change)
- Split into separate centerlines
- Process each branch independently
- Allow user to select which branch to view

### Challenge 2: Rotation Angle Ambiguity
**Problem:** Arbitrary initial orientation of N, B
**Solution:**
- Use consistent reference frame (e.g., align N with "superior" direction)
- Allow user to set reference angle
- Maintain smooth transitions along centerline (minimize twist)

### Challenge 3: Variable Spacing
**Problem:** Centerline points may have non-uniform spacing
**Solution:**
- Resample centerline to uniform spacing (e.g., 0.5mm)
- Use arc-length parametrization
- Cubic spline interpolation for smooth curve

### Challenge 4: Out-of-Bounds Sampling
**Problem:** Cross-section may extend beyond CTA volume
**Solution:**
- Return 0 (air) for out-of-bounds voxels
- Clip cross-section to volume bounds
- Warn user if >20% of cross-section is out-of-bounds

---

## 💡 Future Enhancements

### Voxel-Based Approach (Alternative to Point Clouds)
**When:** If MEDIS TXT not available, or for automatic processing
**Method:**
1. Segmentation mask (SAM-Med3D) → binary volume
2. Voronoi skeletonization → centerline
3. Distance transform → vessel radius at each point
4. Same Frenet-Serret pipeline → cross-sections

**Advantage:** Fully automatic, no manual contours needed

### Advanced CPR Modes
1. **Stretched CPR:** Preserve vessel length (no compression at curves)
2. **Projected CPR:** Maximum intensity projection along curved plane
3. **Multi-path CPR:** Simultaneous display of multiple branches

### Quantitative Analysis
1. **Stenosis detection:** Measure minimum lumen diameter
2. **Plaque burden:** Compare lumen vs. vessel wall areas
3. **Remodeling index:** Vessel wall area / lumen area
4. **Calcium scoring:** Integrate HU values in wall

---

## 📏 Technical Specifications

### Performance Targets
- **Centerline extraction:** <100ms for 200-point centerline
- **Single cross-section:** <10ms (real-time slider interaction)
- **Straightened MPR generation:** <2s for full volume
- **Mesh generation:** Already fast with buildstl.py (<50ms)

### Memory Requirements
- **CTA volume:** ~200 MB (512×512×300 × 2 bytes)
- **Straightened volume:** ~8 MB (64×64×200 × 2 bytes)
- **Meshes:** <5 MB per vessel (STL format)
- **Total:** ~300 MB per case (reasonable for modern browsers)

### Accuracy Considerations
- **Centerline:** ±0.5mm (limited by contour spacing)
- **Cross-section orientation:** ±2° (Frenet-Serret numerical stability)
- **Interpolation error:** <2 HU (trilinear interpolation quality)
- **Sufficient** for clinical visualization and qualitative assessment

---

---

## 🔄 Optional: General Volume Viewer with Free Rotation

### Overview

**Separate from vessel-specific tools**, the platform includes a **general-purpose 3D volume viewer** with **free rotation capabilities**. This is based on the existing `viewer.py` implementation (`C:\Users\steff\Documents\GitHub\allerlei\old\viewer.py`) and provides unrestricted viewing angles for any NIfTI volume.

**Use Cases:**
- General CTA volume inspection without vessel-specific constraints
- Quality control and data verification
- Free exploration before starting vessel analysis
- Educational demonstrations
- Multi-modality volume viewing (MRI, CT, PET, etc.)

**Key Feature:** **Interactive mouse-drag rotation** with real-time volume resampling

---

### Mathematical Foundation: 3D Rotation System

**Based on:** Rodrigues' rotation formula + Euler angles

#### Rotation Matrix from Axis-Angle

**Rodrigues' Formula:**
```
Given: axis n (unit vector), angle θ (radians)
Rotation matrix R:

R = I*cos(θ) + (1 - cos(θ))*n*n^T + [n]_×*sin(θ)

Where [n]_× is the cross-product matrix:
[n]_× = [  0   -n_z   n_y ]
        [ n_z    0   -n_x ]
        [-n_y   n_x    0  ]
```

**Implementation (from viewer.py):**
```python
def cross_product_matrix(v):
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0]
    ])

def matrix_from_axis_angle(n, theta):
    """
    Rodrigues' rotation formula implementation.
    
    Args:
        n: Unit axis vector [x, y, z]
        theta: Rotation angle in radians
    
    Returns:
        3×3 rotation matrix
    """
    return (
        np.eye(3) * np.cos(theta) + 
        (1.0 - np.cos(theta)) * n[:,np.newaxis].dot(n[np.newaxis,:]) + 
        cross_product_matrix(n) * np.sin(theta)
    )
```

#### Euler Angles Extraction

**Purpose:** Convert rotation matrix back to Euler angles (X-Y-Z convention) for display/storage

```python
def euler_from_matrix(R):
    """
    Extract Euler angles from rotation matrix.
    Convention: Rotate around Z, then Y, then X (extrinsic)
    
    Returns:
        [angle_x, angle_y, angle_z] in radians, range (-π, π]
    """
    if np.abs(R[2, 0]) != 1.0:
        # General case: two solutions exist
        angle2 = np.arcsin(R[2, 0])
        angle3 = np.arctan2(-R[2, 1] / np.cos(angle2), R[2, 2] / np.cos(angle2))
        angle1 = np.arctan2(-R[1, 0] / np.cos(angle2), R[0, 0] / np.cos(angle2))
    else:
        # Gimbal lock case
        if R[2, 0] == 1.0:
            angle3 = 0.0
            angle2 = np.pi / 2.0
            angle1 = np.arctan2(R[0, 1], -R[0, 2])
        else:
            angle3 = 0.0
            angle2 = -np.pi / 2.0
            angle1 = np.arctan2(R[0, 1], R[0, 2])
    
    return -np.array([angle1, angle2, angle3])[::-1]
```

---

### Interactive Mouse Drag Rotation Algorithm

**Goal:** Rotate volume in real-time as user drags mouse across viewing panel

**Mouse Drag Event Handling:**
```python
def mouseDragEvent(self, ev):
    """
    Drag mode 0: Free rotation mode
    
    Algorithm:
    1. Compute drag angle in viewing plane
    2. Rotation axis = normal to current view (perpendicular to screen)
    3. Apply in-plane rotation around this axis
    4. Update global rotation matrix
    5. Resample and redisplay all views
    """
    pos, lastPos = ev.pos(), ev.lastPos()
    
    if Viewer.dragmode == 0:  # Rotation mode
        # Center of view
        c = pg.Point(self.boundingRect().center())
        
        # Compute angle between lastPos→center→pos
        # Positive angle = counterclockwise in view
        radinplane = angle(pos, c, lastPos) * (1 if self.spositive else -1)
        
        # Normalize view normal (perpendicular to screen)
        self.normal = self.normal / np.linalg.norm(self.normal)
        
        # Create rotation matrix for in-plane rotation
        rinplane = matrix_from_axis_angle(self.normal, radinplane)
        
        # Compose with existing rotation
        rnew = rinplane.dot(Viewer.R)
        
        # Extract Euler angles for display
        radnew = euler_from_matrix(rnew)  # In radians
        Viewer.eulerxyz = np.degrees(radnew).tolist()
        Viewer.R = rnew
        
        # Update all view normals
        for view in Viewer.instances:
            view.normal = rinplane.dot(view.normal)
        
        # Resample and redisplay
        Viewer.update_all()
```

**Angle Computation:**
```python
def angle(p0, p1, p2):
    """
    Compute signed angle p0-p1-p2 in 2D plane.
    
    Args:
        p0: Mouse position (current)
        p1: Center of view (pivot)
        p2: Mouse position (previous)
    
    Returns:
        Signed angle in radians (-π to π)
    """
    v0 = np.array(p0) - np.array(p1)  # Vector to current pos
    v1 = np.array(p2) - np.array(p1)  # Vector to last pos
    
    # atan2(cross, dot) gives signed angle
    return math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
```

---

### Three-Panel Orthogonal View System

**Layout:** YZ (sagittal), XZ (coronal), XY (axial) views simultaneously

```
┌──────────────────────────────────────────────┐
│  YZ View (Sagittal)  │  XZ View (Coronal)    │
│  Slice through       │  Slice through        │
│  fixed X position    │  fixed Y position     │
├──────────────────────┴───────────────────────┤
│  XY View (Axial)                             │
│  Slice through fixed Z position              │
└──────────────────────────────────────────────┘
```

**Key Feature:** All three views rotate together when dragging on any panel

**View Initialization:**
```python
# View orientations (from viewer.py)
# Format: [x_orient, y_orient, z_orient, rot_dir, sign]
o_lps = [
    [1,  1,  1, -1, 1],  # YZ view (sagittal)
    [1,  1,  1,  1, 1],  # XZ view (coronal)
    [1, -1,  1,  1, 1]   # XY view (axial)
]

# Create three SliceBox instances
Viewer.instances = [
    SliceBox(grid, axis=0, orient=o_lps[0], name='slice0'),  # YZ
    SliceBox(grid, axis=1, orient=o_lps[1], name='slice1'),  # XZ
    SliceBox(grid, axis=2, orient=o_lps[2], name='slice2')   # XY
]
```

---

### Volume Resampling with Rotation

**After each rotation**, the volume must be resampled to show the rotated view:

```python
def update_image(grid, image, i, size, axis, interpolator, 
                 levels, zvalue, compmode, lut, resample, item, defaultval, layer):
    """
    Resample rotated volume for display in one view.
    
    Steps:
    1. Define slice plane perpendicular to view axis
    2. Apply rotation transform
    3. Trilinear interpolation at slice positions
    4. Display as 2D image
    """
    # Set background value for out-of-bounds
    resample.SetDefaultPixelValue(defaultval)
    item.setZValue(zvalue)
    
    # Create coordinate slice (defines sampling plane)
    coord_slice = make_slice(image, grid, i, axis)
    
    # Set as reference for resampling
    resample.SetReferenceImage(coord_slice)
    
    # Execute resampling with rotation
    b = np.squeeze(sitk.GetArrayFromImage(
        resample.Execute(image)
    )).astype(np.float32)
    
    # Handle window/level and NaN for overlays
    if layer > 0 and layer < 9:
        b[b < levels[0]] = np.nan
        b[b > levels[1]] = np.nan
    
    # Display
    item.setImage(b, lut=lut, levels=levels, compositionMode=compmode)
    
    return coord_slice
```

---

### Drag Modes

The viewer supports **three drag modes** (toggle with keyboard):

**Mode 0: Rotation (Default)**
- Drag mouse → rotate volume around view normal
- All three views update simultaneously
- Rotation matrix accumulated over multiple drags

**Mode 1: Paint/Label**
- Drag to paint segmentation mask
- Used for manual corrections
- Requires label layer active (F5 to create)

**Mode 2: Window/Level Adjustment**
- Horizontal drag → adjust window (contrast width)
- Vertical drag → adjust level (brightness center)
- Real-time HU value adjustment

**Toggle:** Press `Space` to cycle through modes

---

### Keyboard Controls

**From viewer.py implementation:**

```
Navigation:
  ↑/↓           : Move slice up/down in current view
  ←/→           : Navigate through time/volume dimension
  Mouse Wheel   : Scroll through slices
  Space         : Toggle drag mode (rotate/paint/window-level)

Rotation:
  Mouse Drag    : Free rotation (when in rotation mode)
  S             : Save current rotation matrix to file
  F2/F3/F4      : Flip X/Y/Z axes

View:
  F7            : Toggle click/drag modes
  F8            : Set drag mode to window/level adjustment
  O             : Play through slices (cine mode)
  A             : Animate through volumes

Export:
  P             : Save screenshot (PNG)
  Q             : Save animated GIF sequence

Other:
  F1            : Show help
  F12           : Exit viewer
  R             : Reload settings from config
```

---

### Integration into Segment Platform

**Navbar Button:** Add "General Viewer" option

```
Top Navbar:
┌─────────────────────────────────────────────────────────────────┐
│ [Segment Logo] [Load CTA] [Load MEDIS] [🔬 Vessel Tool] [🔄 General Viewer] │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow:**
1. User clicks **[🔄 General Viewer]** button
2. Opens separate viewer window/modal
3. Loads current CTA volume
4. Enables free rotation with mouse drag
5. No vessel-specific constraints
6. Can return to vessel tool anytime

**Use Cases:**
- **Quality check:** Inspect CTA volume for artifacts, motion, noise
- **Orientation:** Get familiar with anatomy before vessel tracing
- **Teaching:** Demonstrate 3D cardiac anatomy
- **General viewing:** Any NIfTI volume (not just CTA)

---

### Technical Implementation Options

**Option A: Separate Window (Desktop-like)**
```typescript
// Frontend: Open general viewer in new window
function openGeneralViewer(volume: NVImage) {
  const viewerWindow = window.open(
    '/viewer',
    'GeneralViewer',
    'width=1200,height=800'
  );
  
  viewerWindow.postMessage({
    type: 'loadVolume',
    volume: volume.url
  }, '*');
}
```

**Option B: Modal Overlay**
```typescript
// Frontend: Show viewer as full-screen modal
function showGeneralViewerModal(volume: NVImage) {
  const modal = document.getElementById('general-viewer-modal');
  modal.style.display = 'block';
  
  // Initialize PyQtGraph-like viewer in canvas
  initializeRotatableViewer(modal, volume);
}
```

**Option C: Side Panel Toggle**
```typescript
// Frontend: Replace right panel with viewer
function toggleGeneralViewerPanel() {
  const vesselPanel = document.getElementById('vessel-controls');
  const viewerPanel = document.getElementById('general-viewer');
  
  vesselPanel.style.display = 'none';
  viewerPanel.style.display = 'block';
  
  // Enable rotation mode
  nv.setRotationMode(true);
}
```

**Recommended:** Option A (separate window) for desktop-like experience

---

### Rotation State Management

**Save/Load Rotation:**
```python
# Save rotation state to file (viewer.py: Key 'S')
def save_rotation_state():
    """
    Save current rotation for reproducibility.
    Format: [center_x, center_y, center_z, R_11, R_12, ..., R_33, 
             euler_x, euler_y, euler_z]
    """
    state = np.array([])
    state = np.append(state, Viewer.C)           # Center point
    state = np.append(state, Viewer.R.flatten()) # Rotation matrix
    state = np.append(state, Viewer.eulerxyz)    # Euler angles (degrees)
    
    np.savetxt('rotation_state.txt', state, fmt='%.6f')
    
    return {
        'center': Viewer.C,
        'rotation_matrix': Viewer.R.tolist(),
        'euler_angles_deg': Viewer.eulerxyz
    }
```

**Apply Saved Rotation:**
```python
def load_rotation_state(state_file):
    """Load and apply saved rotation state."""
    state = np.loadtxt(state_file)
    
    Viewer.C = state[:3]                    # Center
    Viewer.R = state[3:12].reshape((3, 3))  # Rotation matrix
    Viewer.eulerxyz = state[12:15].tolist() # Euler angles
    
    # Update all views
    Viewer.update_all()
```

---

### Performance Considerations

**Real-time Requirements:**
- **Target:** 30-60 FPS during rotation
- **Challenge:** Trilinear interpolation for entire volume per frame

**Optimization Strategies:**
1. **Reduce resolution:** Downsample during rotation, full-res on release
2. **GPU acceleration:** WebGL shaders for interpolation
3. **Level of Detail (LOD):** Coarser sampling when rotating fast
4. **Caching:** Pre-compute common rotation angles

**Memory:**
- Original volume: ~200 MB (typical CTA)
- Resampled slices: 3 × (512×512×2 bytes) ≈ 1.5 MB
- Reasonable for modern browsers

---

### Configuration File Format

**From viewer.py config system:**

```json
{
  "filename": ["patient_001.nii.gz"],
  "size": 400,
  "vol": true,
  "click_mode": 0,
  "drag_mode": 0,
  "orientation": [1, 1, 1],
  "padding": [0, 0, 10],
  "thickness": 1.0,
  "perc": [1.0, 99.0],
  "layer0": {
    "colormap": "Greys",
    "interpolator": "linear",
    "zvalue": 0,
    "visible": true,
    "compmode": "SourceOver",
    "level": [0, 100],
    "defval": -2048
  }
}
```

**For Web Implementation:** Convert to JSON config for frontend

---

### Future Enhancements

1. **VR Mode:** Gyroscope-based rotation on mobile/tablet
2. **Multi-touch:** Pinch-to-zoom, two-finger rotation
3. **Preset Views:** Quick buttons for standard anatomical views
4. **Rotation Recording:** Save rotation sequence as animation
5. **Synchronized Multi-Volume:** Rotate multiple volumes together (e.g., CTA + perfusion)

---

## 🎯 Implementation Summary

**General Viewer Features:**
- ✅ Free rotation with mouse drag (Rodrigues' formula)
- ✅ Three-panel orthogonal views (YZ, XZ, XY)
- ✅ Real-time volume resampling
- ✅ Window/level adjustment
- ✅ Save/load rotation state
- ✅ Keyboard shortcuts
- ✅ Export screenshots and animations

**Integration:**
- Add **[🔄 General Viewer]** button to top navbar
- Separate from vessel-specific tools
- Can be opened alongside or instead of vessel analysis
- Uses existing viewer.py algorithms

**Next Steps:**
1. Port viewer.py rotation algorithms to TypeScript
2. Integrate with NiiVue volume rendering
3. Add UI button and modal/window system
4. Test performance with typical CTA volumes

---

**Next Steps:** 
1. Review algorithms for completeness
2. Identify any mathematical issues or edge cases
3. Begin implementation with Algorithm 1 (centerline extraction)
4. Validate each step with real MEDIS TXT data

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
