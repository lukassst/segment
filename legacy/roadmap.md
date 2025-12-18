# 🛠️ Technical Roadmap: Segment 3D + Meta SAM with Niivue

**Project:** Volumetric Cardiovascular Segmentation Platform  
**Target Institution:** Charité – Universitätsmedizin Berlin (Prof. Marc Dewey)  
**Data Sources:** DISCHARGE Trial (25m images) | SCOT-HEART (10m images) | Prostate Trial (4m images)  
**Core Technology:** Niivue (WebGL2) | FastAPI | Mask SAM 3D

---

## Architecture Overview

### Frontend: Niivue as the "Radiologist's Canvas"
**Niivue** is a high-performance WebGL2 engine that enables seamless 3D rendering and interaction directly in the browser—critical for CCTA where sub-millimeter precision is required.

**Zero-Latency Interaction:**
- `onLocationChange` and `onMouseUp` events capture 3D voxel coordinates from user clicks
- Coordinates sent to backend as prompts for real-time segmentation
- **Target latency:** <2 seconds from click to overlay

**Dynamic Overlays:**
- Backend returns segmentation masks as NIfTI or compressed MZ3 meshes
- Niivue renders overlays via `nv.addVolume()` or `nv.addMesh()` with adjustable opacity
- Toggle controls for plaque characterization layers (calcified, non-calcified, low-attenuation)

**Multi-Planar Reconstruction (MPR):**
- Simultaneous axial/coronal/sagittal views synchronized with 3D render
- View coronary tree in 3D while slicing through specific stenoses
- Sub-millimeter precision for plaque assessment

### Backend: Mask SAM 3D "Intelligence Layer"
**Dual-Stage Pipeline:**
1. **nnU-Net Prior ("Artery Prior"):** 
   - Initial pass generates arterial skeleton
   - Provides anatomical context (identifies which vessel is which)
   - Reduces search space for SAM adapter
   
2. **SAM-3D Adapter ("Plaque-Aware Refinement"):** 
   - Takes nnU-Net mask as semantic prompt
   - Refines segmentation for high-risk plaques that standard models miss due to low contrast
   - Outputs vessel lumen, plaque composition, remodeling index, stenosis severity

**FastAPI Strategy:**
- **Asynchronous handling** of large CCTA volumes (25M+ images)
- **Pre-computed feature embeddings** from SAM encoder cached (Redis/disk)
- Enables near-instant "point-to-segmentation" updates in Niivue
- **Endpoints:** `POST /segment/point`, `POST /segment/box`, `GET /volume/{id}`
- **Output formats:** NIfTI volumes or MZ3 meshes for Niivue overlay rendering

---

## Phase 1: Minimal Viable Research (MVR) Tool
**Timeline:** 8-12 weeks

### 1.1 Backend Setup
- [ ] Containerize Mask SAM 3D model (Docker + PyTorch)
- [ ] Implement FastAPI endpoints:
  - `POST /segment/point` - Point-based prompting
  - `POST /segment/box` - Bounding box prompting
  - `GET /volume/{id}` - DICOM to NIfTI conversion
- [ ] Set up embedding cache (Redis/disk-based)
- [ ] Test with single DISCHARGE volume

### 1.2 Frontend Development
- [ ] Initialize Vite + TypeScript project
- [ ] Install `@niivue/niivue` package
- [ ] Implement core features:
  - Volume loading (NIfTI format)
  - Click event capture → 3D coordinate extraction
  - API integration for segmentation requests
  - Overlay rendering with toggle controls
- [ ] Add MPR view synchronization

### 1.3 Integration & Testing
- [ ] Connect frontend ↔ backend via REST API
- [ ] Test interactive loop:
  1. Load DISCHARGE CCTA volume
  2. User clicks coronary artery point
  3. Backend returns 3D segmentation mask
  4. Niivue overlays mask in real-time
- [ ] Optimize latency (target: <2s per segmentation)

---

## Phase 2: Clinical Deployment at Charité
**Timeline:** 12-16 weeks

### 2.1 Infrastructure
- [ ] Deploy on Charité GPU cluster
- [ ] Implement authentication (LDAP/SSO integration)
- [ ] Set up secure DICOM ingestion pipeline
- [ ] Configure HTTPS + VPN access

### 2.2 Active Learning "Refine-as-You-Go" Workflow
- [ ] Deploy secure internal instance at Charité GPU cluster
- [ ] Enable collaborative annotation by research team ("many people who can do things")
- [ ] Add annotation refinement tools in Niivue:
  - Brush tool for mask correction
  - Eraser for false positives
  - Drawing tools for manual refinement
  - Save corrected masks to training database
- [ ] Implement intelligent feedback loop:
  - **AI suggests** segmentations
  - **Human corrects** via intuitive Niivue interface
  - **Model updates** on corrected data (Active Learning)
  - Performance improves iteratively
- [ ] Version control for model iterations with A/B testing

### 2.3 Multi-Trial Data Integration
- [ ] **DISCHARGE (25M images):** Batch process primary trial cohort
- [ ] **SCOT-HEART (10M images):** External validation and collaborative annotation
- [ ] **Prostate (4M images):** Cross-domain validation for foundation model generalization
- [ ] Generate initial segmentations for all datasets
- [ ] Quality control dashboard (Dice scores, manual review flags, cross-trial consistency)

---

## Phase 3: Research-Grade Platform
**Timeline:** 16-24 weeks

### 3.1 Advanced Features
- [ ] **Plaque Characterization:**
  - Low-attenuation plaque detection
  - Positive remodeling quantification
  - Stenosis severity scoring
- [ ] **Longitudinal Analysis:**
  - Multi-timepoint volume registration
  - Plaque progression tracking
  - Digital twin visualization

### 3.2 Validation & Benchmarking
- [ ] Compare against core-lab manual segmentations
- [ ] Calculate performance metrics:
  - Dice coefficient
  - Hausdorff distance
  - Clinical agreement (stenosis grading)
- [ ] Multi-vendor validation (Siemens, GE, Canon)

### 3.3 Publication-Ready Outputs
- [ ] Export quantitative reports (plaque volume, composition)
- [ ] Generate figures for manuscripts
- [ ] Anonymization pipeline for data sharing

---

## Technical Stack Details

### Frontend
```
- Framework: Vite + TypeScript
- 3D Engine: @niivue/niivue (WebGL2)
- UI: React + TailwindCSS + shadcn/ui
- State: Zustand
- Icons: Lucide React
```

### Backend
```
- API: FastAPI (Python 3.10+)
- AI Engine: Mask SAM 3D (PyTorch 2.0+)
- Image Processing: SimpleITK, nibabel
- Cache: Redis
- Container: Docker + NVIDIA Container Toolkit
```

### Infrastructure
```
- Deployment: Charité GPU Cluster (NVIDIA A100/V100)
- Storage: PACS integration + NAS for processed volumes
- Security: VPN + HTTPS + LDAP authentication
```

---

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|-----------|
| Large volume latency | Pre-compute embeddings, progressive loading |
| Browser memory limits | Implement volume streaming, downsample for preview |
| Model accuracy on edge cases | Active learning loop, expert review queue |
| Multi-vendor variability | Normalization pipeline, vendor-specific fine-tuning |

### Deployment Risks
| Risk | Mitigation |
|------|-----------|
| Charité IT approval | Early engagement, security audit, compliance documentation |
| User adoption | Training sessions, intuitive UI, performance demos |
| Data privacy | On-premise deployment, no external data transfer |

---

## Success Metrics

### Technical KPIs
- **Segmentation Speed:** <2 seconds per vessel segment
- **Accuracy:** Dice score >0.85 vs. expert annotations
- **Uptime:** >99% availability during research hours

### Research KPIs
- **Annotation Throughput:** 10x faster than manual segmentation
- **SCOT-HEART Coverage:** >80% of dataset processed in 6 months
- **Publications:** 2+ peer-reviewed papers within 18 months

---

## Next Actions

1. **Immediate (Week 1-2):**
   - Set up development environment
   - Clone Mask SAM 3D repository
   - Initialize Niivue prototype project

2. **Short-term (Month 1):**
   - Complete MVR tool
   - Demo with single DISCHARGE case
   - Present to Prof. Dewey for feedback

3. **Medium-term (Month 2-3):**
   - Deploy to Charité infrastructure
   - Onboard research team
   - Begin SCOT-HEART processing

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-17  
**Owner:** Flow Project Team
