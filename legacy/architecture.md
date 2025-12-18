# Segment Platform Architecture: TypeScript + Niivue v6 + MedSAM3D

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

#### Frontend: Niivue v6+ (Latest)

**Key capabilities** (check latest docs):
- WebGL2 rendering for 3D medical imaging
- Interactive prompting (click-to-segment)
- Multi-planar reconstruction (MPR)
- Mesh overlay rendering
- TypeScript support (improved from v0.62)
- Better performance and API

**What we need to verify**:
- Current version number (v6.x.x?)
- TypeScript type definitions
- Point/box prompt API
- Mesh rendering capabilities
- 4D volume support

#### Backend: MedSAM3D

**Available**:
- Segment Anything Model adapted for 3D medical imaging
- Pre-trained on medical imaging datasets
- Point/box prompting support
- PyTorch implementation

**What we need**:
- FastAPI wrapper for HTTP endpoints
- Feature embedding cache (for speed)
- DICOM → NIfTI conversion pipeline
- Docker containerization

#### Deployment: Hospital Infrastructure

**Charité has**:
- GPU cluster (NVIDIA A100/V100)
- PACS integration
- VPN access
- Secure data storage

**What we need to build**:
- Authentication (LDAP/SSO)
- HTTPS + VPN configuration
- On-premise deployment (no external data transfer)
- GDPR compliance

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
1. Radiologist opens browser → logs in (Charité SSO)
2. Loads DISCHARGE CCTA scan from PACS
3. Clicks on LAD artery → AI segments entire vessel in 2 seconds
4. Clicks on plaque → AI characterizes composition (calcified, lipid, mixed)
5. Exports segmentation for CT-FFR analysis or research database

**Technical flow**:
1. Frontend: Niivue loads NIfTI volume, displays 3D rendering
2. User clicks → frontend captures 3D coordinates
3. Frontend sends coordinates to backend API
4. Backend: MedSAM3D generates segmentation mask
5. Backend returns mask as NIfTI or mesh
6. Frontend: Niivue overlays mask with adjustable opacity

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

**Advantages over vanilla JS** (from `frac`):
- Type safety (catch errors at compile time)
- Better IDE support (autocomplete, refactoring)
- Easier collaboration (explicit interfaces)
- Scales better for large codebase

**Migration from `frac`**:
- Convert .js → .ts
- Add type definitions for Niivue
- Define interfaces for API responses
- Use strict TypeScript config

### Why Niivue v6+?

**Check latest version** (need to verify):
- Improved TypeScript support
- Better performance
- Enhanced mesh rendering
- More flexible API

**What we need to research**:
- Current stable version
- Breaking changes from v0.62
- TypeScript type definitions
- New features we can leverage

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
