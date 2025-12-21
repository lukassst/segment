# MEDIS Vessel Viewer

NiiVue-based viewers for visualizing coronary artery CTA volumes with MEDIS contour overlays.

## Files

| File | Description |
|------|-------------|
| `vessel-viewer.html` | Full-featured viewer with sidebar controls |
| `vessel-viewer-simple.html` | Streamlined single-header version |

## Quick Start

1. **Start a local server** from the `segment/` directory:
   ```bash
   cd c:\Users\lukass\Documents\GitHub\segment
   python -m http.server 8080
   ```

2. **Open in browser:**
   - Simple: http://localhost:8080/code/vessel-viewer-simple.html
   - Full: http://localhost:8080/code/vessel-viewer.html

## Features

### View Modes

- **World Space**: Original CTA volume (512×512×512) with world-coordinate meshes
- **CPR Space**: Straightened vessel (120×120×N) with CPR-aligned meshes

### Controls

#### Simple Viewer
- **World/CPR buttons**: Toggle between coordinate spaces
- **Vessel dropdown**: LAD, RCA, LCX, D1
- **Lumen/Wall checkboxes**: Show/hide inner and outer meshes
- **Mesh α slider**: Mesh opacity (0-100%)
- **X-Ray slider**: Mesh transparency for seeing through surfaces
- **View dropdown**: 3D Render, Multiplanar, Axial, Coronal, Sagittal
- **Crosshair checkbox**: Toggle 3D crosshair

#### Full Viewer (Additional Features)
- **Volume opacity slider**: Control CTA volume transparency
- **Shader buttons**: Apply different mesh shaders (Flat, Phong, Outline, etc.)
- **Info panel**: Display volume dimensions, spacing, mesh vertex counts
- **Point cloud toggles**: Load JSON point clouds (experimental)

### Mouse Interaction

- **Left drag**: Rotate (3D mode) / Pan (2D mode)
- **Right drag**: Zoom
- **Scroll**: Zoom (3D) / Scroll slices (2D)
- **Middle click**: Reset view

## Data Structure

The viewers load data from `../data/` relative to the HTML files:

```
data/
├── 01-BER-0088.nii.gz                    # CTA volume (world space)
├── 01-BER-0088_LAD_inner.gii             # Lumen mesh (world)
├── 01-BER-0088_LAD_outer.gii             # Vessel wall mesh (world)
├── 01-BER-0088_LAD_inner_cpr.gii         # Lumen mesh (CPR space)
├── 01-BER-0088_LAD_outer_cpr.gii         # Vessel wall mesh (CPR space)
├── 01-BER-0088_LAD_cpr_cross.nii.gz      # CPR volume (cross-sectional)
├── 01-BER-0088_LAD_cpr_long.nii.gz       # CPR volume (longitudinal)
└── ... (same pattern for RCA, LCX, D1)
```

## Technical Details

- **NiiVue Version**: Latest from CDN (https://cdn.jsdelivr.net/npm/@niivue/niivue@latest/dist/niivue.umd.js)
- **No build required**: Pure HTML/CSS/JS, runs directly in browser
- **CORS**: Requires local server (not file://) for loading NIfTI/GIfTI files

### Color Scheme

- **Lumen (inner)**: Red `[255, 80, 80, 255]`
- **Vessel Wall (outer)**: Blue `[80, 80, 255, 200]`
- **Background**: Dark blue-gray `[0.05, 0.05, 0.1, 1]`

## Extending

### Adding New Patients

Edit the HTML file and add to the patient dropdown:

```html
<select id="patientSelect">
  <option value="01-BER-0088">01-BER-0088</option>
  <option value="NEW-PATIENT-ID">NEW-PATIENT-ID</option>
</select>
```

Ensure data files follow naming convention: `{patient}_{vessel}_{type}.gii`

### Custom Shaders

Available mesh shaders (full viewer only):
- Flat, Phong, Toon, Outline, Hemi, Matcap, etc.

Apply programmatically:
```javascript
nv.setMeshShader(mesh.id, 'Outline');
```

### Point Cloud Support

JSON point clouds use NiiVue's connectome format:
```json
{
  "nodes": {
    "x": [1.0, 2.0, ...],
    "y": [1.0, 2.0, ...],
    "z": [1.0, 2.0, ...]
  }
}
```

Load with:
```javascript
await nv.loadConnectome(data);
```

## Troubleshooting

**"Failed to fetch" errors:**
- Ensure server is running from `segment/` directory
- Check browser console for specific file paths
- Verify data files exist in `data/` directory

**Meshes not visible:**
- Toggle mesh checkboxes off/on
- Increase mesh opacity slider
- Try different shader (e.g., "Flat")
- Check mesh X-Ray is not at 100%

**CPR mode shows nothing:**
- Verify CPR files exist: `{patient}_{vessel}_cpr_cross.nii.gz`
- Check console for file loading errors
- Ensure vessel is selected (LAD, RCA, LCX, or D1)

**Slow performance:**
- Reduce mesh X-Ray value
- Switch to 2D view mode (Axial/Coronal/Sagittal)
- Disable one of the mesh overlays

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ Requires WebGL 2.0 support

## Related Documentation

- `medis-viewer.md` - Full project documentation
- `medis_to_cpr.py` - CPR volume generation script
- `transform_to_cpr.py` - Mesh/point cloud coordinate transformation
