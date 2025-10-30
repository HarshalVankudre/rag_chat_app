# RapidOCR Docling Models Information

## Models Being Used

Based on the code in `rag/ingest.py`, RapidOCR Docling uses the following ONNX models:

### 1. Detection Model
- **Name**: `ch_PP-OCRv4_det_{flavor}_infer.onnx`
- **Flavor options**: `mobile` (default) or `server`
- **Purpose**: Text detection (finding text regions in images)

### 2. Recognition Model
- **Name**: `ch_PP-OCRv4_rec_{flavor}_infer.onnx`
- **Flavor options**: `mobile` (default) or `server`
- **Purpose**: Text recognition (reading text from detected regions)

### 3. Classification Model
- **Name**: `ch_ppocr_mobile_v2.0_cls_infer.onnx`
- **Purpose**: Text direction classification (determining if text is rotated)

### Model Flavor Configuration
- Controlled by environment variable: `RAPID_OCR_MODEL_FLAVOR`
- Default: `mobile` (lighter, optimized for CPU)
- Alternative: `server` (more accurate but heavier)

## Current Storage Location

### Default Mode (RAPID_OCR_USE_DEFAULT_MODELS=1)

When using default models (which is the current configuration):
- Models are **automatically downloaded** by RapidOCR on first use
- Storage locations vary by platform:
  - **Linux**: `~/.rapidocr/` or `~/.cache/rapidocr/`
  - **Windows**: `%USERPROFILE%\.rapidocr\`
  - **macOS**: `~/.rapidocr/`
- Models are cached locally after first download
- No manual download required

### Custom Mode (RAPID_OCR_USE_DEFAULT_MODELS=0)

When using custom models:
- Expected location: `ocr-models/` directory (relative to workspace root)
- Directory is in `.gitignore` (not tracked in git)
- Currently this directory does not exist in the workspace
- You need to manually place model files here

## How to Get the Models

### Option 1: Automatic Download (Recommended - Current Setup)

**This is the default configuration.** Models download automatically when first used.

1. **No action needed** - models download automatically
2. Set environment variable (optional, already default):
   ```bash
   export RAPID_OCR_USE_DEFAULT_MODELS=1
   export RAPID_OCR_MODEL_FLAVOR=mobile  # or 'server'
   ```
3. When you first run document processing, RapidOCR will:
   - Download models automatically
   - Cache them in the default location
   - Use them for subsequent runs

### Option 2: Manual Download to Custom Directory

If you want to use custom models in the `ocr-models/` directory:

1. **Create the directory**:
   ```bash
   mkdir -p ocr-models
   ```

2. **Download models** from RapidOCR releases:
   - Visit: https://github.com/RapidAI/RapidOCR/releases
   - Download the ONNX model files
   - Or use the RapidOCR CLI to trigger download:
     ```bash
     pip install rapidocr-onnxruntime
     python -c "from rapidocr import RapidOCR; ocr = RapidOCR()"
     ```
   - Then copy models from `~/.rapidocr/` to `ocr-models/`

3. **Place models** in `ocr-models/` directory:
   ```
   ocr-models/
   ├── ch_PP-OCRv4_det_mobile_infer.onnx    # or _server_
   ├── ch_PP-OCRv4_rec_mobile_infer.onnx    # or _server_
   └── ch_ppocr_mobile_v2.0_cls_infer.onnx
   ```

4. **Configure environment**:
   ```bash
   export RAPID_OCR_USE_DEFAULT_MODELS=0
   export RAPID_OCR_MODEL_FLAVOR=mobile  # or 'server'
   ```

### Option 3: Download from HuggingFace

Some RapidOCR models may be available on HuggingFace:
- Search for: "rapidocr" or "PP-OCRv4"
- Download ONNX files manually
- Place in `ocr-models/` directory

## Current Configuration

From `rag/ingest.py`:
- **Default model directory**: `ocr-models/` (line 81)
- **Default behavior**: Uses default models (`RAPID_OCR_USE_DEFAULT_MODELS=1`)
- **Default flavor**: `mobile` (configured via `RAPID_OCR_MODEL_FLAVOR=mobile`)

From `Dockerfile`:
- `RAPID_OCR_MODEL_FLAVOR=mobile` (line 46)
- `RAPID_OCR_ANGLE=0` (line 47) - angle classifier disabled
- `RAPID_OCR_LANGS=en` (line 48) - English only

## Verification

To check where models are currently stored:

```bash
# Check default RapidOCR cache location
ls -la ~/.rapidocr/ 2>/dev/null || echo "Directory doesn't exist yet"
ls -la ~/.cache/rapidocr/ 2>/dev/null || echo "Directory doesn't exist yet"

# Check custom directory
ls -la ocr-models/ 2>/dev/null || echo "Directory doesn't exist"
```

## Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `RAPID_OCR_USE_DEFAULT_MODELS` | `1` | Use default models (auto-download) or custom (`0`) |
| `RAPID_OCR_MODEL_FLAVOR` | `mobile` | Model flavor: `mobile` or `server` |
| `RAPID_OCR_LANGS` | `en` | Languages (comma-separated) |
| `RAPID_OCR_ANGLE` | `0` | Enable angle classifier (`1`) or disable (`0`) |
| `DOCLING_RENDER_DPI` | `150` | PDF render DPI for OCR |

## Notes

- The `ocr-models/` directory is in `.gitignore`, so it won't be tracked in git
- Models are typically 10-50MB each (mobile) or 50-200MB each (server)
- Mobile models are optimized for CPU and lower memory usage
- Server models provide better accuracy but require more resources
