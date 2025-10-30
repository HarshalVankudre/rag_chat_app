#!/usr/bin/env python3
"""Script to check RapidOCR model locations and provide download instructions."""

import os
import sys
from pathlib import Path

def check_rapidocr_models():
    """Check where RapidOCR models are stored and how to get them."""
    
    print("=" * 70)
    print("RapidOCR Docling Model Information")
    print("=" * 70)
    
    # Check if docling is installed
    try:
        import docling
        print(f"✓ Docling is installed: {docling.__version__}")
        print(f"  Location: {docling.__file__}")
    except ImportError:
        print("✗ Docling is not installed")
        print("  Install with: pip install docling>=0.5.0")
        return
    
    # Check if rapidocr is installed
    try:
        import rapidocr
        print(f"✓ RapidOCR is installed")
        print(f"  Location: {rapidocr.__file__}")
        
        # Try to find RapidOCR's default model directory
        rapidocr_path = Path(rapidocr.__file__).parent
        print(f"  RapidOCR package path: {rapidocr_path}")
        
        # Check common model storage locations
        home = Path.home()
        possible_locations = [
            home / ".rapidocr",
            home / ".cache" / "rapidocr",
            home / ".cache" / "onnxruntime",
            Path("/tmp/rapidocr"),
            Path.cwd() / "ocr-models",
        ]
        
        print("\nChecking for model storage locations:")
        for loc in possible_locations:
            if loc.exists():
                print(f"  ✓ Found: {loc}")
                if loc.is_dir():
                    models = list(loc.glob("*.onnx"))
                    if models:
                        print(f"    Models found: {len(models)}")
                        for model in models[:5]:  # Show first 5
                            print(f"      - {model.name}")
                    else:
                        print(f"    No .onnx files found")
            else:
                print(f"  ✗ Not found: {loc}")
        
    except ImportError:
        print("✗ RapidOCR is not directly importable (it's used via Docling)")
    
    # Check current configuration
    print("\n" + "=" * 70)
    print("Current Configuration (from rag/ingest.py)")
    print("=" * 70)
    
    use_default = os.environ.get("RAPID_OCR_USE_DEFAULT_MODELS", "1")
    flavor = os.environ.get("RAPID_OCR_MODEL_FLAVOR", "mobile")
    
    print(f"RAPID_OCR_USE_DEFAULT_MODELS: {use_default}")
    print(f"RAPID_OCR_MODEL_FLAVOR: {flavor}")
    
    if use_default.lower() in ("1", "true", "yes"):
        print("\n✓ Using DEFAULT models (downloaded automatically by RapidOCR)")
        print("  Models are downloaded on first use to:")
        print("    - Linux: ~/.rapidocr/ or ~/.cache/rapidocr/")
        print("    - Windows: %USERPROFILE%\\.rapidocr\\")
        print("    - macOS: ~/.rapidocr/")
        print("\n  Models will be downloaded automatically when first used.")
    else:
        print("\n✓ Using CUSTOM models from 'ocr-models/' directory")
        model_dir = Path("ocr-models")
        if model_dir.exists():
            models = list(model_dir.glob("*.onnx"))
            print(f"  Found {len(models)} model(s):")
            for model in models:
                print(f"    - {model.name}")
        else:
            print(f"  ✗ Directory 'ocr-models/' does not exist")
            print("\n  Expected model files:")
            print(f"    - ocr-models/ch_PP-OCRv4_det_{flavor}_infer.onnx")
            print(f"    - ocr-models/ch_PP-OCRv4_rec_{flavor}_infer.onnx")
            print(f"    - ocr-models/ch_ppocr_mobile_v2.0_cls_infer.onnx")
    
    print("\n" + "=" * 70)
    print("Models Used by RapidOCR Docling")
    print("=" * 70)
    print("1. Detection model: ch_PP-OCRv4_det_{mobile|server}_infer.onnx")
    print("2. Recognition model: ch_PP-OCRv4_rec_{mobile|server}_infer.onnx")
    print("3. Classification model: ch_ppocr_mobile_v2.0_cls_infer.onnx")
    print(f"\nCurrent flavor: {flavor}")
    
    print("\n" + "=" * 70)
    print("How to Download Models")
    print("=" * 70)
    print("""
Option 1: Use Default Models (Recommended - Automatic)
  - Set RAPID_OCR_USE_DEFAULT_MODELS=1 (default)
  - Models download automatically on first use
  - Stored in ~/.rapidocr/ or ~/.cache/rapidocr/

Option 2: Download Models Manually
  - Set RAPID_OCR_USE_DEFAULT_MODELS=0
  - Create ocr-models/ directory
  - Download models from RapidOCR releases:
    https://github.com/RapidAI/RapidOCR/releases
    
  Or use RapidOCR CLI:
    pip install rapidocr-onnxruntime
    python -c "from rapidocr import RapidOCR; ocr = RapidOCR()"
    
  Then copy models from default location to ocr-models/

Option 3: Download from HuggingFace (if available)
  - Some RapidOCR models may be available on HuggingFace
  - Search for "rapidocr" or "PP-OCRv4" models
""")

if __name__ == "__main__":
    check_rapidocr_models()
