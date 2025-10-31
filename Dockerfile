# ---------- STAGE 1: builder (cache wheels) ----------
FROM python:3.10-slim AS builder

# Speed & reliability tweaks
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for building wheels - some packages may need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment in builder
RUN python -m venv /app/.venv

# Activate virtual environment for build
ENV PATH="/app/.venv/bin:$PATH"

# Copy only requirements first to maximize layer caching
COPY requirements.txt requirements.txt

# Upgrade pip and install wheel in the venv
RUN pip install --upgrade pip wheel setuptools

# Build wheels for all dependencies (faster, reproducible installs later)
# Note: Your requirements.txt already includes the extra index for CPU-only torch.
RUN mkdir -p /wheels && \
    pip wheel --wheel-dir /wheels -r requirements.txt


# ---------- STAGE 2: runtime ----------
FROM python:3.10-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU-only optimization settings for RapidOCR and ML libraries
ENV CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    RAPID_OCR_FORCE_CPU=1 \
    RAPID_OCR_REQUIRE_CUDA=0 \
    DOCLING_RENDER_DPI=150 \
    RAPID_OCR_MODEL_FLAVOR=mobile

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Activate virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy wheels from builder
COPY --from=builder /wheels /wheels
COPY requirements.txt requirements.txt

# Install all dependencies from wheels (no network needed, faster)
RUN pip install --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

# Verify critical packages are installed correctly
RUN echo "=== Verifying installed packages ===" && \
    python -c "import streamlit; print(f'✓ streamlit: {streamlit.__version__}')" && \
    python -c "import streamlit_authenticator; print(f'✓ streamlit-authenticator installed')" && \
    python -c "import openai; print(f'✓ openai: {openai.__version__}')" && \
    python -c "import pymongo; print(f'✓ pymongo: {pymongo.__version__}')" && \
    python -c "import pinecone; print(f'✓ pinecone installed')" && \
    python -c "import torch; print(f'✓ torch: {torch.__version__}')" && \
    python -c "import rapidocr; print(f'✓ rapidocr: {rapidocr.__version__}')" && \
    python -c "import rapidocr_onnxruntime; print(f'✓ rapidocr-onnxruntime installed')" && \
    python -c "import onnxruntime; print(f'✓ onnxruntime: {onnxruntime.__version__}')" && \
    python -c "import docling; print(f'✓ docling installed')" && \
    python -c "import onnxruntime; providers = onnxruntime.get_available_providers(); print(f'✓ ONNX providers: {providers}')" && \
    echo "=== All critical packages verified ==="

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs && \
    chmod -R 755 /app

# Healthcheck to ensure Streamlit is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit defaults to 8501
EXPOSE 8501

# Start the app with proper configuration
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true", \
     "--browser.gatherUsageStats=false"]
    