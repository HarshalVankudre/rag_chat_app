# ---------- STAGE 1: builder (cache wheels) ----------
FROM python:3.10-slim AS builder

# Speed & reliability tweaks
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps for building wheels - some packages may need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

 RUN python -m venv /app/.venv

# Copy only requirements first to maximize layer caching
COPY requirements.txt requirements.txt

# Build wheels for all dependencies (faster, reproducible installs later)
# Note: Your requirements.txt already includes the extra index for CPU-only torch.
RUN python -m pip install --upgrade pip wheel \
 && mkdir -p /wheels \
 && pip wheel --wheel-dir /wheels -r requirements.txt


# ---------- STAGE 2: runtime ----------
FROM python:3.10-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# (Optional) add runtime OS deps your app might need; keep minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# CPU-only optimization settings for RapidOCR
ENV CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    RAPID_OCR_FORCE_CPU=1 \
    RAPID_OCR_REQUIRE_CUDA=0 \
    DOCLING_RENDER_DPI=150 \
    RAPID_OCR_MODEL_FLAVOR=mobile

# Create virtual environment


# Activate virtual environment and upgrade pip
ENV PATH="/app/.venv/bin:$PATH"
RUN python -m pip install --upgrade pip

# Copy wheels from builder and install without hitting the network
COPY --from=builder /wheels /wheels
COPY requirements.txt requirements.txt

# Install runtime dependencies into virtual environment
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# Verify RapidOCR packages are installed
RUN echo "=== Verifying RapidOCR packages ===" && \
    python -c "import rapidocr; print(f'✓ rapidocr: {rapidocr.__version__}')" && \
    python -c "import rapidocr_onnxruntime; print(f'✓ rapidocr-onnxruntime installed')" && \
    python -c "import onnxruntime; print(f'✓ onnxruntime: {onnxruntime.__version__}')" && \
    python -c "import onnxruntime; providers = onnxruntime.get_available_providers(); print(f'✓ Available providers: {providers}')" && \
    echo "=== All RapidOCR packages verified ==="

# Copy the rest of your application
COPY . .

# Streamlit defaults to 8501
EXPOSE 8501

# Start the app as root for now (non-root can be added later if needed)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    