# ---- STAGE 1: The Builder ----
# Use a full Python image to build our environment
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install poetry and export plugin (needed for poetry export command)
RUN pip install --no-cache-dir poetry poetry-plugin-export

# Copy only the files needed to install dependencies
# This leverages Docker's cache - if dependencies don't change, this layer is cached
COPY pyproject.toml poetry.lock ./

# Configure poetry for CPU-only builds
RUN poetry config virtualenvs.in-project true

# Create venv first (needed for installing torch into it)
RUN python -m venv /app/.venv

# Install dependencies directly with Poetry (CPU-only torch pinned in pyproject)
# Using --sync keeps the virtualenv lean and reproducible
RUN poetry install --only main --no-root --sync && \
    # Verify CPU-only PyTorch is installed (no GPU acceleration available)
    /app/.venv/bin/python -c "import torch; assert not torch.cuda.is_available(), 'CUDA should not be available'; print(f'✓ PyTorch {torch.__version__} (CPU-only)')" && \
    # Ensure no NVIDIA packages slipped in
    if /app/.venv/bin/pip list | grep -iq '^nvidia'; then \
        echo 'ERROR: NVIDIA packages found!' >&2 && exit 1; \
    else \
        echo '✓ No NVIDIA packages'; \
    fi && \
    # Clean up Poetry caches
    rm -rf /root/.cache/pypoetry /root/.cache/pip

# ---- STAGE 2: The Final Image ----
# Use a slim image for the final, lightweight container (CPU-only, no GPU dependencies)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the virtual environment from the builder stage (not build tools)
COPY --from=builder /app/.venv /app/.venv

# Set the PATH to use the venv's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Verify no NVIDIA packages are present
RUN pip list | grep -i nvidia && echo "ERROR: NVIDIA packages found!" && exit 1 || echo "✓ No NVIDIA packages"

# Clean up any unnecessary files
RUN find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete 2>/dev/null || true

# ========================================
# CPU-ONLY OPTIMIZATION SETTINGS
# ========================================
# Disable GPU fallback attempts
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_CUDA_ARCH_LIST=""

# Optimize thread settings for 4-core CPU
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Force CPU-only execution for PyTorch
ENV PYTORCH_ENABLE_MPS_FALLBACK=0

# Optimize OCR settings for CPU
ENV DOCLING_RENDER_DPI=150
ENV RAPID_OCR_MODEL_FLAVOR=mobile
ENV RAPID_OCR_ANGLE=0
ENV RAPID_OCR_LANGS=en
ENV DOCLING_PAR_WORKERS=2
ENV EMBED_BATCH=64

# Disable unnecessary GPU-related warnings
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Copy application code last (changes most frequently, so this layer invalidates last)
# Use .dockerignore to exclude unnecessary files (logs, cache, etc.)
COPY . .

# Streamlit runs on port 8501 by default
EXPOSE 8501

# The command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

