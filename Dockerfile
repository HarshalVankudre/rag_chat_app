# ---- STAGE 1: The Builder ----
# Use a full Python image to build our environment
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only the files needed to install dependencies
# This leverages Docker's cache.
COPY pyproject.toml poetry.lock ./

# ---- THIS IS THE FIX ----
# Tell poetry to create the venv inside the project folder (at /app/.venv)
# This ensures the COPY command in the next stage will find it.
RUN poetry config virtualenvs.in-project true

# Now, this command will correctly create the venv at /app/.venv
RUN poetry install --without dev --no-root

# ---- STAGE 2: The Final Image ----
# Use a slim image for the final, lightweight container (CPU-only, no GPU dependencies)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
# This command will now succeed!
COPY --from=builder /app/.venv /app/.venv

# Set the PATH to use the venv's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Optimize for CPU-only execution (4 cores, 4GB RAM)
# Disable GPU fallback attempts
ENV CUDA_VISIBLE_DEVICES=""
# Optimize thread settings for 4-core CPU
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
# Optimize OCR settings for CPU
ENV DOCLING_RENDER_DPI=150
ENV RAPID_OCR_MODEL_FLAVOR=mobile
ENV RAPID_OCR_ANGLE=0
ENV RAPID_OCR_LANGS=en
ENV DOCLING_PAR_WORKERS=2
ENV EMBED_BATCH=64

# Copy the rest of your application code
COPY . .

# Streamlit runs on port 8501 by default
EXPOSE 8501

# The command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

