# ---- STAGE 1: The Builder ----
# Use a full Python image to build our environment
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only the files needed to install dependencies
# This leverages Docker's cache.
COPY pyproject.toml poetry.lock ./

# Install dependencies
# --no-dev: Skips dev dependencies (like pytest, ruff)
# --no-root: Skips installing the project itself (it's an app, not a library)
RUN poetry install --no-dev --no-root

# ---- STAGE 2: The Final Image ----
# Use a slim image for the final, lightweight container
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
# This is the key to a clean, isolated environment
COPY --from=builder /app/.venv /app/.venv

# Set the PATH to use the venv's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of your application code
COPY . .

# Streamlit runs on port 8501 by default
EXPOSE 8501

# The command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
