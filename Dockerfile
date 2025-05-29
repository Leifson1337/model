# ARG for base images to allow customization (e.g., for GPU)
ARG BASE_IMAGE=python:3.9-bullseye
ARG FINAL_BASE_IMAGE=python:3.9-slim-bullseye

# --- Stage 1: Builder ---
# Use the build argument for the builder's base image
FROM ${BASE_IMAGE} AS builder

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory for venv creation
WORKDIR /opt

# Create a virtual environment
RUN python -m venv venv

# Activate the virtual environment for subsequent RUN commands in this stage
# Note: ENV PATH is for the final image, RUN commands in builder need to call venv python/pip directly or activate.
# For simplicity in RUN, we'll use the direct path to venv pip.
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file
COPY requirements.txt .

# Install dependencies into the virtual environment
# Upgrade pip first
RUN venv/bin/pip install --no-cache-dir --upgrade pip
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Image ---
# Use the build argument for the final image's base image
FROM ${FINAL_BASE_IMAGE}

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use the Python interpreter from the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user and group for security
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Set the working directory in the container
WORKDIR /app

# Copy the application's code into the container at /app
# Ensure .dockerignore is properly set up to exclude unnecessary files/dirs
# (like .git, .venv from host, __pycache__, local logs/, etc.)
COPY . .

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appgroup /app
# The /opt/venv will be owned by root, which is fine as it should be read-only for the app.

# Switch to the non-root user
USER appuser

# Healthcheck: Checks if the main CLI help command runs successfully.
# Adjust as needed for a more comprehensive health check if the app runs a server.
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
  CMD python main.py --help || exit 1

# Define the command to run on container start via entrypoint.sh.
ENTRYPOINT ["/app/entrypoint.sh"]
# Default command for entrypoint.sh (will be passed as "$@")
# This runs `python main.py --help` if no other command is provided to `docker run`.
CMD ["python", "main.py", "--help"]

# --- Notes for GPU Support ---
# To build a GPU-enabled image:
# 1. Ensure your `requirements.txt` includes GPU versions of libraries
#    (e.g., tensorflow[and-cuda] or torch[cuda]). You might use a separate
#    `requirements-gpu.txt` and conditionally copy/install it:
#    ARG REQS_FILE=requirements.txt
#    COPY ${REQS_FILE} .
#    RUN pip install -r ${REQS_FILE}
#    Then build with: --build-arg REQS_FILE=requirements-gpu.txt
#
# 2. Build the Docker image using --build-arg to specify CUDA-enabled base images:
#    Example:
#    docker build \
#      --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
#      --build-arg FINAL_BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 \
#      -t qlop-app-gpu .
#
#    Choose CUDA base images compatible with your GPU drivers and library versions.
#    The 'devel' image is larger, suitable for building; 'runtime' is smaller for the final stage.
#    Ensure the OS in the CUDA image (e.g., ubuntu22.04) is compatible with Python versions and other packages.
