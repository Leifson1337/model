# Docker Deployment

This document describes how to build and run the project using Docker and Docker Compose.

## Prerequisites

-   Docker installed on your system (Docker Desktop for Windows/Mac, Docker Engine for Linux).
-   Docker Compose (usually included with Docker Desktop, or installable as a plugin for Docker Engine).

## 1. Using Dockerfile Directly

These instructions are for building the Docker image manually and running containers using `docker` commands. For most use cases, especially development and testing, using Docker Compose (see Section 2) is recommended.

### Building the Docker Image

1.  Ensure you have the `Dockerfile` in the project root.
2.  Navigate to the project root directory.
3.  Run the build command:
    ```bash
    docker build -t qlop_app:latest .
    ```
    (Replace `qlop_app:latest` with your desired image name and tag.)

    **GPU Support Build (Example):**
    To build an image with GPU support (assuming your `requirements.txt` and application code are GPU-compatible), you can pass build arguments to specify CUDA-enabled base images:
    ```bash
    docker build \
      --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
      --build-arg FINAL_BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 \
      -t qlop_app:gpu-latest .
    ```
    Refer to the comments in the `Dockerfile` for more details on GPU support.

### Running Docker Containers (Manual)

-   **To run the CLI (e.g., view help):**
    The image uses an `entrypoint.sh` script. By default, it runs `python main.py --help`.
    ```bash
    docker run --rm qlop_app:latest
    ```
    To run another command:
    ```bash
    docker run --rm qlop_app:latest python main.py --version
    ```

-   **To run a specific CLI command with local file system access (e.g., load data):**
    You need to mount volumes for configuration, data, models, and logs, and pass your `.env` file for API keys.
    ```bash
    docker run --rm \
        --env-file .env \
        -v "$(pwd)/config:/app/config:ro" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        qlop_app:latest python main.py load-data --config config/dev.json
    ```
    *(Note: `$(pwd)` is for Linux/macOS. For Windows PowerShell, use `${PWD}`. For Windows CMD, use `%cd%`.)*
    The `:ro` flag makes the config mount read-only in the container, which is good practice.

-   **To run the Streamlit GUI:**
    ```bash
    docker run --rm -p 8501:8501 \
        --env-file .env \
        -v "$(pwd)/config:/app/config:ro" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        qlop_app:latest streamlit run gui.py --server.port 8501 --server.address 0.0.0.0
    ```
    Access the GUI at `http://localhost:8501`.

## 2. Using Docker Compose

Docker Compose simplifies the management of multi-container applications and development environments. We provide separate compose files for different purposes.

### General Docker Compose Commands

-   **Build images and start services:** `docker-compose -f <file.yml> up --build`
-   **Start services (after building/pulling):** `docker-compose -f <file.yml> up`
-   **Start services in detached mode:** `docker-compose -f <file.yml> up -d`
-   **Stop and remove containers, networks, volumes:** `docker-compose -f <file.yml> down`
-   **View logs:** `docker-compose -f <file.yml> logs -f [service_name]`
-   **Run a one-off command in a service container:** `docker-compose -f <file.yml> run --rm <service_name> [command]`

### Development Environment (`docker-compose.dev.yml`)

This configuration is optimized for local development.
-   Mounts the current source code into the container for live reloading.
-   Sets `APP_ENV=development` and `LOG_LEVEL=DEBUG`.
-   By default, it starts the Streamlit GUI on port 8501.

**To start the development environment (Streamlit GUI):**
```bash
docker-compose -f docker-compose.dev.yml up --build
```
Access the GUI at `http://localhost:8501`. Your local code changes will be reflected live.

**To run a CLI command within the development container environment:**
First, ensure the services are up (e.g., `docker-compose -f docker-compose.dev.yml up -d --build`).
Then, execute commands in the running service container:
```bash
docker-compose -f docker-compose.dev.yml exec app-dev python main.py <your_command> [options]
# Example:
docker-compose -f docker-compose.dev.yml exec app-dev python main.py load-data --config config/dev.json
```
Alternatively, use `docker-compose run` for one-off commands using the dev service definition:
```bash
docker-compose -f docker-compose.dev.yml run --rm app-dev python main.py load-data --config config/dev.json
```

### Testing Environment (`docker-compose.test.yml`)

This configuration is for running automated tests within a container.
-   Builds the image using the standard `Dockerfile`.
-   Sets `APP_ENV=test`.
-   The default command runs `python run_tests.py`.
-   Mounts volumes for test reports (`coverage_reports`, `test_results`) and logs.

**To run tests:**
```bash
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
```
The `--abort-on-container-exit` flag will stop `docker-compose` once the test container exits, and `docker-compose` will return the exit code of the test container. This is useful for CI environments.

Test reports (if configured in `pytest.ini` or `run_tests.py` to output to these directories) will be available in `./coverage_reports` and `./test_results` on the host.

### Production-like Environment (`docker-compose.prod.yml`)

This configuration is for running the application from a "clean" image, as it would be in a production-like scenario.
-   It does **not** mount the local source code (uses code baked into the image).
-   Sets `APP_ENV=production` and `LOG_LEVEL=INFO`.
-   Uses named volumes (`qlop_data_prod`, `qlop_models_prod`, `qlop_logs_prod`) for data persistence.

**To run a CLI command in a production-like container:**
Use `docker-compose run` for CLI tasks. This creates a fresh container for the command and removes it afterwards.
```bash
docker-compose -f docker-compose.prod.yml run --rm app-prod python main.py <your_command> --config config/prod.json [options]
# Example:
docker-compose -f docker-compose.prod.yml run --rm app-prod python main.py load-data --config config/prod.json
```

If you were to deploy a server component (e.g., API or GUI for production), you would use:
```bash
# Example: If CMD in Dockerfile was changed to run Streamlit or API server
# docker-compose -f docker-compose.prod.yml up -d --build
```
Ensure your `.env` file is properly configured with production secrets if needed, and that this file is securely managed.

## Healthcheck

The `Dockerfile` includes a `HEALTHCHECK` instruction (e.g., `python main.py --help`). This helps Docker monitor the basic health of the application container. You can view the health status using `docker ps`.

(More details on volume mounts, environment variables, and specific use cases will be added as the project evolves.)
