# Docker Deployment

This document describes how to build and run the project using Docker.

## Prerequisites

-   Docker installed on your system.

## Building the Docker Image

1.  Ensure you have a `Dockerfile` in the project root.
2.  Navigate to the project root directory.
3.  Run the build command:
    ```bash
    docker build -t qlop_app .
    ```
    (Replace `qlop_app` with your desired image name/tag.)

## Running the Docker Container

-   **To run the CLI (e.g., view help):**
    ```bash
    docker run --rm qlop_app python main.py --help
    ```
-   **To run a specific CLI command (e.g., load data):**
    You'll need to mount your configuration files and data directories into the container.
    ```bash
    docker run --rm \
        -v $(pwd)/config:/app/config \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/models:/app/models \
        qlop_app python main.py load-data --config config/dev.json 
    ```
    (Adjust paths and volumes according to your setup and the command being run.)

-   **To run the Streamlit GUI:**
    ```bash
    docker run --rm -p 8501:8501 \
        -v $(pwd)/config:/app/config \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/models:/app/models \
        # Add other necessary volume mounts (e.g., for .env if GUI needs API keys)
        qlop_app streamlit run gui.py --server.port 8501 --server.address 0.0.0.0
    ```
    Access the GUI at `http://localhost:8501`.

(More details on volume mounts, environment variables, and specific use cases will be added.)
