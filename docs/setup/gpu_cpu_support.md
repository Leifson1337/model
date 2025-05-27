# GPU/CPU Support

This document describes how to configure the project for GPU or CPU-only operations.

## TensorFlow / PyTorch

-   **GPU:** Ensure CUDA drivers, CUDA Toolkit, and cuDNN are installed correctly. TensorFlow and PyTorch versions with GPU support (`tensorflow-gpu`, `torch` with CUDA) are required.
-   **CPU:** If no GPU is available or desired, TensorFlow and PyTorch will default to CPU. Ensure you have the CPU-only versions if installation size is a concern.

## Other Libraries

-   **XGBoost, LightGBM, CatBoost:** These libraries can utilize GPUs. Refer to their respective documentation for GPU build/usage instructions. By default, they use CPU.

(More details on environment variables like `CUDA_VISIBLE_DEVICES` or specific build flags will be added.)
