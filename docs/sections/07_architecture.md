## Architecture Overview

This project follows a modular architecture designed for scalability and maintainability.

-   **`src/`**: Contains the core pipeline logic:
    -   `data_management.py`: Data loading and preprocessing.
    -   `feature_engineering.py`: Feature creation and selection.
    -   `modeling.py`: Model definitions, training, and prediction functions.
    -   `evaluation.py`: Model evaluation metrics and plots.
    -   `backtesting.py`: Backtesting strategies and execution.
    -   `tuner.py`: Hyperparameter tuning using Optuna.
    -   `config_models.py`: Pydantic models for configuration validation.
    -   `metadata_utils.py`: Utilities for generating model metadata.
    -   `utils.py`: General utility functions, including logging and model saving/loading.
    -   `sentiment_analysis.py`: News sentiment fetching and analysis.
    -   `fundamental_data.py`: Fetching and processing fundamental stock data.
-   **`main.py`**: CLI entry point using `click`.
-   **`gui.py`**: Streamlit GUI application.
-   **`api/`**: For the future REST/gRPC API:
    -   `controllers.py`: API logic (currently stubs).
    -   `models.py`: API request/response Pydantic models (placeholders).
-   **`config/`**: JSON configuration files for different environments (dev, test, prod).
-   **`models/`**: Default directory for saved trained models and their metadata.
-   **`data/`**: For raw, processed, and feature data.
-   **`notebooks/`**: Jupyter notebooks for experimentation.
-   **`tests/`**: Unit and integration tests.
-   **`docs/`**: Documentation, including these auto-generated README sections.

(A more detailed architecture diagram will be added here.)
