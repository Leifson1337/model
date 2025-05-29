## Key Features

- **Modular Design:** Core components (data, features, models, API) are separated for clarity and maintainability.
- **CLI Interface:** Provides command-line access to pipeline operations (`main.py`).
- **GUI Interface:** Streamlit-based GUI (`gui.py`) for interactive data analysis and model training.
- **Pydantic Configuration:** Type-safe configuration management using Pydantic models (`src/config_models.py`).
- **Versatile Modeling:** Supports XGBoost, LightGBM, CatBoost, LSTM, CNN-LSTM, and Transformer models.
- **Advanced Backtesting:** Includes stubs for walk-forward and dynamic rolling window evaluations.
- **Metadata Tracking:** Generates metadata for trained models (`src/metadata_utils.py`).
- **Sentiment Analysis:** Integrates news sentiment analysis using Hugging Face transformers.
- **Fundamental Data:** Incorporates fundamental stock data.
- **Hyperparameter Tuning:** Optuna integration for optimizing model parameters (`src/tuner.py`).
- **Dockerization Support:** Includes a `Dockerfile` for containerized deployment.
- **CI/CD Readiness:** Basic testing structure (`tests/`) and integrity checks (`check_pipeline_integrity.py`).
- **Auto-generated Documentation:** This README is auto-generated from modular sections.

{{feature_list}}

(More features to be listed as developed)
