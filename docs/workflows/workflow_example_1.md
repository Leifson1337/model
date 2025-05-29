# Workflow Example 1: Training an XGBoost Model

This workflow demonstrates how to train an XGBoost model using the CLI.

1.  **Prepare Configuration:**
    *   Create a `load_data_config.json` (see `src/config_models.py` for `LoadDataConfig`).
    *   Create a `feature_engineering_config.json` (see `src/config_models.py` for `FeatureEngineeringConfig`).
    *   Create a `train_xgboost_config.json` (see `src/config_models.py` for `TrainModelConfig`, specifying "XGBoost" as model type and relevant XGBoost parameters).

2.  **Run Pipeline Steps:**
    ```bash
    python main.py load-data --config path/to/load_data_config.json
    python main.py engineer-features --config path/to/feature_engineering_config.json
    python main.py train-model --config path/to/train_xgboost_config.json
    ```

3.  **Check Output:**
    *   The trained model will be saved in the `models/` directory (or as specified in your `TrainModelConfig`).
    *   Metadata for the model will be saved alongside it (e.g., `model_name.meta.json`).

(More details on specific configurations and expected outputs will be added.)
