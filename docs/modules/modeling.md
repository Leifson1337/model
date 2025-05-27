# Modeling

This section covers the different models implemented and their training/prediction functions.

- **Tree-based Models:** XGBoost, LightGBM, CatBoost
- **Sequence Models:** LSTM, CNN-LSTM, Transformer
- **Other Models:** Prophet (for time series forecasting, if used directly)

Model training and prediction logic is in `src/modeling.py`.
Hyperparameter tuning is handled by `src/tuner.py` using Optuna.

Model configurations are defined in `src/config_models.py` (`TrainModelConfig`, `ModelParamsConfig`).
Saved models and their metadata are typically stored in the `models/` directory. Refer to `src/utils.py` for saving/loading utilities and `src/metadata_utils.py` for metadata generation.
