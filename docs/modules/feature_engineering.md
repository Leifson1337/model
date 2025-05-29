# Feature Engineering

This section details the feature engineering pipeline.

- Technical Indicators (from `ta` library)
- Rolling and Lag Features
- Sentiment Scores (daily aggregated)
- Fundamental Data Features (and their lags/deltas)
- Target Variable Creation

Configuration is managed via `src/config_models.py` (`FeatureEngineeringConfig`).
Refer to `src/feature_engineering.py` for implementation details.
