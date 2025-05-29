# API Reference

## Endpoints
### `POST /analyze-feature-drift`
**Controller Function:** `analyze_feature_drift_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /backtest-strategy`
**Controller Function:** `backtest_strategy_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /compare-models`
**Controller Function:** `compare_models_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /describe-model`
**Controller Function:** `describe_model_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /engineer-features`
**Controller Function:** `engineer_features_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /evaluate-model`
**Controller Function:** `evaluate_model_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /export-model`
**Controller Function:** `export_model_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /get-latest-model-path`
**Controller Function:** `get_latest_model_path_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /list-models`
**Controller Function:** `list_models_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /load-data`
**Controller Function:** `load_data_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

### `POST /train-model`
**Controller Function:** `train_model_controller`

**Description:** No description provided.

**Request Body:** JSON object (`dict`). See controller implementation for expected structure.

**Success Response Body (Example - 200 OK):** Varies. See controller implementation (typically includes `status`, `message`).

**Error Responses:** Standard error responses (e.g., 400, 404, 422, 500) use the `ApiErrorResponse` model structure (see below).

---

## Data Models (Pydantic)
### `AnalyzeDriftApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `current_features_stats_path` | `FilePath` | No | Path to current feature statistics JSON file.. Field details defined in model. |
| `baseline_stats_path` | `FilePath` | No | Path to baseline feature statistics JSON file for comparison.. Field details defined in model. |
| `output_filename_suffix` | `Optional[str]` | Yes | Optional suffix for the drift report file.. Default: `Field(None, example='_drift_report_q2_2024', description='Optional suffix for the drift report file.')` |

### `AnalyzeDriftApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `status` | `str` | No | Field details defined in model. |
| `message` | `str` | No | Field details defined in model. |
| `drift_report_path` | `str` | No | N/A |
| `drift_metrics` | `Dict[str, Any]` | No | Field details defined in model. |

### `ApiErrorResponse`
Standard error response structure for the API.

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `error` | `Dict[str, Any]` | No | Field details defined in model. |
| `status_code` | `int` | No | HTTP status code.. Field details defined in model. |

### `BacktestApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `ohlcv_data_path` | `FilePath` | No | Path to OHLCV data for backtesting.. Field details defined in model. |
| `predictions_path` | `FilePath` | No | Path to model predictions file (aligned with OHLCV data).. Field details defined in model. |
| `strategy_config` | `Dict[str, Any]` | No | Configuration for the backtesting strategy.. Field details defined in model. |
| `results_output_filename_suffix` | `Optional[str]` | Yes | Optional suffix for the backtest results file.. Default: `Field(None, example='_simple_threshold_strategy', description='Optional suffix for the backtest results file.')` |

### `BacktestApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `status` | `str` | No | Field details defined in model. |
| `message` | `str` | No | Field details defined in model. |
| `results` | `Dict[str, Any]` | No | Field details defined in model. |
| `results_output_path` | `str` | No | N/A |

### `CompareModelsApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `model_name` | `str` | No | Field details defined in model. |
| `versions` | `List[str]` | No | Field details defined in model. |

### `CompareModelsApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `comparison_data` | `Dict[str, Any]` | No | N/A |

### `DescribeModelApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `model_name` | `str` | No | Field details defined in model. |
| `version` | `str` | No | Field details defined in model. |

### `DescribeModelApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `details` | `Optional[Dict[str, Any]]` | Yes | Default: `N/A (Required)` |

### `ErrorDetail`
Detailed information about a validation error or other issues.

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `loc` | `Optional[List[str]]` | Yes | Location of the error (e.g., field name).. Default: `Field(None, description='Location of the error (e.g., field name).', example=['body', 'ticker'])` |
| `msg` | `str` | No | Error message.. Field details defined in model. |
| `type` | `Optional[str]` | Yes | Type of error.. Default: `Field(None, description='Type of error.', example='value_error.missing')` |

### `EvaluateModelApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `model_path` | `FilePath` | No | Path to the trained model file.. Field details defined in model. |
| `test_data_path` | `FilePath` | No | Path to the test data file (features + target).. Field details defined in model. |
| `scaler_path` | `Optional[FilePath]` | Yes | Path to the scaler file, if used during training.. Default: `Field(None, description='Path to the scaler file, if used during training.')` |
| `metrics_to_compute` | `Optional[List[str]]` | Yes | Specific metrics to compute. If None, computes all default metrics.. Default: `Field(None, example=['accuracy', 'roc_auc', 'f1_score_macro'], description='Specific metrics to compute. If None, computes all default metrics.')` |
| `shap_summary_plot` | `bool` | No | Generate and save a SHAP summary plot.. Field details defined in model. |

### `EvaluateModelApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `status` | `str` | No | Field details defined in model. |
| `message` | `str` | No | Field details defined in model. |
| `metrics` | `Dict[str, Any]` | No | Field details defined in model. |
| `metrics_output_json_path` | `str` | No | N/A |
| `shap_summary_plot_path` | `Optional[str]` | Yes | Default: `Field(None)` |

### `ExportModelApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `trained_model_path` | `FilePath` | No | Path to the trained model file to be exported.. Field details defined in model. |
| `export_type` | `str` | No | Format to export the model to (e.g., onnx, joblib_zip).. Field details defined in model. |
| `export_filename_suffix` | `Optional[str]` | Yes | Optional suffix for the exported model file name.. Default: `Field(None, example='_for_serving', description='Optional suffix for the exported model file name.')` |

### `ExportModelApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `status` | `str` | No | Field details defined in model. |
| `message` | `str` | No | Field details defined in model. |
| `export_output_path` | `str` | No | N/A |

### `FeatureEngineeringApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `input_data_path` | `FilePath` | No | Path to the input raw data file (e.g., output from load-data).. Field details defined in model. |
| `output_filename_suffix` | `Optional[str]` | Yes | Optional suffix for the output features file name.. Default: `Field(None, example='_features_v2', description='Optional suffix for the output features file name.')` |
| `technical_indicators` | `bool` | No | Enable generation of technical indicators.. Field details defined in model. |
| `rolling_lag_features` | `bool` | No | Enable generation of rolling window and lag features.. Field details defined in model. |
| `sentiment_features` | `bool` | No | Enable sentiment analysis features (requires NewsAPI key).. Field details defined in model. |
| `fundamental_features` | `bool` | No | Enable fundamental data features (requires AlphaVantage key).. Field details defined in model. |
| `target_variable` | `Optional[TargetVariableConfigApi]` | Yes | Configuration for target variable generation.. Default: `Generated by TargetVariableConfigApi` |

### `FeatureEngineeringApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `status` | `str` | No | Field details defined in model. |
| `message` | `str` | No | Field details defined in model. |
| `output_features_path` | `str` | No | N/A |
| `features_generated` | `List[str]` | No | Field details defined in model. |
| `num_features` | `int` | No | Field details defined in model. |
| `data_shape` | `Optional[Tuple[int, int]]` | Yes | Shape of the output DataFrame (rows, columns).. Default: `Field(None, description='Shape of the output DataFrame (rows, columns).')` |

### `GetLatestModelPathApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `model_name` | `str` | No | Field details defined in model. |

### `GetLatestModelPathApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `path` | `Optional[str]` | Yes | Default: `None` |
| `message` | `str` | No | Field details defined in model. |

### `ListModelsApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `model_name_filter` | `Optional[str]` | Yes | Optional filter by model name.. Default: `Field(None, example='xgboost_stock_predictor', description='Optional filter by model name.')` |

### `ListModelsApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `models` | `List[ModelRegistryEntry]` | No | N/A |

### `LoadDataApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `ticker` | `str` | No | Stock ticker symbol.. Field details defined in model. |
| `start_date` | `date` | No | Start date for data fetching (YYYY-MM-DD).. Field details defined in model. |
| `end_date` | `date` | No | End date for data fetching (YYYY-MM-DD).. Field details defined in model. |
| `output_filename_suffix` | `Optional[str]` | Yes | Optional suffix for the output data file name.. Default: `Field(None, example='_daily_v1', description='Optional suffix for the output data file name.')` |

### `LoadDataApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `status` | `str` | No | Field details defined in model. |
| `message` | `str` | No | Field details defined in model. |
| `output_data_path` | `str` | No | N/A |
| `records_processed` | `Optional[int]` | Yes | Default: `Field(None, example=252)` |
| `data_preview` | `Optional[List[Dict[str, Any]]]` | Yes | Preview of the first few records.. Default: `Field(None, description='Preview of the first few records.')` |

### `ModelRegistryEntry`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `model_name_from_config` | `Optional[str]` | Yes | Default: `Field(None, alias='model_name')` |
| `model_version` | `Optional[str]` | Yes | Default: `None` |
| `timestamp_utc` | `Optional[str]` | Yes | Default: `None` |
| `primary_metric_name` | `Optional[str]` | Yes | Default: `None` |
| `primary_metric_value` | `Optional[Any]` | Yes | Default: `None` |
| `meta_json_path` | `Optional[str]` | Yes | Default: `None` |
| `has_feature_importance` | `Optional[bool]` | Yes | Default: `None` |
| `model_params` | `Optional[Dict[str, Any]]` | Yes | Default: `None` |
| `feature_engineering_config` | `Optional[Dict[str, Any]]` | Yes | Default: `None` |
| `training_data_stats` | `Optional[Dict[str, Any]]` | Yes | Default: `None` |
| `evaluation_metrics` | `Optional[Dict[str, Any]]` | Yes | Default: `None` |
| `tags` | `Optional[Dict[str, str]]` | Yes | Default: `None` |
| `notes` | `Optional[str]` | Yes | Default: `None` |
| `model_path` | `Optional[str]` | Yes | Default: `None` |
| `scaler_path` | `Optional[str]` | Yes | Default: `None` |
| `feature_importance_path` | `Optional[str]` | Yes | Default: `None` |

### `StatusResponse`
Generic response for status updates.

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `status` | `str` | No | Field details defined in model. |
| `message` | `str` | No | Field details defined in model. |
| `details` | `Optional[Dict[str, Any]]` | Yes | Default: `Field(None, example={'item_id': 123, 'action': 'processed'})` |

### `TargetVariableConfigApi`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `enabled` | `bool` | No | N/A |
| `days_forward` | `int` | No | Field details defined in model. |
| `threshold` | `float` | No | Field details defined in model. |

### `TrainModelApiRequest`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `input_features_path` | `FilePath` | No | Path to the feature-engineered data file.. Field details defined in model. |
| `model_type` | `str` | No | Type of model to train (e.g., XGBoost, LSTM).. Field details defined in model. |
| `model_params` | `Dict[str, Any]` | No | Hyperparameters for the model.. Field details defined in model. |
| `target_column` | `str` | No | Name of the target variable column.. Field details defined in model. |
| `model_name_suffix` | `Optional[str]` | Yes | Optional suffix for the trained model name/path.. Default: `Field(None, example='_final_production', description='Optional suffix for the trained model name/path.')` |
| `scaler_filename_suffix` | `Optional[str]` | Yes | Optional suffix for the scaler file if applicable (e.g., for NNs).. Default: `Field(None, example='_final_scaler', description='Optional suffix for the scaler file if applicable (e.g., for NNs).')` |

### `TrainModelApiResponse`

| Field | Type | Optional | Default / Description |
|-------|------|----------|-----------------------|
| `status` | `str` | No | Field details defined in model. |
| `message` | `str` | No | Field details defined in model. |
| `model_output_path_base` | `str` | No | N/A |
| `model_type` | `str` | No | Field details defined in model. |
| `training_duration_seconds` | `Optional[float]` | Yes | Default: `Field(None, example=120.5)` |
