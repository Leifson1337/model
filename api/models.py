# api/models.py
from pydantic import BaseModel, Field, FilePath
from typing import Optional, List, Dict, Any, Tuple # Added Tuple
from datetime import date

# --- General Purpose Models ---

class StatusResponse(BaseModel):
    """Generic response for status updates."""
    status: str = Field(..., example="success")
    message: str = Field(..., example="Operation completed successfully.")
    details: Optional[Dict[str, Any]] = Field(None, example={"item_id": 123, "action": "processed"})

class ErrorDetail(BaseModel):
    """Detailed information about a validation error or other issues."""
    loc: Optional[List[str]] = Field(None, description="Location of the error (e.g., field name).", example=["body", "ticker"])
    msg: str = Field(..., description="Error message.", example="Field required")
    type: Optional[str] = Field(None, description="Type of error.", example="value_error.missing")

class ApiErrorResponse(BaseModel):
    """Standard error response structure for the API."""
    error: Dict[str, Any] = Field(..., example={
        "type": "DataValidationError",
        "message": "Invalid input data provided.",
        "details": [{"loc": ["body", "ticker"], "msg": "Field required", "type": "value_error.missing"}]
    })
    status_code: int = Field(..., description="HTTP status code.", example=400)


# --- Load Data Models ---

class LoadDataApiRequest(BaseModel):
    ticker: str = Field(..., example="AAPL", description="Stock ticker symbol.")
    start_date: date = Field(..., example="2022-01-01", description="Start date for data fetching (YYYY-MM-DD).")
    end_date: date = Field(..., example="2023-01-01", description="End date for data fetching (YYYY-MM-DD).")
    output_filename_suffix: Optional[str] = Field(None, example="_daily_v1", description="Optional suffix for the output data file name.")

class LoadDataApiResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Data loaded successfully.")
    output_data_path: str 
    records_processed: Optional[int] = Field(None, example=252)
    data_preview: Optional[List[Dict[str, Any]]] = Field(None, description="Preview of the first few records.")

# --- Feature Engineering Models ---

class TargetVariableConfigApi(BaseModel): 
    enabled: bool = True
    days_forward: int = Field(5, gt=0)
    threshold: float = Field(0.02, gt=0)

class FeatureEngineeringApiRequest(BaseModel):
    input_data_path: FilePath = Field(..., description="Path to the input raw data file (e.g., output from load-data).")
    output_filename_suffix: Optional[str] = Field(None, example="_features_v2", description="Optional suffix for the output features file name.")
    technical_indicators: bool = Field(True, description="Enable generation of technical indicators.")
    rolling_lag_features: bool = Field(True, description="Enable generation of rolling window and lag features.")
    sentiment_features: bool = Field(False, description="Enable sentiment analysis features (requires NewsAPI key).")
    fundamental_features: bool = Field(False, description="Enable fundamental data features (requires AlphaVantage key).")
    target_variable: Optional[TargetVariableConfigApi] = Field(default_factory=TargetVariableConfigApi, description="Configuration for target variable generation.")

class FeatureEngineeringApiResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Features engineered successfully.")
    output_features_path: str 
    features_generated: List[str] = Field(..., example=["SMA_20", "RSI_14", "target"])
    num_features: int = Field(..., example=50)
    data_shape: Optional[Tuple[int, int]] = Field(None, description="Shape of the output DataFrame (rows, columns).")

# --- Train Model Models ---

class TrainModelApiRequest(BaseModel):
    input_features_path: FilePath = Field(..., description="Path to the feature-engineered data file.")
    model_type: str = Field(..., example="XGBoost", description="Type of model to train (e.g., XGBoost, LSTM).")
    model_params: Dict[str, Any] = Field(..., example={"n_estimators": 100, "learning_rate": 0.1}, description="Hyperparameters for the model.")
    target_column: str = Field("target", example="target", description="Name of the target variable column.")
    model_name_suffix: Optional[str] = Field(None, example="_final_production", description="Optional suffix for the trained model name/path.")
    scaler_filename_suffix: Optional[str] = Field(None, example="_final_scaler", description="Optional suffix for the scaler file if applicable (e.g., for NNs).")

class TrainModelApiResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Model trained successfully.")
    model_output_path_base: str 
    model_type: str = Field(..., example="XGBoost")
    training_duration_seconds: Optional[float] = Field(None, example=120.5)

# --- Evaluate Model Models ---

class EvaluateModelApiRequest(BaseModel):
    model_path: FilePath = Field(..., description="Path to the trained model file.")
    test_data_path: FilePath = Field(..., description="Path to the test data file (features + target).")
    scaler_path: Optional[FilePath] = Field(None, description="Path to the scaler file, if used during training.")
    metrics_to_compute: Optional[List[str]] = Field(None, example=["accuracy", "roc_auc", "f1_score_macro"], description="Specific metrics to compute. If None, computes all default metrics.")
    shap_summary_plot: bool = Field(False, description="Generate and save a SHAP summary plot.")

class EvaluateModelApiResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Model evaluation completed.")
    metrics: Dict[str, Any] = Field(..., example={"accuracy": 0.85, "roc_auc": 0.92})
    metrics_output_json_path: str 
    shap_summary_plot_path: Optional[str] = Field(None)

# --- Backtest Models ---

class BacktestApiRequest(BaseModel):
    ohlcv_data_path: FilePath = Field(..., description="Path to OHLCV data for backtesting.")
    predictions_path: FilePath = Field(..., description="Path to model predictions file (aligned with OHLCV data).")
    strategy_config: Dict[str, Any] = Field(..., example={"long_threshold": 0.6, "short_threshold": 0.4}, description="Configuration for the backtesting strategy.")
    results_output_filename_suffix: Optional[str] = Field(None, example="_simple_threshold_strategy", description="Optional suffix for the backtest results file.")

class BacktestApiResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Backtest completed successfully.")
    results: Dict[str, Any] = Field(..., example={"total_return_pct": 15.5, "sharpe_ratio": 1.2})
    results_output_path: str

# --- Export Model Models ---

class ExportModelApiRequest(BaseModel):
    trained_model_path: FilePath = Field(..., description="Path to the trained model file to be exported.")
    export_type: str = Field(..., example="onnx", description="Format to export the model to (e.g., onnx, joblib_zip).")
    export_filename_suffix: Optional[str] = Field(None, example="_for_serving", description="Optional suffix for the exported model file name.")

class ExportModelApiResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Model exported successfully.")
    export_output_path: str

# --- Model Registry Models ---

class ListModelsApiRequest(BaseModel):
    model_name_filter: Optional[str] = Field(None, example="xgboost_stock_predictor", description="Optional filter by model name.")

class ModelRegistryEntry(BaseModel): 
    model_name_from_config: Optional[str] = Field(None, alias="model_name") 
    model_version: Optional[str] = None
    timestamp_utc: Optional[str] = None
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[Any] = None # Allow Any for flexibility
    meta_json_path: Optional[str] = None
    has_feature_importance: Optional[bool] = None
    # Ensure all fields from registry_utils.list_models output are covered
    model_params: Optional[Dict[str, Any]] = None
    feature_engineering_config: Optional[Dict[str, Any]] = None
    training_data_stats: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None
    notes: Optional[str] = None
    model_path: Optional[str] = None # Actual path to model file
    scaler_path: Optional[str] = None # Path to scaler file
    feature_importance_path: Optional[str] = None # Path to FI plot/data

class ListModelsApiResponse(BaseModel):
    models: List[ModelRegistryEntry]

class DescribeModelApiRequest(BaseModel): 
    model_name: str = Field(..., example="xgboost_stock_predictor")
    version: str = Field(..., example="v1.2.0")

class DescribeModelApiResponse(BaseModel):
    details: Optional[Dict[str, Any]]

class CompareModelsApiRequest(BaseModel): 
    model_name: str = Field(..., example="xgboost_stock_predictor")
    versions: List[str] = Field(..., min_length=2, example=["v1.1.0", "v1.2.0"])

class CompareModelsApiResponse(BaseModel):
    comparison_data: Dict[str, Any]

class GetLatestModelPathApiRequest(BaseModel): 
    model_name: str = Field(..., example="xgboost_stock_predictor")

class GetLatestModelPathApiResponse(BaseModel):
    path: Optional[str] = None 
    message: str = Field(..., example="Path retrieved successfully or model not found.")

# --- Feature Drift Analysis Models ---

class AnalyzeDriftApiRequest(BaseModel):
    current_features_stats_path: FilePath = Field(..., description="Path to current feature statistics JSON file.")
    baseline_stats_path: FilePath = Field(..., description="Path to baseline feature statistics JSON file for comparison.")
    output_filename_suffix: Optional[str] = Field(None, example="_drift_report_q2_2024", description="Optional suffix for the drift report file.")

class AnalyzeDriftApiResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Feature drift analysis completed.")
    drift_report_path: str 
    drift_metrics: Dict[str, Any] = Field(..., example={"psi_sum": 0.15, "num_drifted_features": 3})

# Rebuild models that might have forward references if types were defined later
FeatureEngineeringApiResponse.model_rebuild()
ListModelsApiResponse.model_rebuild()
DescribeModelApiResponse.model_rebuild()
CompareModelsApiResponse.model_rebuild()
GetLatestModelPathApiResponse.model_rebuild()
AnalyzeDriftApiResponse.model_rebuild()
