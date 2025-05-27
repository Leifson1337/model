from pydantic import BaseModel, FilePath, DirectoryPath, field_validator, model_validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime # Added for field_validator
import os # Added for os.path operations in __main__
import json # Added for json operations in __main__

class GeneralConfig(BaseModel):
    """General configuration settings."""
    environment: str = "development"
    log_level: str = "INFO"
    # TODO: Add versioning for config itself

class DataPathsConfig(BaseModel):
    """Configuration for data paths."""
    raw_data_dir: Optional[DirectoryPath] = "data/raw/"
    processed_data_dir: Optional[DirectoryPath] = "data/processed/"
    feature_store_path: Optional[DirectoryPath] = "data/feature_store/"
    model_registry_path: Optional[DirectoryPath] = "models/"
    metrics_output_path: Optional[DirectoryPath] = "metrics/"
    predictions_output_path: Optional[DirectoryPath] = "predictions/"
    
    # Example validator for paths (can be more specific if needed)
    @field_validator('*', mode='before')
    @classmethod
    def check_path_string(cls, value):
        if isinstance(value, str):
            # Basic check, pydantic will do more if type is FilePath/DirectoryPath
            return value
        return value

class LoadDataConfig(BaseModel):
    """Configuration for loading data."""
    ticker: str = "AAPL"
    start_date: str = "2020-01-01"
    end_date: str = "2023-01-01"
    # Example: if data is loaded from a file instead of API
    input_csv_path: Optional[FilePath] = None 
    output_parquet_path: Optional[str] = "data/raw/downloaded_stock_data.parquet" # Not FilePath because it might not exist yet

    @field_validator('start_date', 'end_date', mode='before')
    @classmethod
    def validate_date_format(cls, value):
        if isinstance(value, str):
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format.")
        return value
        
class TargetVariableConfig(BaseModel):
    enabled: bool = True
    days_forward: int = 5
    threshold: float = 0.03

class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering."""
    input_data_path: Union[FilePath, str] # str if it's a reference to a previous step's output name
    output_features_path: str # Not FilePath as it will be created
    technical_indicators: bool = True
    rolling_lag_features: bool = True
    sentiment_features: bool = False # Requires NewsAPI key
    fundamental_features: bool = False
    target_variable: TargetVariableConfig = TargetVariableConfig()
    feature_list_to_use: Optional[List[str]] = None # If None, use all generated/selected
    # TODO: Add more specific params for each feature type, e.g., window sizes for rolling features

class ModelParamsConfig(BaseModel):
    """Flexible model parameters."""
    # Common tree model params
    n_estimators: Optional[int] = None
    learning_rate: Optional[float] = None
    max_depth: Optional[int] = None
    
    # Common DL params
    units: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None
    
    # For specific model types like CatBoost
    iterations: Optional[int] = None
    
    # For Logistic Regression
    C: Optional[float] = None

    # Allow any other parameters for flexibility
    class Config:
        extra = 'allow'


class TrainModelConfig(BaseModel):
    """Configuration for model training."""
    input_features_path: Union[FilePath, str]
    model_output_path_base: str # Base name, type and extension will be added
    scaler_output_path: Optional[str] = None # For DL models
    model_type: str # e.g., "XGBoost", "LSTM"
    model_params: ModelParamsConfig = ModelParamsConfig()
    feature_columns_to_use: Optional[List[str]] = None # If None, use all from input_features_path
    target_column: str = "target"
    test_size: float = 0.2
    random_state: Optional[int] = 42
    shuffle_data: bool = False # Time series usually false

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, value):
        supported_models = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "CNN-LSTM", "Transformer", "LogisticRegression"]
        if value not in supported_models:
            raise ValueError(f"Model type '{value}' not recognized. Supported: {supported_models}")
        return value

class EvaluateModelConfig(BaseModel):
    """Configuration for model evaluation."""
    model_path: FilePath
    scaler_path: Optional[FilePath] = None # For DL models
    test_data_path: Union[FilePath, str] # Features + target
    metrics_output_json_path: str
    # TODO: Add specific evaluation metrics to compute, e.g., ["accuracy", "roc_auc", "f1_score"]

class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    ohlcv_data_path: Union[FilePath, str] # Path to OHLCV data for backtesting period
    predictions_path: Union[FilePath, str] # Path to model predictions aligned with OHLCV data
    results_output_path: str
    initial_capital: float = 100000.0
    commission_bps: float = 2.0
    slippage_bps: float = 1.0
    strategy_config: Dict[str, Any] = { # Passed to SignalStrategy in backtesting.py
        "long_threshold": 0.5,
        "short_threshold": -0.5,
        "target_percent": 0.95
    }

class ExportConfig(BaseModel):
    """Configuration for exporting models or other artifacts."""
    trained_model_path: FilePath
    scaler_path: Optional[FilePath] = None # If applicable
    export_type: str = "pickle" # e.g., "pickle", "onnx"
    export_output_path: str # Full path for the exported artifact

    @field_validator('export_type')
    @classmethod
    def validate_export_type(cls, value):
        if value not in ["pickle", "onnx", "joblib"]: # Added joblib
            raise ValueError(f"Export type '{value}' not supported. Use 'pickle', 'joblib', or 'onnx'.")
        return value

# --- Main Configuration Schema ---
# This will be the model used to parse the main config files (dev.json, test.json, prod.json)
# It will compose the specific configurations for each pipeline step.

class PipelineStepConfig(BaseModel):
    """Wrapper to hold config for a specific step, or indicate it's disabled/uses defaults."""
    enabled: bool = True
    config_path: Optional[str] = None # Path to a more detailed JSON for this step
    config_inline: Optional[Dict[str, Any]] = None # Or inline config
    
    # This model_validator allows either config_path or config_inline, but not both if enabled.
    # If not enabled, neither is required.
    @model_validator(mode='before')
    @classmethod
    def check_config_source(cls, values):
        enabled = values.get('enabled', True)
        config_path = values.get('config_path')
        config_inline = values.get('config_inline')

        if enabled:
            if config_path and config_inline:
                raise ValueError("Provide either 'config_path' or 'config_inline', not both.")
            # If enabled, one of them should ideally be present, but this can be a soft requirement
            # if defaults are assumed by the pipeline step runner.
            # For now, just checking they are not both present.
        return values

class GlobalAppConfig(BaseModel):
    """Root configuration model for the entire application."""
    general: GeneralConfig = GeneralConfig()
    paths: DataPathsConfig = DataPathsConfig()
    
    # Pipeline step configurations
    # These can be loaded from separate files or defined inline in the main config
    load_data: Union[LoadDataConfig, PipelineStepConfig] = LoadDataConfig() # Example: load_data can be fully defined here
    engineer_features: Union[FeatureEngineeringConfig, PipelineStepConfig] = PipelineStepConfig()
    train_model: Union[TrainModelConfig, PipelineStepConfig] = PipelineStepConfig()
    evaluate_model: Union[EvaluateModelConfig, PipelineStepConfig] = PipelineStepConfig()
    backtest: Union[BacktestConfig, PipelineStepConfig] = PipelineStepConfig()
    export_model: Union[ExportConfig, PipelineStepConfig] = PipelineStepConfig()

    # Example: if dev.json directly contains LoadDataConfig fields under "load_data"
    # Pydantic will try to parse it. If it's a dict with "enabled": false, it'll use PipelineStepConfig.

    class Config:
        extra = 'ignore' # Ignore extra fields in the config files to prevent errors during initial rollout

# Example of how to use:
# from pathlib import Path
# import json
# def load_app_config(config_file_path: str) -> GlobalAppConfig:
#     config_json = json.loads(Path(config_file_path).read_text())
#     app_config = GlobalAppConfig(**config_json)
#     return app_config

if __name__ == '__main__':
    from datetime import datetime # Required for field_validator

    # --- Example Usage and Validation ---
    print("--- Testing Pydantic Config Models ---")

    # Create necessary directories for test file/path validations
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("ml_models_prod", exist_ok=True) # For GlobalAppConfig test
    os.makedirs("config/steps", exist_ok=True) # For GlobalAppConfig test

    # 1. Example: Valid LoadDataConfig
    valid_load_data_payload = {
        "ticker": "MSFT",
        "start_date": "2021-01-01",
        "end_date": "2022-01-01",
        "output_parquet_path": "data/raw/msft_data.parquet"
    }
    try:
        cfg = LoadDataConfig(**valid_load_data_payload)
        print(f"\nValid LoadDataConfig: {cfg.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"\nError validating valid LoadDataConfig: {e}")

    # 2. Example: Invalid LoadDataConfig (bad date)
    invalid_load_data_payload = {
        "ticker": "GOOG",
        "start_date": "2021/01/01", # Invalid date format
        "end_date": "2022-01-01"
    }
    try:
        cfg = LoadDataConfig(**invalid_load_data_payload)
        print(f"\nValid LoadDataConfig (should fail): {cfg}")
    except Exception as e:
        print(f"\nSuccessfully caught error for invalid LoadDataConfig (date format): {e}")

    # 3. Example: TrainModelConfig with specific model_params
    valid_train_xgb_payload = {
        "input_features_path": "data/processed/features.csv",
        "model_output_path_base": "models/xgb_model_v1",
        "model_type": "XGBoost",
        "model_params": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 5}
    }
    try:
        # Need to ensure FilePath resolves correctly if file doesn't exist.
        # For testing, let's assume paths are valid strings if file check is not desired here.
        # To make FilePath work, the file must exist. We can create dummy files for testing.
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/features.csv", "w") as f: f.write("dummy_feature_data")
        
        cfg = TrainModelConfig(**valid_train_xgb_payload)
        print(f"\nValid TrainModelConfig (XGBoost): {cfg.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"\nError validating valid TrainModelConfig (XGBoost): {e}")
    finally:
        if os.path.exists("data/processed/features.csv"): os.remove("data/processed/features.csv")


    # 4. Example: GlobalAppConfig (simplified for testing)
    with open("config/steps/train_config_detail.json", "w") as f:
        json.dump({
            "input_features_path": "data/processed/features.csv", # This file is created locally in TrainModelConfig test
            "model_output_path_base": "models/detailed_model",
            "model_type": "LightGBM",
            "model_params": {"n_estimators": 80}
        }, f)

    global_config_payload = {
        "general": {"environment": "production", "log_level": "WARN"},
        "paths": {"model_registry_path": "ml_models_prod/"},
        "load_data": { # Fully defined inline
            "ticker": "TSLA",
            "start_date": "2019-01-01",
            "end_date": "2024-01-01",
            "output_parquet_path": "data/raw/tsla_prod_data.parquet"
        },
        "engineer_features": { # Uses PipelineStepConfig to point to another file (not implemented here)
            "enabled": True,
            "config_path": "config/steps/feature_eng_prod_config.json" # This file doesn't exist, so direct parsing would fail if not handled
        },
        "train_model": { # Uses PipelineStepConfig with inline config
            "enabled": True,
            "config_inline": {
                 "input_features_path": "data/processed/features_prod.csv", # File doesn't exist, would fail FilePath validation if TrainModelConfig was directly parsed
                 "model_output_path_base": "models/prod_model",
                 "model_type": "CatBoost",
                 "model_params": {"iterations": 200}
            }
        },
        "evaluate_model": {"enabled": False} # Disabled step
    }
    try:
        cfg = GlobalAppConfig(**global_config_payload)
        print(f"\nValid GlobalAppConfig:\n{cfg.model_dump_json(indent=2)}")
        if isinstance(cfg.load_data, LoadDataConfig):
            print(f"Load Data Ticker (parsed): {cfg.load_data.ticker}")
        
        if isinstance(cfg.train_model, PipelineStepConfig) and cfg.train_model.config_inline:
            # If we want to parse the inline config for train_model:
            # Need to ensure 'data/processed/features_prod.csv' exists for FilePath validation
            os.makedirs("data/processed", exist_ok=True)
            with open("data/processed/features_prod.csv", "w") as f: f.write("dummy_data")
            
            train_step_config = TrainModelConfig(**cfg.train_model.config_inline)
            print(f"\nParsed inline TrainModelConfig: {train_step_config.model_dump_json(indent=2)}")
            
            if os.path.exists("data/processed/features_prod.csv"): os.remove("data/processed/features_prod.csv")

    except Exception as e:
        print(f"\nError validating GlobalAppConfig: {e}")
    finally:
        # Clean up created files and directories
        if os.path.exists("config/steps/train_config_detail.json"): os.remove("config/steps/train_config_detail.json")
        if os.path.isdir("config/steps"): os.rmdir("config/steps")
        # Don't remove data/raw, data/processed, models, ml_models_prod if they might be used by other tests or main.py test
        # For isolated test, we would:
        # if os.path.exists("data/raw/tsla_prod_data.parquet"): os.remove("data/raw/tsla_prod_data.parquet") # If created
        # if os.path.isdir("ml_models_prod"): os.rmdir("ml_models_prod")


    print("\n--- Pydantic Config Models Test Completed ---")
