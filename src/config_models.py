# src/config_models.py
from pydantic import BaseModel, FilePath, DirectoryPath, field_validator, model_validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime # Added for field_validator
import os # Added for os.path operations in __main__
import json # Added for json operations in __main__
from pathlib import Path # Added for path operations in __main__


class GeneralConfig(BaseModel):
    """General configuration settings."""
    environment: str = "development"
    log_level: str = "INFO"

class DataPathsConfig(BaseModel):
    """Configuration for data paths."""
    raw_data_dir: Optional[DirectoryPath] = Path("data/raw/")
    processed_data_dir: Optional[DirectoryPath] = Path("data/processed/")
    feature_store_path: Optional[DirectoryPath] = Path("data/feature_store/")
    model_registry_path: Optional[DirectoryPath] = Path("models/")
    metrics_output_path: Optional[DirectoryPath] = Path("metrics/")
    predictions_output_path: Optional[DirectoryPath] = Path("predictions/")
    
    @field_validator('*', mode='before')
    @classmethod
    def check_path_string(cls, value):
        if isinstance(value, str):
            return Path(value) # Convert strings to Path objects for DirectoryPath/FilePath
        return value

class LoadDataConfig(BaseModel):
    """Configuration for loading data."""
    ticker: str = "AAPL"
    start_date: str = "2020-01-01"
    end_date: str = "2023-01-01"
    input_csv_path: Optional[FilePath] = None 
    output_parquet_path: Optional[Path] = Path("data/raw/downloaded_stock_data.parquet") 

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
    input_data_path: Union[FilePath, Path, str] 
    output_features_path: Union[Path, str] 
    technical_indicators: bool = True
    rolling_lag_features: bool = True
    sentiment_features: bool = False 
    fundamental_features: bool = False
    target_variable: TargetVariableConfig = TargetVariableConfig()
    feature_list_to_use: Optional[List[str]] = None 

class ModelParamsConfig(BaseModel):
    """Flexible model parameters."""
    n_estimators: Optional[int] = None
    learning_rate: Optional[float] = None
    max_depth: Optional[int] = None
    units: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None
    iterations: Optional[int] = None
    C: Optional[float] = None
    class Config: extra = 'allow'

class TrainModelConfig(BaseModel):
    """Configuration for model training."""
    input_features_path: Union[FilePath, Path, str]
    model_output_path_base: Union[Path, str] 
    scaler_output_path: Optional[Union[Path, str]] = None 
    model_type: str 
    model_params: ModelParamsConfig = ModelParamsConfig()
    feature_columns_to_use: Optional[List[str]] = None 
    target_column: str = "target"
    test_size: float = 0.2
    random_state: Optional[int] = 42
    shuffle_data: bool = False 

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
    scaler_path: Optional[FilePath] = None 
    test_data_path: Union[FilePath, Path, str] 
    metrics_output_json_path: Union[Path, str]
    metrics_to_compute: Optional[List[str]] = None 
    # Example: ["accuracy", "roc_auc", "f1_score"]
    # If None, evaluation script calculates all available default metrics.
    target_column: str = "target" # Default target column name
    model_type: Optional[str] = None # To help select predict function if needed, or for logging

    # SHAP related config
    shap_summary_plot_path: Optional[Union[Path, str]] = None # Path to save the SHAP summary plot
    shap_train_data_sample_path: Optional[Union[FilePath, Path, str]] = None # Path to a sample of training data for SHAP

class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    ohlcv_data_path: Union[FilePath, Path, str] 
    predictions_path: Union[FilePath, Path, str] 
    results_output_path: Union[Path, str]
    initial_capital: float = 100000.0
    commission_bps: float = 2.0
    slippage_bps: float = 1.0
    strategy_config: Dict[str, Any] = {
        "long_threshold": 0.5, "short_threshold": -0.5, "target_percent": 0.95
    }

class ExportConfig(BaseModel):
    """Configuration for exporting models or other artifacts."""
    trained_model_path: FilePath
    scaler_path: Optional[FilePath] = None 
    export_type: str = "pickle" 
    export_output_path: Union[Path, str] 

    @field_validator('export_type')
    @classmethod
    def validate_export_type(cls, value):
        if value not in ["pickle", "onnx", "joblib"]: 
            raise ValueError(f"Export type '{value}' not supported. Use 'pickle', 'joblib', or 'onnx'.")
        return value

class PipelineStepConfig(BaseModel):
    """Wrapper to hold config for a specific step."""
    enabled: bool = True
    config_path: Optional[str] = None 
    config_inline: Optional[Dict[str, Any]] = None 
    
    @model_validator(mode='before')
    @classmethod
    def check_config_source(cls, values):
        enabled = values.get('enabled', True)
        config_path, config_inline = values.get('config_path'), values.get('config_inline')
        if enabled and config_path and config_inline:
            raise ValueError("Provide either 'config_path' or 'config_inline', not both.")
        return values

class GlobalAppConfig(BaseModel):
    """Root configuration model for the entire application."""
    general: GeneralConfig = GeneralConfig()
    paths: DataPathsConfig = DataPathsConfig()
    load_data: Union[LoadDataConfig, PipelineStepConfig] = LoadDataConfig()
    engineer_features: Union[FeatureEngineeringConfig, PipelineStepConfig] = PipelineStepConfig()
    train_model: Union[TrainModelConfig, PipelineStepConfig] = PipelineStepConfig()
    evaluate_model: Union[EvaluateModelConfig, PipelineStepConfig] = PipelineStepConfig()
    backtest: Union[BacktestConfig, PipelineStepConfig] = PipelineStepConfig()
    export_model: Union[ExportConfig, PipelineStepConfig] = PipelineStepConfig()
    class Config: extra = 'ignore'


if __name__ == '__main__':
    print("--- Testing Pydantic Config Models ---")

    test_artifact_base_dir = Path("temp_config_models_test_artifacts")
    data_raw_dir_for_test = test_artifact_base_dir / "data/raw"
    data_processed_dir_for_test = test_artifact_base_dir / "data/processed"
    models_dir_for_test = test_artifact_base_dir / "models"
    ml_models_prod_for_test = test_artifact_base_dir / "ml_models_prod"
    config_steps_dir_for_test = test_artifact_base_dir / "config/steps"

    for p in [data_raw_dir_for_test, data_processed_dir_for_test, models_dir_for_test, ml_models_prod_for_test, config_steps_dir_for_test]:
        p.mkdir(parents=True, exist_ok=True)

    print("\n1. Example: Valid LoadDataConfig")
    valid_load_data_payload = {
        "ticker": "MSFT", "start_date": "2021-01-01", "end_date": "2022-01-01",
        "output_parquet_path": str(data_raw_dir_for_test / "msft_data.parquet")
    }
    try:
        cfg_ld = LoadDataConfig(**valid_load_data_payload)
        print(f"Valid LoadDataConfig:\n{cfg_ld.model_dump_json(indent=2)}")
    except Exception as e: print(f"Error: {e}")

    print("\n2. Example: Invalid LoadDataConfig (bad date)")
    invalid_load_data_payload = {"ticker": "GOOG", "start_date": "2021/01/01", "end_date": "2022-01-01"}
    try: LoadDataConfig(**invalid_load_data_payload)
    except Exception as e: print(f"Successfully caught error for invalid date: {e}")

    print("\n3. Example: TrainModelConfig with specific model_params")
    dummy_features_path_train = data_processed_dir_for_test / "features.csv"
    with open(dummy_features_path_train, "w") as f: f.write("dummy_feature_data")
    valid_train_xgb_payload = {
        "input_features_path": str(dummy_features_path_train),
        "model_output_path_base": str(models_dir_for_test / "xgb_model_v1"),
        "model_type": "XGBoost",
        "model_params": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 5}
    }
    try:
        cfg_train = TrainModelConfig(**valid_train_xgb_payload)
        print(f"Valid TrainModelConfig (XGBoost):\n{cfg_train.model_dump_json(indent=2)}")
    except Exception as e: print(f"Error: {e}")
    finally: os.remove(dummy_features_path_train)

    print("\n4. Example: GlobalAppConfig (simplified for testing)")
    dummy_train_detail_path = config_steps_dir_for_test / "train_config_detail.json"
    with open(dummy_train_detail_path, "w") as f:
        json.dump({"input_features_path": str(dummy_features_path_train), "model_output_path_base": str(models_dir_for_test / "detailed"), "model_type": "LightGBM"}, f)
    
    global_config_payload = {
        "general": {"environment": "production", "log_level": "WARN"},
        "paths": {"model_registry_path": str(ml_models_prod_for_test)},
        "load_data": {"ticker": "TSLA", "output_parquet_path": str(data_raw_dir_for_test / "tsla_prod.parquet")},
        "engineer_features": {"enabled": True, "config_path": str(config_steps_dir_for_test / "feature_eng_prod.json")},
        "train_model": {"enabled": True, "config_inline": {
            "input_features_path": "placeholder.csv", # Will be replaced for validation
            "model_output_path_base": str(models_dir_for_test / "prod_model"), "model_type": "CatBoost", "model_params": {"iterations": 200}}},
        "evaluate_model": {"enabled": False}
    }
    try:
        cfg_global = GlobalAppConfig(**global_config_payload)
        print(f"\nValid GlobalAppConfig (initial parse):\n{cfg_global.model_dump_json(indent=2)}")
        
        dummy_prod_features_path = data_processed_dir_for_test / "features_prod.csv"
        with open(dummy_prod_features_path, "w") as f: f.write("dummy_data")
        
        if isinstance(cfg_global.train_model, PipelineStepConfig) and cfg_global.train_model.config_inline:
            cfg_global.train_model.config_inline["input_features_path"] = str(dummy_prod_features_path) # Correct path for validation
            train_step_config = TrainModelConfig(**cfg_global.train_model.config_inline)
            print(f"\nParsed inline TrainModelConfig from Global:\n{train_step_config.model_dump_json(indent=2)}")
        
        if os.path.exists(dummy_prod_features_path): os.remove(dummy_prod_features_path)
    except Exception as e:
        print(f"\nError validating GlobalAppConfig: {e}")
        import traceback; traceback.print_exc()
    finally:
        if os.path.exists(dummy_train_detail_path): os.remove(dummy_train_detail_path)

    print("\n--- Testing EvaluateModelConfig with metrics_to_compute ---")
    dummy_eval_model_dir = models_dir_for_test / "eval_model_test_dir"
    dummy_eval_model_dir.mkdir(parents=True, exist_ok=True)
    dummy_eval_model_path = dummy_eval_model_dir / "eval_model.pkl"
    dummy_eval_data_path = data_processed_dir_for_test / "eval_data.csv"
    with open(dummy_eval_model_path, "w") as f: f.write("dummy model")
    with open(dummy_eval_data_path, "w") as f: f.write("dummy data")

    eval_config_payload = {
        "model_path": str(dummy_eval_model_path),
        "test_data_path": str(dummy_eval_data_path),
        "metrics_output_json_path": str(test_artifact_base_dir / "metrics/eval_output_test.json"),
        "metrics_to_compute": ["accuracy", "roc_auc"],
        "target_column": "custom_target", 
        "model_type": "XGBoost",
        "shap_summary_plot_path": str(test_artifact_base_dir / "plots/shap_summary_test.png"), # Example path
        "shap_train_data_sample_path": str(dummy_eval_data_path) # Using eval data as dummy sample for test
    }
    try:
        eval_cfg = EvaluateModelConfig(**eval_config_payload)
        print(f"Valid EvaluateModelConfig with SHAP fields:\n{eval_cfg.model_dump_json(indent=2)}")
        assert eval_cfg.metrics_to_compute == ["accuracy", "roc_auc"]
        assert eval_cfg.target_column == "custom_target"
        assert eval_cfg.model_type == "XGBoost"
        assert eval_cfg.shap_summary_plot_path == str(test_artifact_base_dir / "plots/shap_summary_test.png")
        assert eval_cfg.shap_train_data_sample_path == str(dummy_eval_data_path)
    except Exception as e:
        print(f"Error validating EvaluateModelConfig: {e}"); import traceback; traceback.print_exc()
    finally:
        if os.path.exists(dummy_eval_model_path): os.remove(dummy_eval_model_path)
        if os.path.exists(dummy_eval_data_path): os.remove(dummy_eval_data_path)
        try:
            if dummy_eval_model_dir.exists() and not list(dummy_eval_model_dir.iterdir()):
                dummy_eval_model_dir.rmdir()
        except OSError: pass

    print("\n--- Overall Test Artifact Cleanup ---")
    try:
        import shutil
        if test_artifact_base_dir.exists():
            shutil.rmtree(test_artifact_base_dir)
            print(f"Cleaned up base test artifact directory: {test_artifact_base_dir}")
    except Exception as e: print(f"Error during final cleanup of {test_artifact_base_dir}: {e}")
    print("\n--- All Config Model Tests Completed ---")
