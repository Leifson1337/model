import pytest
from pydantic import ValidationError as PydanticValidationError # Alias to avoid confusion if we have a custom one
from pathlib import Path
import json
import sys
from typing import Dict, Any

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config_models import (
    GlobalAppConfig, DataPathsConfig, LoggingConfig,
    LoadDataConfig, TargetVariableConfig, FeatureEngineeringConfig,
    ModelParamsConfig, TrainModelConfig, EvaluateModelConfig,
    BacktestStrategyConfig, BacktestConfig, ExportConfig
)
from main import load_and_validate_config # Function to test from main.py
from src.exceptions import ConfigError # Custom exception expected from load_and_validate_config

# --- Test Data for Pydantic Models ---

VALID_LOAD_DATA_CONFIG = {
    "ticker": "AAPL", "start_date": "2022-01-01", "end_date": "2023-01-01",
    "output_parquet_path": "data/raw/aapl.parquet"
}
VALID_TARGET_VAR_CONFIG = {"enabled": True, "days_forward": 5, "threshold": 0.02}
VALID_FEATURE_ENG_CONFIG = {
    "input_data_path": "data/raw/aapl.parquet",
    "output_features_path": "data/processed/aapl_features.parquet",
    "technical_indicators": True, "rolling_lag_features": True,
    "target_variable": VALID_TARGET_VAR_CONFIG
}
VALID_MODEL_PARAMS_CONFIG = {"XGBoost": {"n_estimators": 100, "learning_rate": 0.1}} # Example for XGBoost
VALID_TRAIN_MODEL_CONFIG = {
    "input_features_path": "data/processed/aapl_features.parquet",
    "model_output_path_base": "models/aapl_xgb",
    "model_type": "XGBoost",
    "model_params": {"n_estimators": 100, "learning_rate": 0.1}, # Directly pass compatible dict
    "target_column": "target"
}
VALID_EVALUATE_MODEL_CONFIG = {
    "model_path": "models/aapl_xgb/xgb_model.pkl",
    "test_data_path": "data/processed/aapl_test_features.parquet",
    "metrics_output_json_path": "data/reports/aapl_xgb_metrics.json"
}
VALID_BACKTEST_STRATEGY_CONFIG = {"long_threshold": 0.5, "short_threshold": 0.5, "target_percent": 0.02}
VALID_BACKTEST_CONFIG = {
    "ohlcv_data_path": "data/raw/aapl.parquet",
    "predictions_path": "data/predictions/aapl_xgb_preds.csv",
    "results_output_path": "data/reports/backtests/aapl_xgb_backtest.json",
    "strategy_config": VALID_BACKTEST_STRATEGY_CONFIG
}
VALID_EXPORT_CONFIG = {
    "trained_model_path": "models/aapl_xgb/xgb_model.pkl",
    "export_type": "onnx",
    "export_output_path": "models/exported/aapl_xgb.onnx"
}
VALID_GLOBAL_APP_CONFIG = {
    "app_name": "TestApp",
    "version": "1.0.0",
    "paths": {"data_dir": "data/", "model_registry_path": "models/"},
    "logging": {"log_level": "DEBUG", "log_file": "logs/app_test.log"}
}

# --- Pydantic Model Tests ---

@pytest.mark.parametrize("config_class, valid_data, invalid_data_scenarios", [
    (LoadDataConfig, VALID_LOAD_DATA_CONFIG, [
        ({"ticker": "AAPL"}, "Missing required fields start_date, end_date, output_parquet_path"),
        ({**VALID_LOAD_DATA_CONFIG, "start_date": "invalid-date"}, "Invalid date format for start_date"),
    ]),
    (TargetVariableConfig, VALID_TARGET_VAR_CONFIG, [
        ({"enabled": True, "days_forward": -1}, "days_forward must be > 0"),
    ]),
    (FeatureEngineeringConfig, VALID_FEATURE_ENG_CONFIG, [
        ({"input_data_path": "path.parquet"}, "Missing output_features_path and others"),
        ({**VALID_FEATURE_ENG_CONFIG, "target_variable": {"enabled": True, "days_forward": -5}}, "Invalid nested target_variable"),
    ]),
    (ModelParamsConfig, VALID_MODEL_PARAMS_CONFIG, [ # This model is flexible, hard to make truly "invalid" beyond type
        ({"XGBoost": "not_a_dict"}, "Model params should be a dict"),
    ]),
    (TrainModelConfig, VALID_TRAIN_MODEL_CONFIG, [
        ({"input_features_path": "path.pq", "model_type": "XGBoost"}, "Missing model_output_path_base, model_params"),
        ({**VALID_TRAIN_MODEL_CONFIG, "model_type": None}, "model_type cannot be None"),
    ]),
    (EvaluateModelConfig, VALID_EVALUATE_MODEL_CONFIG, [
        ({"model_path": "model.pkl"}, "Missing test_data_path, metrics_output_json_path"),
    ]),
    (BacktestStrategyConfig, VALID_BACKTEST_STRATEGY_CONFIG, [
        ({"long_threshold": "high"}, "Invalid type for long_threshold"),
    ]),
    (BacktestConfig, VALID_BACKTEST_CONFIG, [
        ({"ohlcv_data_path": "path.pq"}, "Missing predictions_path, results_output_path, strategy_config"),
        ({**VALID_BACKTEST_CONFIG, "strategy_config": {"invalid_key": True}}, "Invalid strategy_config structure"),
    ]),
    (ExportConfig, VALID_EXPORT_CONFIG, [
        ({"trained_model_path": "model.pkl"}, "Missing export_type, export_output_path"),
    ]),
    (DataPathsConfig, {"data_dir": "d/", "model_registry_path": "m/"}, [
        ({}, "Missing all fields"),
    ]),
    (LoggingConfig, {"log_level": "INFO", "log_file": "f.log"}, [
        ({"log_level": "INVALID"}, "Invalid log_level choice"),
    ]),
    (GlobalAppConfig, VALID_GLOBAL_APP_CONFIG, [
        ({"app_name": "Test"}, "Missing version, paths, logging"),
        ({**VALID_GLOBAL_APP_CONFIG, "paths": {"data_dir": 123}}, "Invalid type for paths.data_dir"),
    ]),
])
def test_pydantic_config_models(config_class, valid_data: Dict[str, Any], invalid_data_scenarios: list):
    # Test successful validation
    model_instance = config_class(**valid_data)
    for key, value in valid_data.items():
        # For nested models, Pydantic converts dicts to model instances.
        # Compare dicts for simplicity if the source was a dict.
        if isinstance(getattr(model_instance, key), BaseModel) and isinstance(value, dict):
            assert getattr(model_instance, key).model_dump() == value
        else:
            assert getattr(model_instance, key) == value

    # Test validation failures
    for invalid_data, description in invalid_data_scenarios:
        with pytest.raises(PydanticValidationError, match=description.split(" ")[0] if description else None): # Basic match on first word or more specific regex
            config_class(**invalid_data)
            pytest.fail(f"PydanticValidationError not raised for scenario: {description} with data {invalid_data}")


# --- Tests for load_and_validate_config function from main.py ---

def test_load_and_validate_config_success(tmp_path: Path):
    config_data = VALID_LOAD_DATA_CONFIG.copy()
    config_file = tmp_path / "valid_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    loaded_model = load_and_validate_config(str(config_file), LoadDataConfig, command_name="test_load")
    assert isinstance(loaded_model, LoadDataConfig)
    assert loaded_model.ticker == config_data["ticker"]
    assert str(loaded_model.output_parquet_path) == config_data["output_parquet_path"] # Path objects are compared via str

def test_load_and_validate_config_file_not_found():
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_and_validate_config("non_existent_file.json", LoadDataConfig, command_name="test_notfound")

def test_load_and_validate_config_invalid_json(tmp_path: Path):
    config_file = tmp_path / "invalid_json.json"
    with open(config_file, 'w') as f:
        f.write("{'ticker': 'AAPL', 'start_date': ") # Malformed JSON

    with pytest.raises(ConfigError, match="Invalid JSON format"):
        load_and_validate_config(str(config_file), LoadDataConfig, command_name="test_badjson")

def test_load_and_validate_config_pydantic_validation_error(tmp_path: Path):
    config_data = {"ticker": "MSFT"} # Missing required fields for LoadDataConfig
    config_file = tmp_path / "pydantic_error_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    with pytest.raises(ConfigError, match="Configuration validation failed"): # Or more specific Pydantic error message part
        load_and_validate_config(str(config_file), LoadDataConfig, command_name="test_pydanticerr")

def test_load_and_validate_config_no_path_provided():
    with pytest.raises(ConfigError, match="No configuration file path was provided"):
        load_and_validate_config(None, LoadDataConfig, command_name="test_nopath")

def test_load_and_validate_config_inline_config(tmp_path: Path):
    """Tests loading when 'config_inline' is present in the JSON."""
    config_data_wrapper = {"config_inline": VALID_LOAD_DATA_CONFIG.copy()}
    config_file = tmp_path / "inline_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data_wrapper, f)
    
    loaded_model = load_and_validate_config(str(config_file), LoadDataConfig, command_name="test_inline")
    assert isinstance(loaded_model, LoadDataConfig)
    assert loaded_model.ticker == VALID_LOAD_DATA_CONFIG["ticker"]

def test_load_and_validate_config_path_redirection_warning(tmp_path: Path, caplog):
    """Tests loading when 'config_path' (redirection) is present in the JSON."""
    config_data_wrapper = {"config_path": "some_other_path.json", **VALID_LOAD_DATA_CONFIG}
    config_file = tmp_path / "redirect_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data_wrapper, f)
    
    # The current `load_and_validate_config` logs a warning for redirection but still parses the current file.
    # It doesn't actually load `some_other_path.json`.
    loaded_model = load_and_validate_config(str(config_file), LoadDataConfig, command_name="test_redirect")
    assert isinstance(loaded_model, LoadDataConfig)
    assert loaded_model.ticker == VALID_LOAD_DATA_CONFIG["ticker"] # Ensures it parsed the current file's data
    assert "points to another config path" in caplog.text # Check for the warning
```
