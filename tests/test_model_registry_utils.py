import pytest
import json
from pathlib import Path
from typing import Dict, Any, List, Generator
from datetime import datetime, timezone, timedelta
import shutil # For cleaning up if needed, though tmp_path handles it
import uuid # For unique enough version strings

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.model_registry_utils import (
    register_model, 
    get_model_details, 
    list_models, 
    get_latest_model_version_path,
    compare_model_versions,
    _format_param_value # If testing private helper, though usually not recommended
)
from src.config_models import ModelMetadata, FeatureEngineeringConfig, ModelParamsConfig, EvaluationMetrics
from src.exceptions import ModelRegistrationError, ModelNotFoundError

# --- Fixtures ---

@pytest.fixture
def dummy_models_dir(tmp_path: Path) -> Path:
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    return models_root

@pytest.fixture
def model_registry_file(dummy_models_dir: Path) -> Path:
    return dummy_models_dir / "model_registry.jsonl"

def create_dummy_meta_file(
    base_models_dir: Path,
    model_name: str, 
    version: str, 
    timestamp_utc_str: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    feature_config: Dict[str, Any],
    model_filename: str = "model.joblib",
    notes: Optional[str] = None
) -> Path:
    version_dir = base_models_dir / model_name / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    meta_content = ModelMetadata(
        model_name_from_config=model_name,
        model_version=version,
        timestamp_utc=timestamp_utc_str,
        model_path=str(version_dir / model_filename), # Relative to project root conceptually
        scaler_path=str(version_dir / "scaler.pkl") if "scaler" in params else None, # Example
        feature_importance_path=str(version_dir / "feature_importance.png") if params.get("create_fi_plot", False) else None,
        primary_metric_name=next(iter(metrics)) if metrics else "accuracy", # First metric key
        primary_metric_value=next(iter(metrics.values())) if metrics else 0.0,
        evaluation_metrics=EvaluationMetrics(**metrics),
        model_params=ModelParamsConfig.model_validate(params), # Using model_validate for Pydantic v2
        feature_engineering_config=FeatureEngineeringConfig.model_validate(feature_config),
        training_data_stats={"shape": [1000, 10], "description": "Dummy training data stats"},
        tags={"stage": "development", "task": "classification"},
        notes=notes or f"Notes for {model_name} {version}"
    )
    
    meta_file_path = version_dir / "model.meta.json"
    with open(meta_file_path, 'w') as f:
        f.write(meta_content.model_dump_json(indent=4)) # Pydantic v2
    
    # Create dummy model file
    (version_dir / model_filename).touch()
    if meta_content.scaler_path:
        Path(meta_content.scaler_path).touch()

    return meta_file_path


@pytest.fixture
def populated_model_registry(dummy_models_dir: Path, model_registry_file: Path) -> Path:
    """Creates a model registry with a few sample models and versions."""
    registry_entries = []

    # Model A
    ts_a1_str = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    meta_a1_path = create_dummy_meta_file(
        dummy_models_dir, "ModelA", "v1.0.0_abc1", ts_a1_str,
        {"accuracy": 0.85, "f1_score": 0.82},
        {"model_type": "XGBoost", "n_estimators": 100, "learning_rate": 0.1},
        {"input_data_path": "data/raw/a.parquet", "output_features_path": "data/proc/a_f.parquet", "technical_indicators": True}
    )
    registry_entries.append(ModelMetadata.model_validate_json(meta_a1_path.read_text()).model_dump(mode='json'))

    ts_a2_str = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    meta_a2_path = create_dummy_meta_file(
        dummy_models_dir, "ModelA", "v1.1.0_def2", ts_a2_str,
        {"accuracy": 0.88, "f1_score": 0.85, "roc_auc": 0.92},
        {"model_type": "XGBoost", "n_estimators": 150, "learning_rate": 0.05, "max_depth": 5},
        {"input_data_path": "data/raw/a.parquet", "output_features_path": "data/proc/a_f.parquet", "technical_indicators": True, "rolling_lag_features": True}
    )
    registry_entries.append(ModelMetadata.model_validate_json(meta_a2_path.read_text()).model_dump(mode='json'))

    # Model B
    ts_b1_str = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    meta_b1_path = create_dummy_meta_file(
        dummy_models_dir, "ModelB", "v0.9.0_xyz3", ts_b1_str,
        {"loss": 0.05, "val_accuracy": 0.91},
        {"model_type": "LSTM", "units": 50, "epochs": 10, "scaler": "MinMaxScaler"},
        {"input_data_path": "data/raw/b.parquet", "output_features_path": "data/proc/b_f.parquet", "sequence_length": 20}
    )
    registry_entries.append(ModelMetadata.model_validate_json(meta_b1_path.read_text()).model_dump(mode='json'))
    
    # Model A again, but older (should not be latest)
    ts_a0_str = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    meta_a0_path = create_dummy_meta_file(
        dummy_models_dir, "ModelA", "v0.9.5_ghi0", ts_a0_str,
        {"accuracy": 0.80, "f1_score": 0.78},
        {"model_type": "XGBoost", "n_estimators": 50, "learning_rate": 0.2},
        {"input_data_path": "data/raw/a_old.parquet", "output_features_path": "data/proc/a_old_f.parquet", "technical_indicators": False}
    )
    registry_entries.append(ModelMetadata.model_validate_json(meta_a0_path.read_text()).model_dump(mode='json'))


    with open(model_registry_file, "w") as f:
        for entry_dict in registry_entries:
            # The registry stores the ModelMetadata dict itself, not ModelRegistryEntry
            f.write(json.dumps(entry_dict) + "\n")
            
    return dummy_models_dir


# --- Tests for compare_model_versions ---

def test_compare_model_versions_basic_diff(populated_model_registry: Path):
    model_name = "ModelA"
    versions_to_compare = ["v1.0.0_abc1", "v1.1.0_def2"]
    
    # Need to tell utils where the registry is
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        comparison_result = compare_model_versions(model_name, versions_to_compare)

    assert model_name in comparison_result
    assert len(comparison_result[model_name]["versions_compared"]) == 2
    
    diff = comparison_result[model_name]["differences"]
    assert "evaluation_metrics" in diff
    assert diff["evaluation_metrics"]["v1.0.0_abc1"]["accuracy"] == 0.85
    assert diff["evaluation_metrics"]["v1.1.0_def2"]["accuracy"] == 0.88
    assert "roc_auc" in diff["evaluation_metrics"]["v1.1.0_def2"] # Present in v1.1.0, not v1.0.0
    
    assert "model_params" in diff
    assert diff["model_params"]["v1.0.0_abc1"]["n_estimators"] == 100
    assert diff["model_params"]["v1.1.0_def2"]["n_estimators"] == 150
    assert "max_depth" in diff["model_params"]["v1.1.0_def2"]

    assert "feature_engineering_config" in diff
    assert "rolling_lag_features" not in diff["feature_engineering_config"]["v1.0.0_abc1"] # or False
    assert diff["feature_engineering_config"]["v1.1.0_def2"]["rolling_lag_features"] is True

def test_compare_model_versions_no_diff(populated_model_registry: Path):
    # To test no diff, we'd need two identical meta files for different versions,
    # or compare a version against itself (though util might prevent this).
    # Let's register one version and try to compare it to itself.
    model_name = "ModelB"
    versions_to_compare = ["v0.9.0_xyz3", "v0.9.0_xyz3"] # Comparing same version

    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        with pytest.raises(ValueError, match="At least two unique versions are required for comparison."):
             compare_model_versions(model_name, versions_to_compare) # Should raise error or return specific no-diff result

    # Test with two versions that are programmatically made identical (except version string)
    ts_b2_str = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    # Create a new meta file for ModelB, v0.9.1, but with identical content to v0.9.0_xyz3
    meta_b1_content = ModelMetadata.model_validate_json((populated_model_registry / "ModelB" / "v0.9.0_xyz3" / "model.meta.json").read_text())
    
    create_dummy_meta_file(
        populated_model_registry, "ModelB", "v0.9.1_xyz4", ts_b2_str,
        meta_b1_content.evaluation_metrics.model_dump(),
        meta_b1_content.model_params.model_dump(),
        meta_b1_content.feature_engineering_config.model_dump(),
        notes=meta_b1_content.notes
    )
    # Manually add to registry for this test
    with open(populated_model_registry / "model_registry.jsonl", "a") as f:
        new_entry = meta_b1_content.model_copy(update={"model_version": "v0.9.1_xyz4", "timestamp_utc": ts_b2_str})
        new_entry.model_path = str(populated_model_registry / "ModelB" / "v0.9.1_xyz4" / "model.joblib") # Adjust path
        f.write(new_entry.model_dump_json() + "\n")

    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        comparison_result_no_diff = compare_model_versions("ModelB", ["v0.9.0_xyz3", "v0.9.1_xyz4"])
    
    assert "ModelB" in comparison_result_no_diff
    # Expect 'differences' to be empty or indicate no substantive changes
    # The current compare_model_versions always shows full values, so "no diff" means identical values.
    # We'd need to check if the values for each key under 'differences' are the same for both versions.
    # For simplicity here, we'll just check that the structure is there.
    # A more advanced diff function would only show actual changes.
    assert "evaluation_metrics" in comparison_result_no_diff["ModelB"]["differences"]
    # Check that the values are indeed the same, indicating no functional difference for these fields
    metrics_v1 = comparison_result_no_diff["ModelB"]["differences"]["evaluation_metrics"]["v0.9.0_xyz3"]
    metrics_v2 = comparison_result_no_diff["ModelB"]["differences"]["evaluation_metrics"]["v0.9.1_xyz4"]
    # Exclude 'model_version' and 'timestamp_utc' from direct comparison of dicts if they are part of metrics dict
    # Or ensure the comparison logic in the util itself is what's being tested.
    # The current compare_model_versions just lists all values, so they should be identical if "no diff".
    assert metrics_v1 == metrics_v2
    # Similarly for other compared fields like model_params, feature_engineering_config

def test_compare_model_versions_three_versions(populated_model_registry: Path):
    model_name = "ModelA"
    versions_to_compare = ["v0.9.5_ghi0", "v1.0.0_abc1", "v1.1.0_def2"]
    
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        comparison_result = compare_model_versions(model_name, versions_to_compare)

    assert model_name in comparison_result
    assert len(comparison_result[model_name]["versions_compared"]) == 3
    assert "v0.9.5_ghi0" in comparison_result[model_name]["differences"]["evaluation_metrics"]
    assert "v1.0.0_abc1" in comparison_result[model_name]["differences"]["evaluation_metrics"]
    assert "v1.1.0_def2" in comparison_result[model_name]["differences"]["evaluation_metrics"]

def test_compare_model_versions_non_existent_model(populated_model_registry: Path):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        # The compare_model_versions function itself calls get_model_details, which raises ModelNotFoundError.
        # This test ensures that ModelNotFoundError is indeed raised when the model doesn't exist.
        with pytest.raises(ModelNotFoundError, match="Model 'NonExistentModel' not found in registry."):
            compare_model_versions("NonExistentModel", ["v1", "v2"])


def test_compare_model_versions_non_existent_version(populated_model_registry: Path):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        with pytest.raises(ModelNotFoundError, match="Version 'vNonExistent' not found for model 'ModelA'."):
            compare_model_versions("ModelA", ["v1.0.0_abc1", "vNonExistent"])

def test_compare_model_versions_malformed_meta_file(populated_model_registry: Path):
    model_name = "ModelA"
    version_good = "v1.0.0_abc1"
    version_bad_meta = "v_bad_meta" # We'll create this one with malformed JSON

    # Create a malformed meta file
    bad_meta_dir = populated_model_registry / model_name / version_bad_meta
    bad_meta_dir.mkdir(parents=True, exist_ok=True)
    (bad_meta_dir / "model.joblib").touch()
    with open(bad_meta_dir / "model.meta.json", "w") as f:
        f.write("{'this_is': 'not_valid_json_because_of_single_quotes',")
    
    # Add to registry (even if meta is bad, registry might list it)
    with open(populated_model_registry / "model_registry.jsonl", "a") as f:
        entry_dict = {
            "model_name_from_config": model_name, "model_version": version_bad_meta, 
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(bad_meta_dir / "model.joblib"),
            "meta_json_path": str(bad_meta_dir / "model.meta.json")
        }
        f.write(json.dumps(entry_dict) + "\n")

    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        # get_model_details (called by compare_model_versions) should handle this.
        # It might return None for the bad version, or raise an error.
        # If it returns None, compare_model_versions might then raise ModelNotFoundError for the version.
        # Let's assume it logs an error and effectively treats the version as not found or unloadable.
        with pytest.raises(ModelNotFoundError, match=f"Version '{version_bad_meta}' for model '{model_name}' could not be loaded or its metadata is invalid."):
             compare_model_versions(model_name, [version_good, version_bad_meta])


# --- Tests for CLI 'models compare' command ---
from click.testing import CliRunner
from main import cli as main_cli

@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()

def test_cli_models_compare_success(populated_model_registry: Path, cli_runner: CliRunner):
    model_name = "ModelA"
    versions = ["v1.0.0_abc1", "v1.1.0_def2"]
    
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'compare', model_name] + versions)
    
    assert result.exit_code == 0, f"CLI Error: {result.output}"
    output_json = json.loads(result.output)
    assert model_name in output_json
    assert len(output_json[model_name]["versions_compared"]) == 2
    assert "evaluation_metrics" in output_json[model_name]["differences"]

def test_cli_models_compare_insufficient_versions(populated_model_registry: Path, cli_runner: CliRunner):
    model_name = "ModelA"
    versions = ["v1.0.0_abc1"] # Only one version
    
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'compare', model_name] + versions)
    
    assert result.exit_code != 0 # Should fail as CLI expects at least two versions
    assert "At least two versions required" in result.output

def test_cli_models_compare_non_existent_model(populated_model_registry: Path, cli_runner: CliRunner):
    model_name = "NonExistentModel"
    versions = ["v1", "v2"]
    
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'compare', model_name] + versions)
        
    assert result.exit_code != 0 # Expect non-zero exit due to ModelNotFoundError in backend
    # The CLI command's error handling (from main.py refactor) should catch this.
    # Check for a message indicating model not found or error during operation.
    assert "Error" in result.output or "not found" in result.output.lower()


# Placeholder for further tests for other utils if this file is meant for all registry utils
# For now, focusing on compare_model_versions

# --- Tests for other model registry utils ---

def test_register_model_success(dummy_models_dir: Path, model_registry_file: Path):
    model_name = "NewModel"
    version = "v1.0_test"
    ts_now_str = datetime.now(timezone.utc).isoformat()
    
    # Create a dummy meta file first, as register_model reads from it
    # (or it can take ModelMetadata object directly, depending on design)
    # Current register_model takes a ModelMetadata object.
    
    version_dir = dummy_models_dir / model_name / version
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path_dummy = version_dir / "model.dat"
    model_path_dummy.touch()

    metadata_obj = ModelMetadata(
        model_name_from_config=model_name,
        model_version=version,
        timestamp_utc=ts_now_str,
        model_path=str(model_path_dummy),
        evaluation_metrics=EvaluationMetrics(accuracy=0.99),
        model_params=ModelParamsConfig.model_validate({"type": "test", "param": 1}),
        feature_engineering_config=FeatureEngineeringConfig.model_validate({
            "input_data_path": "dummy_in.parquet", "output_features_path": "dummy_out.parquet"
        })
    )
    
    with patch('src.model_registry_utils.MODELS_DIR', dummy_models_dir), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        
        # Ensure meta file is saved where register_model expects it, if it reads it.
        # Or, if register_model saves it, that's part of what's tested.
        # The current register_model expects the meta_json_path to be set and file to exist.
        # Let's simulate that:
        meta_file_for_register = version_dir / "model.meta.json" # This is the path register_model will try to use
        with open(meta_file_for_register, "w") as f:
            f.write(metadata_obj.model_dump_json())
        
        # Update metadata_obj to point to this expected meta_json_path for the registry entry
        metadata_obj.meta_json_path = str(meta_file_for_register)


        registered_meta = register_model(metadata_obj)

    assert registered_meta.model_version == version
    assert model_registry_file.exists()
    
    with open(model_registry_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    registered_entry_dict = json.loads(lines[0])
    assert registered_entry_dict["model_name_from_config"] == model_name
    assert registered_entry_dict["model_version"] == version
    assert registered_entry_dict["meta_json_path"] == str(meta_file_for_register)


def test_register_model_already_exists_error(populated_model_registry: Path, model_registry_file: Path):
    # Try to register ModelA v1.0.0_abc1 again
    existing_meta_path = populated_model_registry / "ModelA" / "v1.0.0_abc1" / "model.meta.json"
    metadata_obj = ModelMetadata.model_validate_json(existing_meta_path.read_text())
    metadata_obj.meta_json_path = str(existing_meta_path) # Ensure this is set for the check

    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        with pytest.raises(ModelRegistrationError, match="already registered"):
            register_model(metadata_obj)

def test_get_model_details_success(populated_model_registry: Path):
    model_name = "ModelA"
    version = "v1.1.0_def2"
    expected_meta_path = populated_model_registry / model_name / version / "model.meta.json"
    expected_metadata = ModelMetadata.model_validate_json(expected_meta_path.read_text())

    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        details = get_model_details(model_name, version)
    
    assert details is not None
    # Compare relevant fields, ModelMetadata objects should be comparable if Pydantic models
    assert details["model_name_from_config"] == expected_metadata.model_name_from_config
    assert details["model_version"] == expected_metadata.model_version
    assert details["evaluation_metrics"]["accuracy"] == expected_metadata.evaluation_metrics.accuracy

def test_get_model_details_model_not_found(populated_model_registry: Path):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        with pytest.raises(ModelNotFoundError, match="Model 'NonExistent' not found in registry."):
            get_model_details("NonExistent", "v1.0")

def test_get_model_details_version_not_found(populated_model_registry: Path):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        with pytest.raises(ModelNotFoundError, match="Version 'vNonExistent' not found for model 'ModelA'."):
            get_model_details("ModelA", "vNonExistent")

def test_get_model_details_meta_file_missing(populated_model_registry: Path, model_registry_file: Path):
    model_name = "ModelC_NoMeta"
    version = "v1.0"
    
    # Add an entry to registry but don't create the meta file
    entry_dict = {
        "model_name_from_config": model_name, "model_version": version, 
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(populated_model_registry / model_name / version / "model.dat"), # Dummy path
        "meta_json_path": str(populated_model_registry / model_name / version / "model.meta.json") # This file won't exist
    }
    with open(model_registry_file, "a") as f:
        f.write(json.dumps(entry_dict) + "\n")
    
    (populated_model_registry / model_name / version).mkdir(parents=True, exist_ok=True)
    (populated_model_registry / model_name / version / "model.dat").touch()


    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        # Expect get_model_details to log an error and return None or raise specific error
        # Current implementation of get_model_details returns None if meta_json_path is None or file not found
        # and logs an error. If it returns None, subsequent code (like compare) might raise ModelNotFound.
        # For direct call, it should return None.
        details = get_model_details(model_name, version)
        assert details is None # Or assert that a specific error is logged if that's the contract

def test_list_models_populated(populated_model_registry: Path, model_registry_file: Path):
    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        models = list_models()
    
    assert len(models) >= 3 # Based on populated_model_registry fixture (ModelA has 3, ModelB has 1)
    model_names_found = {m['model_name_from_config'] for m in models}
    assert "ModelA" in model_names_found
    assert "ModelB" in model_names_found

def test_list_models_filter(populated_model_registry: Path, model_registry_file: Path):
    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        models_a = list_models(model_name_filter="ModelA")
    assert len(models_a) == 3 # ModelA has three versions in fixture
    assert all(m['model_name_from_config'] == "ModelA" for m in models_a)

def test_list_models_empty_registry(dummy_models_dir: Path, model_registry_file: Path):
    # Ensure registry file is empty or doesn't exist
    if model_registry_file.exists(): model_registry_file.unlink()
    model_registry_file.touch() # Create empty file

    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        models = list_models()
    assert models == []

def test_list_models_malformed_registry(dummy_models_dir: Path, model_registry_file: Path):
    with open(model_registry_file, "w") as f:
        f.write("this is not json\n")
        # Add a valid line to see if it's skipped or if the whole thing fails
        valid_entry = {"model_name_from_config": "GoodModel", "model_version": "v1"}
        f.write(json.dumps(valid_entry) + "\n")

    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        models = list_models() # Should skip the malformed line and log error
    
    assert len(models) == 1 
    assert models[0]['model_name_from_config'] == "GoodModel"


def test_get_latest_model_version_path_success(populated_model_registry: Path, model_registry_file: Path):
    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        latest_meta_path_a = get_latest_model_version_path("ModelA")
        latest_meta_path_b = get_latest_model_version_path("ModelB")

    assert latest_meta_path_a is not None
    assert "ModelA" in str(latest_meta_path_a)
    assert "v1.1.0_def2" in str(latest_meta_path_a) # This is the latest by timestamp

    assert latest_meta_path_b is not None
    assert "ModelB" in str(latest_meta_path_b)
    assert "v0.9.0_xyz3" in str(latest_meta_path_b)

def test_get_latest_model_version_path_not_found(populated_model_registry: Path, model_registry_file: Path):
    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        latest_path = get_latest_model_version_path("NonExistentModelForPath")
    assert latest_path is None


# --- CLI Tests for Model Registry ---

def test_cli_models_list_all(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'list'])
    assert result.exit_code == 0
    assert "ModelA" in result.output
    assert "v1.1.0_def2" in result.output
    assert "ModelB" in result.output
    assert "v0.9.0_xyz3" in result.output

def test_cli_models_list_filtered(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'list', '--name', 'ModelB'])
    assert result.exit_code == 0
    assert "ModelA" not in result.output
    assert "ModelB" in result.output
    assert "v0.9.0_xyz3" in result.output


def test_cli_models_describe_success(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'describe', 'ModelA', 'v1.1.0_def2'])
    assert result.exit_code == 0
    output_json = json.loads(result.output)
    assert output_json["model_name_from_config"] == "ModelA"
    assert output_json["model_version"] == "v1.1.0_def2"
    assert output_json["evaluation_metrics"]["accuracy"] == 0.88

def test_cli_models_describe_not_found(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'describe', 'ModelA', 'vNonExistent'])
    assert result.exit_code != 0 # Handled by main.py's error handling
    assert "Model version not found" in result.output or "Error" in result.output


def test_cli_models_get_latest_path_success(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'get-latest-path', 'ModelA'])
    assert result.exit_code == 0
    assert "ModelA" in result.output
    assert "v1.1.0_def2" in result.output # Latest version of ModelA
    assert "model.meta.json" in result.output

def test_cli_models_get_latest_path_not_found(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'get-latest-path', 'NonExistentForPath'])
    assert result.exit_code != 0 # Handled by main.py
    assert "Model not found" in result.output or "Error" in result.output

# --- Tests for other model registry utils ---

def test_register_model_success(dummy_models_dir: Path, model_registry_file: Path):
    model_name = "NewModel"
    version = "v1.0_test"
    ts_now_str = datetime.now(timezone.utc).isoformat()
    
    # Create a dummy meta file first, as register_model reads from it
    # (or it can take ModelMetadata object directly, depending on design)
    # Current register_model takes a ModelMetadata object.
    
    version_dir = dummy_models_dir / model_name / version
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path_dummy = version_dir / "model.dat"
    model_path_dummy.touch()

    metadata_obj = ModelMetadata(
        model_name_from_config=model_name,
        model_version=version,
        timestamp_utc=ts_now_str,
        model_path=str(model_path_dummy),
        evaluation_metrics=EvaluationMetrics(accuracy=0.99),
        model_params=ModelParamsConfig.model_validate({"type": "test", "param": 1}),
        feature_engineering_config=FeatureEngineeringConfig.model_validate({
            "input_data_path": "dummy_in.parquet", "output_features_path": "dummy_out.parquet"
        })
    )
    
    with patch('src.model_registry_utils.MODELS_DIR', dummy_models_dir), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        
        # Ensure meta file is saved where register_model expects it, if it reads it.
        # Or, if register_model saves it, that's part of what's tested.
        # The current register_model expects the meta_json_path to be set and file to exist.
        # Let's simulate that:
        meta_file_for_register = version_dir / "model.meta.json" # This is the path register_model will try to use
        with open(meta_file_for_register, "w") as f:
            f.write(metadata_obj.model_dump_json())
        
        # Update metadata_obj to point to this expected meta_json_path for the registry entry
        metadata_obj.meta_json_path = str(meta_file_for_register)


        registered_meta = register_model(metadata_obj)

    assert registered_meta.model_version == version
    assert model_registry_file.exists()
    
    with open(model_registry_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    registered_entry_dict = json.loads(lines[0])
    assert registered_entry_dict["model_name_from_config"] == model_name
    assert registered_entry_dict["model_version"] == version
    assert registered_entry_dict["meta_json_path"] == str(meta_file_for_register)


def test_register_model_already_exists_error(populated_model_registry: Path, model_registry_file: Path):
    # Try to register ModelA v1.0.0_abc1 again
    existing_meta_path = populated_model_registry / "ModelA" / "v1.0.0_abc1" / "model.meta.json"
    metadata_obj = ModelMetadata.model_validate_json(existing_meta_path.read_text())
    metadata_obj.meta_json_path = str(existing_meta_path) # Ensure this is set for the check

    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        with pytest.raises(ModelRegistrationError, match="already registered"):
            register_model(metadata_obj)

def test_get_model_details_success(populated_model_registry: Path):
    model_name = "ModelA"
    version = "v1.1.0_def2"
    expected_meta_path = populated_model_registry / model_name / version / "model.meta.json"
    expected_metadata = ModelMetadata.model_validate_json(expected_meta_path.read_text())

    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        details = get_model_details(model_name, version)
    
    assert details is not None
    # Compare relevant fields, ModelMetadata objects should be comparable if Pydantic models
    assert details["model_name_from_config"] == expected_metadata.model_name_from_config
    assert details["model_version"] == expected_metadata.model_version
    assert details["evaluation_metrics"]["accuracy"] == expected_metadata.evaluation_metrics.accuracy

def test_get_model_details_model_not_found(populated_model_registry: Path):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        with pytest.raises(ModelNotFoundError, match="Model 'NonExistent' not found in registry."):
            get_model_details("NonExistent", "v1.0")

def test_get_model_details_version_not_found(populated_model_registry: Path):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        with pytest.raises(ModelNotFoundError, match="Version 'vNonExistent' not found for model 'ModelA'."):
            get_model_details("ModelA", "vNonExistent")

def test_get_model_details_meta_file_missing(populated_model_registry: Path, model_registry_file: Path):
    model_name = "ModelC_NoMeta"
    version = "v1.0"
    
    # Add an entry to registry but don't create the meta file
    entry_dict = {
        "model_name_from_config": model_name, "model_version": version, 
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(populated_model_registry / model_name / version / "model.dat"), # Dummy path
        "meta_json_path": str(populated_model_registry / model_name / version / "model.meta.json") # This file won't exist
    }
    with open(model_registry_file, "a") as f:
        f.write(json.dumps(entry_dict) + "\n")
    
    (populated_model_registry / model_name / version).mkdir(parents=True, exist_ok=True)
    (populated_model_registry / model_name / version / "model.dat").touch()


    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        # Expect get_model_details to log an error and return None or raise specific error
        # Current implementation of get_model_details returns None if meta_json_path is None or file not found
        # and logs an error. If it returns None, subsequent code (like compare) might raise ModelNotFound.
        # For direct call, it should return None.
        details = get_model_details(model_name, version)
        assert details is None # Or assert that a specific error is logged if that's the contract

def test_list_models_populated(populated_model_registry: Path, model_registry_file: Path):
    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        models = list_models()
    
    assert len(models) >= 3 # Based on populated_model_registry fixture (ModelA has 3, ModelB has 1)
    model_names_found = {m['model_name_from_config'] for m in models}
    assert "ModelA" in model_names_found
    assert "ModelB" in model_names_found

def test_list_models_filter(populated_model_registry: Path, model_registry_file: Path):
    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        models_a = list_models(model_name_filter="ModelA")
    assert len(models_a) == 3 # ModelA has three versions in fixture
    assert all(m['model_name_from_config'] == "ModelA" for m in models_a)

def test_list_models_empty_registry(dummy_models_dir: Path, model_registry_file: Path):
    # Ensure registry file is empty or doesn't exist
    if model_registry_file.exists(): model_registry_file.unlink()
    model_registry_file.touch() # Create empty file

    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        models = list_models()
    assert models == []

def test_list_models_malformed_registry(dummy_models_dir: Path, model_registry_file: Path):
    with open(model_registry_file, "w") as f:
        f.write("this is not json\n")
        # Add a valid line to see if it's skipped or if the whole thing fails
        valid_entry = {"model_name_from_config": "GoodModel", "model_version": "v1"}
        f.write(json.dumps(valid_entry) + "\n")

    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        models = list_models() # Should skip the malformed line and log error
    
    assert len(models) == 1 
    assert models[0]['model_name_from_config'] == "GoodModel"


def test_get_latest_model_version_path_success(populated_model_registry: Path, model_registry_file: Path):
    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        latest_meta_path_a = get_latest_model_version_path("ModelA")
        latest_meta_path_b = get_latest_model_version_path("ModelB")

    assert latest_meta_path_a is not None
    assert "ModelA" in str(latest_meta_path_a)
    assert "v1.1.0_def2" in str(latest_meta_path_a) # This is the latest by timestamp

    assert latest_meta_path_b is not None
    assert "ModelB" in str(latest_meta_path_b)
    assert "v0.9.0_xyz3" in str(latest_meta_path_b)

def test_get_latest_model_version_path_not_found(populated_model_registry: Path, model_registry_file: Path):
    with patch('src.model_registry_utils.MODEL_REGISTRY_FILE', model_registry_file):
        latest_path = get_latest_model_version_path("NonExistentModelForPath")
    assert latest_path is None


# --- CLI Tests for Model Registry ---

def test_cli_models_list_all(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'list'])
    assert result.exit_code == 0
    assert "ModelA" in result.output
    assert "v1.1.0_def2" in result.output
    assert "ModelB" in result.output
    assert "v0.9.0_xyz3" in result.output

def test_cli_models_list_filtered(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'list', '--name', 'ModelB'])
    assert result.exit_code == 0
    assert "ModelA" not in result.output
    assert "ModelB" in result.output
    assert "v0.9.0_xyz3" in result.output


def test_cli_models_describe_success(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'describe', 'ModelA', 'v1.1.0_def2'])
    assert result.exit_code == 0
    output_json = json.loads(result.output)
    assert output_json["model_name_from_config"] == "ModelA"
    assert output_json["model_version"] == "v1.1.0_def2"
    assert output_json["evaluation_metrics"]["accuracy"] == 0.88

def test_cli_models_describe_not_found(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'describe', 'ModelA', 'vNonExistent'])
    assert result.exit_code != 0 # Handled by main.py's error handling
    assert "Model version not found" in result.output or "Error" in result.output


def test_cli_models_get_latest_path_success(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'get-latest-path', 'ModelA'])
    assert result.exit_code == 0
    assert "ModelA" in result.output
    assert "v1.1.0_def2" in result.output # Latest version of ModelA
    assert "model.meta.json" in result.output

def test_cli_models_get_latest_path_not_found(populated_model_registry: Path, cli_runner: CliRunner):
    with patch('src.model_registry_utils.MODELS_DIR', populated_model_registry), \
         patch('src.model_registry_utils.MODEL_REGISTRY_FILE', populated_model_registry / "model_registry.jsonl"):
        result = cli_runner.invoke(main_cli, ['models', 'get-latest-path', 'NonExistentForPath'])
    assert result.exit_code != 0 # Handled by main.py
    assert "Model not found" in result.output or "Error" in result.output
```
