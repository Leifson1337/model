# src/metadata_utils.py
import json
import hashlib
import os
from datetime import datetime, timezone
import subprocess
import importlib.metadata
from typing import Optional, Dict, List, Any
from pydantic import BaseModel # For type hinting training_config_obj

# For testing in __main__
from src.config_models import TrainModelConfig, ModelParamsConfig


def calculate_file_hash(filepath: str, hash_algorithm: str = 'sha256') -> str:
    """Calculates the hash of a file."""
    hasher = hashlib.new(hash_algorithm)
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_git_commit_hash() -> Optional[str]:
    """Gets the current Git commit hash (full)."""
    try:
        check_git_repo_cmd = ['git', 'rev-parse', '--is-inside-work-tree']
        is_git_repo_proc = subprocess.run(check_git_repo_cmd, capture_output=True, text=True, check=False)
        if not (is_git_repo_proc.returncode == 0 and is_git_repo_proc.stdout.strip() == "true"):
            print("Not a Git repository or git command issue. Cannot get commit hash.")
            return None
            
        cmd = ['git', 'rev-parse', 'HEAD']
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return process.stdout.strip()
    except FileNotFoundError:
        print("Git command not found. Cannot get commit hash.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error getting Git commit hash: {e.stderr.strip()}")
        return None
    except Exception as e:
        print(f"Unexpected error getting Git commit hash: {e}")
        return None

def get_library_versions(libs: List[str]) -> Dict[str, Optional[str]]:
    """Gets versions of specified Python libraries."""
    versions = {}
    for lib_name in libs:
        try:
            versions[lib_name] = importlib.metadata.version(lib_name)
        except importlib.metadata.PackageNotFoundError:
            versions[lib_name] = "not_found"
        except Exception as e:
            print(f"Error getting version for {lib_name}: {e}")
            versions[lib_name] = "error"
    return versions

def generate_reproducibility_hash(
    model_content_hash: str,
    git_commit_hash_val: Optional[str],
    library_versions_val: Dict[str, Optional[str]],
    training_config_json_str: Optional[str]
) -> str:
    """Generates a reproducibility hash from various components."""
    hasher = hashlib.sha256()
    hasher.update(model_content_hash.encode())
    if git_commit_hash_val:
        hasher.update(git_commit_hash_val.encode())
    
    sorted_lib_versions_json = json.dumps(library_versions_val, sort_keys=True)
    hasher.update(sorted_lib_versions_json.encode())
    
    if training_config_json_str:
        hasher.update(training_config_json_str.encode())
        
    return hasher.hexdigest()


def generate_model_metadata(
    model_filepath: str,
    metrics: Dict[str, Any],
    feature_config: Dict[str, Any], # This can contain 'feature_importance_file' from modeling.py
    model_version: str,
    model_name_from_config: str,
    training_config_obj: Optional[BaseModel] = None,
    feature_importance_artifact: Optional[str] = None # New dedicated parameter
) -> str:
    """
    Generates and saves metadata for a given model file.
    """
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file not found: {model_filepath}")

    model_dir = os.path.dirname(model_filepath)
    os.makedirs(model_dir, exist_ok=True) 

    model_content_hash_val = calculate_file_hash(model_filepath)
    timestamp_utc_val = datetime.now(timezone.utc).isoformat()
    git_commit_hash_val = get_git_commit_hash()
    key_libs = [
        "scikit-learn", "pandas", "numpy", "tensorflow", "xgboost", 
        "lightgbm", "catboost", "prophet", "torch", "joblib", "GitPython",
        "pydantic", "ta", "newsapi-python", "transformers", "optuna", "backtrader"
    ]
    library_versions_val = get_library_versions(key_libs)

    training_config_filepath_relative = None
    training_config_json_for_hash = None
    if training_config_obj:
        training_config_filepath_relative = "training_run_config.json"
        full_training_config_path = os.path.join(model_dir, training_config_filepath_relative)
        try:
            with open(full_training_config_path, 'w') as f:
                json.dump(training_config_obj.model_dump(mode='json'), f, indent=4)
            training_config_dict_for_hash = training_config_obj.model_dump(mode='json')
            training_config_json_for_hash = json.dumps(training_config_dict_for_hash, sort_keys=True)
            print(f"Training configuration saved to: {full_training_config_path}")
        except Exception as e:
            print(f"Error saving training configuration to JSON: {e}")
            training_config_filepath_relative = None
            training_config_json_for_hash = None

    actual_repro_hash = generate_reproducibility_hash(
        model_content_hash=model_content_hash_val,
        git_commit_hash_val=git_commit_hash_val,
        library_versions_val=library_versions_val,
        training_config_json_str=training_config_json_for_hash
    )

    # Remove "feature_importance_file" from feature_config if it exists, as it's now a top-level field
    # (This assumes modeling.py might still pass it via feature_config before being updated)
    feature_importance_path_from_fc = feature_config.pop("feature_importance_file", None)
    # Use dedicated parameter if provided, else fallback to one from feature_config (for transition)
    final_feature_importance_artifact = feature_importance_artifact if feature_importance_artifact is not None else feature_importance_path_from_fc

    metadata = {
        "model_name_from_config": model_name_from_config,
        "model_version": model_version,
        "model_filename": os.path.basename(model_filepath), 
        "model_filepath_full": os.path.abspath(model_filepath), 
        "timestamp_utc": timestamp_utc_val,
        "model_content_hash": {"algorithm": "sha256", "hash_value": model_content_hash_val},
        "metrics": metrics,
        "feature_config_used": feature_config, 
        "training_config_file": training_config_filepath_relative,
        "feature_importance_file": final_feature_importance_artifact, # New top-level field
        "reproducibility_details": {
            "reproducibility_hash_sha256": actual_repro_hash,
            "git_commit_hash": git_commit_hash_val,
            "library_versions": library_versions_val,
            "notes": "Reproducibility hash combines model content, git commit, library versions, and training config."
        },
        "schema_version_metadata": "1.2.0" # Incremented schema version
    }
    # Remove feature_importance_file if it's None to keep metadata clean
    if metadata["feature_importance_file"] is None:
        del metadata["feature_importance_file"]


    base, _ = os.path.splitext(model_filepath)
    metadata_filepath = f"{base}.meta.json"
    
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Enhanced metadata (v1.2.0) successfully generated and saved to: {metadata_filepath}")
    return metadata_filepath


if __name__ == '__main__':
    print("--- Testing Enhanced Model Metadata Generation (v1.2.0) ---")

    MODELS_TEST_DIR = "temp_test_models_metadata"
    MODEL_NAME_CONFIG = "XGBoost" 
    MODEL_VERSION_STR = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}_testgit123"
    
    versioned_model_dir = os.path.join(MODELS_TEST_DIR, MODEL_NAME_CONFIG, MODEL_VERSION_STR)
    os.makedirs(versioned_model_dir, exist_ok=True)
    
    dummy_model_filename = "dummy_model.joblib"
    dummy_model_filepath = os.path.join(versioned_model_dir, dummy_model_filename)
    with open(dummy_model_filepath, 'w') as f:
        f.write("This is a dummy model file for testing metadata generation.")
    print(f"Created dummy model file: {dummy_model_filepath}")

    dummy_train_params = ModelParamsConfig(n_estimators=120, learning_rate=0.08)
    dummy_training_config = TrainModelConfig(
        input_features_path="dummy_path/features.csv", 
        model_output_path_base=MODELS_TEST_DIR, 
        model_type=MODEL_NAME_CONFIG, 
        model_params=dummy_train_params,
        target_column="dummy_target"
    )
    
    sample_metrics = {"accuracy": 0.96, "f1_score": 0.93}
    sample_feature_config = {
        "features_used": ["test_feat1", "test_feat2"],
        "preprocessing_steps": ["scaling_standard_test"]
        # "feature_importance_file": "feature_importances.json" # Old way, will be removed
    }
    sample_feature_importance_file = "feature_importances.json" # New way via dedicated param

    print(f"\nGenerating metadata for: {dummy_model_filepath}")
    metadata_file_path = None # Initialize for finally block
    training_config_on_disk_path = None # Initialize

    try:
        metadata_file_path = generate_model_metadata(
            model_filepath=dummy_model_filepath,
            metrics=sample_metrics,
            feature_config=sample_feature_config.copy(), # Pass a copy as it might be modified
            model_version=MODEL_VERSION_STR,
            model_name_from_config=MODEL_NAME_CONFIG,
            training_config_obj=dummy_training_config,
            feature_importance_artifact=sample_feature_importance_file # Test new parameter
        )
        print(f"\nSuccessfully generated metadata file: {metadata_file_path}")

        print("\nVerifying content of the metadata file...")
        with open(metadata_file_path, 'r') as f:
            generated_metadata = json.load(f)
        print(f"Metadata content:\n{json.dumps(generated_metadata, indent=2)}")

        assert generated_metadata["model_name_from_config"] == MODEL_NAME_CONFIG
        assert generated_metadata["model_version"] == MODEL_VERSION_STR
        assert "reproducibility_hash_sha256" in generated_metadata["reproducibility_details"]
        assert "feature_importance_file" in generated_metadata
        assert generated_metadata["feature_importance_file"] == sample_feature_importance_file
        assert "feature_importance_file" not in generated_metadata["feature_config_used"] # Check it was removed
        
        training_config_on_disk_path = os.path.join(os.path.dirname(metadata_file_path), generated_metadata["training_config_file"])
        assert os.path.exists(training_config_on_disk_path)
        print(f"Verified training_run_config.json exists at: {training_config_on_disk_path}")
        print("Content verification successful.")

    except Exception as e:
        print(f"\nAn error occurred during metadata generation or verification: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(dummy_model_filepath): os.remove(dummy_model_filepath)
        if training_config_on_disk_path and os.path.exists(training_config_on_disk_path): os.remove(training_config_on_disk_path)
        if metadata_file_path and os.path.exists(metadata_file_path): os.remove(metadata_file_path)
        if os.path.exists(versioned_model_dir) and not os.listdir(versioned_model_dir): os.rmdir(versioned_model_dir)
        model_type_dir = os.path.join(MODELS_TEST_DIR, MODEL_NAME_CONFIG)
        if os.path.exists(model_type_dir) and not os.listdir(model_type_dir): os.rmdir(model_type_dir)
        if os.path.exists(MODELS_TEST_DIR) and not os.listdir(MODELS_TEST_DIR): os.rmdir(MODELS_TEST_DIR)
        print(f"Cleaned up test directory: {MODELS_TEST_DIR}")

    print("\n--- Enhanced Metadata Generation Test (v1.2.0) Completed ---")
