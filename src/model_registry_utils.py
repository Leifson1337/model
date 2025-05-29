# src/model_registry_utils.py
import json
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import fcntl # For file locking (Unix-like systems)
import time # For retry logic with locking
from collections import defaultdict # For compare_model_versions

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_REGISTRY_BASE_DIR = Path("models") # Base directory where versioned models are stored
MODEL_REGISTRY_FILE = MODEL_REGISTRY_BASE_DIR / "model_registry.jsonl"

# Ensure the base models directory exists
MODEL_REGISTRY_BASE_DIR.mkdir(parents=True, exist_ok=True)


def _acquire_lock(lock_file_path: Path, timeout: int = 10):
    start_time = time.time()
    while True:
        try:
            lock_file = open(lock_file_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file 
        except (IOError, BlockingIOError): 
            if time.time() - start_time >= timeout:
                logger.error(f"Timeout acquiring lock for {lock_file_path}")
                return None 
            time.sleep(0.1) 
        except Exception as e:
            logger.error(f"Unexpected error acquiring lock for {lock_file_path}: {e}")
            return None

def _release_lock(lock_file_obj):
    if lock_file_obj:
        fcntl.flock(lock_file_obj.fileno(), fcntl.LOCK_UN)
        lock_file_obj.close()

def register_model(meta_json_path: str) -> bool:
    lock_file_path = MODEL_REGISTRY_FILE.with_suffix(".lock")
    lock_file_obj = None
    try:
        meta_path_obj = Path(meta_json_path).resolve()
        if not meta_path_obj.exists() or not meta_path_obj.is_file():
            logger.error(f"Metadata JSON file not found: {meta_json_path}")
            return False

        with open(meta_path_obj, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        model_name = metadata.get("model_name_from_config")
        model_version = metadata.get("model_version")
        timestamp_utc = metadata.get("timestamp_utc")
        metrics_data = metadata.get("metrics", {})
        feature_importance_file = metadata.get("feature_importance_file") # Get new field
        
        primary_metric_name, primary_metric_value = None, None
        if "accuracy" in metrics_data: primary_metric_name, primary_metric_value = "accuracy", metrics_data["accuracy"]
        elif "f1_score" in metrics_data: primary_metric_name, primary_metric_value = "f1_score", metrics_data["f1_score"]
        elif "auc" in metrics_data: primary_metric_name, primary_metric_value = "auc", metrics_data["auc"]
        elif metrics_data: first_key = next(iter(metrics_data)); primary_metric_name, primary_metric_value = first_key, metrics_data[first_key]
        
        try:
            project_root_candidate = MODEL_REGISTRY_BASE_DIR.resolve().parent
            relative_meta_path = str(meta_path_obj.relative_to(project_root_candidate))
        except ValueError: relative_meta_path = meta_json_path 

        if not all([model_name, model_version, timestamp_utc]):
            logger.error(f"Missing essential fields in metadata: {meta_json_path}"); return False

        registry_entry = {
            "model_name_from_config": model_name, "model_version": model_version,
            "timestamp_utc": timestamp_utc, "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value, "meta_json_path": relative_meta_path,
            "has_feature_importance": bool(feature_importance_file) # Add boolean flag
        }
        
        lock_file_obj = _acquire_lock(lock_file_path)
        if not lock_file_obj: logger.error(f"Could not acquire lock. Registration failed for {meta_json_path}."); return False

        with open(MODEL_REGISTRY_FILE, 'a+', encoding='utf-8') as f: f.write(json.dumps(registry_entry) + '\n')
        logger.info(f"Model '{model_name}' v'{model_version}' registered from '{relative_meta_path}' (FI: {registry_entry['has_feature_importance']}).")
        return True
    except Exception as e: logger.error(f"Failed to register model from {meta_json_path}: {e}"); return False
    finally:
        if lock_file_obj: _release_lock(lock_file_obj)
        if lock_file_path.exists():
            try: os.remove(lock_file_path)
            except OSError: pass

def list_models(model_name_from_config: Optional[str] = None) -> List[Dict[str, Any]]:
    if not MODEL_REGISTRY_FILE.exists(): logger.info("Registry file not found."); return []
    models = []
    try:
        with open(MODEL_REGISTRY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if model_name_from_config:
                        if entry.get("model_name_from_config") == model_name_from_config: models.append(entry)
                    else: models.append(entry)
                except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line: {line.strip()}")
        return models
    except Exception as e: logger.error(f"Error reading model registry: {e}"); return []

def get_model_details(model_name_from_config: str, version: str) -> Optional[Dict[str, Any]]:
    models = list_models(model_name_from_config=model_name_from_config)
    for entry in models:
        if entry.get("model_version") == version:
            meta_json_path_str = entry.get("meta_json_path")
            if not meta_json_path_str: logger.error(f"No meta_json_path for {model_name_from_config} v{version}"); return None
            project_root = MODEL_REGISTRY_BASE_DIR.resolve().parent 
            full_meta_path = (project_root / meta_json_path_str).resolve()
            if not full_meta_path.exists(): logger.error(f"Metadata file {full_meta_path} not found."); return None
            try:
                with open(full_meta_path, 'r', encoding='utf-8') as f: return json.load(f)
            except Exception as e: logger.error(f"Error reading metadata {full_meta_path}: {e}"); return None
    logger.warning(f"Model '{model_name_from_config}' version '{version}' not found."); return None

def get_latest_model_version_path(model_name_from_config: str) -> Optional[str]:
    models = list_models(model_name_from_config=model_name_from_config)
    if not models: logger.info(f"No versions for model '{model_name_from_config}'."); return None
    try:
        latest = sorted(models, key=lambda x: x.get("model_version", ""), reverse=True)[0]
        return latest.get("meta_json_path")
    except Exception as e: logger.error(f"Error sorting versions for {model_name_from_config}: {e}"); return None

def compare_model_versions(model_name_from_config: str, versions_to_compare: List[str]) -> Dict[str, Any]:
    comparison_output = {"model_name": model_name_from_config, "versions_compared": [], "comparison_data": defaultdict(lambda: defaultdict(dict)), "errors": {}}
    all_metadata = {}
    for version_str in versions_to_compare:
        details = get_model_details(model_name_from_config, version_str)
        if details: all_metadata[version_str] = details; comparison_output["versions_compared"].append(version_str)
        else: comparison_output["errors"][version_str] = "Metadata not found or error loading."; logger.warning(f"Could not retrieve metadata for {model_name_from_config} v {version_str}.")
    if not all_metadata: logger.info(f"No valid metadata for versions of {model_name_from_config}."); return comparison_output
    
    for version, meta in all_metadata.items():
        for metric_name, metric_value in meta.get("metrics", {}).items():
            comparison_output["comparison_data"]["metrics"][metric_name][version] = metric_value
        fc = meta.get("feature_config_used", {}); features_list = fc.get("features_used", [])
        comparison_output["comparison_data"]["feature_config"]["features_used_count"][version] = len(features_list)
        comparison_output["comparison_data"]["feature_config"]["features_sample"][version] = features_list[:5] if features_list else "N/A"
        comparison_output["comparison_data"]["training_config_file"]["path"][version] = meta.get("training_config_file", "N/A")
        # Added feature_importance_file to comparison
        comparison_output["comparison_data"]["feature_importance_file"]["path"][version] = meta.get("feature_importance_file", "N/A")

        repro = meta.get("reproducibility_details", {})
        comparison_output["comparison_data"]["reproducibility"]["git_commit_hash"][version] = repro.get("git_commit_hash", "N/A")
        comparison_output["comparison_data"]["reproducibility"]["repro_hash"][version] = repro.get("reproducibility_hash_sha256", "N/A")
        lib_versions = repro.get("library_versions", {})
        comparison_output["comparison_data"]["reproducibility"]["versions_pandas"][version] = lib_versions.get("pandas", "N/A")
        comparison_output["comparison_data"]["reproducibility"]["versions_sklearn"][version] = lib_versions.get("scikit-learn", "N/A")

    comparison_output["comparison_data"] = {k: dict(v) for k, v in comparison_output["comparison_data"].items()}
    for main_key, sub_dict in comparison_output["comparison_data"].items():
        comparison_output["comparison_data"][main_key] = {k: dict(v) for k, v in sub_dict.items()}
    return comparison_output

if __name__ == '__main__':
    print("--- Testing Model Registry Utilities (with feature_importance handling) ---")
    TEST_MODELS_BASE = Path("temp_registry_test_models_main_fi")
    _orig_mrbd, _orig_mrf = MODEL_REGISTRY_BASE_DIR, MODEL_REGISTRY_FILE
    MODEL_REGISTRY_BASE_DIR, MODEL_REGISTRY_FILE = TEST_MODELS_BASE, TEST_MODELS_BASE / "model_registry.jsonl"
    MODEL_REGISTRY_BASE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using temporary registry file: {MODEL_REGISTRY_FILE}")

    project_root_for_test = TEST_MODELS_BASE.resolve().parent
    
    dummy_meta_templates_for_fi_test = [
        { # Version 1 - with feature importance
            "model_name_from_config": "fi_test_model", "model_version": "v20230301000000_fi1",
            "timestamp_utc": "2023-03-01T00:00:00Z", "metrics": {"accuracy": 0.80},
            "model_filename_for_meta": "model_v1_fi.joblib",
            "feature_config_used": {"features_used": ["A", "B"]},
            "training_config_file": "training_config_v1.json",
            "feature_importance_file": "feature_importances_v1.json", # Has FI
            "reproducibility_details": {"git_commit_hash": "fi_git1"}
        },
        { # Version 2 - without feature importance
            "model_name_from_config": "fi_test_model", "model_version": "v20230302000000_fi2",
            "timestamp_utc": "2023-03-02T00:00:00Z", "metrics": {"accuracy": 0.85},
            "model_filename_for_meta": "model_v2_nofi.joblib",
            "feature_config_used": {"features_used": ["A", "B", "C"]},
            "training_config_file": "training_config_v2.json",
            # No feature_importance_file field
            "reproducibility_details": {"git_commit_hash": "fi_git2"}
        }
    ]
    
    meta_paths_to_register_fi_test = []
    for content_template in dummy_meta_templates_for_fi_test:
        content = content_template.copy()
        model_version_dir = MODEL_REGISTRY_BASE_DIR / content["model_name_from_config"] / content["model_version"]
        model_version_dir.mkdir(parents=True, exist_ok=True)
        actual_model_file_path = model_version_dir / content["model_filename_for_meta"]
        actual_meta_file_path = model_version_dir / "model.meta.json" 
        with open(actual_model_file_path, "w") as f_model: f_model.write("dummy")
        content["model_filepath_full"] = str(actual_model_file_path.resolve())
        content["model_filename"] = content["model_filename_for_meta"]
        with open(actual_meta_file_path, 'w') as f: json.dump(content, f, indent=2)
        meta_paths_to_register_fi_test.append(str(actual_meta_file_path))

    try:
        print("\n1. Testing register_model with feature importance field...")
        for path in meta_paths_to_register_fi_test:
            success = register_model(path)
            print(f"Registered '{Path(path).name}' from '{Path(path).parent.parent.name}/{Path(path).parent.name}': {success}")
            assert success
        
        assert MODEL_REGISTRY_FILE.exists()
        with open(MODEL_REGISTRY_FILE, 'r') as f: lines = f.readlines()
        assert len(lines) == len(dummy_meta_templates_for_fi_test)
        
        registered_entries = [json.loads(line) for line in lines]
        assert registered_entries[0]["has_feature_importance"] is True
        assert registered_entries[1]["has_feature_importance"] is False
        print("Registry entries correctly reflect 'has_feature_importance'.")

        print("\n2. Testing list_models output...")
        listed_models = list_models("fi_test_model")
        assert len(listed_models) == 2
        assert listed_models[0]["has_feature_importance"] is True
        assert listed_models[1]["has_feature_importance"] is False
        print(f"Listed models for 'fi_test_model':")
        for entry in listed_models: print(f"  - {entry['model_version']}, Has FI: {entry['has_feature_importance']}")

        print("\n3. Testing compare_model_versions with feature importance field...")
        comparison = compare_model_versions("fi_test_model", ["v20230301000000_fi1", "v20230302000000_fi2"])
        print("Comparison Result for FI test:")
        print(json.dumps(comparison, indent=2))
        assert comparison["comparison_data"]["feature_importance_file"]["path"]["v20230301000000_fi1"] == "feature_importances_v1.json"
        assert comparison["comparison_data"]["feature_importance_file"]["path"]["v20230302000000_fi2"] == "N/A"
        print("compare_model_versions correctly shows feature_importance_file presence.")

    finally:
        print("\nCleaning up test files and directories for FI test...")
        if MODEL_REGISTRY_FILE.exists(): MODEL_REGISTRY_FILE.unlink()
        lock_file = MODEL_REGISTRY_FILE.with_suffix(".lock")
        if lock_file.exists(): lock_file.unlink(missing_ok=True)
        for i, path_str in enumerate(meta_paths_to_register_fi_test):
            meta_file = Path(path_str)
            model_filename = dummy_meta_templates_for_fi_test[i]["model_filename_for_meta"]
            model_file = meta_file.parent / model_filename
            if model_file.exists(): model_file.unlink(missing_ok=True)
            if meta_file.exists(): meta_file.unlink(missing_ok=True)
            try:
                if meta_file.parent.exists() and not list(meta_file.parent.iterdir()): meta_file.parent.rmdir() 
                if meta_file.parent.parent.exists() and not list(meta_file.parent.parent.iterdir()): meta_file.parent.parent.rmdir()
            except OSError as e: print(f"Warning: Cleanup error {meta_file.parent.parent if meta_file.parent.parent else meta_file.parent}: {e}")
        if TEST_MODELS_BASE.exists():
            is_empty_or_hidden = not any(item.name for item in TEST_MODELS_BASE.iterdir() if not item.name.startswith('.'))
            if is_empty_or_hidden:
                try: TEST_MODELS_BASE.rmdir(); print(f"Cleaned up: {TEST_MODELS_BASE}")
                except OSError as e: print(f"Warning: Could not remove {TEST_MODELS_BASE}: {e}.")
            else: print(f"{TEST_MODELS_BASE} not empty/hidden, not removed.")
        else: print(f"{TEST_MODELS_BASE} already removed.")
        MODEL_REGISTRY_BASE_DIR, MODEL_REGISTRY_FILE = _orig_mrbd, _orig_mrf
    
    print("\n--- Model Registry Utilities Tests (including FI) Completed ---")
