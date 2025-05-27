import json
import hashlib
import os
from datetime import datetime, timezone

def calculate_file_hash(filepath: str, hash_algorithm: str = 'sha256') -> str:
    """Calculates the hash of a file."""
    hasher = hashlib.new(hash_algorithm)
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192): # Read in 8KB chunks
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_model_metadata(model_filepath: str, metrics: dict, feature_config: dict, reproducibility_hash_placeholder: str = "placeholder_repro_hash_123") -> str:
    """
    Generates and saves metadata for a given model file.

    Args:
        model_filepath: Path to the model file.
        metrics: Dictionary of model performance metrics.
        feature_config: Dictionary of feature configuration used for the model.
        reproducibility_hash_placeholder: Placeholder for the reproducibility hash.

    Returns:
        Path to the created metadata JSON file.
    """
    # TODO: Input validation for arguments using Pydantic models or validate_call if appropriate.
    #       - model_filepath: Check if it's a valid path string (Pydantic's FilePath would do this).
    #       - metrics: Validate structure (e.g., dict with specific keys like 'accuracy', 'f1_score').
    #                  Consider a Pydantic model for metrics.
    #       - feature_config: Validate structure (e.g., dict with 'features_used', 'preprocessing_steps').
    #                         Ideally, this would be a dump of a FeatureEngineeringConfig Pydantic model.
    #       - reproducibility_hash_placeholder: Ensure it's a string.

    if not os.path.exists(model_filepath):
        # TODO: Log this error.
        raise FileNotFoundError(f"Model file not found: {model_filepath}")

    try:
        model_hash = calculate_file_hash(model_filepath)
        timestamp = datetime.now(timezone.utc).isoformat() # ISO 8601 format

        # TODO: Enhance reproducibility_hash generation.
        # Current `reproducibility_hash_placeholder` is just a placeholder.
        # A more robust hash could include:
        # 1. Hash of the training configuration (e.g., TrainModelConfig instance from src.config_models).
        #    - Convert Pydantic model to a canonical JSON string (sort keys), then hash.
        # 2. Hash of feature engineering configuration (e.g., FeatureEngineeringConfig).
        # 3. Hashes of any scalers or transformers used (e.g., hash of the saved scaler file).
        # 4. Versions of key libraries (e.g., sklearn, tensorflow, pandas, numpy, ta, yfinance).
        #    - This can be collected using pkg_resources or importlib.metadata.
        # 5. Hash of the specific dataset URI or a hash of the data itself if feasible (for small datasets).
        # 6. Git commit hash of the codebase version used for training.
        # Example stub for a more complex hash function is at the end of this file.
        
        # TODO: Standardize the structure of `metrics` and `feature_config` using Pydantic models from `src.config_models`
        #       For example, feature_config should ideally be `feature_config_model.model_dump()`
        
        metadata = {
            "model_name": os.path.basename(base), # Added model name
            "model_filepath_relative": model_filepath, # Store relative path if appropriate for portability
            "model_filepath_absolute": os.path.abspath(model_filepath),
            "timestamp_utc": timestamp,
            "model_content_hash": {"algorithm": "sha256", "hash_value": model_hash},
            "metrics": metrics, 
            "feature_config_used": feature_config, 
            "training_config_reference": "placeholder_path_to_actual_training_config.json", # TODO: Link to actual training config file or its hash
            "reproducibility_details": {
                "current_reproducibility_hash": reproducibility_hash_placeholder, # This is the placeholder passed in
                "notes": "This is a basic hash. See TODOs in metadata_utils.py for enhancement ideas like hashing configs and lib versions.",
                # "data_source_version_or_hash": "...", # Future
                # "code_version_git_commit": "...", # Future
                # "library_versions_snapshot": {"sklearn": "x.y.z", ...}, # Future
                # "full_pipeline_reproducibility_hash": "...", # Future: A hash combining all above elements
            },
            "schema_version_metadata": "1.0.0" # Version of this metadata schema itself
        }

        base, ext = os.path.splitext(model_filepath)
        # Ensure .meta.json is added correctly, even if original model has no extension or multiple dots.
        metadata_filepath = f"{base}.meta.json"
        if ext.lower() == ".json" and base.endswith(".meta"): # Avoid model.meta.meta.json
             metadata_filepath = f"{os.path.splitext(base)[0]}.meta.json"


        # Ensure the directory for the metadata file exists (should be same as model)
        os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)

        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # TODO: Log successful metadata creation.
        print(f"Metadata successfully generated and saved to: {metadata_filepath}")
        return metadata_filepath

    except FileNotFoundError: # Already handled above for model_filepath, but good for completeness if other files are involved.
        # TODO: Log specific file not found if different from model_filepath
        raise
    except Exception as e:
        # TODO: Log this error.
        print(f"Error generating model metadata for {model_filepath}: {e}")
        # Consider raising a custom exception for metadata errors or returning a structured error object.
        raise # Or return None / handle as per fault tolerance strategy for the pipeline step.


# TODO: Stub for enhanced reproducibility hash generation
# def generate_enhanced_reproducibility_hash(
#       training_config: TrainModelConfig, # Pydantic model
#       feature_config: FeatureEngineeringConfig, # Pydantic model
#       # scaler_object: Optional[Any] = None, # Or path to saved scaler
#       # data_hash: Optional[str] = None, 
#       # code_version_git_commit: Optional[str] = None
#   ) -> str:
#     """
#     Generates a comprehensive reproducibility hash.
#     Combines hashes of model config, feature config, scalers, library versions, etc.
#     """
#     import hashlib
#     import json
#     # import importlib.metadata
# 
#     hasher = hashlib.sha256()
# 
#     # 1. Training Config
#     hasher.update(training_config.model_dump_json(sort_keys=True).encode())
# 
#     # 2. Feature Config
#     hasher.update(feature_config.model_dump_json(sort_keys=True).encode())
# 
#     # 3. Scaler (if applicable) - hash its parameters or the saved file
#     # if scaler_object and hasattr(scaler_object, 'get_params'):
#     #     hasher.update(json.dumps(scaler_object.get_params(), sort_keys=True).encode())
# 
#     # 4. Library Versions (example for a few key ones)
#     # key_libraries = ["scikit-learn", "pandas", "numpy", "tensorflow", "xgboost", "lightgbm", "catboost", "ta"]
#     # lib_versions = {}
#     # for lib in key_libraries:
#     #     try:
#     #         lib_versions[lib] = importlib.metadata.version(lib)
#     #     except importlib.metadata.PackageNotFoundError:
#     #         lib_versions[lib] = "not_found"
#     # hasher.update(json.dumps(lib_versions, sort_keys=True).encode())
# 
#     # 5. Data Hash (if provided)
#     # if data_hash:
#     #     hasher.update(data_hash.encode())
# 
#     # 6. Code Version (Git commit hash, if provided)
#     # if code_version_git_commit:
#     #     hasher.update(code_version_git_commit.encode())
# 
#     return hasher.hexdigest()


if __name__ == '__main__':
    # Example Usage (for testing within this script)
    print("Running example usage of generate_model_metadata...")

    # Ensure models directory exists
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    print(f"Ensured '{models_dir}' directory exists.")

    # Create a dummy model file
    dummy_model_path = os.path.join(models_dir, "sample_model.joblib")
    with open(dummy_model_path, 'w') as f:
        f.write("This is a dummy model file for testing metadata generation.")
    print(f"Created dummy model file: {dummy_model_path}")

    # Define sample metrics and feature config
    sample_metrics = {"accuracy": 0.95, "f1_score": 0.92, "precision": 0.90, "recall": 0.94}
    sample_feature_config = {
        "features_used": ["age", "income", "education_level"],
        "preprocessing_steps": ["scaling_standard", "one_hot_encode_categorical"],
        "hyperparameters_for_feature_selection": {"k_best": 10}
    }
    custom_repro_hash = "custom_repro_hash_for_sample_model_v1.0"

    print(f"\nGenerating metadata for: {dummy_model_path}")
    print(f"With metrics: {sample_metrics}")
    print(f"With feature_config: {sample_feature_config}")
    print(f"With custom reproducibility_hash: {custom_repro_hash}")

    try:
        metadata_file_path = generate_model_metadata(
            dummy_model_path,
            sample_metrics,
            sample_feature_config,
            reproducibility_hash_placeholder=custom_repro_hash
        )
        print(f"\nSuccessfully generated metadata file: {metadata_file_path}")

        # Verify content
        print("\nVerifying content of the metadata file...")
        with open(metadata_file_path, 'r') as f:
            generated_metadata = json.load(f)

        print(f"Metadata content:\n{json.dumps(generated_metadata, indent=4)}")

        # Check some key fields
        if generated_metadata["model_filepath"] == os.path.abspath(dummy_model_path) and \
           generated_metadata["metrics"]["accuracy"] == 0.95 and \
           generated_metadata["feature_config"]["features_used"][0] == "age" and \
           generated_metadata["reproducibility_hash"] == custom_repro_hash and \
           "model_hash" in generated_metadata and \
           "timestamp" in generated_metadata:
            print("\nContent verification successful: Key fields match expected values.")
        else:
            print("\nContent verification failed: Some key fields do not match expected values.")
            # Detailed comparison for debugging
            if generated_metadata["model_filepath"] != os.path.abspath(dummy_model_path):
                print(f"  Mismatch in model_filepath: Expected '{os.path.abspath(dummy_model_path)}', Got '{generated_metadata['model_filepath']}'")
            if generated_metadata["metrics"]["accuracy"] != 0.95:
                 print(f"  Mismatch in metrics.accuracy: Expected 0.95, Got {generated_metadata['metrics']['accuracy']}")
            if generated_metadata["feature_config"]["features_used"][0] != "age":
                 print(f"  Mismatch in feature_config.features_used[0]: Expected 'age', Got {generated_metadata['feature_config']['features_used'][0]}")
            if generated_metadata["reproducibility_hash"] != custom_repro_hash:
                 print(f"  Mismatch in reproducibility_hash: Expected '{custom_repro_hash}', Got '{generated_metadata['reproducibility_hash']}'")


    except Exception as e:
        print(f"\nAn error occurred during metadata generation or verification: {e}")

    # Test with a non-existent model file
    print("\nTesting with a non-existent model file...")
    try:
        generate_model_metadata("models/non_existent_model.pkl", {}, {})
    except FileNotFoundError as e:
        print(f"Successfully caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    print("\nExample usage completed.")
