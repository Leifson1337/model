# src/pipeline_utils.py
import os
import subprocess
from datetime import datetime
import logging # Added for logging

# Configure logging for this module (or use a shared logger if available)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_model_version_str() -> str:
    """
    Generates a model version string: v<YYYYMMDDHHMMSS>_<short_git_hash>.
    Uses "nogit" as fallback if Git hash cannot be obtained.
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    git_hash_short = "nogit"  # Default fallback

    try:
        # Ensure the command is run where .git directory is expected (e.g., project root)
        # If this script is in src/, .git is usually ../.git
        # For simplicity, assume `git` command works from current working dir if it's project root.
        # If running from a different CWD, this might need adjustment (e.g. specify cwd in subprocess.run)
        
        # Check if current directory is a git repository
        check_git_repo_cmd = ['git', 'rev-parse', '--is-inside-work-tree']
        is_git_repo_proc = subprocess.run(check_git_repo_cmd, capture_output=True, text=True, check=False)

        if is_git_repo_proc.returncode == 0 and is_git_repo_proc.stdout.strip() == "true":
            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            process = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to handle non-zero exit
            
            if process.returncode == 0:
                git_hash_short = process.stdout.strip()
            else:
                logger.warning(f"Failed to get Git hash: {process.stderr.strip()}. Using fallback '{git_hash_short}'.")
        else:
            logger.warning("Not inside a Git work tree or `git` command issue. Using fallback 'nogit'.")
            if is_git_repo_proc.stderr:
                 logger.warning(f"Git check stderr: {is_git_repo_proc.stderr.strip()}")


    except FileNotFoundError:
        logger.warning("Git command not found. Using fallback 'nogit'.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting Git hash: {e}. Using fallback 'nogit'.")
        
    return f"v{timestamp}_{git_hash_short}"


def get_versioned_model_paths(base_path: str, model_type_or_name: str, version_str: str, model_filename: str) -> tuple[str, str]:
    """
    Constructs versioned paths for saving a model and its artifacts.

    Args:
        base_path: Base directory for model output (e.g., "models/experiment_group_A").
                   This typically comes from TrainModelConfig.model_output_path_base.
        model_type_or_name: Specific type or name of the model (e.g., "XGBoost", "LSTM_attention_v2").
                            This will be used as a subdirectory under base_path.
        version_str: The generated version string (e.g., from get_model_version_str()).
        model_filename: The actual filename of the model (e.g., "model.joblib", "model.h5").

    Returns:
        A tuple containing:
            - full_model_file_path (str): The complete path for the model file.
            - versioned_directory_path (str): The path to the directory containing this versioned model
                                              (e.g., for saving metadata or other artifacts).
    """
    # Ensure model_type_or_name and version_str are valid for directory names
    # Basic sanitization (can be expanded if needed)
    safe_model_type_or_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in model_type_or_name)
    safe_version_str = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in version_str)

    versioned_directory_path = os.path.join(base_path, safe_model_type_or_name, safe_version_str)
    
    try:
        os.makedirs(versioned_directory_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {versioned_directory_path}")
    except OSError as e:
        logger.error(f"Could not create directory {versioned_directory_path}: {e}")
        # Depending on desired fault tolerance, could raise error or return None paths
        raise # For now, re-raise as path creation is critical

    full_model_file_path = os.path.join(versioned_directory_path, model_filename)
    
    return full_model_file_path, versioned_directory_path


if __name__ == '__main__':
    # Test get_model_version_str()
    print("--- Testing get_model_version_str() ---")
    version_string = get_model_version_str()
    print(f"Generated version string: {version_string}")
    assert "v" in version_string
    assert "_" in version_string
    # Check if git hash part is reasonable (alphanumeric or "nogit")
    git_part = version_string.split('_')[-1]
    assert git_part == "nogit" or git_part.isalnum()

    # Test get_versioned_model_paths()
    print("\n--- Testing get_versioned_model_paths() ---")
    base = "test_models_output"
    model_type = "MyCoolModel/TypeA" # With special char to test sanitization
    version = get_model_version_str() # Use a real version string
    filename = "model_weights.h5"

    full_path, dir_path = get_versioned_model_paths(base, model_type, version, filename)
    print(f"Base path: {base}")
    print(f"Model type/name: {model_type}")
    print(f"Version string: {version}")
    print(f"Model filename: {filename}")
    print(f"Full model path: {full_path}")
    print(f"Versioned directory path: {dir_path}")

    assert os.path.exists(dir_path) # Check if directory was created
    expected_safe_model_type = "MyCoolModel_TypeA"
    assert expected_safe_model_type in full_path
    assert version in full_path # Version string itself is already somewhat sanitized by its generator
    assert filename in full_path
    assert base in full_path

    # Clean up test directory
    if os.path.exists(dir_path):
        # Clean up the content first if any, then remove dirs.
        # For this test, just removing the created directory structure.
        try:
            os.remove(full_path) # if a dummy file was created for some reason (not in this test)
        except OSError:
            pass
        os.rmdir(dir_path) # Remove version_str dir
        os.rmdir(os.path.join(base, expected_safe_model_type)) # Remove model_type_or_name dir
        if not os.listdir(base): # Remove base if empty
             os.rmdir(base)
    print(f"\nCleaned up test directory: {base}")
    
    print("\n--- Tests for pipeline_utils.py completed ---")
