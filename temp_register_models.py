import sys
sys.path.append('.') # Add project root to sys.path for src imports
from src.model_registry_utils import register_model, MODEL_REGISTRY_FILE
from pathlib import Path

def main():
    # Ensure the registry file is initially empty for a clean test
    if MODEL_REGISTRY_FILE.exists():
        MODEL_REGISTRY_FILE.unlink()
        print(f"Removed existing registry file: {MODEL_REGISTRY_FILE}")
    MODEL_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)


    meta_paths_to_register = [
        "models/XGBoostTest/v20230101000000_abc123/model.meta.json",
        "models/XGBoostTest/v20230102000000_def456/model.meta.json",
        "models/LSTMTest/v20230103000000_ghi789/model.meta.json"
    ]

    print("Registering dummy models...")
    for path_str in meta_paths_to_register:
        if Path(path_str).exists():
            success = register_model(path_str)
            print(f"Registered '{path_str}': {success}")
        else:
            print(f"Meta file not found, cannot register: {path_str}")

if __name__ == "__main__":
    main()
