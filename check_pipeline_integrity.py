# check_pipeline_integrity.py
import os
import json

def check_config_files():
    """Checks if essential configuration files exist."""
    print("Checking for configuration files (dev.json, test.json, prod.json)...")
    config_dir = "config"
    expected_files = ["dev.json", "test.json", "prod.json"]
    all_found = True
    if not os.path.isdir(config_dir):
        print(f"Error: Configuration directory '{config_dir}' not found.")
        return False
    
    for f_name in expected_files:
        f_path = os.path.join(config_dir, f_name)
        if os.path.exists(f_path):
            try:
                with open(f_path, 'r') as f:
                    json.load(f) # Try parsing to check if valid JSON
                print(f"  Found and validated: {f_path}")
            except json.JSONDecodeError:
                print(f"  Found but INVALID JSON: {f_path}")
                all_found = False
            except Exception as e:
                print(f"  Error reading {f_path}: {e}")
                all_found = False
        else:
            print(f"  Missing: {f_path}")
            all_found = False
    if all_found:
        print("All essential configuration files found and are valid JSON.")
    else:
        print("Some configuration files are missing or invalid.")
    return all_found

def check_main_script_executable():
    """Checks if main.py exists and is executable (basic check)."""
    print("\nChecking for main.py...")
    main_py_path = "main.py"
    if os.path.exists(main_py_path):
        print(f"  Found: {main_py_path}")
        # In a real CI, you might run `python main.py --help` here
        # For now, just checking existence.
        return True
    else:
        print(f"  Error: {main_py_path} not found.")
        return False

def check_critical_dirs():
    """Checks for the presence of critical directories like src/, api/, tests/."""
    print("\nChecking for critical directories (src/, api/, tests/, models/, data/ (placeholder))...")
    # Adding models/ and data/ as they are often critical, even if data/ is gitignored
    # For data/, we'd typically check for placeholder files or structure if data itself is not in git.
    dirs_to_check = ["src", "api", "tests", "models", "data"] 
    all_found = True
    for d_name in dirs_to_check:
        if os.path.isdir(d_name):
            print(f"  Found directory: {d_name}/")
            if d_name == "data" and not os.listdir(d_name): # Example: check if data has subdirs or .gitkeep
                 # Check for .gitkeep in data/raw and data/processed
                raw_gitkeep = os.path.join(d_name, "raw", ".gitkeep")
                processed_gitkeep = os.path.join(d_name, "processed", ".gitkeep")
                if not (os.path.exists(raw_gitkeep) and os.path.exists(processed_gitkeep)):
                    print(f"  Warning: Directory '{d_name}/' is present but expected .gitkeep files in 'data/raw' or 'data/processed' might be missing.")
                    # This is a soft warning, not a failure for this basic check
        else:
            print(f"  Error: Directory '{d_name}/' not found.")
            all_found = False
    if all_found:
        print("All critical directories found.")
    else:
        print("Some critical directories are missing.")
    return all_found
    
def main():
    print("--- Simulating Pipeline Integrity Check ---")
    
    results = {
        "config_files_ok": False,
        "main_script_ok": False,
        "critical_dirs_ok": False
    }

    results["config_files_ok"] = check_config_files()
    results["main_script_ok"] = check_main_script_executable()
    results["critical_dirs_ok"] = check_critical_dirs()

    # Overall integrity status
    overall_ok = all(results.values())

    if overall_ok:
        print("\nPipeline Integrity Check: PASSED")
    else:
        print("\nPipeline Integrity Check: FAILED")
        # In a real CI, this would exit with a non-zero code
        # For this simulation, we just print the status.
        # sys.exit(1) 

    print("\n--- Integrity Check Simulation Complete ---")

if __name__ == "__main__":
    main()
