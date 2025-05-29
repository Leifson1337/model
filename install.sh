#!/bin/bash
# Installation script for Linux/macOS

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u

echo "Starting QLOP Project installation..."

# --- Configuration ---
PYTHON_CMD_DEFAULT="python3" # Default Python command
VENV_DIR=".venv"       # Common name for virtual environment directory
SKIP_VENV=0
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=9

# --- Helper Functions ---
check_command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_success() {
    echo -e "\033[0;32mSUCCESS: $1\033[0m"
}

print_error() {
    echo -e "\033[0;31mERROR: $1\033[0m"
}

print_warning() {
    echo -e "\033[0;33mWARNING: $1\033[0m"
}

# --- Argument Parsing ---
PYTHON_CMD="${PYTHON_CMD_DEFAULT}" # Set PYTHON_CMD to default, can be overridden by args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-venv) SKIP_VENV=1 ;;
        --python-cmd)
            if [[ -z "${2-}" ]]; then print_error "Error: --python-cmd requires a non-empty argument."; exit 1; fi
            PYTHON_CMD="$2"; shift ;;
        --venv-dir)
            if [[ -z "${2-}" ]]; then print_error "Error: --venv-dir requires a non-empty argument."; exit 1; fi
            VENV_DIR="$2"; shift ;;
        *) print_warning "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Dependency Checks ---
# Python
if ! check_command_exists "$PYTHON_CMD"; then
    print_error "$PYTHON_CMD is not available. Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ and ensure it's in your PATH (or use --python-cmd)."
    exit 1
fi
PY_VERSION_FULL=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
PY_MAJOR=$(echo "$PY_VERSION_FULL" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION_FULL" | cut -d. -f2)

if ! (("$PY_MAJOR" > "$MIN_PYTHON_MAJOR")) && ! (("$PY_MAJOR" == "$MIN_PYTHON_MAJOR" && "$PY_MINOR" >= "$MIN_PYTHON_MINOR")); then
    print_error "Python version $PY_VERSION_FULL found. Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+."
    exit 1
fi
print_success "Python check passed: $($PYTHON_CMD --version)"

# Pip
if ! "$PYTHON_CMD" -m pip --version > /dev/null 2>&1; then
    print_error "pip is not available for $PYTHON_CMD. Please ensure pip is installed for your Python environment."
    exit 1
fi
print_success "pip check passed: ($($PYTHON_CMD -m pip --version))"

# Docker
if ! check_command_exists docker; then
    print_warning "Docker is not found. Some functionalities like building or running Docker containers will not be available."
    print_warning "Please visit https://docs.docker.com/get-docker/ to install Docker."
    # Not exiting, as Docker might not be critical for all users.
else
    print_success "Docker check passed: $(docker --version)"
fi

# --- Virtual Environment Setup ---
ACTIVATED_VENV=0
if [ "$SKIP_VENV" -eq 0 ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment '$VENV_DIR' already exists."
        read -r -p "Use existing virtual environment '$VENV_DIR'? (y/n): " USE_EXISTING_VENV
        if [[ "$USE_EXISTING_VENV" != "y" && "$USE_EXISTING_VENV" != "Y" ]]; then
            print_error "User chose not to use existing virtual environment. Please remove or rename '$VENV_DIR' or use --skip-venv."
            exit 1
        fi
    else
        echo "Creating virtual environment in '$VENV_DIR'..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        print_success "Virtual environment created."
    fi

    echo "Activating virtual environment: $VENV_DIR/bin/activate"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    ACTIVATED_VENV=1 # Mark that we activated a venv we might need to deactivate on error
else
    echo "Skipping virtual environment creation as per --skip-venv."
    if [[ -z "${VIRTUAL_ENV-}" ]]; then # Check if VIRTUAL_ENV is unset or empty
        print_warning "You are installing packages globally because --skip-venv was used and no virtual environment is active."
        read -r -p "Are you sure you want to proceed with global installation? (y/n): " CONFIRM_GLOBAL
        if [[ "$CONFIRM_GLOBAL" != "y" && "$CONFIRM_GLOBAL" != "Y" ]]; then
            print_error "Installation aborted by user."
            exit 1
        fi
    else
        echo "Using existing active virtual environment: $VIRTUAL_ENV"
    fi
fi
PYTHON_IN_VENV_CMD="$PYTHON_CMD" # If venv is used, this will be the venv python
PIP_IN_VENV_CMD="$PYTHON_CMD -m pip"

# --- Install Dependencies ---
echo "Upgrading pip..."
$PIP_IN_VENV_CMD install --upgrade pip

echo "Installing dependencies from requirements.txt..."
if ! $PIP_IN_VENV_CMD install -r requirements.txt; then
    print_error "Failed to install dependencies from requirements.txt."
    [ "$ACTIVATED_VENV" -eq 1 ] && deactivate && echo "Deactivated virtual environment due to error."
    exit 1
fi
print_success "Dependencies installed."

# --- Install Project in Editable Mode with Dev Extras ---
echo "Installing project in editable mode with [dev] extras..."
if ! $PIP_IN_VENV_CMD install -e ".[dev]"; then
    print_error "Failed to install project in editable mode with [dev] extras."
    [ "$ACTIVATED_VENV" -eq 1 ] && deactivate && echo "Deactivated virtual environment due to error."
    exit 1
fi
print_success "Project installed in editable mode with [dev] extras."

# --- Basic Verification ---
echo "Running basic verification (main.py --help)..."
if ! "$PYTHON_IN_VENV_CMD" main.py --help > /dev/null; then
    print_error "Verification step (main.py --help) failed."
    [ "$ACTIVATED_VENV" -eq 1 ] && deactivate && echo "Deactivated virtual environment due to error."
    exit 1
fi
print_success "Basic verification (main.py --help) passed."

echo "Running a quick smoke test using pytest..."
# Ensure pytest is available (should be installed with [dev] extras)
if ! $PIP_IN_VENV_CMD show pytest > /dev/null 2>&1; then
    echo "pytest not found, installing it for smoke test..."
    $PIP_IN_VENV_CMD install pytest
fi
# Run a specific, simple test that should always pass if setup is correct.
# Assuming test_example.py has a test named test_example_success
SMOKE_TEST_SELECTOR="tests/test_example.py::test_example_success" 
# If test_example.py doesn't exist or doesn't have this test, this will fail.
# A more robust way might be to have a dedicated smoke test tag.
echo "Attempting to run smoke test: pytest -k test_example_success"
if ! "$PYTHON_IN_VENV_CMD" -m pytest -k "test_example_success" tests/test_example.py; then
    print_warning "Smoke test (pytest -k test_example_success) failed or test not found. This might indicate issues with the test setup or core functionality."
    print_warning "Continuing installation, but please review test output."
    # Not exiting on smoke test failure to allow installation to complete, but user should be warned.
else
    print_success "Smoke test passed."
fi


print_success "QLOP Project installation and basic verification completed!"
if [ "$SKIP_VENV" -eq 0 ] && [ "$ACTIVATED_VENV" -eq 1 ]; then
    echo "The virtual environment '$VENV_DIR' is currently active."
    echo "To deactivate it, run: deactivate"
    echo "To use the project later, activate the virtual environment: source $VENV_DIR/bin/activate"
elif [ "$SKIP_VENV" -eq 0 ] && [ "$ACTIVATED_VENV" -eq 0 ]; then # Used existing venv
     echo "Remember to activate your virtual environment '$VENV_DIR' if not already active: source $VENV_DIR/bin/activate"
fi
echo "You can now use the 'qlop-cli' command (if installed by setup.py) or '$PYTHON_IN_VENV_CMD main.py'."
