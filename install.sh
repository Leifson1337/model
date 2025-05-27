#!/bin/bash
# Installation script for Linux/macOS

echo "Starting QLOP Project installation..."

# --- Configuration ---
PYTHON_CMD="python3" # Change to "python" if python3 is not your default for Python 3
VENV_DIR="venv_qlop"
SKIP_VENV=0

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
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-venv) SKIP_VENV=1 ;;
        --python) PYTHON_CMD="$2"; shift ;;
        *) print_warning "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Check for Python ---
if ! check_command_exists "$PYTHON_CMD"; then
    print_error "$PYTHON_CMD is not available. Please install Python 3.9+ and ensure it's in your PATH."
    exit 1
fi
echo "Using Python: $($PYTHON_CMD --version)"

# --- Virtual Environment Setup ---
if [ "$SKIP_VENV" -eq 0 ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment '$VENV_DIR' already exists. Using it."
    else
        echo "Creating virtual environment in '$VENV_DIR'..."
        if ! "$PYTHON_CMD" -m venv "$VENV_DIR"; then
            print_error "Failed to create virtual environment. Please check your Python venv module."
            exit 1
        fi
        print_success "Virtual environment created."
    fi

    echo "Activating virtual environment..."
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        print_error "Failed to activate virtual environment."
        exit 1
    fi
else
    echo "Skipping virtual environment creation as per --skip-venv."
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "You are installing packages globally. It is highly recommended to use a virtual environment."
    else
        echo "Using existing active virtual environment: $VIRTUAL_ENV"
    fi
fi

# --- Install Dependencies ---
echo "Installing dependencies from requirements.txt..."
"$PYTHON_CMD" -m pip install --upgrade pip
if ! "$PYTHON_CMD" -m pip install -r requirements.txt; then
    print_error "Failed to install dependencies from requirements.txt."
    # Deactivate venv if we created it
    [ "$SKIP_VENV" -eq 0 ] && deactivate && echo "Deactivated virtual environment due to error."
    exit 1
fi
print_success "Dependencies installed."

# --- Install Project in Editable Mode with Dev Extras ---
echo "Installing project in editable mode with [dev] extras..."
if ! "$PYTHON_CMD" -m pip install -e ".[dev]"; then
    print_error "Failed to install project in editable mode with [dev] extras."
    [ "$SKIP_VENV" -eq 0 ] && deactivate && echo "Deactivated virtual environment due to error."
    exit 1
fi
print_success "Project installed in editable mode with [dev] extras."

# --- Basic Verification ---
echo "Running basic verification (python main.py --help)..."
if ! "$PYTHON_CMD" main.py --help > /dev/null; then
    print_error "Verification step (python main.py --help) failed."
    [ "$SKIP_VENV" -eq 0 ] && deactivate && echo "Deactivated virtual environment due to error."
    exit 1
fi
print_success "Basic verification (main.py --help) passed."

echo "Running basic verification (run_tests.py)..."
# Note: run_tests.py installs pytest if not present.
# It will output its own success/failure messages.
if ! "$PYTHON_CMD" run_tests.py; then
    print_error "Verification step (run_tests.py) failed. Check output above for details."
    [ "$SKIP_VENV" -eq 0 ] && deactivate && echo "Deactivated virtual environment due to error."
    exit 1
fi
# run_tests.py prints its own success, so we don't double print here.

print_success "QLOP Project installation and basic verification completed!"
if [ "$SKIP_VENV" -eq 0 ]; then
    echo "To deactivate the virtual environment, run: deactivate"
    echo "To use the project, activate the virtual environment: source $VENV_DIR/bin/activate"
fi
echo "You can now use the 'qlop-cli' command or 'python main.py'."
