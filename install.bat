@echo off
echo Starting QLOP Project installation for Windows...

REM --- Configuration ---
set PYTHON_CMD=python
set VENV_DIR=venv_qlop
set SKIP_VENV=0

REM --- Helper Functions (Batch doesn't have easy functions like shell scripts) ---
REM For simplicity, error checking will be inline.

REM --- Argument Parsing (Basic) ---
REM This is a simplified argument parsing for batch.
REM For more complex parsing, PowerShell or external tools might be better.
if "%1"=="--skip-venv" set SKIP_VENV=1
if "%1"=="--python" set PYTHON_CMD=%2

REM --- Check for Python ---
echo Checking for Python using command: %PYTHON_CMD% --version
%PYTHON_CMD% --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: %PYTHON_CMD% is not available or not in PATH. Please install Python 3.9+.
    goto :eof
)
echo Using Python:
%PYTHON_CMD% --version

REM --- Virtual Environment Setup ---
if %SKIP_VENV% equ 0 (
    if exist "%VENV_DIR%" (
        echo Virtual environment '%VENV_DIR%' already exists. Using it.
    ) else (
        echo Creating virtual environment in '%VENV_DIR%'...
        %PYTHON_CMD% -m venv "%VENV_DIR%"
        if %errorlevel% neq 0 (
            echo ERROR: Failed to create virtual environment.
            goto :eof
        )
        echo SUCCESS: Virtual environment created.
    )

    echo Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to activate virtual environment.
        goto :eof
    )
) else (
    echo Skipping virtual environment creation as per --skip-venv.
    if "%VIRTUAL_ENV%"=="" (
        echo WARNING: You are installing packages globally. It is highly recommended to use a virtual environment.
    ) else (
        echo Using existing active virtual environment: %VIRTUAL_ENV%
    )
)

REM --- Install Dependencies ---
echo Installing/upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip
echo Installing dependencies from requirements.txt...
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies from requirements.txt.
    if %SKIP_VENV% equ 0 (
        call "%VENV_DIR%\Scripts\deactivate.bat"
        echo Deactivated virtual environment due to error.
    )
    goto :eof
)
echo SUCCESS: Dependencies installed.

REM --- Install Project in Editable Mode with Dev Extras ---
echo Installing project in editable mode with [dev] extras...
%PYTHON_CMD% -m pip install -e .[dev]
if %errorlevel% neq 0 (
    echo ERROR: Failed to install project in editable mode with [dev] extras.
    if %SKIP_VENV% equ 0 (
        call "%VENV_DIR%\Scripts\deactivate.bat"
        echo Deactivated virtual environment due to error.
    )
    goto :eof
)
echo SUCCESS: Project installed in editable mode with [dev] extras.

REM --- Basic Verification ---
echo Running basic verification (python main.py --help)...
%PYTHON_CMD% main.py --help > nul
if %errorlevel% neq 0 (
    echo ERROR: Verification step (python main.py --help) failed.
    if %SKIP_VENV% equ 0 (
        call "%VENV_DIR%\Scripts\deactivate.bat"
        echo Deactivated virtual environment due to error.
    )
    goto :eof
)
echo SUCCESS: Basic verification (main.py --help) passed.

echo Running basic verification (run_tests.py)...
REM run_tests.py installs pytest if not present.
%PYTHON_CMD% run_tests.py
if %errorlevel% neq 0 (
    echo ERROR: Verification step (run_tests.py) failed. Check output above for details.
    if %SKIP_VENV% equ 0 (
        call "%VENV_DIR%\Scripts\deactivate.bat"
        echo Deactivated virtual environment due to error.
    )
    goto :eof
)
echo SUCCESS: run_tests.py completed (check its output for specific test results).


echo.
echo SUCCESS: QLOP Project installation and basic verification completed!
if %SKIP_VENV% equ 0 (
    echo To deactivate the virtual environment, run: call "%VENV_DIR%\Scripts\deactivate.bat"
    echo To use the project, activate the virtual environment: call "%VENV_DIR%\Scripts\activate.bat"
)
echo You can now use the 'qlop-cli' command or 'python main.py'.

:eof
echo.
