@echo off
setlocal enabledelayedexpansion

echo Starting QLOP Project installation for Windows...

REM --- Configuration ---
set PYTHON_CMD_DEFAULT=python
set VENV_DIR=.venv
set SKIP_VENV=0
set MIN_PYTHON_MAJOR=3
set MIN_PYTHON_MINOR=9
set PYTHON_CMD=%PYTHON_CMD_DEFAULT%

REM --- Argument Parsing (Basic) ---
:ParseArgs
IF "%~1"=="" GOTO ArgsDone
IF /I "%~1"=="--skip-venv" (SET SKIP_VENV=1 & SHIFT & GOTO ParseArgs)
IF /I "%~1"=="--python-cmd" (
    IF "%~2"=="" (echo ERROR: --python-cmd requires an argument. & EXIT /B 1)
    SET PYTHON_CMD=%~2
    SHIFT
    SHIFT
    GOTO ParseArgs
)
IF /I "%~1"=="--venv-dir" (
    IF "%~2"=="" (echo ERROR: --venv-dir requires an argument. & EXIT /B 1)
    SET VENV_DIR=%~2
    SHIFT
    SHIFT
    GOTO ParseArgs
)
echo WARNING: Unknown parameter passed: %1
EXIT /B 1
:ArgsDone

REM --- Dependency Checks ---
echo.
echo --- Checking Prerequisites ---

REM Python and Pip
echo Checking for Python %MIN_PYTHON_MAJOR%.%MIN_PYTHON_MINOR%+ ...
%PYTHON_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (%MIN_PYTHON_MAJOR%, %MIN_PYTHON_MINOR%) else 1)"
if errorlevel 1 (
    echo ERROR: Python %MIN_PYTHON_MAJOR%.%MIN_PYTHON_MINOR%+ is not available using command '%PYTHON_CMD%'.
    echo Please install Python %MIN_PYTHON_MAJOR%.%MIN_PYTHON_MINOR%+ and ensure it's in your PATH, or use --python-cmd.
    goto :failure
)
%PYTHON_CMD% -m pip --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available for %PYTHON_CMD%. Please ensure pip is installed.
    goto :failure
)
echo SUCCESS: Python and pip checks passed. (%PYTHON_CMD% %PYTHON_VERSION%)

REM Docker
echo Checking for Docker...
docker --version > nul 2>&1
if errorlevel 1 (
    echo WARNING: Docker is not found. Some functionalities will not be available.
    echo Please visit https://docs.docker.com/docker-for-windows/install/ to install Docker Desktop.
) else (
    echo SUCCESS: Docker check passed. (docker --version)
)

REM --- Virtual Environment Setup ---
echo.
echo --- Setting up Virtual Environment ---
set ACTIVATED_VENV=0
if %SKIP_VENV% equ 0 (
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Virtual environment '%VENV_DIR%' already exists.
        CHOICE /C YN /M "Use existing virtual environment '%VENV_DIR%'?"
        IF ERRORLEVEL 2 (
            echo ERROR: User chose not to use existing virtual environment. Please remove or rename '%VENV_DIR%' or use --skip-venv.
            goto :failure
        )
        echo Using existing virtual environment.
    ) else (
        echo Creating virtual environment in '%VENV_DIR%'...
        %PYTHON_CMD% -m venv "%VENV_DIR%"
        if errorlevel 1 (
            echo ERROR: Failed to create virtual environment.
            goto :failure
        )
        echo SUCCESS: Virtual environment created.
    )

    echo Activating virtual environment: %VENV_DIR%\Scripts\activate.bat
    call "%VENV_DIR%\Scripts\activate.bat"
    set ACTIVATED_VENV=1
) else (
    echo Skipping virtual environment creation as per --skip-venv.
    if "%VIRTUAL_ENV%"=="" (
        echo WARNING: You are installing packages globally because --skip-venv was used and no virtual environment is active.
        CHOICE /C YN /M "Are you sure you want to proceed with global installation?"
        IF ERRORLEVEL 2 (
            echo ERROR: Installation aborted by user.
            goto :failure
        )
    ) else (
        echo Using existing active virtual environment: %VIRTUAL_ENV%
    )
)
set PYTHON_IN_VENV_CMD=%PYTHON_CMD%
set PIP_IN_VENV_CMD=%PYTHON_CMD% -m pip

REM --- Install Dependencies ---
echo.
echo --- Installing Dependencies ---
echo Upgrading pip...
%PIP_IN_VENV_CMD% install --upgrade pip
if errorlevel 1 ( echo ERROR: Failed to upgrade pip. & goto :handle_error )

echo Installing dependencies from requirements.txt...
%PIP_IN_VENV_CMD% install -r requirements.txt
if errorlevel 1 ( echo ERROR: Failed to install dependencies from requirements.txt. & goto :handle_error )
echo SUCCESS: Dependencies installed.

REM --- Install Project in Editable Mode with Dev Extras ---
echo Installing project in editable mode with [dev] extras...
%PIP_IN_VENV_CMD% install -e .[dev]
if errorlevel 1 ( echo ERROR: Failed to install project in editable mode. & goto :handle_error )
echo SUCCESS: Project installed in editable mode with [dev] extras.

REM --- Basic Verification ---
echo.
echo --- Running Basic Verification ---
echo Verifying (main.py --help)...
%PYTHON_IN_VENV_CMD% main.py --help > nul
if errorlevel 1 ( echo ERROR: Verification (main.py --help) failed. & goto :handle_error )
echo SUCCESS: Verification (main.py --help) passed.

echo Running a quick smoke test using pytest...
%PIP_IN_VENV_CMD% show pytest > nul 2>&1
if errorlevel 1 (
    echo pytest not found, installing it for smoke test...
    %PIP_IN_VENV_CMD% install pytest
    if errorlevel 1 ( echo WARNING: Failed to install pytest for smoke test. Skipping smoke test. & goto :skip_smoke_test_failure )
)
%PYTHON_IN_VENV_CMD% -m pytest -k "test_example_success" tests/test_example.py
if errorlevel 1 (
    echo WARNING: Smoke test (pytest -k test_example_success) failed or test not found.
    echo This might indicate issues with the test setup or core functionality.
    echo Continuing installation, but please review test output.
) else (
    echo SUCCESS: Smoke test passed.
)
:skip_smoke_test_failure

echo.
echo SUCCESS: QLOP Project installation and basic verification completed!
if %SKIP_VENV% equ 0 (
    if %ACTIVATED_VENV% equ 1 (
        echo The virtual environment '%VENV_DIR%' is currently active.
        echo To deactivate it, run: call "%VENV_DIR%\Scripts\deactivate.bat"
    )
    echo To use the project later, activate the virtual environment: call "%VENV_DIR%\Scripts\activate.bat"
)
echo You can now use the 'qlop-cli' command (if installed by setup.py) or '%PYTHON_IN_VENV_CMD% main.py'.
goto :eof

:handle_error
if %ACTIVATED_VENV% equ 1 (
    echo Deactivating virtual environment due to error...
    call "%VENV_DIR%\Scripts\deactivate.bat"
)
goto :failure

:failure
echo.
echo ERROR: Installation failed. Please check the messages above.
exit /B 1

:eof
echo.
endlocal
