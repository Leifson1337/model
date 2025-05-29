import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os

# Assuming project structure where logs directory is at the project root
# Adjust if utils.py or config.py defines a central log path determination logic
try:
    # Try to use the logging setup from utils to ensure consistency if possible for log dir
    from .utils import setup_logging # Relative import if cli_monitor is in src
    from .config import LOGS_DIR as CENTRAL_LOGS_DIR # Assuming LOGS_DIR is defined in config.py
    
    # Ensure setup_logging is called if it hasn't been (e.g. if cli_monitor is used standalone early)
    if not logging.getLogger().hasHandlers():
        setup_logging()
    
    # Construct path relative to the centrally defined LOGS_DIR
    # This assumes LOGS_DIR is relative to project root.
    project_root = Path(__file__).resolve().parent.parent 
    CLI_EXECUTION_LOG_DIR = project_root / CENTRAL_LOGS_DIR
except ImportError:
    # Fallback if utils or config cannot be imported (e.g. running script standalone for tests without full src path setup)
    CLI_EXECUTION_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

CLI_EXECUTION_LOG_FILE = CLI_EXECUTION_LOG_DIR / "cli_executions.jsonl"
logger = logging.getLogger(__name__) # Use standard logger for any issues within cli_monitor itself

def _ensure_log_dir_exists():
    """Ensures the log directory exists."""
    try:
        CLI_EXECUTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"[CLI_Monitor] Failed to create log directory {CLI_EXECUTION_LOG_DIR}: {e}", exc_info=True)
        # Optionally, could fallback to a simpler logging mechanism or disable if dir creation fails.

def start_cli_run(command_name: str, args: Dict[str, Any], run_id: Optional[str] = None) -> tuple[str, datetime]:
    """
    Logs the start of a CLI command execution.

    Args:
        command_name: The name of the CLI command being executed.
        args: A dictionary of arguments passed to the command. Sensitive args should be pre-filtered.
        run_id: Optional existing run_id. If None, a new one will be generated.

    Returns:
        A tuple containing (run_id, start_timestamp_utc).
    """
    _ensure_log_dir_exists()
    
    current_run_id = run_id or str(uuid.uuid4())
    start_time_utc = datetime.now(timezone.utc)
    
    log_entry = {
        "run_id": current_run_id,
        "command_name": command_name,
        "args": args, # Consider filtering sensitive args if any
        "status": "start",
        "start_timestamp_utc": start_time_utc.isoformat(),
        "pid": os.getpid()
    }
    
    try:
        with open(CLI_EXECUTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except IOError as e:
        logger.error(f"[CLI_Monitor] Failed to write start log for run_id {current_run_id}: {e}", exc_info=True)
        # Depending on policy, could re-raise or just log the failure to monitor.
        
    return current_run_id, start_time_utc

def end_cli_run(
    run_id: str, 
    command_name: str, # Pass command_name again for consistency in the end log
    status: str,  # "success" or "failure"
    exit_code: int,
    start_time_utc: datetime,
    error_message: Optional[str] = None,
    output_summary: Optional[Dict[str, Any]] = None
):
    """
    Logs the end of a CLI command execution.

    Args:
        run_id: The unique ID for this run (from start_cli_run).
        command_name: The name of the CLI command.
        status: "success" or "failure".
        exit_code: The exit code of the command.
        start_time_utc: The UTC timestamp when the command started.
        error_message: Optional error message if the command failed.
        output_summary: Optional dictionary with summary of outputs (e.g., paths to artifacts).
    """
    _ensure_log_dir_exists()
    
    end_time_utc = datetime.now(timezone.utc)
    duration = end_time_utc - start_time_utc
    
    log_entry = {
        "run_id": run_id,
        "command_name": command_name,
        "status": status,
        "exit_code": exit_code,
        "start_timestamp_utc": start_time_utc.isoformat(), # Included for easier correlation
        "end_timestamp_utc": end_time_utc.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "pid": os.getpid()
    }
    if error_message:
        log_entry["error_message"] = error_message
    if output_summary:
        log_entry["output_summary"] = output_summary
        
    try:
        with open(CLI_EXECUTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except IOError as e:
        logger.error(f"[CLI_Monitor] Failed to write end log for run_id {run_id}: {e}", exc_info=True)

if __name__ == "__main__":
    # Example Usage (for testing the monitor itself)
    print(f"CLI Execution Log File will be at: {CLI_EXECUTION_LOG_FILE}")
    
    # Simulate a command run
    cmd_args = {"config": "/path/to/conf.json", "debug": True, "api_key": "sensitive_value_removed"}
    run_id_1, start_time_1 = start_cli_run("load-data", cmd_args)
    print(f"Started run: {run_id_1} for load-data")
    
    # Simulate some work
    import time
    time.sleep(0.1) 
    
    # Simulate success
    end_cli_run(run_id_1, "load-data", "success", 0, start_time_1, output_summary={"output_path": "/data/output.parquet"})
    print(f"Ended run: {run_id_1}")

    # Simulate another command run that fails
    cmd_args_2 = {"model_name": "MyModel", "version": "v1.2"}
    run_id_2, start_time_2 = start_cli_run("train-model", cmd_args_2)
    print(f"Started run: {run_id_2} for train-model")
    
    time.sleep(0.2)
    
    try:
        # Simulate an error
        raise ValueError("Something went wrong during training")
    except ValueError as e:
        end_cli_run(run_id_2, "train-model", "failure", 1, start_time_2, error_message=str(e))
        print(f"Ended run (failed): {run_id_2}")

    print(f"\nCheck the log file: {CLI_EXECUTION_LOG_FILE}")
    if CLI_EXECUTION_LOG_FILE.exists():
        with open(CLI_EXECUTION_LOG_FILE, "r") as f:
            for line in f:
                print(line.strip())
```
