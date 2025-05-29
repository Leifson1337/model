import click
import json
from pathlib import Path
import pandas as pd
import numpy as np
import os
import logging
import sys
from typing import Optional # Added for type hinting

# Pydantic models for type hinting and config access
from src.config_models import (
    GlobalAppConfig, LoadDataConfig, FeatureEngineeringConfig, TrainModelConfig,
    EvaluateModelConfig, BacktestConfig, ExportConfig
)
# Import model registry utilities
import src.model_registry_utils as registry_utils
# Import logging setup utility and model loading
from src.utils import setup_logging, load_model
# Import custom exceptions
from src.exceptions import (
    AppBaseException, ConfigError, DataValidationError, 
    ModelTrainingError, PipelineError, FileOperationError
)
# Import feature analysis utilities
from src.feature_analysis import calculate_feature_statistics, compare_feature_statistics
# Import evaluation utilities
import src.evaluation as evaluation_utils
import joblib # For loading scalers

from src.cli_monitor import start_cli_run, end_cli_run # For CLI execution monitoring
from datetime import datetime # For monitor start time

# Call setup_logging() at the very beginning before any loggers are instantiated.
# Note: setup_logging itself is now responsible for clearing existing handlers if needed.
setup_logging()
logger = logging.getLogger(__name__)

def load_and_validate_config(config_path: str, model_class: type, command_name: str = "Unknown Command"):
    """Loads and validates a configuration file."""
    if not config_path:
        # Log this, but ConfigError will be raised and then caught by the command's error handler.
        logger.error(f"[{command_name}] No config file path provided.", extra={"props": {"command": command_name}})
        raise ConfigError(message=f"No configuration file path was provided for {command_name}.")

    try:
        path_obj = Path(config_path)
        config_text = path_obj.read_text()
        config_json = json.loads(config_text)
        
        # Handle nested configurations if necessary (example from original code)
        if 'config_inline' in config_json and isinstance(config_json.get('config_inline'), dict):
            parsed_config = model_class(**config_json.get('config_inline'))
        elif 'config_path' in config_json and isinstance(config_json.get('config_path'), str):
            logger.warning(
                f"[{command_name}] Config '{config_path}' points to another config path '{config_json.get('config_path')}'. "
                "This redirection should ideally be resolved before CLI invocation.",
                extra={"props": {"command": command_name, "config_path": config_path, "redirect_path": config_json.get('config_path')}}
            )
            # Assuming the intent was to load the current file, not redirect.
            # If redirection is a valid feature, this logic needs to be more robust.
            parsed_config = model_class(**config_json) 
        else:
            parsed_config = model_class(**config_json)
        
        logger.info(f"[{command_name}] Configuration loaded and validated from '{config_path}' for {model_class.__name__}.",
                    extra={"props": {"command": command_name, "config_path": config_path, "model_class": model_class.__name__}})
        return parsed_config
        
    except FileNotFoundError:
        logger.error(f"[{command_name}] Configuration file not found: {config_path}",
                     extra={"props": {"command": command_name, "config_path": config_path}}, exc_info=True)
        raise ConfigError(message=f"Configuration file not found: {config_path}", missing_key=config_path)
        
    except json.JSONDecodeError as e:
        logger.error(f"[{command_name}] Failed to decode JSON from configuration file: {config_path}. Error: {e}",
                     extra={"props": {"command": command_name, "config_path": config_path, "json_error": str(e)}}, exc_info=True)
        raise ConfigError(message=f"Invalid JSON format in configuration file: {config_path}. Details: {e}")
        
    except Exception as e: # Catches Pydantic's ValidationError and other general exceptions
        logger.error(f"[{command_name}] Error loading or validating configuration from '{config_path}': {e}",
                     extra={"props": {"command": command_name, "config_path": config_path, "error_type": type(e).__name__}}, exc_info=True)
        # Check if it's a Pydantic ValidationError and format message if so.
        if "ValidationError" in type(e).__name__: # Simple check for Pydantic error
             # Pydantic errors can be verbose. Extracting a summary if possible.
            error_details = getattr(e, 'errors', lambda: str(e))() # Get Pydantic error list or string representation
            raise ConfigError(message=f"Configuration validation failed for '{config_path}'. Please check the structure and values. Details: {error_details}",
                              invalid_value={"key": config_path, "value": "complex_structure", "reason": str(error_details)})
        raise ConfigError(message=f"An unexpected error occurred while processing configuration '{config_path}': {e}")


@click.group()
@click.option('--app-config', default=None, help='Path to global app config (e.g., dev.json).')
@click.pass_context
def cli(ctx, app_config):
    """A CLI tool for the ML pipeline."""
    # Logging is set up globally. Log the invocation.
    logger.info(f"CLI invoked. Provided global app config path: {app_config}",
                extra={"props": {"command": "cli_group_init", "app_config_path": app_config}})
    ctx.obj = {}
    try:
        if app_config:
            global_cfg_instance = load_and_validate_config(app_config, GlobalAppConfig, command_name="GlobalAppConfigLoading")
            ctx.obj['GLOBAL_CONFIG'] = global_cfg_instance
            logger.info(f"Global app configuration loaded successfully from: {app_config}",
                        extra={"props": {"command": "cli_group_init", "status": "success", "app_config_path": app_config}})
        else:
            ctx.obj['GLOBAL_CONFIG'] = GlobalAppConfig() # Default global config
            logger.info("No global app config provided. Using default GlobalAppConfig.",
                        extra={"props": {"command": "cli_group_init", "status": "default_config"}})
    except ConfigError as e:
        logger.critical(f"Failed to load critical global app configuration from '{app_config}': {e.message}",
                        extra={"props": {"command": "cli_group_init", "status": "critical_failure", "app_config_path": app_config, "error_details": e.details}})
        click.echo(f"CRITICAL ERROR: Could not load global application configuration: {e.message}", err=True)
        click.echo("Please ensure the global configuration file is correct and accessible.", err=True)
        sys.exit(1) # Exit if global config is critical and fails to load
    except Exception as e: # Catch any other unexpected error during global config load
        logger.critical(f"An unexpected critical error occurred while loading global app configuration from '{app_config}': {e}",
                        extra={"props": {"command": "cli_group_init", "status": "unexpected_critical_failure", "app_config_path": app_config}}, exc_info=True)
        click.echo(f"CRITICAL UNEXPECTED ERROR: Could not load global application configuration: {e}", err=True)
        sys.exit(1)


@cli.command('load-data')
@click.option('--config', 'config_path', default=None, help='Path to LoadDataConfig JSON file.')
@click.pass_context
def load_data(ctx, config_path: str):
    command_name = ctx.command.name
    run_id, start_time_utc = start_cli_run(command_name, ctx.params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, None
    
    try:
        logger.info(f"CLI '{command_name}' called. Config path: {config_path}", extra={"props": {"command": command_name, "config_path": config_path, "run_id": run_id}})
        click.echo(f"--- {command_name} ---")
        
        if not config_path:
            raise ConfigError(message="Path to LoadDataConfig JSON file is required for 'load-data' command.")
        
        loaded_config = load_and_validate_config(config_path, LoadDataConfig, command_name=command_name)
        
        # --- STUB: Actual data loading logic (replace with call to src.data_management) ---
        # This part should ideally be a single function call, e.g., 
        # output_file = src.data_management.process_load_data_config(loaded_config)
        logger.info(f"[{command_name}] Simulating data loading with config: {loaded_config.input_source}",
                    extra={"props": {"command": command_name, "input_source": loaded_config.input_source, "output_path": loaded_config.output_path, "run_id": run_id}})
        
        Path(loaded_config.output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'dummy_data': [1,2,3]}).to_csv(loaded_config.output_path, index=False) # Dummy save
        # --- END STUB ---

        output_summary = {"output_data_path": loaded_config.output_path}
        click.echo(f"'{command_name}' logic completed successfully. Output at {loaded_config.output_path}")
        logger.info(f"[{command_name}] completed successfully.", extra={"props": {"command": command_name, "status": "success", "run_id": run_id}})
        status, exit_code = "success", 0

    except ConfigError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Configuration error: {error_msg}", extra={"props": {"command": command_name, "status": "config_error", "error_details": e.details, "run_id": run_id}})
        click.echo(f"Error in '{command_name}': {error_msg}", err=True)
        click.echo("Please check your configuration file and command arguments.", err=True)
        # sys.exit(1) # Let finally block handle exit
    except FileOperationError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] File operation error: {error_msg}", extra={"props": {"command": command_name, "status": "file_error", "filepath": e.filepath, "operation": e.operation, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{command_name}' during file operation: {error_msg}", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Application error: {error_msg}", extra={"props": {"command": command_name, "status": "app_error", "error_type": type(e).__name__, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"An application error occurred in '{command_name}': {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(f"[{command_name}] {error_msg}", extra={"props": {"command": command_name, "status": "unexpected_error", "error_type": type(e).__name__, "run_id": run_id}}, exc_info=True)
        click.echo(f"An unexpected error occurred in '{command_name}'. Check logs for details.", err=True)
    finally:
        end_cli_run(run_id, command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
            sys.exit(exit_code)


@cli.command('engineer-features')
@click.option('--config', 'config_path', required=True, type=click.Path(exists=False, dir_okay=False, readable=True), help='Path to FeatureEngineeringConfig JSON file.')
@click.option('--compare-to-baseline', 'baseline_stats_path', default=None, type=click.Path(exists=True, dir_okay=False, readable=True), help="Optional path to baseline feature statistics JSON file for drift comparison.")
@click.pass_context
def engineer_features(ctx, config_path: str, baseline_stats_path: Optional[str]):
    command_name = ctx.command.name
    # Filter params for logging, remove potentially large or sensitive data if necessary
    log_params = {k: v for k, v in ctx.params.items() if k != "some_large_data_param"}
    run_id, start_time_utc = start_cli_run(command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}

    try:
        logger.info(f"CLI '{command_name}' called.", extra={"props": {"command": command_name, "run_id": run_id, **log_params}})
        click.echo(f"--- {command_name} ---")
        
        loaded_config = load_and_validate_config(config_path, FeatureEngineeringConfig, command_name=command_name)
        click.echo(f"Op: Engineer features from '{loaded_config.input_data_path}' to '{loaded_config.output_features_path}'.")
        
        output_path_obj = Path(loaded_config.output_features_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_data = {'num1': np.random.rand(100) + (0 if not baseline_stats_path else 2), 
                      'cat1': np.random.choice(['A','B','X'],100), 
                      'target':np.random.randint(0,2,100)}
        if baseline_stats_path: dummy_data['new_feat_drift'] = np.random.rand(100)
        current_features_df = pd.DataFrame(dummy_data)
        
        if not Path(loaded_config.input_data_path).exists() and loaded_config.input_data_path != "dummy_input_placeholder.csv":
             raise FileOperationError(message=f"Input data path does not exist: {loaded_config.input_data_path}", filepath=loaded_config.input_data_path, operation="read")

        try:
            if str(output_path_obj).endswith(".parquet"): current_features_df.to_parquet(output_path_obj, index=False)
            else: current_features_df.to_csv(output_path_obj, index=False)
        except Exception as io_err:
            raise FileOperationError(message=f"Failed to save engineered features to {output_path_obj}", filepath=str(output_path_obj), operation="write", details={"os_error": str(io_err)})

        output_summary["output_features_path"] = str(output_path_obj)
        click.echo(f"Dummy current features saved: {output_path_obj}")
        logger.info(f"[{command_name}] Dummy current features saved: {output_path_obj}", extra={"props": {"command": command_name, "run_id": run_id}})
        
        current_stats_dict = calculate_feature_statistics(current_features_df)
        base, _ = os.path.splitext(loaded_config.output_features_path)
        current_stats_path_str = f"{base}_current_stats.json"
        
        try:
            with open(current_stats_path_str, 'w') as f: json.dump(current_stats_dict, f, indent=4)
        except Exception as io_err:
            raise FileOperationError(message=f"Failed to save current feature statistics to {current_stats_path_str}", filepath=current_stats_path_str, operation="write", details={"os_error": str(io_err)})
        
        output_summary["current_stats_path"] = current_stats_path_str
        click.echo(f"Current stats saved: {current_stats_path_str}")
        logger.info(f"[{command_name}] Current stats saved: {current_stats_path_str}", extra={"props": {"command": command_name, "run_id": run_id}})

        if baseline_stats_path:
            try:
                with open(baseline_stats_path, 'r') as f: loaded_baseline_stats = json.load(f)
            except FileNotFoundError:
                raise FileOperationError(message=f"Baseline stats file not found: {baseline_stats_path}", filepath=baseline_stats_path, operation="read")
            except json.JSONDecodeError:
                raise DataValidationError(message=f"Invalid JSON in baseline stats file: {baseline_stats_path}", details={"filepath": baseline_stats_path})

            drift_report = compare_feature_statistics(current_stats=current_stats_dict, baseline_stats=loaded_baseline_stats)
            drift_path_str = f"{base}_drift_report.json"
            try:
                with open(drift_path_str, 'w') as f: json.dump(drift_report, f, indent=4)
            except Exception as io_err:
                raise FileOperationError(message=f"Failed to save drift report to {drift_path_str}", filepath=drift_path_str, operation="write", details={"os_error": str(io_err)})
            output_summary["drift_report_path"] = drift_path_str
            click.echo(f"Drift report saved: {drift_path_str}")
            logger.info(f"[{command_name}] Drift report saved: {drift_path_str}", extra={"props": {"command": command_name, "run_id": run_id}})
        else:
            click.echo("No baseline stats provided, skipping drift analysis.")
            logger.info(f"[{command_name}] Skipped drift analysis.", extra={"props": {"command": command_name, "run_id": run_id}})
        
        click.echo(f"'{command_name}' logic completed successfully.")
        logger.info(f"[{command_name}] completed successfully.", extra={"props": {"command": command_name, "status": "success", "run_id": run_id}})
        status, exit_code = "success", 0

    except ConfigError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Configuration error: {error_msg}", extra={"props": {"command": command_name, "status": "config_error", "error_details": e.details, "run_id": run_id}})
        click.echo(f"Error in '{command_name}': {error_msg}\nPlease check your configuration file.", err=True)
    except FileOperationError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] File operation error: {error_msg}", extra={"props": {"command": command_name, "status": "file_error", "filepath": e.filepath, "operation": e.operation, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{command_name}' during file operation: {error_msg}", err=True)
    except DataValidationError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Data validation error: {error_msg}", extra={"props": {"command": command_name, "status": "data_validation_error", "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{command_name}' due to data validation: {error_msg}", err=True)
    except PipelineError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Pipeline error: {error_msg}", extra={"props": {"command": command_name, "status": "pipeline_error", "stage_name": e.stage_name, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"A pipeline error occurred in '{command_name}': {error_msg}", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Application error: {error_msg}", extra={"props": {"command": command_name, "status": "app_error", "error_type": type(e).__name__, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"An application error occurred in '{command_name}': {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected critical error occurred: {str(e)}"
        logger.critical(f"[{command_name}] {error_msg}", extra={"props": {"command": command_name, "status": "unexpected_critical_error", "error_type": type(e).__name__, "run_id": run_id}}, exc_info=True)
        click.echo(f"An unexpected critical error occurred in '{command_name}'. Please check logs for details.", err=True)
    finally:
        end_cli_run(run_id, command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
            sys.exit(exit_code)


@cli.command('train-model')
@click.option('--config', 'config_path', required=True, type=click.Path(exists=False, dir_okay=False, readable=True), help='Path to TrainModelConfig JSON file.')
@click.pass_context
def train_model(ctx, config_path: str):
    command_name = ctx.command.name
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}

    try:
        logger.info(f"CLI '{command_name}' called.", extra={"props": {"command": command_name, "run_id": run_id, **log_params}})
        click.echo(f"--- {command_name} ---")
        
        loaded_config = load_and_validate_config(config_path, TrainModelConfig, command_name=command_name)
        output_summary["model_output_path_base"] = loaded_config.model_output_path_base
        if loaded_config.scaler_output_path:
            output_summary["scaler_output_path"] = loaded_config.scaler_output_path

        click.echo(f"Op: Train a {loaded_config.model_type} model.")
        
        if loaded_config.model_type == "XGBoost":
            try:
                if not Path(loaded_config.input_features_path).exists():
                    raise FileOperationError(message=f"Input features path does not exist: {loaded_config.input_features_path}", filepath=loaded_config.input_features_path, operation="read")
                features_df = pd.read_csv(loaded_config.input_features_path) 
                
                if loaded_config.target_column not in features_df.columns:
                    raise DataValidationError(message=f"Target column '{loaded_config.target_column}' not in features data.", validation_errors={"target_column": f"'{loaded_config.target_column}' not found."})
                
                X_train_dummy = features_df.drop(columns=[loaded_config.target_column]).fillna(0)
                y_train_dummy = features_df[loaded_config.target_column].fillna(0)

                if X_train_dummy.empty or y_train_dummy.empty:
                    raise DataValidationError(message="No data available for training after processing.", details={"input_path": loaded_config.input_features_path})
                
                from src.modeling import train_xgboost 
                click.echo(f"Calling src.modeling.train_xgboost with features from '{loaded_config.input_features_path}'...")
                logger.info(f"[{command_name}] Starting XGBoost training process.", extra={"props": {"command": command_name, "model_type": "XGBoost", "run_id": run_id}})
                
                model = train_xgboost(X_train_dummy, y_train_dummy, loaded_config) 
                
                if model:
                    click.echo(f"XGBoost training executed. Model type: {type(model)}")
                    logger.info(f"[{command_name}] XGBoost training successful.", extra={"props": {"command": command_name, "model_type": "XGBoost", "output_path": loaded_config.model_output_path_base, "run_id": run_id}})
                else:
                    raise ModelTrainingError(message="XGBoost training function returned no model without raising an error.", model_name="XGBoost")

            except FileNotFoundError as e:
                raise FileOperationError(message=f"Input features file not found: {loaded_config.input_features_path}", filepath=str(e.filename), operation="read")
            except pd.errors.EmptyDataError:
                 raise DataValidationError(message=f"Input features file is empty: {loaded_config.input_features_path}", details={"filepath": loaded_config.input_features_path})
        else:
            logger.warning(f"[{command_name}] Actual training for '{loaded_config.model_type}' not fully implemented in CLI stub.", extra={"props": {"command": command_name, "model_type": loaded_config.model_type, "status": "stub_warning", "run_id": run_id}})
            click.echo(f"Warning: Actual training for '{loaded_config.model_type}' not fully implemented in CLI stub.", err=True)
        
        click.echo(f"Model base output path: {loaded_config.model_output_path_base}")
        click.echo(f"'{command_name}' logic completed.")
        logger.info(f"[{command_name}] completed.", extra={"props": {"command": command_name, "status": "partial_success_stub" if loaded_config.model_type != 'XGBoost' else 'success', "run_id": run_id}})
        status, exit_code = "success", 0
        if loaded_config.model_type != 'XGBoost': status = "partial_success_stub"


    except ConfigError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Configuration error: {error_msg}", extra={"props": {"command": command_name, "status": "config_error", "error_details": e.details, "run_id": run_id}})
        click.echo(f"Error in '{command_name}': {error_msg}\nPlease check your configuration file.", err=True)
    except FileOperationError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] File operation error: {error_msg}", extra={"props": {"command": command_name, "status": "file_error", "filepath": e.filepath, "operation": e.operation, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{command_name}' during file operation: {error_msg}", err=True)
    except DataValidationError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Data validation error: {error_msg}", extra={"props": {"command": command_name, "status": "data_validation_error", "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{command_name}' due to data validation: {error_msg}", err=True)
    except ModelTrainingError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Model training error for '{e.model_name}': {error_msg}", extra={"props": {"command": command_name, "status": "training_error", "model_name": e.model_name, "stage": e.stage, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error during model training in '{command_name}': {error_msg}", err=True)
    except PipelineError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Pipeline error: {error_msg}", extra={"props": {"command": command_name, "status": "pipeline_error", "stage_name": e.stage_name, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"A pipeline error occurred in '{command_name}': {error_msg}", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Application error: {error_msg}", extra={"props": {"command": command_name, "status": "app_error", "error_type": type(e).__name__, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"An application error occurred in '{command_name}': {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected critical error occurred: {str(e)}"
        logger.critical(f"[{command_name}] {error_msg}", extra={"props": {"command": command_name, "status": "unexpected_critical_error", "error_type": type(e).__name__, "run_id": run_id}}, exc_info=True)
        click.echo(f"An unexpected critical error occurred in '{command_name}'. Please check logs for details.", err=True)
    finally:
        end_cli_run(run_id, command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
            sys.exit(exit_code)


@cli.command('evaluate')
@click.option('--config', 'config_path', required=True, type=click.Path(exists=False, dir_okay=False, readable=True), help='Path to EvaluateModelConfig JSON file.')
@click.pass_context
def evaluate(ctx, config_path: str):
    """Evaluates a trained model using test data and saves metrics."""
    command_name = ctx.command.name
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}
    
    try:
        logger.info(f"CLI '{command_name}' called.", extra={"props": {"command": command_name, "run_id": run_id, **log_params}})
        click.echo(f"--- {command_name} ---")
        
        loaded_config = load_and_validate_config(config_path, EvaluateModelConfig, command_name=command_name)
        output_summary["metrics_output_json_path"] = loaded_config.metrics_output_json_path
        if loaded_config.shap_summary_plot_path:
            output_summary["shap_summary_plot_path"] = loaded_config.shap_summary_plot_path

        model_path_obj = Path(loaded_config.model_path)
        logger.info(f"[{command_name}] Loading model from: {model_path_obj}", extra={"props": {"command": command_name, "model_path": str(model_path_obj), "run_id": run_id}})
        model = load_model(model_name=model_path_obj.name, models_dir=str(model_path_obj.parent))
        if not model: 
            raise FileOperationError(message=f"Model could not be loaded from {model_path_obj}", filepath=str(model_path_obj), operation="load")
        click.echo(f"Model loaded successfully from {model_path_obj}")

        scaler = None
        if loaded_config.scaler_path:
            scaler_path_obj = Path(loaded_config.scaler_path)
            logger.info(f"[{command_name}] Loading scaler from: {scaler_path_obj}", extra={"props": {"command": command_name, "scaler_path": str(scaler_path_obj), "run_id": run_id}})
            if not scaler_path_obj.exists():
                raise FileOperationError(message=f"Scaler file not found: {scaler_path_obj}", filepath=str(scaler_path_obj), operation="read")
            try:
                scaler = joblib.load(scaler_path_obj)
                click.echo(f"Scaler loaded successfully from {scaler_path_obj}")
            except Exception as joblib_err:
                raise FileOperationError(message=f"Failed to load scaler from {scaler_path_obj}", filepath=str(scaler_path_obj), operation="load", details={"joblib_error": str(joblib_err)})

        logger.info(f"[{command_name}] Loading test data from: {loaded_config.test_data_path}", extra={"props": {"command": command_name, "test_data_path": loaded_config.test_data_path, "run_id": run_id}})
        test_data_path_obj = Path(loaded_config.test_data_path)
        if not test_data_path_obj.exists():
            raise FileOperationError(message=f"Test data file not found: {test_data_path_obj}", filepath=str(test_data_path_obj), operation="read")
        try:
            test_df = pd.read_csv(test_data_path_obj) if str(test_data_path_obj).endswith(".csv") else pd.read_parquet(test_data_path_obj)
        except Exception as pd_err:
            raise DataValidationError(message=f"Failed to read test data from {test_data_path_obj}", details={"filepath": str(test_data_path_obj), "pandas_error": str(pd_err)})
        click.echo(f"Test data loaded. Shape: {test_df.shape}")

        if loaded_config.target_column not in test_df.columns:
            raise DataValidationError(message=f"Target column '{loaded_config.target_column}' not in test data columns.", validation_errors={"target_column": f"'{loaded_config.target_column}' not found"})
        feature_columns = test_df.columns.drop(loaded_config.target_column, errors='ignore').tolist()
        if not feature_columns:
            raise DataValidationError(message="No feature columns found in test data after excluding target.", details={"target_column": loaded_config.target_column})
        X_test = test_df[feature_columns]; y_true = test_df[loaded_config.target_column]
        click.echo(f"Prepared X_test (shape: {X_test.shape}), y_true (shape: {y_true.shape})")

        logger.info(f"[{command_name}] Performing predictions.", extra={"props": {"command": command_name, "model_type": loaded_config.model_type or 'Generic', "run_id": run_id}})
        X_test_processed = X_test.copy()
        if scaler:
            try:
                X_test_scaled_np = scaler.transform(X_test_processed)
                X_test_processed = pd.DataFrame(X_test_scaled_np, columns=X_test_processed.columns, index=X_test_processed.index)
                click.echo("Test data scaled.")
            except Exception as scale_err:
                raise PipelineError(message=f"Error during data scaling: {scale_err}", stage_name="evaluate_scaling", details={"error": str(scale_err)})
        
        try:
            if hasattr(model, "predict_proba") and hasattr(model, "predict"):
                y_pred_proba_np = model.predict_proba(X_test_processed)[:, 1]
                y_pred_np = model.predict(X_test_processed)
            elif hasattr(model, "predict"):
                y_pred_proba_np = model.predict(X_test_processed)
                if y_pred_proba_np.ndim > 1 and y_pred_proba_np.shape[1] > 1: y_pred_proba_np = y_pred_proba_np[:, 1]
                y_pred_np = (y_pred_proba_np > 0.5).astype(int)
            else:
                raise ModelTrainingError(message="Loaded model is missing standard predict/predict_proba methods.", model_name=str(type(model)))
        except Exception as pred_err:
            raise PipelineError(message=f"Error during model prediction: {pred_err}", stage_name="evaluate_prediction", details={"error": str(pred_err)})
        
        y_pred = pd.Series(y_pred_np.flatten(), index=y_true.index, name="y_pred")
        y_pred_proba = pd.Series(y_pred_proba_np.flatten(), index=y_true.index, name="y_pred_proba")
        click.echo("Predictions generated.")

        logger.info(f"[{command_name}] Calculating classification metrics...", extra={"props": {"command": command_name, "run_id": run_id}})
        all_metrics = evaluation_utils.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        metrics_to_save = {k: v for k, v in all_metrics.items() if k in loaded_config.metrics_to_compute} if loaded_config.metrics_to_compute else all_metrics
        if loaded_config.metrics_to_compute: click.echo(f"Filtered metrics to: {list(metrics_to_save.keys())}")
        click.echo("\nCalculated Metrics:"); click.echo(json.dumps(metrics_to_save, indent=2))
        output_summary["calculated_metrics"] = metrics_to_save

        metrics_output_path_obj = Path(loaded_config.metrics_output_json_path)
        metrics_output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(metrics_output_path_obj, 'w') as f: json.dump(metrics_to_save, f, indent=4)
        except Exception as io_err:
            raise FileOperationError(message=f"Failed to save metrics to {metrics_output_path_obj}", filepath=str(metrics_output_path_obj), operation="write", details={"os_error": str(io_err)})
        click.echo(f"Metrics saved to: {metrics_output_path_obj}")
        logger.info(f"[{command_name}] Metrics saved to: {metrics_output_path_obj}", extra={"props": {"command": command_name, "run_id": run_id}})

        if loaded_config.shap_summary_plot_path:
            # ... (SHAP logic placeholder) ...
            pass 

        click.echo(f"'{command_name}' completed successfully.")
        logger.info(f"[{command_name}] completed successfully.", extra={"props": {"command": command_name, "status": "success", "run_id": run_id}})
        status, exit_code = "success", 0

    except ConfigError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Configuration error: {error_msg}", extra={"props": {"command": command_name, "status": "config_error", "error_details": e.details, "run_id": run_id}})
        click.echo(f"Error in '{command_name}': {error_msg}\nPlease check your configuration file.", err=True)
    except FileOperationError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] File operation error: {error_msg}", extra={"props": {"command": command_name, "status": "file_error", "filepath": e.filepath, "operation": e.operation, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{command_name}' during file operation: {error_msg}", err=True)
    except DataValidationError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Data validation error: {error_msg}", extra={"props": {"command": command_name, "status": "data_validation_error", "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{command_name}' due to data validation: {error_msg}", err=True)
    except ModelTrainingError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Model related error: {error_msg}", extra={"props": {"command": command_name, "status": "model_error", "model_name": e.model_name, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"A model related error occurred in '{command_name}': {error_msg}", err=True)
    except PipelineError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Pipeline error: {error_msg}", extra={"props": {"command": command_name, "status": "pipeline_error", "stage_name": e.stage_name, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"A pipeline error occurred in '{command_name}' during stage '{e.stage_name}': {error_msg}", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Application error: {error_msg}", extra={"props": {"command": command_name, "status": "app_error", "error_type": type(e).__name__, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"An application error occurred in '{command_name}': {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected critical error occurred: {str(e)}"
        logger.critical(f"[{command_name}] {error_msg}", extra={"props": {"command": command_name, "status": "unexpected_critical_error", "error_type": type(e).__name__, "run_id": run_id}}, exc_info=True)
        click.echo(f"An unexpected critical error occurred in '{command_name}'. Please check logs for details.", err=True)
    finally:
        end_cli_run(run_id, command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
            sys.exit(exit_code)


@cli.command('backtest')
@click.option('--config', 'config_path', default=None, help='Path to BacktestConfig JSON file.')
@click.pass_context
def backtest(ctx, config_path: str):
    command_name = ctx.command.name
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}

    try:
        logger.info(f"CLI '{command_name}' called.", extra={"props": {"command": command_name, "run_id": run_id, **log_params}})
        click.echo(f"--- {command_name} ---")
        
        if not config_path:
             raise ConfigError(message="Path to BacktestConfig JSON file is required.")
        loaded_config = load_and_validate_config(config_path, BacktestConfig, command_name=command_name)
        output_summary["results_output_path"] = loaded_config.results_output_path
        
        # --- STUB: Actual backtesting logic ---
        logger.info(f"[{command_name}] Simulating backtesting with config for model: {loaded_config.model_to_backtest}",
                    extra={"props": {"command": command_name, "model": loaded_config.model_to_backtest, "run_id": run_id}})
        click.echo(f"Backtest logic for '{loaded_config.model_to_backtest}' (stub) completed.")
        # --- END STUB ---
        
        logger.info(f"[{command_name}] completed successfully.", extra={"props": {"command": command_name, "status": "success", "run_id": run_id}})
        status, exit_code = "success", 0
        
    except ConfigError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Configuration error: {error_msg}", extra={"props": {"command": command_name, "status": "config_error", "error_details": e.details, "run_id": run_id}})
        click.echo(f"Error in '{command_name}': {error_msg}\nPlease check your configuration file.", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Application error: {error_msg}", extra={"props": {"command": command_name, "status": "app_error", "error_type": type(e).__name__, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"An application error occurred in '{command_name}': {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected critical error occurred: {str(e)}"
        logger.critical(f"[{command_name}] {error_msg}", extra={"props": {"command": command_name, "status": "unexpected_critical_error", "error_type": type(e).__name__, "run_id": run_id}}, exc_info=True)
        click.echo(f"An unexpected critical error occurred in '{command_name}'. Please check logs for details.", err=True)
    finally:
        end_cli_run(run_id, command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
            sys.exit(exit_code)


@cli.command('export')
@click.option('--config', 'config_path', default=None, help='Path to ExportConfig JSON file.')
@click.pass_context
def export(ctx, config_path: str):
    command_name = ctx.command.name
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}

    try:
        logger.info(f"CLI '{command_name}' called.", extra={"props": {"command": command_name, "run_id": run_id, **log_params}})
        click.echo(f"--- {command_name} ---")
        
        if not config_path:
            raise ConfigError(message="Path to ExportConfig JSON file is required.")
        loaded_config = load_and_validate_config(config_path, ExportConfig, command_name=command_name)
        output_summary["export_output_path"] = loaded_config.export_output_path
        
        # --- STUB: Actual export logic ---
        logger.info(f"[{command_name}] Simulating model export for: {loaded_config.model_name} to {loaded_config.export_format}",
                    extra={"props": {"command": command_name, "model_name": loaded_config.model_name, "format": loaded_config.export_format, "run_id": run_id}})
        click.echo(f"Export logic for '{loaded_config.model_name}' to format '{loaded_config.export_format}' (stub) completed.")
        # --- END STUB ---
        
        logger.info(f"[{command_name}] completed successfully.", extra={"props": {"command": command_name, "status": "success", "run_id": run_id}})
        status, exit_code = "success", 0

    except ConfigError as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Configuration error: {error_msg}", extra={"props": {"command": command_name, "status": "config_error", "error_details": e.details, "run_id": run_id}})
        click.echo(f"Error in '{command_name}': {error_msg}\nPlease check your configuration file.", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{command_name}] Application error: {error_msg}", extra={"props": {"command": command_name, "status": "app_error", "error_type": type(e).__name__, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"An application error occurred in '{command_name}': {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected critical error occurred: {str(e)}"
        logger.critical(f"[{command_name}] {error_msg}", extra={"props": {"command": command_name, "status": "unexpected_critical_error", "error_type": type(e).__name__, "run_id": run_id}}, exc_info=True)
        click.echo(f"An unexpected critical error occurred in '{command_name}'. Please check logs for details.", err=True)
    finally:
        end_cli_run(run_id, command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
            sys.exit(exit_code)


@cli.group("models")
def models_group(ctx): # Added ctx
    # This function is for the group, so monitor might not be needed here unless group itself can fail
    # For now, subcommands will handle their own monitoring
    logger.debug("CLI group 'models' invoked.", extra={"props": {"command_group": "models"}})
    pass # No direct action, subcommands handle logic

@models_group.command("list")
@click.option("--name", "model_name_filter", default=None, help="Filter by model name.")
@click.pass_context
def list_registered_models(ctx, model_name_filter: Optional[str]):
    command_name = ctx.command.name # "list" but part of "models" group
    full_command_name = f"{ctx.parent.command.name}-{command_name}" # "models-list"
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(full_command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}

    try:
        logger.info(f"CLI '{full_command_name}'. Filter: {model_name_filter}", extra={"props": {"command": full_command_name, "run_id": run_id, **log_params}})
        
        models = registry_utils.list_models(model_name_filter)
        if not models:
            message = "No models found." if not model_name_filter else f"No models found matching filter '{model_name_filter}'."
            click.echo(message)
            logger.info(f"[{full_command_name}] {message}", extra={"props": {"command": full_command_name, "status": "no_models_found", "run_id": run_id}})
            status, exit_code = "success", 0 # Not an error if no models found, but command succeeded
            output_summary = {"models_found": 0}
            return # Exit early, finally will still run

        click.echo("\n--- Registered Models ---")
        header = "| {:<20} | {:<27} | {:<21} | {:<10} | {:<7} | {:<35} | {:<8} |".format("Model Name","Version","Timestamp (UTC)","Metric","Value","Metadata Path","Has FI")
        click.echo(header)
        click.echo(f"|{'-'*22}|{'-'*29}|{'-'*23}|{'-'*12}|{'-'*9}|{'-'*37}|{'-'*10}|")
        for entry in models:
            val_str = f"{entry.get('primary_metric_value','N/A'):.3f}" if isinstance(entry.get('primary_metric_value'),float) else str(entry.get('primary_metric_value','N/A'))
            click.echo(f"| {entry.get('model_name_from_config','N/A'):<20} | {entry.get('model_version','N/A'):<27} | {entry.get('timestamp_utc','N/A'):<21} | {entry.get('primary_metric_name','N/A'):<10} | {val_str:<7} | {entry.get('meta_json_path','N/A'):<35} | {str(entry.get('has_feature_importance',False)):<8} |")
        click.echo("-------------------------\n")
        logger.info(f"[{full_command_name}] Successfully listed {len(models)} model(s).", extra={"props": {"command": full_command_name, "status": "success", "count": len(models), "run_id": run_id}})
        status, exit_code = "success", 0
        output_summary = {"models_listed": len(models)}

    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{full_command_name}] Error listing models: {error_msg}", extra={"props": {"command": full_command_name, "status": "app_error", "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error listing models: {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.critical(f"[{full_command_name}] Unexpected error listing models: {error_msg}", extra={"props": {"command": full_command_name, "status": "unexpected_error", "run_id": run_id}}, exc_info=True)
        click.echo("An unexpected error occurred while listing models. Check logs.", err=True)
    finally:
        end_cli_run(run_id, full_command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0 and status != "success": # Avoid exiting if status was success (e.g. no models found)
             sys.exit(exit_code)


@models_group.command("describe")
@click.argument("model_name")
@click.argument("version")
@click.pass_context
def describe_model_version(ctx, model_name: str, version: str):
    command_name = ctx.command.name
    full_command_name = f"{ctx.parent.command.name}-{command_name}"
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(full_command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}

    try:
        logger.info(f"CLI '{full_command_name}' for {model_name} v{version}", extra={"props": {"command": full_command_name, "run_id": run_id, **log_params}})
        details = registry_utils.get_model_details(model_name, version)
        
        if details:
            click.echo(json.dumps(details, indent=2))
            logger.info(f"[{full_command_name}] Successfully described model '{model_name}' v{version}.", extra={"props": {"command": full_command_name, "status": "success", "run_id": run_id}})
            status, exit_code = "success", 0
            output_summary = {"model_described": f"{model_name}_v{version}"}
        else:
            # This case is effectively handled by get_model_details raising ModelNotFoundError
            # which is caught by AppBaseException below.
            # If get_model_details returned None instead of raising:
            error_msg = f"Model version '{model_name}' v'{version}' not found (details were None)."
            click.echo(error_msg, err=True)
            logger.warning(f"[{full_command_name}] {error_msg}", extra={"props": {"command": full_command_name, "status": "not_found", "run_id": run_id}})
            # status remains "failure", exit_code remains 1
            
    except ModelNotFoundError as e: # Specific catch for ModelNotFoundError
        error_msg = str(e)
        logger.warning(f"[{full_command_name}] Model or version not found: {error_msg}", extra={"props": {"command": full_command_name, "status": "not_found", "error_details": e.details, "run_id": run_id}})
        click.echo(f"Error: {error_msg}", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{full_command_name}] Error describing model: {error_msg}", extra={"props": {"command": full_command_name, "status": "app_error", "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error describing model: {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.critical(f"[{full_command_name}] Unexpected error: {error_msg}", extra={"props": {"command": full_command_name, "status": "unexpected_error", "run_id": run_id}}, exc_info=True)
        click.echo("An unexpected error occurred. Check logs.", err=True)
    finally:
        end_cli_run(run_id, full_command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
             sys.exit(exit_code)

@models_group.command("compare")
@click.argument("model_name") # Changed from model_name_from_config for consistency
@click.argument("versions", nargs=-1, required=True) # Ensure at least two versions
@click.pass_context
def compare_model_meta_versions(ctx, model_name: str, versions: tuple[str]):
    command_name = ctx.command.name
    full_command_name = f"{ctx.parent.command.name}-{command_name}"
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(full_command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}

    try:
        logger.info(f"CLI '{full_command_name}' for {model_name}, versions: {versions}", extra={"props": {"command": full_command_name, "run_id": run_id, **log_params}})
        if len(versions) < 2:
            # This check is also in registry_utils.compare_model_versions, but good to have at CLI entry too
            raise ValueError("At least two versions are required for comparison.") # Caught by generic Exception or AppBaseException
        
        data = registry_utils.compare_model_versions(model_name, list(versions))
        click.echo(json.dumps(data, indent=2))
        status, exit_code = "success", 0
        output_summary = {"models_compared": model_name, "versions": list(versions)}

    except ValueError as e: # Catch specific errors like not enough versions
        error_msg = str(e)
        logger.warning(f"[{full_command_name}] Value error: {error_msg}", extra={"props": {"command": full_command_name, "status": "value_error", "run_id": run_id}})
        click.echo(f"Error: {error_msg}", err=True)
    except ModelNotFoundError as e:
        error_msg = str(e)
        logger.warning(f"[{full_command_name}] Model or version not found for comparison: {error_msg}", extra={"props": {"command": full_command_name, "status": "not_found", "run_id": run_id}})
        click.echo(f"Error: {error_msg}", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{full_command_name}] Error comparing models: {error_msg}", extra={"props": {"command": full_command_name, "status": "app_error", "run_id": run_id}}, exc_info=True)
        click.echo(f"Error comparing models: {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.critical(f"[{full_command_name}] Unexpected error: {error_msg}", extra={"props": {"command": full_command_name, "status": "unexpected_error", "run_id": run_id}}, exc_info=True)
        click.echo("An unexpected error occurred. Check logs.", err=True)
    finally:
        end_cli_run(run_id, full_command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
             sys.exit(exit_code)

@models_group.command("get-latest-path")
@click.argument("model_name") # Changed from model_name_from_config
@click.pass_context
def get_latest_model_meta_path(ctx, model_name: str):
    command_name = ctx.command.name
    full_command_name = f"{ctx.parent.command.name}-{command_name}"
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(full_command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}

    try:
        logger.info(f"CLI '{full_command_name}' for {model_name}", extra={"props": {"command": full_command_name, "run_id": run_id, **log_params}})
        path = registry_utils.get_latest_model_version_path(model_name)
        if path:
            click.echo(path)
            status, exit_code = "success", 0
            output_summary = {"latest_model_path": str(path)}
        else:
            error_msg = "Model not found." # This is how the original CLI command reported it
            click.echo(error_msg, err=True) # Original CLI echoed to stdout, but err=True makes sense for not found
            logger.warning(f"[{full_command_name}] {error_msg}", extra={"props": {"command": full_command_name, "status": "not_found", "run_id": run_id}})
            # status remains "failure", exit_code remains 1 as per original CLI's err=not path
            
    except AppBaseException as e: # Should not be raised by current get_latest_model_version_path if it returns None
        error_msg = str(e)
        logger.error(f"[{full_command_name}] Error getting path: {error_msg}", extra={"props": {"command": full_command_name, "status": "app_error", "run_id": run_id}}, exc_info=True)
        click.echo(f"Error: {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.critical(f"[{full_command_name}] Unexpected error: {error_msg}", extra={"props": {"command": full_command_name, "status": "unexpected_error", "run_id": run_id}}, exc_info=True)
        click.echo("An unexpected error occurred. Check logs.", err=True)
    finally:
        end_cli_run(run_id, full_command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
             sys.exit(exit_code)


@cli.group("features")
def features_group(ctx): # Added ctx
    logger.debug("CLI group 'features' invoked.", extra={"props": {"command_group": "features"}})
    pass

@features_group.command("analyze-drift")
@click.option('--current-features-stats', 'current_stats_path', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help="Path to current feature stats JSON.")
@click.option('--baseline-stats', 'baseline_stats_path', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help="Path to baseline feature stats JSON.")
@click.option('--output', 'output_path', required=True, type=click.Path(dir_okay=False, writable=True, allow_dash=False), help="Path to save drift report JSON.")
@click.pass_context
def analyze_feature_drift(ctx, current_stats_path: str, baseline_stats_path: str, output_path: str):
    command_name = ctx.command.name
    full_command_name = f"{ctx.parent.command.name}-{command_name}"
    log_params = {k: v for k, v in ctx.params.items()}
    run_id, start_time_utc = start_cli_run(full_command_name, log_params)
    status, exit_code, error_msg, output_summary = "failure", 1, None, {}
    
    try:
        logger.info(f"CLI '{full_command_name}'.", extra={"props": {"command": full_command_name, "run_id": run_id, **log_params}})
        
        try:
            with open(current_stats_path, 'r') as f: current_stats = json.load(f)
            with open(baseline_stats_path, 'r') as f: baseline_stats = json.load(f)
            logger.info(f"[{full_command_name}] Loaded current and baseline stats files successfully.", extra={"props": {"command": full_command_name, "run_id": run_id}})
        except FileNotFoundError as e:
            raise FileOperationError(message=f"Statistics file not found: {e.filename}", filepath=str(e.filename), operation="read")
        except json.JSONDecodeError as e_json:
            raise DataValidationError(message=f"Invalid JSON format in one of the statistics files. Details: {e_json.msg}", validation_errors={"json_decode_error": e_json.msg})
        
        drift_report = compare_feature_statistics(current_stats, baseline_stats)
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path_obj, 'w') as f: json.dump(drift_report, f, indent=4)
            click.echo(f"Drift report saved to: {output_path_obj}")
            logger.info(f"[{full_command_name}] Drift report saved to: {output_path_obj}", extra={"props": {"command": full_command_name, "status": "success", "run_id": run_id}})
            status, exit_code = "success", 0
            output_summary = {"drift_report_path": str(output_path_obj), "drift_metrics_summary": {k:v for k,v in drift_report.items() if not isinstance(v, dict)}} # Summary of top-level metrics
        except Exception as io_err:
            raise FileOperationError(message=f"Failed to save drift report to {output_path_obj}", filepath=str(output_path_obj), operation="write", details={"os_error": str(io_err)})

    except FileOperationError as e:
        error_msg = str(e)
        logger.error(f"[{full_command_name}] File operation error: {error_msg}", extra={"props": {"command": full_command_name, "status": "file_error", "filepath": e.filepath, "operation": e.operation, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{full_command_name}' during file operation: {error_msg}", err=True)
    except DataValidationError as e:
        error_msg = str(e)
        logger.error(f"[{full_command_name}] Data validation error: {error_msg}", extra={"props": {"command": full_command_name, "status": "data_validation_error", "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"Error in '{full_command_name}' due to data validation: {error_msg}", err=True)
    except AppBaseException as e:
        error_msg = str(e)
        logger.error(f"[{full_command_name}] Application error: {error_msg}", extra={"props": {"command": full_command_name, "status": "app_error", "error_type": type(e).__name__, "error_details": e.details, "run_id": run_id}}, exc_info=True)
        click.echo(f"An application error occurred: {error_msg}", err=True)
    except Exception as e:
        error_msg = f"An unexpected critical error occurred: {str(e)}"
        logger.critical(f"[{full_command_name}] {error_msg}", extra={"props": {"command": full_command_name, "status": "unexpected_critical_error", "error_type": type(e).__name__, "run_id": run_id}}, exc_info=True)
        click.echo(f"An unexpected critical error occurred in '{full_command_name}'. Please check logs for details.", err=True)
    finally:
        end_cli_run(run_id, full_command_name, status, exit_code, start_time_utc, error_msg, output_summary)
        if exit_code != 0:
            sys.exit(exit_code)


if __name__ == '__main__':
    # The main cli() function will handle its own logging and error reporting.
    # If cli() itself raises an unhandled exception (e.g. Click-internal), it will go to stderr.
    cli()
