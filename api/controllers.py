# This file will contain the core logic for handling API requests and
# interacting with the backend services (e.g., functions in src/).
# These controller functions are designed to be called by API route handlers
# (e.g., from a FastAPI or Flask app) or gRPC service implementations.

# For a REST API (e.g., using FastAPI):
# - Each controller function would typically be called by a route handler
#   (e.g., @app.post("/train")).
# - Request data (parsed from JSON by Pydantic models defined in api/models.py)
#   would be passed as arguments to these functions. These Pydantic models in api/models.py
#   would define the API request/response schema.
# - The controller would then map this API request model to the relevant
#   Pydantic configuration model from src.config_models (e.g., LoadDataConfig).
# - This src.config_models instance would be used to call the backend service.
# - The return values from these functions (often Pydantic models from api/models.py) would be
#   serialized to JSON and sent as HTTP responses.

# For a gRPC API:
# - Similar logic, mapping protobuf messages to/from Pydantic models.

import logging
# Assuming src.exceptions and relevant Pydantic models are accessible
# For example, if api is a package: from ..src.exceptions import ...
# For simplicity, let's assume they are importable directly or adjust path as needed.
from src.exceptions import AppBaseException, ConfigError, DataValidationError, ModelTrainingError, PipelineError, FileOperationError, APIError
# Core Pydantic models from src
from src.config_models import (
    LoadDataConfig, FeatureEngineeringConfig, TargetVariableConfig,
    TrainModelConfig, EvaluateModelConfig, BacktestConfig, ExportConfig
)
# API specific Pydantic models
from api.models import (
    LoadDataApiRequest, LoadDataApiResponse,
    FeatureEngineeringApiRequest, FeatureEngineeringApiResponse, TargetVariableConfigApi,
    TrainModelApiRequest, TrainModelApiResponse,
    EvaluateModelApiRequest, EvaluateModelApiResponse,
    BacktestApiRequest, BacktestApiResponse,
    ExportModelApiRequest, ExportModelApiResponse,
    ListModelsApiRequest, ListModelsApiResponse, ModelRegistryEntry,
    DescribeModelApiRequest, DescribeModelApiResponse,
    CompareModelsApiRequest, CompareModelsApiResponse,
    GetLatestModelPathApiRequest, GetLatestModelPathApiResponse,
    AnalyzeDriftApiRequest, AnalyzeDriftApiResponse,
    StatusResponse, ApiErrorResponse # General models
)
# Assumed src module imports - these might need adjustment or new functions in src
from src import data_management
from src import feature_engineering
from src import modeling
from src import evaluation
from src import backtesting
from src import model_registry_utils
from src import feature_analysis
from src import utils # For potential generic helpers like save/load model if not part of specific modules

import pandas as pd # For creating dummy data preview in load_data_controller
from pathlib import Path # For path manipulations
import os # For os.path.join, etc.

logger = logging.getLogger(__name__)

# Helper to create a standardized error response (already present from previous task)
def _create_error_response(message: str, status_code: int, error_type: str = "APIError", details: Optional[dict] = None):
    # ... (implementation remains the same)
    response = {
        "error": {
            "type": error_type,
            "message": message,
            "details": details or {}
        }
    }
    return response, status_code

def load_data_controller(api_request_data: dict):
    controller_name = "load_data_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data_summary": {k:v for k,v in api_request_data.items() if k != 'data'}}})
    try:
        # 1. Input Validation
        try:
            api_req_model = LoadDataApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Configuration Mapping
        # Define output path based on request or defaults
        project_root = Path(utils.__file__).resolve().parent.parent # Assuming utils.py is in src/
        raw_data_dir = project_root / "data" / "raw"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{api_req_model.ticker.lower()}{api_req_model.output_filename_suffix or ''}.parquet"
        core_config_output_path = str(raw_data_dir / output_filename)

        try:
            core_config = LoadDataConfig(
                ticker=api_req_model.ticker,
                start_date=api_req_model.start_date.strftime("%Y-%m-%d"),
                end_date=api_req_model.end_date.strftime("%Y-%m-%d"),
                output_parquet_path=core_config_output_path
                # data_source could be mapped here if part of api_req_model and core_config
            )
        except Exception as e: # Pydantic's ValidationError for core_config
            logger.error(f"[{controller_name}] Core config mapping failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "core_config_error": str(e)}})
            raise APIError(message=f"Internal configuration error: {e}", status_code=500, details={"config_mapping_error": str(e)})

        # 3. Call Core Logic
        # Assuming a function like `download_stock_data_from_config` exists or is created in `src.data_management`
        # For now, let's simulate this and assume it returns a DataFrame.
        # In reality, this function would handle the download and save to core_config.output_parquet_path
        
        # --- Placeholder for src.data_management.download_stock_data_from_config(core_config) ---
        # This function should:
        # - Take LoadDataConfig as input.
        # - Download data.
        # - Save it to core_config.output_parquet_path.
        # - Return number of records, or raise specific exceptions.
        # Example simulation:
        logger.info(f"[{controller_name}] Simulating data download for ticker {core_config.ticker} to {core_config.output_parquet_path}",
                    extra={"props": {"controller": controller_name, "config": core_config.model_dump()}})
        
        # Create a dummy parquet file for illustration
        num_records_processed = 100 # Simulate record count
        dummy_df_data = {
            'Date': pd.to_datetime(pd.date_range(core_config.start_date, periods=num_records_processed, freq='B')),
            'Open': [i*1.0 for i in range(num_records_processed)],
            'High': [i*1.1 for i in range(num_records_processed)],
            'Low': [i*0.9 for i in range(num_records_processed)],
            'Close': [i*1.05 for i in range(num_records_processed)],
            'Volume': [i*1000 for i in range(num_records_processed)]
        }
        dummy_df = pd.DataFrame(dummy_df_data)
        dummy_df.to_parquet(core_config.output_parquet_path, index=False)
        # --- End Placeholder ---

        # 4. Response Formatting
        preview_df = dummy_df.head() # In real scenario, load from file if function doesn't return df
        response_payload = LoadDataApiResponse(
            status="success",
            message=f"Data for {api_req_model.ticker} loaded and saved to {core_config.output_parquet_path}",
            output_data_path=core_config.output_parquet_path,
            records_processed=num_records_processed,
            data_preview=preview_df.to_dict(orient="records")
        )
        return response_payload.model_dump(), 200

    except APIError as e: # Catch APIError explicitly to forward its status code
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except (DataValidationError, ConfigError, FileOperationError, ExternalServiceError) as e: # Specific app exceptions
        logger.error(f"[{controller_name}] Application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": getattr(e, 'details', str(e))}})
        status_code = 422 if isinstance(e, DataValidationError) else 500 # Unprocessable Entity for validation
        return _create_error_response(e.message, status_code, error_type=type(e).__name__, details=getattr(e, 'details', None))
    except AppBaseException as e: # Other custom app exceptions
        logger.error(f"[{controller_name}] Base application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details}})
        return _create_error_response(e.message, 500, error_type=type(e).__name__, details=e.details)
    except Exception as e: # Catch-all for unexpected errors
        logger.critical(f"[{controller_name}] Unexpected critical error: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response("An unexpected internal server error occurred.", 500, error_type="InternalServerError")


def engineer_features_controller(api_request_data: dict):
    controller_name = "engineer_features_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data_summary": {k:v for k,v in api_request_data.items() if 'path' not in k}}}) # Avoid logging full paths if too verbose
    try:
        # 1. Input Validation
        try:
            api_req_model = FeatureEngineeringApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Configuration Mapping
        project_root = Path(utils.__file__).resolve().parent.parent
        processed_data_dir = project_root / "data" / "processed"
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename based on input filename and suffix
        input_filename_stem = Path(api_req_model.input_data_path).stem
        output_filename = f"{input_filename_stem}{api_req_model.output_filename_suffix or '_features'}.parquet"
        core_config_output_path = str(processed_data_dir / output_filename)

        try:
            # Map TargetVariableConfigApi to core TargetVariableConfig
            core_target_config = None
            if api_req_model.target_variable:
                core_target_config = TargetVariableConfig(**api_req_model.target_variable.model_dump())

            core_config = FeatureEngineeringConfig(
                input_data_path=str(api_req_model.input_data_path), # Ensure it's str for Pydantic if FilePath was used in API model
                output_features_path=core_config_output_path,
                technical_indicators=api_req_model.technical_indicators,
                rolling_lag_features=api_req_model.rolling_lag_features,
                sentiment_features=api_req_model.sentiment_features,
                fundamental_features=api_req_model.fundamental_features,
                target_variable=core_target_config
                # ticker might need to be passed if sentiment/fundamental require it and it's not in input_data_path name
            )
        except Exception as e: # Pydantic's ValidationError for core_config
            logger.error(f"[{controller_name}] Core config mapping failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "core_config_error": str(e)}})
            raise APIError(message=f"Internal configuration error: {e}", status_code=500, details={"config_mapping_error": str(e)})

        # 3. Call Core Logic
        # Assume a function like `generate_features_from_config(config: FeatureEngineeringConfig)` exists in `src.feature_engineering`
        # This function should load data from config.input_data_path, generate features, and save to config.output_features_path.
        # It should return a list of generated feature names and the shape of the output data.
        
        # --- Placeholder for src.feature_engineering.generate_features_from_config(core_config) ---
        logger.info(f"[{controller_name}] Simulating feature engineering from {core_config.input_data_path} to {core_config.output_features_path}",
                    extra={"props": {"controller": controller_name, "config": core_config.model_dump()}})
        # Dummy feature generation
        try:
            input_df = pd.read_parquet(core_config.input_data_path)
        except Exception as e_load:
            raise FileOperationError(message=f"Failed to load input data from {core_config.input_data_path}: {e_load}", filepath=core_config.input_data_path, operation="read")
        
        # Simulate adding features
        generated_feature_names = list(input_df.columns)
        if core_config.technical_indicators: generated_feature_names.extend(["SMA_20", "RSI_14"])
        if core_config.rolling_lag_features: generated_feature_names.extend(["lag_1", "rolling_mean_5"])
        if core_config.target_variable and core_config.target_variable.enabled: generated_feature_names.append("target")
        
        # Create a dummy output DataFrame
        output_df = pd.DataFrame(index=input_df.index)
        for col in generated_feature_names:
            if col in input_df.columns: output_df[col] = input_df[col]
            else: output_df[col] = [i for i in range(len(input_df))] # dummy values
        
        output_df.to_parquet(core_config.output_features_path, index=False)
        num_output_features = len(output_df.columns)
        output_data_shape = output_df.shape
        # --- End Placeholder ---

        # 4. Response Formatting
        response_payload = FeatureEngineeringApiResponse(
            status="success",
            message=f"Features engineered and saved to {core_config.output_features_path}",
            output_features_path=core_config.output_features_path,
            features_generated=list(set(generated_feature_names)), # Unique names
            num_features=num_output_features,
            data_shape=output_data_shape
        )
        return response_payload.model_dump(), 200

    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except (DataValidationError, ConfigError, FileOperationError, PipelineError) as e:
        logger.error(f"[{controller_name}] Application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": getattr(e, 'details', str(e))}})
        status_code = 422 if isinstance(e, DataValidationError) else 500
        return _create_error_response(e.message, status_code, error_type=type(e).__name__, details=getattr(e, 'details', None))
    except AppBaseException as e:
        logger.error(f"[{controller_name}] Base application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details}})
        return _create_error_response(e.message, 500, error_type=type(e).__name__, details=e.details)
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response("An unexpected internal server error occurred.", 500, error_type="InternalServerError")


def train_model_controller(api_request_data: dict):
    controller_name = "train_model_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data_summary": {k:v for k,v in api_request_data.items() if 'path' not in k and 'params' not in k}}})
    try:
        # 1. Input Validation
        try:
            api_req_model = TrainModelApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Configuration Mapping
        project_root = Path(utils.__file__).resolve().parent.parent
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Construct model output base path
        # Example: models/xgboost_stockpredictor_final_production
        model_base_name = f"{api_req_model.model_type.lower()}_{Path(api_req_model.input_features_path).stem.replace('_features', '')}{api_req_model.model_name_suffix or ''}"
        core_model_output_path_base = str(models_dir / model_base_name)
        
        core_scaler_output_path = None
        if api_req_model.model_type in ["LSTM", "CNN-LSTM", "Transformer"]: # Example types needing scalers
             scaler_filename = f"{model_base_name}{api_req_model.scaler_filename_suffix or '_scaler'}.pkl"
             core_scaler_output_path = str(models_dir / model_base_name / scaler_filename) # Scaler inside model's dir
             (models_dir / model_base_name).mkdir(parents=True, exist_ok=True)


        try:
            core_config = TrainModelConfig(
                input_features_path=str(api_req_model.input_features_path),
                model_output_path_base=core_model_output_path_base, # This is a base, specific filenames added by training script
                scaler_output_path=core_scaler_output_path,
                model_type=api_req_model.model_type,
                model_params=api_req_model.model_params,
                target_column=api_req_model.target_column
            )
        except Exception as e: # Pydantic's ValidationError for core_config
            logger.error(f"[{controller_name}] Core config mapping failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "core_config_error": str(e)}})
            raise APIError(message=f"Internal configuration error: {e}", status_code=500, details={"config_mapping_error": str(e)})

        # 3. Call Core Logic
        # Assume a function like `train_model_from_config(config: TrainModelConfig)` exists in `src.modeling`
        # This function would handle the actual training, saving the model and scaler (if any).
        # It might return training duration or other metadata.
        
        # --- Placeholder for src.modeling.train_model_from_config(core_config) ---
        logger.info(f"[{controller_name}] Simulating model training for type {core_config.model_type} from {core_config.input_features_path}",
                    extra={"props": {"controller": controller_name, "config": core_config.model_dump()}})
        
        # Dummy model and scaler file creation
        Path(core_config.model_output_path_base).mkdir(parents=True, exist_ok=True) # Ensure base dir exists
        # Actual model path would be like core_config.model_output_path_base / f"{core_config.model_type}_model.pkl" (or .h5)
        # This path is determined *inside* the src.modeling.train_model_from_config typically.
        # For the response, we return the base path.
        
        # Simulate training time
        training_duration = 60.5 
        # --- End Placeholder ---

        # 4. Response Formatting
        response_payload = TrainModelApiResponse(
            status="success",
            message=f"Model training for {api_req_model.model_type} initiated. Artifacts will be based at {core_config.model_output_path_base}",
            model_output_path_base=core_config.model_output_path_base,
            model_type=api_req_model.model_type,
            training_duration_seconds=training_duration
        )
        return response_payload.model_dump(), 200

    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except (ModelTrainingError, DataValidationError, ConfigError, FileOperationError) as e:
        logger.error(f"[{controller_name}] Application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": getattr(e, 'details', str(e))}})
        status_code = 422 if isinstance(e, DataValidationError) else 500
        return _create_error_response(e.message, status_code, error_type=type(e).__name__, details=getattr(e, 'details', None))
    except AppBaseException as e:
        logger.error(f"[{controller_name}] Base application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details}})
        return _create_error_response(e.message, 500, error_type=type(e).__name__, details=e.details)
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response("An unexpected internal server error occurred.", 500, error_type="InternalServerError")

# Similar error handling for other controllers: evaluate_model, backtest_strategy, export_model

def evaluate_model_controller(api_request_data: dict):
    controller_name = "evaluate_model_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data_summary": {k:v for k,v in api_request_data.items() if 'path' not in k}}})
    try:
        # 1. Input Validation
        try:
            api_req_model = EvaluateModelApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Configuration Mapping
        project_root = Path(utils.__file__).resolve().parent.parent
        reports_dir = project_root / "data" / "reports" # Or a dedicated "evaluations" directory
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Construct metrics output path
        model_name_stem = Path(api_req_model.model_path).stem
        metrics_filename = f"{model_name_stem}_evaluation_metrics.json"
        core_metrics_output_path = str(reports_dir / metrics_filename)

        shap_plot_output_path = None
        if api_req_model.shap_summary_plot:
            shap_plot_filename = f"{model_name_stem}_shap_summary.png"
            shap_plot_output_path = str(reports_dir / shap_plot_filename)

        try:
            core_config = EvaluateModelConfig(
                model_path=str(api_req_model.model_path),
                test_data_path=str(api_req_model.test_data_path),
                scaler_path=str(api_req_model.scaler_path) if api_req_model.scaler_path else None,
                metrics_output_json_path=core_metrics_output_path,
                metrics_to_compute=api_req_model.metrics_to_compute,
                shap_summary_plot_path=shap_plot_output_path,
                # model_type might be needed by evaluation script for SHAP, could be passed in API request or inferred
            )
        except Exception as e: # Pydantic's ValidationError for core_config
            logger.error(f"[{controller_name}] Core config mapping failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "core_config_error": str(e)}})
            raise APIError(message=f"Internal configuration error: {e}", status_code=500, details={"config_mapping_error": str(e)})

        # 3. Call Core Logic
        # Assume `src.evaluation.evaluate_model_from_config(config: EvaluateModelConfig)`
        # This function would:
        # - Load model (and scaler if path provided).
        # - Load test data.
        # - Make predictions.
        # - Calculate metrics.
        # - Save metrics to core_config.metrics_output_json_path.
        # - Optionally generate and save SHAP plot to core_config.shap_summary_plot_path.
        # - Return calculated metrics and path to SHAP plot if generated.
        
        # --- Placeholder for src.evaluation.evaluate_model_from_config(core_config) ---
        logger.info(f"[{controller_name}] Simulating model evaluation for model {core_config.model_path}",
                    extra={"props": {"controller": controller_name, "config": core_config.model_dump()}})
        
        # Dummy metrics calculation
        calculated_metrics = {"accuracy": 0.85, "precision": 0.80, "recall": 0.88, "f1_score": 0.84}
        if core_config.metrics_to_compute: # Filter if specific metrics requested
            calculated_metrics = {k: v for k, v in calculated_metrics.items() if k in core_config.metrics_to_compute}
        
        # Save dummy metrics
        with open(core_config.metrics_output_json_path, 'w') as f:
            json.dump(calculated_metrics, f, indent=4)
            
        actual_shap_plot_path = None
        if core_config.shap_summary_plot_path:
            # Create a dummy SHAP plot image file
            try:
                Path(core_config.shap_summary_plot_path).touch() # Create empty file as placeholder
                actual_shap_plot_path = core_config.shap_summary_plot_path
                logger.info(f"[{controller_name}] Dummy SHAP plot saved to {actual_shap_plot_path}", extra={"props": {"controller": controller_name}})
            except Exception as e_shap:
                logger.error(f"[{controller_name}] Error creating dummy SHAP plot: {e_shap}", exc_info=True)
        # --- End Placeholder ---

        # 4. Response Formatting
        response_payload = EvaluateModelApiResponse(
            status="success",
            message="Model evaluation completed.",
            metrics=calculated_metrics,
            metrics_output_json_path=core_config.metrics_output_json_path,
            shap_summary_plot_path=actual_shap_plot_path
        )
        return response_payload.model_dump(), 200

    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except (DataValidationError, ConfigError, FileOperationError, PipelineError) as e: # FileOperationError for model/data not found
        status_code = 404 if isinstance(e, FileOperationError) else (422 if isinstance(e, DataValidationError) else 500)
        logger.error(f"[{controller_name}] Application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": getattr(e, 'details', str(e))}})
        return _create_error_response(e.message, status_code, error_type=type(e).__name__, details=getattr(e, 'details', None))
    except AppBaseException as e:
        logger.error(f"[{controller_name}] Base application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details}})
        return _create_error_response(e.message, 500, error_type=type(e).__name__, details=e.details)
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response("An unexpected internal server error occurred.", 500, error_type="InternalServerError")


def backtest_strategy_controller(api_request_data: dict):
    controller_name = "backtest_strategy_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data_summary": {k:v for k,v in api_request_data.items() if 'path' not in k}}})
    try:
        # 1. Input Validation
        try:
            api_req_model = BacktestApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Configuration Mapping
        project_root = Path(utils.__file__).resolve().parent.parent
        backtest_reports_dir = project_root / "data" / "reports" / "backtests"
        backtest_reports_dir.mkdir(parents=True, exist_ok=True)

        # Construct results output path
        # Example: data/reports/backtests/predictions_stem_simple_threshold_strategy_results.json
        predictions_stem = Path(api_req_model.predictions_path).stem
        results_filename = f"{predictions_stem}{api_req_model.results_output_filename_suffix or ''}_results.json"
        core_results_output_path = str(backtest_reports_dir / results_filename)

        try:
            core_config = BacktestConfig(
                ohlcv_data_path=str(api_req_model.ohlcv_data_path),
                predictions_path=str(api_req_model.predictions_path),
                strategy_config=api_req_model.strategy_config,
                results_output_path=core_results_output_path
                # initial_capital could be part of api_req_model and mapped here
            )
        except Exception as e: # Pydantic's ValidationError for core_config
            logger.error(f"[{controller_name}] Core config mapping failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "core_config_error": str(e)}})
            raise APIError(message=f"Internal configuration error: {e}", status_code=500, details={"config_mapping_error": str(e)})

        # 3. Call Core Logic
        # Assume `src.backtesting.run_backtest_from_config(config: BacktestConfig)`
        # This function would:
        # - Load OHLCV data and predictions.
        # - Apply the strategy defined in strategy_config.
        # - Calculate backtesting metrics (e.g., total return, Sharpe ratio).
        # - Save detailed results/trades to core_config.results_output_path.
        # - Return a summary of the results.
        
        # --- Placeholder for src.backtesting.run_backtest_from_config(core_config) ---
        logger.info(f"[{controller_name}] Simulating backtest with predictions from {core_config.predictions_path}",
                    extra={"props": {"controller": controller_name, "config": core_config.model_dump()}})
        
        # Dummy results calculation
        backtest_results_summary = {
            "total_return_pct": 12.5, 
            "sharpe_ratio": 1.1, 
            "max_drawdown_pct": -8.2,
            "num_trades": 55
        }
        
        # Save dummy results to file
        with open(core_config.results_output_path, 'w') as f:
            json.dump({"summary": backtest_results_summary, "trades_log": [{"trade_1": "..."}, {"trade_2": "..."}]}, f, indent=4)
        # --- End Placeholder ---

        # 4. Response Formatting
        response_payload = BacktestApiResponse(
            status="success",
            message="Backtest completed successfully.",
            results=backtest_results_summary,
            results_output_path=core_config.results_output_path
        )
        return response_payload.model_dump(), 200

    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except (DataValidationError, ConfigError, FileOperationError, PipelineError) as e:
        status_code = 404 if isinstance(e, FileOperationError) else (422 if isinstance(e, DataValidationError) else 500)
        logger.error(f"[{controller_name}] Application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": getattr(e, 'details', str(e))}})
        return _create_error_response(e.message, status_code, error_type=type(e).__name__, details=getattr(e, 'details', None))
    except AppBaseException as e:
        logger.error(f"[{controller_name}] Base application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details}})
        return _create_error_response(e.message, 500, error_type=type(e).__name__, details=e.details)
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response("An unexpected internal server error occurred.", 500, error_type="InternalServerError")


def export_model_controller(api_request_data: dict):
    controller_name = "export_model_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data_summary": {k:v for k,v in api_request_data.items() if 'path' not in k}}})
    try:
        # 1. Input Validation
        try:
            api_req_model = ExportModelApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Configuration Mapping
        project_root = Path(utils.__file__).resolve().parent.parent
        exported_models_dir = project_root / "models" / "exported"
        exported_models_dir.mkdir(parents=True, exist_ok=True)

        # Construct export output path
        # Example: models/exported/trained_model_stem_for_serving.onnx
        trained_model_stem = Path(api_req_model.trained_model_path).stem
        export_filename = f"{trained_model_stem}{api_req_model.export_filename_suffix or ''}.{api_req_model.export_type.lower().replace('joblib_zip', 'zip')}"
        core_export_output_path = str(exported_models_dir / export_filename)

        try:
            core_config = ExportConfig(
                trained_model_path=str(api_req_model.trained_model_path),
                export_type=api_req_model.export_type,
                export_output_path=core_export_output_path
                # input_sample_path could be mapped here if needed for ONNX
            )
        except Exception as e: # Pydantic's ValidationError for core_config
            logger.error(f"[{controller_name}] Core config mapping failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "core_config_error": str(e)}})
            raise APIError(message=f"Internal configuration error: {e}", status_code=500, details={"config_mapping_error": str(e)})

        # 3. Call Core Logic
        # Assume `src.utils.export_model_artifact(config: ExportConfig)` or similar in a relevant module.
        # This function would:
        # - Load the trained model.
        # - Perform the export (e.g., convert to ONNX, or zip joblib/pickle).
        # - Save the exported artifact to core_config.export_output_path.
        
        # --- Placeholder for src.utils.export_model_artifact(core_config) ---
        logger.info(f"[{controller_name}] Simulating model export of {core_config.trained_model_path} to {core_config.export_output_path} as {core_config.export_type}",
                    extra={"props": {"controller": controller_name, "config": core_config.model_dump()}})
        
        # Dummy export artifact creation
        try:
            if not Path(core_config.trained_model_path).exists(): # Check if model to export exists
                 raise FileOperationError(message=f"Trained model not found at {core_config.trained_model_path}", filepath=core_config.trained_model_path, operation="read")
            Path(core_config.export_output_path).touch() # Create empty file as placeholder
            logger.info(f"[{controller_name}] Dummy exported model artifact created at {core_config.export_output_path}", extra={"props": {"controller": controller_name}})
        except Exception as e_export:
            # This might be a FileOperationError if the trained_model_path doesn't exist,
            # or other errors during a real export process.
            logger.error(f"[{controller_name}] Error during dummy export artifact creation: {e_export}", exc_info=True)
            raise PipelineError(message=f"Failed during model export process: {e_export}", stage_name="model_export")
        # --- End Placeholder ---

        # 4. Response Formatting
        response_payload = ExportModelApiResponse(
            status="success",
            message=f"Model exported successfully as {api_req_model.export_type} to {core_config.export_output_path}",
            export_output_path=core_config.export_output_path
        )
        return response_payload.model_dump(), 200

    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except (DataValidationError, ConfigError, FileOperationError, PipelineError) as e:
        status_code = 404 if isinstance(e, FileOperationError) else (422 if isinstance(e, DataValidationError) else 500)
        logger.error(f"[{controller_name}] Application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": getattr(e, 'details', str(e))}})
        return _create_error_response(e.message, status_code, error_type=type(e).__name__, details=getattr(e, 'details', None))
    except AppBaseException as e:
        logger.error(f"[{controller_name}] Base application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details}})
        return _create_error_response(e.message, 500, error_type=type(e).__name__, details=e.details)
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response("An unexpected internal server error occurred.", 500, error_type="InternalServerError")

# api/controllers.py

# --- Conceptual API Design Notes ---
#
# API Versioning:
# This API, when fully implemented (e.g., using FastAPI or Flask-RESTx), would benefit
# from versioning to manage changes and ensure backward compatibility for clients.
# Common strategies include:
# 1. URL Path Versioning: e.g., /api/v1/load-data, /api/v2/load-data
#    - Implemented by structuring routes under versioned routers or blueprints.
#    - FastAPI example: app.include_router(router_v1, prefix="/api/v1")
# 2. Header Versioning: Clients specify version in an HTTP header (e.g., `Accept: application/vnd.myapi.v1+json`).
#    - Requires middleware to parse the header and dispatch to the correct controller logic.
#
# The choice of versioning strategy depends on specific needs. Path versioning is often simpler to start with.
# Versioning would apply to request/response models in `api/models.py` as well, potentially having
# different Pydantic models for different API versions if breaking changes are introduced.

# Swagger/OpenAPI Documentation:
# If this API were built using a modern Python framework like FastAPI, OpenAPI (formerly Swagger)
# documentation would be automatically generated based on:
# - Path operations (routes) defined.
# - Pydantic models used for request bodies, response bodies, and query/path parameters.
# - Docstrings of the path operation functions and Pydantic models.
#
# For example, in FastAPI:
# from fastapi import FastAPI
# app = FastAPI(title="QLOP API", version="1.0.0", description="API for Quantitative Leverage Opportunity Predictor")
#
# # @app.post("/load-data", response_model=LoadDataApiResponse, tags=["Data Management"])
# # async def handle_load_data(request: LoadDataApiRequest):
# #     # ... logic using load_data_controller ...
# #     pass
#
# This would automatically provide:
# - An OpenAPI schema at `/openapi.json`.
# - Interactive API documentation via Swagger UI at `/docs`.
# - Alternative documentation via ReDoc at `/redoc`.
#
# Pydantic models in `api/models.py` are crucial as they define the schema for requests and responses,
# which is then reflected in the OpenAPI specification.
#
# --- End Conceptual API Design Notes ---


# Add Optional to imports if not already there
from typing import Optional

# --- Model Registry Controllers ---

def list_models_controller(api_request_data: dict):
    controller_name = "list_models_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data": api_request_data}})
    try:
        # 1. Input Validation (optional filter)
        api_req_model = ListModelsApiRequest(**api_request_data) # Handles if request is empty or has the filter

        # 2. Call Core Logic (from src.model_registry_utils)
        # Assuming model_registry_utils.list_models can accept None for filter
        models_list = model_registry_utils.list_models(model_name_filter=api_req_model.model_name_filter)
        
        # Ensure models_list is a list of dicts that can be parsed by ModelRegistryEntry
        # The list_models function in model_registry_utils.py already returns a list of dicts.
        # Pydantic will validate each entry when creating ListModelsApiResponse.

        # 3. Response Formatting
        response_payload = ListModelsApiResponse(models=models_list)
        return response_payload.model_dump(), 200

    except APIError as e: # Should not happen here if ListModelsApiRequest is simple
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except Exception as e: # Catch-all for unexpected errors during model listing
        logger.critical(f"[{controller_name}] Unexpected critical error: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        # This could be an issue with model_registry_utils.list_models or Pydantic parsing if entries are malformed
        return _create_error_response(f"An unexpected error occurred while listing models: {str(e)}", 500, error_type="InternalServerError")

def describe_model_controller(api_request_data: dict): # Takes dict to align with other controllers for now
    controller_name = "describe_model_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data": api_request_data}})
    try:
        # 1. Input Validation
        try:
            api_req_model = DescribeModelApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Call Core Logic
        model_details = model_registry_utils.get_model_details(
            model_name_from_config=api_req_model.model_name, # Pydantic model uses model_name
            version=api_req_model.version
        )

        if model_details is None:
            logger.warning(f"[{controller_name}] Model not found: {api_req_model.model_name} v{api_req_model.version}", extra={"props": {"controller": controller_name}})
            raise APIError(message=f"Model '{api_req_model.model_name}' version '{api_req_model.version}' not found.", status_code=404)

        # 3. Response Formatting
        response_payload = DescribeModelApiResponse(details=model_details)
        return response_payload.model_dump(), 200
        
    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error for model {api_request_data.get('model_name')} v{api_request_data.get('version')}: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response(f"An unexpected error occurred: {str(e)}", 500, error_type="InternalServerError")

def compare_models_controller(api_request_data: dict):
    controller_name = "compare_models_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data": api_request_data}})
    try:
        # 1. Input Validation
        try:
            api_req_model = CompareModelsApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Call Core Logic
        comparison_data = model_registry_utils.compare_model_versions(
            model_name_from_config=api_req_model.model_name,
            versions_to_compare=api_req_model.versions
        )
        # compare_model_versions should raise an error if versions are not found or not enough versions.
        # Or handle it here based on its return value. Assuming it returns a dict or raises.

        # 3. Response Formatting
        response_payload = CompareModelsApiResponse(comparison_data=comparison_data)
        return response_payload.model_dump(), 200

    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except FileNotFoundError as e: # Example if compare_model_versions raises this for missing meta files
        logger.error(f"[{controller_name}] File not found during model comparison: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        raise APIError(message=f"One or more model versions not found for comparison: {e}", status_code=404)
    except ValueError as e: # Example if compare_model_versions raises this for bad input (e.g. not enough versions)
        logger.warning(f"[{controller_name}] Value error during model comparison: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        raise APIError(message=str(e), status_code=400) # Bad request
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error for model {api_request_data.get('model_name')} versions {api_request_data.get('versions')}: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response(f"An unexpected error occurred: {str(e)}", 500, error_type="InternalServerError")

def get_latest_model_path_controller(api_request_data: dict):
    controller_name = "get_latest_model_path_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data": api_request_data}})
    try:
        # 1. Input Validation
        try:
            api_req_model = GetLatestModelPathApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Call Core Logic
        # This util function returns the path to the metadata JSON, not the model file itself directly.
        # The API response model `GetLatestModelPathApiResponse` has `path: Optional[str]`.
        # We might need to clarify if this path should be the model artifact or metadata.
        # Assuming metadata path for now as per registry_utils.
        latest_path = model_registry_utils.get_latest_model_version_path(model_name_from_config=api_req_model.model_name)

        # 3. Response Formatting
        if latest_path:
            response_payload = GetLatestModelPathApiResponse(path=str(latest_path), message="Latest model metadata path retrieved.")
            return response_payload.model_dump(), 200
        else:
            logger.warning(f"[{controller_name}] No model found with name: {api_req_model.model_name}", extra={"props": {"controller": controller_name}})
            # Return 200 with message, or 404? API spec for this response model implies 200 with message.
            response_payload = GetLatestModelPathApiResponse(path=None, message=f"No model found with name '{api_req_model.model_name}'.")
            return response_payload.model_dump(), 200 # Or 404 if preferred
            # raise APIError(message=f"No model found with name '{api_req_model.model_name}'.", status_code=404)


    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error for model {api_request_data.get('model_name')}: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response(f"An unexpected error occurred: {str(e)}", 500, error_type="InternalServerError")

# --- Feature Drift Analysis Controller ---

def analyze_feature_drift_controller(api_request_data: dict):
    controller_name = "analyze_feature_drift_controller"
    logger.info(f"[{controller_name}] called.", extra={"props": {"controller": controller_name, "request_data_summary": {k:v for k,v in api_request_data.items() if 'path' not in k}}})
    try:
        # 1. Input Validation
        try:
            api_req_model = AnalyzeDriftApiRequest(**api_request_data)
        except Exception as e: # Pydantic's ValidationError
            logger.warning(f"[{controller_name}] Request validation failed: {e}", exc_info=True, extra={"props": {"controller": controller_name, "validation_error": str(e)}})
            raise APIError(message=f"Invalid request payload: {e}", status_code=400, details={"validation_errors": e.errors() if hasattr(e, 'errors') else str(e)})

        # 2. Configuration / Path Mapping
        project_root = Path(utils.__file__).resolve().parent.parent
        drift_reports_dir = project_root / "data" / "reports" / "feature_drift"
        drift_reports_dir.mkdir(parents=True, exist_ok=True)

        # Construct drift report output path
        # Example: data/reports/feature_drift/current_vs_baseline_drift_report_q2_2024.json
        current_stem = Path(api_req_model.current_features_stats_path).stem.replace('_current_stats', '')
        baseline_stem = Path(api_req_model.baseline_stats_path).stem.replace('_baseline_stats', '')
        report_filename = f"{current_stem}_vs_{baseline_stem}{api_req_model.output_filename_suffix or ''}_drift_report.json"
        core_drift_report_output_path = str(drift_reports_dir / report_filename)
        
        # The core function feature_analysis.compare_feature_statistics directly takes stats dicts, not paths.
        # So, we need to load these JSON files first.

        try:
            with open(api_req_model.current_features_stats_path, 'r') as f_curr:
                current_stats = json.load(f_curr)
            with open(api_req_model.baseline_stats_path, 'r') as f_base:
                baseline_stats = json.load(f_base)
        except FileNotFoundError as e:
            logger.error(f"[{controller_name}] Statistics file not found: {e.filename}", exc_info=True)
            raise APIError(message=f"Statistics file not found: {e.filename}", status_code=404, details={"filepath": str(e.filename)})
        except json.JSONDecodeError as e:
            logger.error(f"[{controller_name}] Error decoding JSON from statistics file: {e}", exc_info=True)
            raise APIError(message=f"Invalid JSON in statistics file: {e.msg}", status_code=400, details={"json_error": e.msg})


        # 3. Call Core Logic
        # src.feature_analysis.compare_feature_statistics(current_stats: dict, baseline_stats: dict) -> dict
        logger.info(f"[{controller_name}] Comparing feature statistics.", 
                    extra={"props": {"controller": controller_name, 
                                     "current_stats_path": str(api_req_model.current_features_stats_path),
                                     "baseline_stats_path": str(api_req_model.baseline_stats_path)}})
        
        drift_metrics_result = feature_analysis.compare_feature_statistics(
            current_stats=current_stats,
            baseline_stats=baseline_stats
        )
        
        # Save the drift report
        try:
            with open(core_drift_report_output_path, 'w') as f_report:
                json.dump(drift_metrics_result, f_report, indent=4)
            logger.info(f"[{controller_name}] Drift report saved to {core_drift_report_output_path}", extra={"props": {"controller": controller_name}})
        except Exception as e_save:
            logger.error(f"[{controller_name}] Failed to save drift report: {e_save}", exc_info=True)
            # Decide if this is a critical failure for the API response or just a logging issue.
            # For now, we'll still return the metrics but log the save error.
            # If saving is critical, raise FileOperationError here.

        # 4. Response Formatting
        response_payload = AnalyzeDriftApiResponse(
            status="success",
            message="Feature drift analysis completed.",
            drift_report_path=core_drift_report_output_path, # Path where it was saved
            drift_metrics=drift_metrics_result # The actual metrics
        )
        return response_payload.model_dump(), 200

    except APIError as e:
        logger.error(f"[{controller_name}] API Error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details, "status_code": e.status_code}})
        return _create_error_response(e.message, e.status_code, error_type=type(e).__name__, details=e.details)
    except (DataValidationError, ConfigError, FileOperationError, PipelineError) as e: # FileOperationError for loading stats
        status_code = 404 if isinstance(e, FileOperationError) else (422 if isinstance(e, DataValidationError) else 500)
        logger.error(f"[{controller_name}] Application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": getattr(e, 'details', str(e))}})
        return _create_error_response(e.message, status_code, error_type=type(e).__name__, details=getattr(e, 'details', None))
    except AppBaseException as e:
        logger.error(f"[{controller_name}] Base application error: {e.message}", exc_info=True, extra={"props": {"controller": controller_name, "error_details": e.details}})
        return _create_error_response(e.message, 500, error_type=type(e).__name__, details=e.details)
    except Exception as e:
        logger.critical(f"[{controller_name}] Unexpected critical error: {e}", exc_info=True, extra={"props": {"controller": controller_name}})
        return _create_error_response("An unexpected internal server error occurred.", 500, error_type="InternalServerError")
