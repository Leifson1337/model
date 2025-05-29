# gui.py
import streamlit as st
from datetime import datetime, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import tempfile
import os
import sys # To ensure using the same python executable
import logging # Added for logging

from src import config
from src.utils import setup_logging # To ensure logging is configured if GUI is run standalone
from src.exceptions import ConfigError, FileOperationError, AppBaseException # Custom exceptions
from src.config_models import ( # Import Pydantic models for config generation
    LoadDataConfig,
    FeatureEngineeringConfig,
    TrainModelConfig,
    EvaluateModelConfig,
    BacktestConfig,
    ExportConfig,
    ModelParamsConfig # If needed for constructing parts of other configs
)
# Direct src imports will be gradually replaced or used only for data structures/display
# from src.data_management import download_stock_data 
# from src.feature_engineering import add_technical_indicators, add_rolling_lag_features, create_target_variable
# from src.sentiment_analysis import get_daily_sentiment_scores 
# from src.fundamental_data import add_fundamental_features_to_data
# from src import modeling # train_X, predict_X functions are here
# from src import utils # save_model, load_model
# from src.evaluation import plot_roc_auc, plot_confusion_matrix 

# --- Helper Functions for CLI Interaction ---

def create_temp_config_file(config_data: dict) -> str:
    """Creates a temporary JSON config file and returns its path."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir='.') as tmp_file:
            json.dump(config_data, tmp_file, indent=4)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error creating temporary config file: {e}")
    logger.error(f"[GUI] Error creating temporary config file: {e}", exc_info=True, extra={"props": {"gui_event": "create_temp_config_file"}})
        return None

def run_cli_command(command_parts: list[str]) -> tuple[str, str, int]:
    """Runs a CLI command using subprocess and returns output, error, and return code."""
    command_str = ' '.join(command_parts)
    logger.info(f"[GUI] Executing CLI command: {command_str}", extra={"props": {"gui_event": "run_cli_command", "command": command_str}})
    try:
        full_command = [sys.executable] + command_parts
        st.info(f"Executing: {command_str}")
        process = subprocess.run(full_command, capture_output=True, text=True, check=False)
        
        stdout_output = process.stdout if process.stdout else ""
        stderr_output = process.stderr if process.stderr else ""
        
        if stdout_output:
            st.subheader("CLI Output (stdout):")
            st.text_area("stdout", stdout_output, height=150, key=f"stdout_{command_parts[1]}_{datetime.now().timestamp()}")
            logger.debug(f"[GUI] CLI stdout for '{command_str}':\n{stdout_output}", extra={"props": {"gui_event": "run_cli_command_stdout", "command": command_str}})
        
        if stderr_output:
            st.subheader("CLI Output (stderr):")
            st.text_area("stderr", stderr_output, height=150, key=f"stderr_{command_parts[1]}_{datetime.now().timestamp()}")
            # Log stderr as warning or error based on return code
            if process.returncode != 0:
                logger.error(f"[GUI] CLI stderr for '{command_str}' (Return Code: {process.returncode}):\n{stderr_output}",
                             extra={"props": {"gui_event": "run_cli_command_stderr_error", "command": command_str, "return_code": process.returncode}})
            else:
                logger.warning(f"[GUI] CLI stderr for '{command_str}' (Return Code: 0):\n{stderr_output}",
                               extra={"props": {"gui_event": "run_cli_command_stderr_warning", "command": command_str, "return_code": process.returncode}})


        if process.returncode != 0:
            st.error(f"Command '{command_str}' failed with exit code {process.returncode}.")
            logger.error(f"[GUI] Command '{command_str}' failed with exit code {process.returncode}.",
                         extra={"props": {"gui_event": "run_cli_command_failure", "command": command_str, "return_code": process.returncode}})
        else:
            logger.info(f"[GUI] Command '{command_str}' executed successfully.",
                        extra={"props": {"gui_event": "run_cli_command_success", "command": command_str, "return_code": process.returncode}})
            
        return stdout_output, stderr_output, process.returncode
    except Exception as e:
        st.error(f"Exception during CLI command execution for '{command_str}': {e}")
        logger.critical(f"[GUI] Exception during CLI command execution for '{command_str}': {e}",
                        exc_info=True, extra={"props": {"gui_event": "run_cli_command_exception", "command": command_str}})
        return "", str(e), -1 # Indicate failure with -1

# --- End Helper Functions ---
# Initialize logger for the GUI module
# Ensure setup_logging is called, especially if gui.py can be run standalone
# If main.py (or another entry point) always runs first and calls setup_logging, this might be redundant.
# However, for robustness during development or direct gui.py execution:
if not logging.getLogger().hasHandlers(): # Check if root logger is already configured
    setup_logging()
logger = logging.getLogger(__name__)


from src.data_management import download_stock_data # Keep for now for direct use if needed, or for comparison
# These imports are now mostly for reference or for parts of the GUI not yet refactored.
# The goal is to have the core logic triggered via main.py CLI calls.
from src.feature_engineering import add_technical_indicators, add_rolling_lag_features, create_target_variable # Keep for now
from src.sentiment_analysis import get_daily_sentiment_scores # Keep for now
from src.fundamental_data import add_fundamental_features_to_data # Keep for now
from src import modeling # Keep for now
from src import utils # Keep for now
from src.evaluation import plot_roc_auc, plot_confusion_matrix # Keep for now for plotting

from sklearn.model_selection import train_test_split # Keep for now
from sklearn.metrics import accuracy_score, classification_report # Keep for now

# --- Page Configuration ---
st.set_page_config(page_title="Quantitative Leverage Predictor", layout="wide")
st.title("📈 Quantitative Leverage Opportunity Predictor")
st.markdown("Welcome to the advanced stock analysis and prediction tool.")

# --- Session State Initialization ---
default_ticker = config.DEFAULT_TICKERS[0] if config.DEFAULT_TICKERS else "AAPL"
try: default_start_date = datetime.strptime(config.DEFAULT_START_DATE, "%Y-%m-%d").date()
except: default_start_date = datetime(2020, 1, 1).date()
try: default_end_date = datetime.strptime(config.DEFAULT_END_DATE, "%Y-%m-%d").date()
except: default_end_date = date.today()

session_defaults = {
    'selected_ticker': default_ticker,
    'start_date': default_start_date,
    'end_date': default_end_date,
    'stock_data': None,
    'feature_data': None,
    'load_data_tab1_clicked': False,
    'generate_technical': True,
    'generate_rolling_lag': True,
    'generate_sentiment': False,
    'generate_fundamental': False,
    'generate_target': True,
    'selected_model_type_tab3': "XGBoost", 
    'trained_model_info': {}, 
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Sidebar ---
st.sidebar.header("Global Settings")
    st.session_state.selected_ticker = st.sidebar.selectbox("Select Stock Ticker:", options=config.DEFAULT_TICKERS, key='sb_selected_ticker_gui', 
    index=config.DEFAULT_TICKERS.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in config.DEFAULT_TICKERS else 0)
if isinstance(st.session_state.start_date, str): 
    try: st.session_state.start_date = datetime.strptime(st.session_state.start_date, "%Y-%m-%d").date()
    except ValueError: st.session_state.start_date = default_start_date # Fallback
if isinstance(st.session_state.end_date, str): 
    try: st.session_state.end_date = datetime.strptime(st.session_state.end_date, "%Y-%m-%d").date()
    except ValueError: st.session_state.end_date = default_end_date # Fallback
if st.session_state.end_date < st.session_state.start_date: 
    st.session_state.end_date = st.session_state.start_date
    logger.warning("[GUI] Corrected end_date to match start_date as it was earlier.", extra={"props": {"gui_event": "date_correction"}})

st.session_state.start_date = st.sidebar.date_input("Start Date:", value=st.session_state.start_date, min_value=datetime(2010,1,1).date(), max_value=date.today(), key='sb_start_date_gui')
st.session_state.end_date = st.sidebar.date_input("End Date:", value=st.session_state.end_date, min_value=st.session_state.start_date, max_value=date.today(), key='sb_end_date_gui')

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Load & Analyze Data", "🛠️ Feature Engineering", "🧠 Model Ops", "📈 Backtest & Export"])

with tab1: 
    st.header("Load & Analyze Stock Data")
    st.write(f"**Selected Ticker:** {st.session_state.selected_ticker}, **Date Range:** {st.session_state.start_date.strftime('%Y-%m-%d')} to {st.session_state.end_date.strftime('%Y-%m-%d')}")
    if st.button("Load Data via CLI", key="load_data_tab1_cli_button"):
        st.session_state.load_data_tab1_clicked = True
        temp_config_path = None
        try:
            load_data_params = LoadDataConfig(
                ticker=st.session_state.selected_ticker,
                start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                end_date=st.session_state.end_date.strftime("%Y-%m-%d"),
                output_parquet_path=f"data/raw/{st.session_state.selected_ticker.lower()}_gui_data.parquet"
            )
            config_payload = load_data_params.model_dump(mode='json')
            logger.info("[GUI] LoadDataConfig created for CLI.", extra={"props": {"gui_event": "load_data_config_create", "config": config_payload}})
            temp_config_path = create_temp_config_file(config_payload)
            
            if temp_config_path:
                cli_command_parts = ["main.py", "load-data", "--config", temp_config_path]
                with st.spinner("Requesting data load from CLI..."):
                    stdout_cli, stderr_cli, returncode = run_cli_command(cli_command_parts)
                
                if returncode == 0:
                    st.success("CLI 'load-data' command executed. See output above.")
                    st.session_state.stock_data = None 
                    st.warning("Note: GUI data display is bypassed when using CLI for load. Check CLI output for status.")
                else:
                    st.error("CLI 'load-data' command failed. Check CLI output stderr for details.")
            else:
                st.error("Failed to create temporary configuration file for 'load-data'.")

        except ConfigError as e: # Catch specific Pydantic validation error if LoadDataConfig fails
            st.error(f"Configuration Error for Load Data: {e.message}")
            logger.error(f"[GUI] Pydantic ConfigError for LoadDataConfig: {e.message}", exc_info=True, extra={"props": {"gui_event": "load_data_config_error", "error_details": e.details}})
        except Exception as e: # Catch other errors during this process
            st.error(f"An unexpected error occurred during 'Load Data via CLI' setup: {e}")
            logger.error(f"[GUI] Unexpected error in 'Load Data via CLI' button: {e}", exc_info=True, extra={"props": {"gui_event": "load_data_button_error"}})
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                try:
                    os.remove(temp_config_path)
                    logger.info(f"[GUI] Cleaned up temp config file: {temp_config_path}", extra={"props": {"gui_event": "temp_file_cleanup"}})
                except Exception as e_remove:
                    st.warning(f"Could not remove temporary config file {temp_config_path}: {e_remove}")
                    logger.warning(f"[GUI] Failed to remove temp config file {temp_config_path}: {e_remove}", exc_info=True, extra={"props": {"gui_event": "temp_file_cleanup_error"}})
                
    # Existing data display logic
    if st.session_state.stock_data is not None: 
        st.subheader("Preview (Direct Load - Bypassed by CLI)"); st.dataframe(st.session_state.stock_data.head())
        if 'Close' in st.session_state.stock_data: st.subheader("Price Chart (Direct Load)"); st.line_chart(st.session_state.stock_data['Close'])
        if 'Volume' in st.session_state.stock_data: st.subheader("Volume Chart (Direct Load)"); st.bar_chart(st.session_state.stock_data['Volume'])
        st.subheader("Statistics (Direct Load)"); st.dataframe(st.session_state.stock_data.describe())
    elif st.session_state.load_data_tab1_clicked: st.info("Data loading via CLI attempted. Check CLI output above for status. Direct GUI data display is not updated by CLI operations.")
    else: st.info("Click 'Load Data via CLI' to fetch data using the backend pipeline. Direct data loading into GUI is disabled for this version.")


with tab2: 
    st.header("Feature Engineering Options")
    # This tab assumes data might have been loaded into st.session_state.stock_data by *some* means
    # For CLI integration, it's best if engineer-features uses a path from its config.
    st.info("Feature engineering via CLI will assume data is loaded by a previous 'load-data' CLI step, using paths from its configuration.")
    
    st.subheader("Select Features to Generate (via CLI)")
    st.session_state.generate_technical = st.checkbox("Technical Indicators", st.session_state.generate_technical, key="cb_tech_cli")
    st.session_state.generate_rolling_lag = st.checkbox("Rolling & Lag", st.session_state.generate_rolling_lag, key="cb_roll_cli")
    st.session_state.generate_sentiment = st.checkbox("Sentiment (NewsAPI Key needed - Ensure API key is in config/env)", st.session_state.generate_sentiment, key="cb_sent_cli")
    st.session_state.generate_fundamental = st.checkbox("Fundamental Data (AlphaVantage Key needed - Ensure API key is in config/env)", st.session_state.generate_fundamental, key="cb_fund_cli")
    st.session_state.generate_target = st.checkbox("Target Variable", st.session_state.generate_target, key="cb_target_cli")

    if st.button("Generate Features via CLI", key="gen_features_tab2_cli_button"):
        temp_config_path = None
        try:
            input_path_placeholder = f"data/raw/{st.session_state.selected_ticker.lower()}_gui_data.parquet" 
            output_path_placeholder = f"data/processed/{st.session_state.selected_ticker.lower()}_gui_features.parquet"

            feature_eng_params = FeatureEngineeringConfig(
                input_data_path=input_path_placeholder, 
                output_features_path=output_path_placeholder,
                technical_indicators=st.session_state.generate_technical,
                rolling_lag_features=st.session_state.generate_rolling_lag,
                sentiment_features=st.session_state.generate_sentiment,
                fundamental_features=st.session_state.generate_fundamental,
                target_variable={"enabled": st.session_state.generate_target, "days_forward": 5, "threshold": 0.03}
            )
            config_payload = feature_eng_params.model_dump(mode='json')
            logger.info("[GUI] FeatureEngineeringConfig created for CLI.", extra={"props": {"gui_event": "feature_eng_config_create", "config": config_payload}})
            temp_config_path = create_temp_config_file(config_payload)
            
            if temp_config_path:
                cli_command_parts = ["main.py", "engineer-features", "--config", temp_config_path]
                with st.spinner("Requesting feature engineering from CLI..."):
                    stdout_cli, stderr_cli, returncode = run_cli_command(cli_command_parts)
                
                if returncode == 0:
                    st.success("CLI 'engineer-features' command executed.")
                    st.session_state.feature_data = None 
                    st.warning("Note: GUI feature data display is bypassed. Check CLI output.")
                else:
                    st.error("CLI 'engineer-features' command failed. Check CLI output stderr for details.")
            else:
                st.error("Failed to create temporary configuration file for 'engineer-features'.")

        except ConfigError as e:
            st.error(f"Configuration Error for Feature Engineering: {e.message}")
            logger.error(f"[GUI] Pydantic ConfigError for FeatureEngineeringConfig: {e.message}", exc_info=True, extra={"props": {"gui_event": "feature_eng_config_error", "error_details": e.details}})
        except Exception as e:
            st.error(f"An unexpected error occurred during 'Generate Features via CLI' setup: {e}")
            logger.error(f"[GUI] Unexpected error in 'Generate Features via CLI' button: {e}", exc_info=True, extra={"props": {"gui_event": "feature_eng_button_error"}})
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                try:
                    os.remove(temp_config_path)
                    logger.info(f"[GUI] Cleaned up temp config file: {temp_config_path}", extra={"props": {"gui_event": "temp_file_cleanup"}})
                except Exception as e_remove:
                    st.warning(f"Could not remove temporary config file {temp_config_path}: {e_remove}")
                    logger.warning(f"[GUI] Failed to remove temp config file {temp_config_path}: {e_remove}", exc_info=True, extra={"props": {"gui_event": "temp_file_cleanup_error"}})
                
    if st.session_state.feature_data is not None:
        st.subheader("Preview with Features (Direct Load - Bypassed by CLI)"); st.dataframe(st.session_state.feature_data.head())
        st.write(f"Shape: {st.session_state.feature_data.shape}, Nulls: {st.session_state.feature_data.isnull().sum().sum()}")
    else:
        st.info("Feature data is not directly loaded into the GUI when using CLI operations.")


with tab3: # Renamed to "Model Ops"
    st.header("Model Training and Evaluation via CLI")
    st.info("This tab uses the CLI to train and evaluate models. Ensure features are generated first (via CLI).")

    st.subheader("Train Model")
    available_models_cli = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "CNN-LSTM", "Transformer"]
    st.session_state.selected_model_type_tab3 = st.selectbox(
        "Choose a Model for CLI Training:",
        options=available_models_cli,
        index=available_models_cli.index(st.session_state.selected_model_type_tab3),
        key="model_choice_tab3_cli"
    )

    model_params_payload = {
        "XGBoost": {"n_estimators": 100, "learning_rate": 0.1}, "LightGBM": {"n_estimators": 100, "learning_rate": 0.1},
        "CatBoost": {"iterations": 100, "learning_rate": 0.1},
        "LSTM": {"units": 50, "epochs": 10, "batch_size": 32, "sequence_length": 20},
        "CNN-LSTM": {"filters": 32, "kernel_size": 3, "lstm_units": 50, "epochs": 10, "batch_size": 32, "sequence_length": 20},
        "Transformer": {"head_size": 128, "num_heads": 4, "ff_dim": 2, "num_transformer_blocks": 2, "mlp_units": [64], "epochs": 5, "batch_size": 32, "sequence_length": 20}
    }
    selected_cli_model_params = model_params_payload.get(st.session_state.selected_model_type_tab3, {})

    if st.button(f"Train {st.session_state.selected_model_type_tab3} Model via CLI", key=f"train_{st.session_state.selected_model_type_tab3}_cli_btn"):
        temp_config_path = None
        try:
            input_feat_path_placeholder = f"data/processed/{st.session_state.selected_ticker.lower()}_gui_features.parquet"
            model_output_base_placeholder = f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_model"
            scaler_output_placeholder = f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_scaler.pkl" if st.session_state.selected_model_type_tab3 in ["LSTM", "CNN-LSTM", "Transformer"] else None
            
            train_model_pydantic_params = TrainModelConfig(
                input_features_path=input_feat_path_placeholder, model_output_path_base=model_output_base_placeholder,
                scaler_output_path=scaler_output_placeholder, model_type=st.session_state.selected_model_type_tab3,
                model_params=selected_cli_model_params, target_column="target"
            )
            config_payload = train_model_pydantic_params.model_dump(mode='json')
            logger.info("[GUI] TrainModelConfig created for CLI.", extra={"props": {"gui_event": "train_model_config_create", "config": config_payload}})
            temp_config_path = create_temp_config_file(config_payload)

            if temp_config_path:
                cli_command_parts = ["main.py", "train-model", "--config", temp_config_path]
                with st.spinner(f"Requesting {st.session_state.selected_model_type_tab3} model training from CLI..."):
                    stdout_cli, stderr_cli, returncode = run_cli_command(cli_command_parts)
                if returncode == 0:
                    st.success(f"CLI 'train-model' for {st.session_state.selected_model_type_tab3} executed.")
                    st.warning("Note: GUI model display/evaluation is bypassed. Check CLI output.")
                else:
                    st.error(f"CLI 'train-model' for {st.session_state.selected_model_type_tab3} failed. Check stderr output.")
            else:
                st.error("Failed to create temporary configuration file for 'train-model'.")
        except ConfigError as e:
            st.error(f"Configuration Error for Model Training: {e.message}")
            logger.error(f"[GUI] Pydantic ConfigError for TrainModelConfig: {e.message}", exc_info=True, extra={"props": {"gui_event": "train_model_config_error", "error_details": e.details}})
        except Exception as e:
            st.error(f"An unexpected error occurred during 'Train Model via CLI' setup: {e}")
            logger.error(f"[GUI] Unexpected error in 'Train Model via CLI' button: {e}", exc_info=True, extra={"props": {"gui_event": "train_model_button_error"}})
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                try: os.remove(temp_config_path); logger.info(f"[GUI] Cleaned up temp config file: {temp_config_path}", extra={"props": {"gui_event": "temp_file_cleanup"}})
                except Exception as e_remove: st.warning(f"Could not remove temp config: {e_remove}"); logger.warning(f"[GUI] Failed to remove temp config: {e_remove}", exc_info=True, extra={"props": {"gui_event": "temp_file_cleanup_error"}})
    
    st.markdown("---")
    st.subheader("Evaluate Model")
    st.info("To evaluate a model trained via CLI, use the 'evaluate' command with a config pointing to the trained model and test data.")
    
    # Simplified evaluation config - assuming model and scaler paths are known from training step (using placeholders)
    default_eval_model_path = f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_model/{st.session_state.selected_model_type_tab3.lower()}_model.pkl" # Example structure
    if st.session_state.selected_model_type_tab3 in ["LSTM", "CNN-LSTM", "Transformer"]: default_eval_model_path += ".keras" # Adjust for Keras
    
    eval_model_path_input = st.text_input("Path to trained model for evaluation:", default_eval_model_path)
    eval_test_data_path_input = st.text_input("Path to test data (features + target .csv/.parquet):", f"data/processed/{st.session_state.selected_ticker.lower()}_gui_features.parquet") # Assume test data is the full feature set for now

    if st.button("Evaluate Model via CLI", key="eval_model_cli_btn"):
        temp_config_path_eval = None
        try:
            scaler_path_placeholder_eval = None
            if st.session_state.selected_model_type_tab3 in ["LSTM", "CNN-LSTM", "Transformer"]:
                 scaler_path_placeholder_eval = f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_model/{st.session_state.selected_model_type_tab3.lower()}_scaler.pkl"
            
            eval_params = EvaluateModelConfig(
                model_path=eval_model_path_input, 
                scaler_path=scaler_path_placeholder_eval,
                test_data_path=eval_test_data_path_input, 
                metrics_output_json_path=f"data/reports/{st.session_state.selected_ticker.lower()}_evaluation_metrics_gui.json",
                model_type=st.session_state.selected_model_type_tab3 # Pass model_type for context
            )
            config_payload_eval = eval_params.model_dump(mode='json')
            logger.info("[GUI] EvaluateModelConfig created for CLI.", extra={"props": {"gui_event": "eval_model_config_create", "config": config_payload_eval}})
            temp_config_path_eval = create_temp_config_file(config_payload_eval)

            if temp_config_path_eval:
                cli_command_parts_eval = ["main.py", "evaluate", "--config", temp_config_path_eval]
                with st.spinner("Requesting model evaluation from CLI..."):
                    run_cli_command(cli_command_parts_eval) # Output handled by function
            else:
                st.error("Failed to create temporary configuration file for 'evaluate-model'.")
        except ConfigError as e:
            st.error(f"Configuration Error for Model Evaluation: {e.message}")
            logger.error(f"[GUI] Pydantic ConfigError for EvaluateModelConfig: {e.message}", exc_info=True, extra={"props": {"gui_event": "eval_model_config_error", "error_details": e.details}})
        except FileNotFoundError as e: # Specifically for files that should exist for evaluation config
            st.error(f"File not found for evaluation configuration: {e}")
            logger.error(f"[GUI] FileNotFoundError for EvaluateModelConfig: {e}", exc_info=True, extra={"props": {"gui_event": "eval_model_file_not_found_error"}})
        except Exception as e:
            st.error(f"An unexpected error occurred during 'Evaluate Model via CLI' setup: {e}")
            logger.error(f"[GUI] Unexpected error in 'Evaluate Model via CLI' button: {e}", exc_info=True, extra={"props": {"gui_event": "eval_model_button_error"}})
        finally:
            if temp_config_path_eval and os.path.exists(temp_config_path_eval):
                try: os.remove(temp_config_path_eval); logger.info(f"[GUI] Cleaned up temp config file: {temp_config_path_eval}", extra={"props": {"gui_event": "temp_file_cleanup"}})
                except Exception as e_remove: st.warning(f"Could not remove temp config: {e_remove}"); logger.warning(f"[GUI] Failed to remove temp config: {e_remove}", exc_info=True, extra={"props": {"gui_event": "temp_file_cleanup_error"}})


with tab4: # Renamed to "Backtest & Export"
    st.header("Backtesting and Model Export (CLI Hooks)")
    st.info("This tab demonstrates hooks for CLI-based backtesting and model exporting.")

    st.subheader("Run Backtest")
    # Input fields for backtest configuration
    backtest_ohlcv_path = st.text_input("OHLCV Data Path for Backtest:", f"data/raw/{st.session_state.selected_ticker.lower()}_gui_data.parquet") # Example from load_data output
    backtest_predictions_path = st.text_input("Predictions File Path for Backtest:", f"data/predictions/{st.session_state.selected_ticker.lower()}_predictions_gui.csv") # Needs to be generated by a predict CLI step
    backtest_results_output = st.text_input("Output Path for Backtest Results:", f"data/reports/{st.session_state.selected_ticker.lower()}_backtest_results_gui.json")

    if st.button("Run Backtest via CLI", key="run_backtest_cli_btn"):
        temp_config_path_backtest = None
        try:
            backtest_params = BacktestConfig(
                ohlcv_data_path=backtest_ohlcv_path, 
                predictions_path=backtest_predictions_path, 
                results_output_path=backtest_results_output,
                strategy_config={"long_threshold": 0.6, "short_threshold": 0.4, "target_percent": 0.02} # Example strategy
            )
            config_payload_backtest = backtest_params.model_dump(mode='json')
            logger.info("[GUI] BacktestConfig created for CLI.", extra={"props": {"gui_event": "backtest_config_create", "config": config_payload_backtest}})
            temp_config_path_backtest = create_temp_config_file(config_payload_backtest)

            if temp_config_path_backtest:
                cli_command_parts_backtest = ["main.py", "backtest", "--config", temp_config_path_backtest]
                with st.spinner("Requesting backtest from CLI..."):
                    run_cli_command(cli_command_parts_backtest)
            else:
                st.error("Failed to create temporary configuration file for 'backtest'.")
        except ConfigError as e:
            st.error(f"Configuration Error for Backtest: {e.message}")
            logger.error(f"[GUI] Pydantic ConfigError for BacktestConfig: {e.message}", exc_info=True, extra={"props": {"gui_event": "backtest_config_error", "error_details": e.details}})
        except Exception as e:
            st.error(f"An unexpected error occurred during 'Run Backtest via CLI' setup: {e}")
            logger.error(f"[GUI] Unexpected error in 'Run Backtest via CLI' button: {e}", exc_info=True, extra={"props": {"gui_event": "backtest_button_error"}})
        finally:
            if temp_config_path_backtest and os.path.exists(temp_config_path_backtest):
                try: os.remove(temp_config_path_backtest); logger.info(f"[GUI] Cleaned up temp config file: {temp_config_path_backtest}", extra={"props": {"gui_event": "temp_file_cleanup"}})
                except Exception as e_remove: st.warning(f"Could not remove temp config: {e_remove}"); logger.warning(f"[GUI] Failed to remove temp config: {e_remove}", exc_info=True, extra={"props": {"gui_event": "temp_file_cleanup_error"}})
    
    st.markdown("---")
    st.subheader("Export Model")
    # Input fields for model export
    export_model_path = st.text_input("Path to Trained Model to Export:", f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_model/{st.session_state.selected_model_type_tab3.lower()}_model.pkl") # Example
    export_type_selection = st.selectbox("Export Format:", ["pickle", "onnx", "joblib_zip"], key="export_format_gui")
    export_output_path = st.text_input("Output Path for Exported Model:", f"models/exported/{st.session_state.selected_ticker.lower()}_{st.session_state.selected_model_type_tab3.lower()}_gui.{export_type_selection}")
    
    if st.button("Export Model via CLI", key="export_model_cli_btn"):
        temp_config_path_export = None
        try:
            export_params = ExportConfig(
                trained_model_path=export_model_path, 
                export_type=export_type_selection, 
                export_output_path=export_output_path
            )
            config_payload_export = export_params.model_dump(mode='json')
            logger.info("[GUI] ExportConfig created for CLI.", extra={"props": {"gui_event": "export_config_create", "config": config_payload_export}})
            temp_config_path_export = create_temp_config_file(config_payload_export)

            if temp_config_path_export:
                cli_command_parts_export = ["main.py", "export", "--config", temp_config_path_export]
                with st.spinner("Requesting model export from CLI..."):
                    run_cli_command(cli_command_parts_export)
            else:
                st.error("Failed to create temporary configuration file for 'export-model'.")
        except ConfigError as e:
            st.error(f"Configuration Error for Model Export: {e.message}")
            logger.error(f"[GUI] Pydantic ConfigError for ExportConfig: {e.message}", exc_info=True, extra={"props": {"gui_event": "export_config_error", "error_details": e.details}})
        except Exception as e:
            st.error(f"An unexpected error occurred during 'Export Model via CLI' setup: {e}")
            logger.error(f"[GUI] Unexpected error in 'Export Model via CLI' button: {e}", exc_info=True, extra={"props": {"gui_event": "export_button_error"}})
        finally:
            if temp_config_path_export and os.path.exists(temp_config_path_export):
                try: os.remove(temp_config_path_export); logger.info(f"[GUI] Cleaned up temp config file: {temp_config_path_export}", extra={"props": {"gui_event": "temp_file_cleanup"}})
                except Exception as e_remove: st.warning(f"Could not remove temp config: {e_remove}"); logger.warning(f"[GUI] Failed to remove temp config: {e_remove}", exc_info=True, extra={"props": {"gui_event": "temp_file_cleanup_error"}})

st.sidebar.markdown("---")
st.sidebar.info("Quantitative Leverage Opportunity Predictor GUI. Use CLI for backend operations.")
