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

from src import config
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
        return None

def run_cli_command(command_parts: list[str]) -> tuple[str, str, int]:
    """Runs a CLI command using subprocess and returns output, error, and return code."""
    try:
        # Use sys.executable to ensure using the same python interpreter
        # that streamlit is running with. This helps with virtual environments.
        full_command = [sys.executable] + command_parts
        st.info(f"Executing: {' '.join(full_command)}")
        process = subprocess.run(full_command, capture_output=True, text=True, check=False) # check=False to handle non-zero exit codes manually
        
        stdout_output = process.stdout if process.stdout else ""
        stderr_output = process.stderr if process.stderr else ""
        
        if stdout_output:
            st.subheader("CLI Output (stdout):")
            st.text_area("stdout", stdout_output, height=150, key=f"stdout_{command_parts[1]}_{datetime.now().timestamp()}")
        
        if stderr_output:
            st.subheader("CLI Output (stderr):")
            st.text_area("stderr", stderr_output, height=150, key=f"stderr_{command_parts[1]}_{datetime.now().timestamp()}")

        if process.returncode != 0:
            st.error(f"Command '{' '.join(full_command)}' failed with exit code {process.returncode}.")
            
        return stdout_output, stderr_output, process.returncode
    except Exception as e:
        st.error(f"Exception during CLI command execution: {e}")
        return "", str(e), -1 # Indicate failure with -1

# --- End Helper Functions ---


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
st.session_state.selected_ticker = st.sidebar.selectbox("Select Stock Ticker:", options=config.DEFAULT_TICKERS, key='sb_selected_ticker_gui_final_final_final', 
    index=config.DEFAULT_TICKERS.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in config.DEFAULT_TICKERS else 0)
if isinstance(st.session_state.start_date, str): st.session_state.start_date = datetime.strptime(st.session_state.start_date, "%Y-%m-%d").date()
if isinstance(st.session_state.end_date, str): st.session_state.end_date = datetime.strptime(st.session_state.end_date, "%Y-%m-%d").date()
if st.session_state.end_date < st.session_state.start_date: st.session_state.end_date = st.session_state.start_date
st.session_state.start_date = st.sidebar.date_input("Start Date:", value=st.session_state.start_date, min_value=datetime(2010,1,1).date(), max_value=date.today(), key='sb_start_date_gui_final_final_final')
st.session_state.end_date = st.sidebar.date_input("End Date:", value=st.session_state.end_date, min_value=st.session_state.start_date, max_value=date.today(), key='sb_end_date_gui_final_final_final')

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Load & Analyze Data", "🛠️ Feature Engineering", "🧠 Train & Evaluate Model", "📈 Backtest & Visualize"])

with tab1: 
    st.header("Load & Analyze Stock Data")
    st.write(f"**Selected Ticker:** {st.session_state.selected_ticker}, **Date Range:** {st.session_state.start_date.strftime('%Y-%m-%d')} to {st.session_state.end_date.strftime('%Y-%m-%d')}")
    if st.button("Load Data via CLI", key="load_data_tab1_cli_button"):
        st.session_state.load_data_tab1_clicked = True # Keep this for UI flow
        
        # 1. Collect parameters for config and structure with Pydantic model
        try:
            load_data_params = LoadDataConfig(
                ticker=st.session_state.selected_ticker,
                start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                end_date=st.session_state.end_date.strftime("%Y-%m-%d"),
                # Assuming default output_parquet_path from model is acceptable or set it if needed
                output_parquet_path=f"data/raw/{st.session_state.selected_ticker.lower()}_gui_data.parquet" # Example
            )
            config_payload = load_data_params.model_dump(mode='json') # Get dict for JSON serialization
        except Exception as e_pydantic: # Catch Pydantic validation error or other issues
            st.error(f"Error creating LoadDataConfig: {e_pydantic}")
            temp_config_path = None # Ensure it's None so CLI call is skipped
        else:
            # 2. Create temporary config file
            temp_config_path = create_temp_config_file(config_payload)
        
        if temp_config_path:
            # 3. Construct CLI command
            cli_command_parts = ["main.py", "load-data", "--config", temp_config_path]
            
            # 4. Run command
            with st.spinner("Requesting data load from CLI..."):
                stdout_cli, stderr_cli, returncode = run_cli_command(cli_command_parts)
            
            # 5. Display output (handled by run_cli_command)
            if returncode == 0:
                st.success("CLI 'load-data' command executed. See output above.")
                # For now, we don't get data back directly.
                # The GUI's internal st.session_state.stock_data will NOT be updated by this CLI call.
                # This part needs further design if data needs to be passed back to Streamlit state.
                # For this subtask, displaying CLI output is the primary goal.
                st.session_state.stock_data = None # Explicitly set to None or handle differently
                st.warning("Note: GUI data display is bypassed when using CLI for load. Check CLI output for status.")

            # 6. Clean up temp config file
            try:
                os.remove(temp_config_path)
            except Exception as e:
                st.warning(f"Could not remove temporary config file {temp_config_path}: {e}")
                
    # Existing data display logic (will not be populated by the CLI call for now)
    if st.session_state.stock_data is not None: # This block will likely not execute after CLI call
        st.subheader("Preview (Direct Load - Bypassed by CLI)"); st.dataframe(st.session_state.stock_data.head())
        if 'Close' in st.session_state.stock_data: st.subheader("Price Chart"); st.line_chart(st.session_state.stock_data['Close'])
        if 'Volume' in st.session_state.stock_data: st.subheader("Volume Chart"); st.bar_chart(st.session_state.stock_data['Volume'])
        st.subheader("Statistics"); st.dataframe(st.session_state.stock_data.describe())
    elif st.session_state.load_data_tab1_clicked: st.info("Attempted data load. Check CLI output if 'Load Data via CLI' was used.")
    else: st.info("Click 'Load Data via CLI' to use the backend pipeline.")


with tab2: 
    st.header("Feature Engineering Options")
    # This tab assumes data might have been loaded into st.session_state.stock_data by *some* means
    # or that feature engineering can run on a predefined dataset path from config.
    # For CLI integration, it's best if engineer-features uses a path from its config.
    if st.session_state.get('stock_data_loaded_via_cli', False) or st.session_state.stock_data is None:
        st.info("Feature engineering via CLI will assume data is loaded by a previous 'load-data' CLI step, referenced in its config.")
    else:
    # Feature selection checkboxes
    st.subheader("Select Features to Generate (via CLI)")
    st.session_state.generate_technical = st.checkbox("Technical Indicators", st.session_state.generate_technical, key="cb_tech_cli")
    st.session_state.generate_rolling_lag = st.checkbox("Rolling & Lag", st.session_state.generate_rolling_lag, key="cb_roll_cli")
    st.session_state.generate_sentiment = st.checkbox("Sentiment (NewsAPI Key needed)", st.session_state.generate_sentiment, key="cb_sent_cli")
    st.session_state.generate_fundamental = st.checkbox("Fundamental Data", st.session_state.generate_fundamental, key="cb_fund_cli")
    st.session_state.generate_target = st.checkbox("Target Variable", st.session_state.generate_target, key="cb_target_cli")

    if st.button("Generate Features via CLI", key="gen_features_tab2_cli_button"):
        # 1. Collect parameters and structure with Pydantic model
        try:
            # Placeholder paths - these should be managed in a real pipeline (e.g., from a global config or prior step output)
            # For GUI interaction, it might be that load-data step output a known filename that engineer-features consumes.
            input_path_placeholder = f"data/raw/{st.session_state.selected_ticker.lower()}_gui_data.parquet" # Matching example from load_data
            output_path_placeholder = f"data/processed/{st.session_state.selected_ticker.lower()}_gui_features.parquet"

            feature_eng_params = FeatureEngineeringConfig(
                input_data_path=input_path_placeholder, 
                output_features_path=output_path_placeholder,
                technical_indicators=st.session_state.generate_technical,
                rolling_lag_features=st.session_state.generate_rolling_lag,
                sentiment_features=st.session_state.generate_sentiment,
                fundamental_features=st.session_state.generate_fundamental,
                target_variable={ # Using dict here as TargetVariableConfig is nested
                    "enabled": st.session_state.generate_target,
                    "days_forward": 5, # Example, make configurable if needed
                    "threshold": 0.03  # Example, make configurable if needed
                }
                # ticker for fundamental/sentiment is not explicitly in FeatureEngineeringConfig root,
                # but could be passed if those steps need it directly (currently internal to them).
            )
            config_payload = feature_eng_params.model_dump(mode='json')
        except Exception as e_pydantic:
            st.error(f"Error creating FeatureEngineeringConfig: {e_pydantic}")
            temp_config_path = None
        else:
            # 2. Create temp config
            temp_config_path = create_temp_config_file(config_payload)
        
        if temp_config_path:
            # 3. Construct command
            cli_command_parts = ["main.py", "engineer-features", "--config", temp_config_path]
            
            # 4. Run command
            with st.spinner("Requesting feature engineering from CLI..."):
                stdout_cli, stderr_cli, returncode = run_cli_command(cli_command_parts)
            
            if returncode == 0:
                st.success("CLI 'engineer-features' command executed.")
                st.session_state.feature_data = None # Clear any old directly-loaded feature data
                st.warning("Note: GUI feature data display is bypassed. Check CLI output.")

            # 5. Clean up
            try:
                os.remove(temp_config_path)
            except Exception as e:
                st.warning(f"Could not remove temporary config file {temp_config_path}: {e}")
                
    # Display of feature_data (will not be populated by CLI for now)
    if st.session_state.feature_data is not None:
        st.subheader("Preview with Features (Direct Load - Bypassed by CLI)"); st.dataframe(st.session_state.feature_data.head())
        st.write(f"Shape: {st.session_state.feature_data.shape}, Nulls: {st.session_state.feature_data.isnull().sum().sum()}")


with tab3:
    st.header("Model Training and Evaluation via CLI")
    st.info("This tab will use the CLI to train models. Ensure features are generated first (via CLI).")

    st.subheader("1. Select Model Type")
    available_models_cli = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "CNN-LSTM", "Transformer"] 
    st.session_state.selected_model_type_tab3 = st.selectbox("Choose a Model for CLI Training:", 
        options=available_models_cli,
        index=available_models_cli.index(st.session_state.selected_model_type_tab3), 
        key="model_choice_tab3_cli")

    # Example parameters - in a real scenario, these would be more dynamic
    model_params_payload = {
        "XGBoost": {"n_estimators": 100, "learning_rate": 0.1},
        "LightGBM": {"n_estimators": 100, "learning_rate": 0.1},
        "CatBoost": {"iterations": 100, "learning_rate": 0.1},
        "LSTM": {"units": 50, "epochs": 10, "batch_size": 32, "sequence_length": 20},
        "CNN-LSTM": {"filters": 32, "kernel_size": 3, "lstm_units": 50, "epochs": 10, "batch_size": 32, "sequence_length": 20},
        "Transformer": {"head_size": 128, "num_heads": 4, "ff_dim": 2, "num_transformer_blocks": 2, "mlp_units": [64], "epochs": 5, "batch_size": 32, "sequence_length": 20} # Simplified
    }

    selected_cli_model_params = model_params_payload.get(st.session_state.selected_model_type_tab3, {})

    if st.button(f"Train {st.session_state.selected_model_type_tab3} Model via CLI", key=f"train_{st.session_state.selected_model_type_tab3}_cli_btn"):
        # 1. Collect params and structure with Pydantic model
        try:
            # Placeholder paths
            input_feat_path_placeholder = f"data/processed/{st.session_state.selected_ticker.lower()}_gui_features.parquet" # Matching example from engineer_features
            model_output_base_placeholder = f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_model"
            scaler_output_placeholder = f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_scaler.pkl" if st.session_state.selected_model_type_tab3 in ["LSTM", "CNN-LSTM", "Transformer"] else None

            # Ensure model_params are structured correctly for ModelParamsConfig
            # The selected_cli_model_params is already a dict.
            # Pydantic will validate it when creating TrainModelConfig.
            
            train_model_pydantic_params = TrainModelConfig(
                input_features_path=input_feat_path_placeholder,
                model_output_path_base=model_output_base_placeholder,
                scaler_output_path=scaler_output_placeholder,
                model_type=st.session_state.selected_model_type_tab3,
                model_params=selected_cli_model_params, # This should be a dict compatible with ModelParamsConfig
                target_column="target" # Assuming 'target' is standard
            )
            config_payload = train_model_pydantic_params.model_dump(mode='json')
        except Exception as e_pydantic:
            st.error(f"Error creating TrainModelConfig: {e_pydantic}")
            temp_config_path = None
        else:
            # 2. Create temp config
            temp_config_path = create_temp_config_file(config_payload)

        if temp_config_path:
            # 3. Construct command
            cli_command_parts = ["main.py", "train-model", "--config", temp_config_path]
            
            # 4. Run command
            with st.spinner(f"Requesting {st.session_state.selected_model_type_tab3} model training from CLI..."):
                stdout_cli, stderr_cli, returncode = run_cli_command(cli_command_parts)

            if returncode == 0:
                st.success(f"CLI 'train-model' for {st.session_state.selected_model_type_tab3} executed.")
                st.warning("Note: GUI model display/evaluation is bypassed. Check CLI output.")
            
            # 5. Clean up
            try:
                os.remove(temp_config_path)
            except Exception as e:
                st.warning(f"Could not remove temporary config file {temp_config_path}: {e}")
    
    st.markdown("---")
    st.subheader("Evaluate Model (CLI - Placeholder)")
    # This would require a separate 'evaluate' CLI command and a way to reference the trained model
    st.info("To evaluate a model trained via CLI, you would typically use an 'evaluate' CLI command, passing a config that points to the trained model and test data.")
    if st.button("Evaluate Model via CLI (Placeholder)", key="eval_model_cli_btn"):
        try:
            model_path_placeholder = f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_model..." # Needs correct extension
            scaler_path_placeholder = f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_scaler.pkl" if st.session_state.selected_model_type_tab3 in ["LSTM", "CNN-LSTM", "Transformer"] else None
            
            eval_params = EvaluateModelConfig(
                model_path=model_path_placeholder, # This expects a file, will fail if not existing. For CLI stub, allow string.
                scaler_path=scaler_path_placeholder, # Same, expects file.
                test_data_path="path/to/test_features_with_target.csv", # Placeholder
                metrics_output_json_path="evaluation_metrics_gui.json"
            )
            config_payload_eval = eval_params.model_dump(mode='json')
        except Exception as e_pydantic:
            st.error(f"Error creating EvaluateModelConfig: {e_pydantic}")
            # To make this work for CLI stub where files don't exist yet, change FilePath to str in Pydantic model or create dummy files.
            # For now, we'll let it potentially show an error if paths are validated strictly.
            # A better approach for stubs: use string types in Pydantic models for paths that are outputs of previous steps.
            st.warning("Note: Path validation for model/scaler might cause errors if files don't exist. This is a placeholder.")
            # Fallback to dict if Pydantic model creation fails for placeholder.
            config_payload_eval = {
                "model_path": model_path_placeholder,
                "scaler_path": scaler_path_placeholder,
                "test_data_path": "path/to/test_features_with_target.csv",
                "metrics_output_json_path": "evaluation_metrics_gui.json"
            }

        temp_config_path_eval = create_temp_config_file(config_payload_eval)
        if temp_config_path_eval:
            cli_command_parts_eval = ["main.py", "evaluate", "--config", temp_config_path_eval]
            with st.spinner("Requesting model evaluation from CLI..."):
                run_cli_command(cli_command_parts_eval)
            try:
                os.remove(temp_config_path_eval)
            except Exception as e:
                st.warning(f"Could not remove temporary config file {temp_config_path_eval}: {e}")


with tab4: 
    st.header("Backtesting and Visualization (CLI Hooks)")
    st.info("This tab will demonstrate hooks for CLI-based backtesting.")

    st.subheader("Run Backtest via CLI")
    backtest_strategy_config_example = {
        "long_threshold": 0.6,
        "short_threshold": -0.4,
        "target_percent": 0.9
    }
    backtest_data_config_example = {
        "ohlcv_data_path": "path/to/ohlcv_for_backtest.csv", # Needs to be available
        "predictions_path": "path/to/model_predictions_for_backtest.csv" # Needs to be generated by 'export' or 'predict' CLI command
    }

    if st.button("Run Backtest via CLI (Placeholder)", key="run_backtest_cli_btn"):
        try:
            backtest_params = BacktestConfig(
                ohlcv_data_path="path/to/ohlcv_for_backtest_gui.csv", # Placeholder
                predictions_path="path/to/model_predictions_for_backtest_gui.csv", # Placeholder
                results_output_path="backtest_results_gui.json",
                strategy_config=backtest_strategy_config_example # Already a dict
            )
            config_payload_backtest = backtest_params.model_dump(mode='json')
        except Exception as e_pydantic:
            st.error(f"Error creating BacktestConfig: {e_pydantic}")
            config_payload_backtest = { # Fallback for placeholder
                "ohlcv_data_path": "path/to/ohlcv_for_backtest_gui.csv",
                "predictions_path": "path/to/model_predictions_for_backtest_gui.csv",
                "results_output_path": "backtest_results_gui.json",
                "strategy_config": backtest_strategy_config_example
            }
        temp_config_path_backtest = create_temp_config_file(config_payload_backtest)
        if temp_config_path_backtest:
            cli_command_parts_backtest = ["main.py", "backtest", "--config", temp_config_path_backtest]
            with st.spinner("Requesting backtest from CLI..."):
                run_cli_command(cli_command_parts_backtest)
            try:
                os.remove(temp_config_path_backtest)
            except Exception as e:
                st.warning(f"Could not remove temporary config file {temp_config_path_backtest}: {e}")
    
    st.subheader("Export Model/Predictions via CLI")
    if st.button("Export Model via CLI (Placeholder)", key="export_model_cli_btn"):
        try:
            export_params = ExportConfig(
                trained_model_path=f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_model...", # Placeholder, needs correct extension
                export_type="pickle", 
                export_output_path=f"exported_models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_exported"
            )
            config_payload_export = export_params.model_dump(mode='json')
        except Exception as e_pydantic:
            st.error(f"Error creating ExportConfig: {e_pydantic}")
            config_payload_export = { # Fallback for placeholder
                 "trained_model_path": f"models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_cli_model...",
                 "export_type": "pickle", 
                 "export_output_path": f"exported_models/{st.session_state.selected_ticker}_{st.session_state.selected_model_type_tab3.lower()}_gui_exported"
            }
        temp_config_path_export = create_temp_config_file(config_payload_export)
        if temp_config_path_export:
            cli_command_parts_export = ["main.py", "export", "--config", temp_config_path_export]
            with st.spinner("Requesting model export from CLI..."):
                run_cli_command(cli_command_parts_export)
            try:
                os.remove(temp_config_path_export)
            except Exception as e:
                st.warning(f"Could not remove temporary config file {temp_config_path_export}: {e}")

st.sidebar.markdown("---")
st.sidebar.info("App for stock analysis and prediction.")
