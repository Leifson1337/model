import click
import json
from pathlib import Path
import pandas as pd 
import numpy as np 
import os 
import logging 

# Pydantic models for type hinting and config access
from src.config_models import (
    GlobalAppConfig, LoadDataConfig, FeatureEngineeringConfig, TrainModelConfig,
    EvaluateModelConfig, BacktestConfig, ExportConfig
)
# Import model registry utilities
import src.model_registry_utils as registry_utils
# Import logging setup utility and model loading
from src.utils import setup_logging, load_model 
# Import feature analysis utilities
from src.feature_analysis import calculate_feature_statistics, compare_feature_statistics
# Import evaluation utilities
import src.evaluation as evaluation_utils 
import joblib # For loading scalers

setup_logging() 
logger = logging.getLogger(__name__) 

def load_and_validate_config(config_path: str, model_class: type):
    if not config_path:
        click.echo("Error: No config file provided.", err=True); return None 
    try:
        config_json = json.loads(Path(config_path).read_text())
        if 'config_inline' in config_json and isinstance(config_json.get('config_inline'), dict):
            parsed_config = model_class(**config_json.get('config_inline'))
        elif 'config_path' in config_json and isinstance(config_json.get('config_path'), str):
             click.echo(f"Warning: Config '{config_path}' points to '{config_json.get('config_path')}'. Resolve before CLI.", err=True)
             parsed_config = model_class(**config_json)
        else: parsed_config = model_class(**config_json)
        logger.info(f"Loaded config from '{config_path}' for {model_class.__name__}.")
        return parsed_config
    except FileNotFoundError: click.echo(f"Error: Config file '{config_path}' not found.", err=True)
    except json.JSONDecodeError: click.echo(f"Error: Could not decode JSON from '{config_path}'.", err=True)
    except Exception as e: # Catch Pydantic's ValidationError and other general exceptions
        click.echo(f"Error during config validation or loading for '{config_path}':\n{e}", err=True)
    return None

@click.group()
@click.option('--app-config', default=None, help='Path to global app config (e.g., dev.json).')
@click.pass_context
def cli(ctx, app_config):
    """A CLI tool for the ML pipeline."""
    logger.info(f"CLI invoked. Global app config: {app_config}")
    ctx.obj = {}
    if app_config:
        try:
            global_cfg_json = json.loads(Path(app_config).read_text())
            ctx.obj['GLOBAL_CONFIG'] = GlobalAppConfig(**global_cfg_json)
            logger.info(f"Global app config loaded from: {app_config}")
        except Exception as e:
            click.echo(f"Error loading global app config '{app_config}': {e}", err=True)
            logger.error(f"Error loading global app config '{app_config}': {e}")
            ctx.obj['GLOBAL_CONFIG'] = GlobalAppConfig() 
    else: ctx.obj['GLOBAL_CONFIG'] = GlobalAppConfig() 

@cli.command('load-data')
@click.option('--config', default=None, help='Path to LoadDataConfig JSON file.')
@click.pass_context
def load_data(ctx, config):
    logger.info(f"CLI 'load-data' called. Config: {config}"); click.echo("--- load-data ---")
    loaded_config = load_and_validate_config(config, LoadDataConfig)
    if loaded_config: click.echo("load-data logic (stub) completed.")
    else: click.echo("load-data failed (config issues).", err=True)

@cli.command('engineer-features')
@click.option('--config', 'config_path', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to FeatureEngineeringConfig JSON file.')
@click.option('--compare-to-baseline', 'baseline_stats_path', default=None, type=click.Path(exists=True, dir_okay=False, readable=True), help="Optional path to baseline feature statistics JSON file for drift comparison.")
@click.pass_context
def engineer_features(ctx, config_path: str, baseline_stats_path: Optional[str]):
    logger.info(f"CLI 'engineer-features' called. Config: {config_path}, Baseline: {baseline_stats_path}")
    click.echo("--- engineer-features ---")
    loaded_config = load_and_validate_config(config_path, FeatureEngineeringConfig)
    if not loaded_config: click.echo("engineer-features failed (config issues).", err=True); return
    click.echo(f"Op: Engineer features from '{loaded_config.input_data_path}' to '{loaded_config.output_features_path}'.")
    output_path_obj = Path(loaded_config.output_features_path); output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    dummy_data = {'num1': np.random.rand(100) + (0 if not baseline_stats_path else 2), 'cat1': np.random.choice(['A','B','X'],100), 'target':np.random.randint(0,2,100)}
    if baseline_stats_path: dummy_data['new_feat_drift'] = np.random.rand(100)
    current_features_df = pd.DataFrame(dummy_data)
    try:
        if str(output_path_obj).endswith(".parquet"): current_features_df.to_parquet(output_path_obj, index=False)
        else: current_features_df.to_csv(output_path_obj, index=False)
        click.echo(f"Dummy current features saved: {output_path_obj}"); logger.info(f"Dummy current features saved: {output_path_obj}")
        current_stats_dict = calculate_feature_statistics(current_features_df)
        base, _ = os.path.splitext(loaded_config.output_features_path); current_stats_path_str = f"{base}_current_stats.json"
        with open(current_stats_path_str, 'w') as f: json.dump(current_stats_dict, f, indent=4)
        click.echo(f"Current stats saved: {current_stats_path_str}"); logger.info(f"Current stats saved: {current_stats_path_str}")
        if baseline_stats_path:
            with open(baseline_stats_path, 'r') as f: loaded_baseline_stats = json.load(f)
            drift_report = compare_feature_statistics(current_stats=current_stats_dict, baseline_stats=loaded_baseline_stats)
            drift_path_str = f"{base}_drift_report.json"
            with open(drift_path_str, 'w') as f: json.dump(drift_report, f, indent=4)
            click.echo(f"Drift report saved: {drift_path_str}"); logger.info(f"Drift report saved: {drift_path_str}")
        else: click.echo("No baseline stats provided, skipping drift analysis."); logger.info("Skipped drift analysis.")
    except Exception as e: click.echo(f"Error in engineer-features: {e}", err=True); logger.error(f"Error: {e}"); return
    click.echo("engineer-features logic completed.")

@cli.command('train-model')
@click.option('--config', 'config_path', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to TrainModelConfig JSON file.')
@click.pass_context
def train_model(ctx, config_path: str):
    logger.info(f"CLI 'train-model' called. Config: {config_path}"); click.echo("--- train-model ---")
    loaded_config = load_and_validate_config(config_path, TrainModelConfig)
    if not loaded_config: click.echo("train-model failed (config issues).", err=True); return
    click.echo(f"Op: Train a {loaded_config.model_type} model.")
    if loaded_config.model_type == "XGBoost": 
        try:
            features_df = pd.read_csv(loaded_config.input_features_path)
            if loaded_config.target_column not in features_df.columns:
                click.echo(f"Error: Target col '{loaded_config.target_column}' not in {loaded_config.input_features_path}", err=True); return
            X_train_dummy = features_df.drop(columns=[loaded_config.target_column]).fillna(0)
            y_train_dummy = features_df[loaded_config.target_column].fillna(0)
            if X_train_dummy.empty or y_train_dummy.empty: click.echo(f"Error: No data from {loaded_config.input_features_path}.", err=True); return
            from src.modeling import train_xgboost
            click.echo(f"Calling src.modeling.train_xgboost from {loaded_config.input_features_path}...")
            model = train_xgboost(X_train_dummy, y_train_dummy, loaded_config)
            if model: click.echo(f"XGBoost training executed. Model type: {type(model)}")
            else: click.echo("XGBoost training returned no model.", err=True)
        except Exception as e: click.echo(f"Training error: {e}", err=True); logger.error(f"Training error: {e}")
    else: click.echo(f"Actual training for '{loaded_config.model_type}' not fully implemented in CLI stub.", err=True)
    click.echo(f"Model base output path: {loaded_config.model_output_path_base}"); click.echo("train-model logic completed.")

@cli.command('evaluate') 
@click.option('--config', 'config_path', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to EvaluateModelConfig JSON file.')
@click.pass_context
def evaluate(ctx, config_path: str):
    """Evaluates a trained model using test data and saves metrics."""
    logger.info(f"CLI 'evaluate' called. Config: {config_path}"); click.echo("--- Model Evaluation ---")
    loaded_config = load_and_validate_config(config_path, EvaluateModelConfig)
    if not loaded_config: click.echo("Model evaluation failed (config issues).", err=True); return

    try:
        model_path_obj = Path(loaded_config.model_path)
        logger.info(f"Loading model from: {model_path_obj}")
        model = load_model(model_name=model_path_obj.name, models_dir=str(model_path_obj.parent))
        if not model: raise FileNotFoundError(f"Model could not be loaded from {model_path_obj}")
        click.echo(f"Model loaded successfully from {model_path_obj}")

        scaler = None
        if loaded_config.scaler_path:
            scaler_path_obj = Path(loaded_config.scaler_path)
            logger.info(f"Loading scaler from: {scaler_path_obj}")
            if not scaler_path_obj.exists(): raise FileNotFoundError(f"Scaler file not found: {scaler_path_obj}")
            scaler = joblib.load(scaler_path_obj); click.echo(f"Scaler loaded successfully from {scaler_path_obj}")

        logger.info(f"Loading test data from: {loaded_config.test_data_path}")
        test_data_path_obj = Path(loaded_config.test_data_path)
        if not test_data_path_obj.exists(): raise FileNotFoundError(f"Test data file not found: {test_data_path_obj}")
        test_df = pd.read_csv(test_data_path_obj) if str(test_data_path_obj).endswith(".csv") else pd.read_parquet(test_data_path_obj)
        click.echo(f"Test data loaded. Shape: {test_df.shape}")

        if loaded_config.target_column not in test_df.columns: raise ValueError(f"Target '{loaded_config.target_column}' not in test data.")
        feature_columns = test_df.columns.drop(loaded_config.target_column, errors='ignore').tolist()
        if not feature_columns: raise ValueError("No feature columns found in test data.")
        X_test = test_df[feature_columns]; y_true = test_df[loaded_config.target_column]
        click.echo(f"Prepared X_test (shape: {X_test.shape}), y_true (shape: {y_true.shape})")

        logger.info(f"Performing predictions (model type from config: {loaded_config.model_type or 'Generic'}).")
        X_test_processed = X_test.copy() # Keep original X_test for SHAP if needed before scaling
        if scaler: 
            X_test_scaled_np = scaler.transform(X_test_processed)
            X_test_processed = pd.DataFrame(X_test_scaled_np, columns=X_test_processed.columns, index=X_test_processed.index)
            click.echo("Test data scaled.")
        
        if hasattr(model, "predict_proba") and hasattr(model, "predict"): 
            y_pred_proba_np = model.predict_proba(X_test_processed)[:, 1]
            y_pred_np = model.predict(X_test_processed)
        elif hasattr(model, "predict"): 
            y_pred_proba_np = model.predict(X_test_processed)
            if y_pred_proba_np.ndim > 1 and y_pred_proba_np.shape[1] > 1: y_pred_proba_np = y_pred_proba_np[:, 1]
            y_pred_np = (y_pred_proba_np > 0.5).astype(int)
        else: raise AttributeError("Model missing standard predict/predict_proba methods.")
        
        y_pred = pd.Series(y_pred_np.flatten(), index=y_true.index, name="y_pred")
        y_pred_proba = pd.Series(y_pred_proba_np.flatten(), index=y_true.index, name="y_pred_proba")
        click.echo("Predictions generated.")

        logger.info("Calculating classification metrics..."); all_metrics = evaluation_utils.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        metrics_to_save = {k: v for k, v in all_metrics.items() if k in loaded_config.metrics_to_compute} if loaded_config.metrics_to_compute else all_metrics
        if loaded_config.metrics_to_compute: click.echo(f"Filtered metrics to: {list(metrics_to_save.keys())}")
        click.echo("\nCalculated Metrics:"); click.echo(json.dumps(metrics_to_save, indent=2))

        metrics_output_path_obj = Path(loaded_config.metrics_output_json_path)
        metrics_output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_path_obj, 'w') as f: json.dump(metrics_to_save, f, indent=4)
        click.echo(f"Metrics saved to: {metrics_output_path_obj}"); logger.info(f"Metrics saved to: {metrics_output_path_obj}")

        # SHAP Value Plot Generation
        if loaded_config.shap_summary_plot_path:
            if not evaluation_utils.SHAP_INSTALLED: # Check if SHAP is available
                logger.warning("SHAP library not installed. Skipping SHAP summary plot generation.")
                click.echo("Warning: SHAP library not installed. Skipping SHAP summary plot.", err=True)
            else:
                logger.info(f"Attempting to generate SHAP summary plot to: {loaded_config.shap_summary_plot_path}")
                X_train_sample_df = None
                if loaded_config.shap_train_data_sample_path:
                    shap_sample_path_obj = Path(loaded_config.shap_train_data_sample_path)
                    if shap_sample_path_obj.exists():
                        try:
                            if str(shap_sample_path_obj).endswith(".parquet"): X_train_sample_df = pd.read_parquet(shap_sample_path_obj)
                            else: X_train_sample_df = pd.read_csv(shap_sample_path_obj)
                            # Ensure X_train_sample_df has the same columns as X_test (features)
                            X_train_sample_df = X_train_sample_df[feature_columns] if feature_columns else X_train_sample_df 
                            click.echo(f"Loaded SHAP training sample data from: {shap_sample_path_obj}")
                        except Exception as e_shap_load:
                            logger.error(f"Error loading SHAP training sample data from {shap_sample_path_obj}: {e_shap_load}")
                            click.echo(f"Error loading SHAP sample data: {e_shap_load}", err=True)
                    else:
                        logger.warning(f"SHAP training sample data file not found: {shap_sample_path_obj}")
                        click.echo(f"Warning: SHAP sample data file not found at {shap_sample_path_obj}", err=True)
                
                # X_data for SHAP should be the same features used for prediction (X_test before scaling if explainer handles scaling, or X_test_processed if explainer needs scaled data)
                # TreeExplainers usually handle unscaled data well. Kernel/Deep might need scaled.
                # For simplicity, using X_test (original, unscaled features). If model is Keras, X_train_sample should also be unscaled.
                evaluation_utils.generate_shap_summary_plot(
                    model=model,
                    X_data=X_test, # Use original X_test for tree models, ensure consistency for others
                    model_type=loaded_config.model_type or "Unknown", # Pass model_type from config
                    output_plot_path=str(loaded_config.shap_summary_plot_path),
                    X_train_sample=X_train_sample_df
                )
                plot_path_obj = Path(loaded_config.shap_summary_plot_path)
                if plot_path_obj.exists(): click.echo(f"SHAP summary plot saved to: {plot_path_obj}")
                else: click.echo("SHAP summary plot generation might have failed (check logs).", err=True)

    except Exception as e: click.echo(f"Evaluation error: {e}", err=True); logger.error(f"Evaluation error: {e}", exc_info=True)
    else: click.echo("Model evaluation completed successfully.")

@cli.command('backtest') 
@click.option('--config', default=None, help='Path to BacktestConfig JSON file.')
@click.pass_context
def backtest(ctx, config): logger.info(f"CLI 'backtest' called. Config: {config}"); click.echo("--- backtest ---"); loaded_config = load_and_validate_config(config, BacktestConfig); click.echo("backtest logic (stub) completed." if loaded_config else "backtest failed.", err=not loaded_config)

@cli.command('export') 
@click.option('--config', default=None, help='Path to ExportConfig JSON file.')
@click.pass_context
def export(ctx, config): logger.info(f"CLI 'export' called. Config: {config}"); click.echo("--- export ---"); loaded_config = load_and_validate_config(config, ExportConfig); click.echo("export logic (stub) completed." if loaded_config else "export failed.", err=not loaded_config)

@cli.group("models") 
def models_group(): logger.debug("CLI group 'models' invoked."); pass
@models_group.command("list")
@click.option("--name", "model_name_from_config", default=None, help="Filter by model name.")
@click.pass_context
def list_registered_models(ctx,model_name_from_config:str):
    logger.info(f"CLI 'models list'. Filter: {model_name_from_config}")
    models=registry_utils.list_models(model_name_from_config)
    if not models: click.echo("No models found." if not model_name_from_config else f"No models for '{model_name_from_config}'."); return
    click.echo("\n--- Registered Models ---")
    header="| {:<20} | {:<27} | {:<21} | {:<10} | {:<7} | {:<35} | {:<8} |".format("Model Name","Version","Timestamp (UTC)","Metric","Value","Metadata Path","Has FI")
    click.echo(f"|{'-'*22}|{'-'*29}|{'-'*23}|{'-'*12}|{'-'*9}|{'-'*37}|{'-'*10}|")
    for entry in models:
        val_str=f"{entry.get('primary_metric_value','N/A'):.3f}" if isinstance(entry.get('primary_metric_value'),float) else str(entry.get('primary_metric_value','N/A'))
        click.echo(f"| {entry.get('model_name_from_config','N/A'):<20} | {entry.get('model_version','N/A'):<27} | {entry.get('timestamp_utc','N/A'):<21} | {entry.get('primary_metric_name','N/A'):<10} | {val_str:<7} | {entry.get('meta_json_path','N/A'):<35} | {str(entry.get('has_feature_importance',False)):<8} |")
    click.echo("-------------------------\n")
@models_group.command("describe")
@click.argument("model_name_from_config")
@click.argument("version")
@click.pass_context
def describe_model_version(ctx,model_name_from_config:str,version:str):
    logger.info(f"CLI 'models describe' for {model_name_from_config} v{version}"); details=registry_utils.get_model_details(model_name_from_config,version)
    click.echo(json.dumps(details,indent=2) if details else "Model version not found.", err=not details)
@models_group.command("compare")
@click.argument("model_name_from_config")
@click.argument("versions",nargs=-1)
@click.pass_context
def compare_model_meta_versions(ctx,model_name_from_config:str,versions:tuple[str]):
    logger.info(f"CLI 'models compare' for {model_name_from_config}, versions: {versions}")
    if len(versions)<2:click.echo("Error: At least two versions required.",err=True);return
    data=registry_utils.compare_model_versions(model_name_from_config,list(versions));click.echo(json.dumps(data,indent=2))
@models_group.command("get-latest-path")
@click.argument("model_name_from_config")
@click.pass_context
def get_latest_model_meta_path(ctx,model_name_from_config:str):
    logger.info(f"CLI 'models get-latest-path' for {model_name_from_config}");path=registry_utils.get_latest_model_version_path(model_name_from_config)
    click.echo(path if path else "Model not found.",err=not path)

@cli.group("features")
def features_group(): """Manages feature analysis and drift detection.""" ;logger.debug("CLI group 'features' invoked.");pass
@features_group.command("analyze-drift")
@click.option('--current-features-stats','current_stats_path',required=True,type=click.Path(exists=True,dir_okay=False,readable=True),help="Path to current feature stats JSON.")
@click.option('--baseline-stats','baseline_stats_path',required=True,type=click.Path(exists=True,dir_okay=False,readable=True),help="Path to baseline feature stats JSON.")
@click.option('--output','output_path',required=True,type=click.Path(dir_okay=False,writable=True,allow_dash=False),help="Path to save drift report JSON.")
@click.pass_context
def analyze_feature_drift(ctx,current_stats_path:str,baseline_stats_path:str,output_path:str):
    """Compares current feature statistics against a baseline to detect drift."""
    logger.info(f"CLI 'features analyze-drift'. Current:{current_stats_path}, Baseline:{baseline_stats_path}, Output:{output_path}")
    try:
        with open(current_stats_path,'r') as f:current_stats=json.load(f)
        with open(baseline_stats_path,'r') as f:baseline_stats=json.load(f)
        logger.info("Loaded current and baseline stats files.")
    except Exception as e:click.echo(f"Error loading stats files: {e}",err=True);logger.error(f"Error loading stats: {e}");return
    drift_report=compare_feature_statistics(current_stats,baseline_stats)
    output_path_obj=Path(output_path);output_path_obj.parent.mkdir(parents=True,exist_ok=True)
    try:
        with open(output_path_obj,'w') as f:json.dump(drift_report,f,indent=4)
        click.echo(f"Drift report saved to: {output_path_obj}");logger.info(f"Drift report saved: {output_path_obj}")
    except Exception as e:click.echo(f"Error saving drift report: {e}",err=True);logger.error(f"Error saving report: {e}")

if __name__ == '__main__':
    cli()
