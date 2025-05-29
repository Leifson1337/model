# src/evaluation.py
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import logging
from typing import Dict, Any, Optional, List # Ensure List is imported
import os # For __main__ test file operations
from sklearn.metrics import (
    RocCurveDisplay, 
    ConfusionMatrixDisplay, 
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# SHAP and Matplotlib for plotting
try:
    import shap
    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False
    print("Warning: `shap` library not found. SHAP value calculations will be skipped.")


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Helper functions for MAPE and SMAPE --- (Keep existing functions)
def mean_absolute_percentage_error(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> float:
    y_true_np, y_pred_np = np.array(y_true_np), np.array(y_pred_np)
    mask = y_true_np != 0
    if not np.any(mask): logger.warning("MAPE: All true values are zero, returning np.nan."); return np.nan
    return np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100

def symmetric_mean_absolute_percentage_error(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> float:
    y_true_np, y_pred_np = np.array(y_true_np), np.array(y_pred_np)
    numerator = np.abs(y_pred_np - y_true_np)
    denominator = (np.abs(y_true_np) + np.abs(y_pred_np)) / 2.0
    mask = denominator != 0
    smape_values = np.zeros_like(denominator)
    if np.any(mask): smape_values[mask] = numerator[mask] / denominator[mask]
    if not np.any(mask) and np.all(numerator==0): logger.warning("SMAPE: All true/pred are zero, returning 0.0."); return 0.0
    elif not np.any(mask): logger.warning("SMAPE: All denominators zero, but not all numerators. Returning np.nan."); return np.nan
    return np.mean(smape_values) * 100


def calculate_classification_metrics(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series) -> Dict[str, Any]:
    metrics_dict: Dict[str, Any] = {}
    if not (isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series) and isinstance(y_pred_proba, pd.Series)):
        logger.error("Inputs must be pandas Series."); return {"error": "Inputs must be pandas Series."}
    if not (len(y_true) == len(y_pred) == len(y_pred_proba)):
        logger.error("Input series must have same length."); return {"error": "Input series must have same length."}
    if y_true.empty: logger.warning("y_true is empty."); return {"warning": "y_true is empty."}

    try: metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)
    except Exception as e: logger.error(f"Error accuracy: {e}"); metrics_dict["accuracy"] = np.nan
    try: metrics_dict["precision"] = precision_score(y_true, y_pred, zero_division=0)
    except Exception as e: logger.error(f"Error precision: {e}"); metrics_dict["precision"] = np.nan
    try: metrics_dict["recall"] = recall_score(y_true, y_pred, zero_division=0)
    except Exception as e: logger.error(f"Error recall: {e}"); metrics_dict["recall"] = np.nan
    try: metrics_dict["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e: logger.error(f"Error f1_score: {e}"); metrics_dict["f1_score"] = np.nan
    try:
        if len(np.unique(y_true)) > 1: metrics_dict["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        else: logger.warning("ROC AUC undefined (1 class in y_true)."); metrics_dict["roc_auc"] = np.nan
    except Exception as e: logger.error(f"Error roc_auc: {e}"); metrics_dict["roc_auc"] = np.nan
    
    y_true_np, y_pred_np = y_true.to_numpy(), y_pred.to_numpy()
    try: metrics_dict["mape"] = mean_absolute_percentage_error(y_true_np, y_pred_np)
    except Exception as e: logger.error(f"Error mape: {e}"); metrics_dict["mape"] = np.nan
    try: metrics_dict["smape"] = symmetric_mean_absolute_percentage_error(y_true_np, y_pred_np)
    except Exception as e: logger.error(f"Error smape: {e}"); metrics_dict["smape"] = np.nan
        
    logger.info(f"Calculated classification metrics: {metrics_dict}")
    return metrics_dict


def generate_shap_summary_plot(model: Any, X_data: pd.DataFrame, model_type: str, 
                               output_plot_path: str, X_train_sample: Optional[pd.DataFrame] = None):
    global SHAP_INSTALLED 
    if not SHAP_INSTALLED: logger.error("SHAP library not installed. Cannot generate SHAP plot."); return
    logger.info(f"Generating SHAP summary plot for '{model_type}'. Output: {output_plot_path}")
    try:
        explainer = None
        if model_type.lower() in ["xgboost", "lgbm", "catboost", "lightgbm"]:
            explainer = shap.TreeExplainer(model, data=X_train_sample) if X_train_sample is not None else shap.TreeExplainer(model)
            if X_train_sample is None: logger.warning("TreeExplainer init without X_train_sample (background data).")
        elif model_type.lower() == "keras":
            if X_train_sample is None: raise ValueError("X_train_sample required for Keras DeepExplainer.")
            explainer = shap.DeepExplainer(model, X_train_sample)
        else: 
            logger.info(f"Using KernelExplainer for model type '{model_type}'. This can be slow.")
            if X_train_sample is None: raise ValueError("X_train_sample required for KernelExplainer.")
            if not hasattr(model, "predict_proba"): raise AttributeError("Model needs predict_proba for KernelExplainer.")
            explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
        if not explainer: logger.error(f"Could not init SHAP explainer for '{model_type}'."); return

        logger.info(f"Calculating SHAP values for X_data shape: {X_data.shape}")
        shap_values_obj = explainer(X_data) 
        
        shap_values_for_plot = None
        if isinstance(shap_values_obj.values, list) and len(shap_values_obj.values) == 2:
            shap_values_for_plot = shap_values_obj.values[1]; logger.info("Using SHAP values for positive class (index 1).")
        elif isinstance(shap_values_obj.values, np.ndarray): 
            shap_values_for_plot = shap_values_obj.values; logger.info("SHAP values are single array, using as is.")
        else: logger.error(f"Unexpected SHAP values format: {type(shap_values_obj.values)}."); return

        plt.figure(); shap.summary_plot(shap_values_for_plot, X_data, show=False, plot_type="bar")
        plt.title(f"SHAP Summary Plot ({model_type})"); plt.savefig(output_plot_path, bbox_inches='tight'); plt.clf(); plt.close()
        logger.info(f"SHAP summary plot saved to {output_plot_path}")
    except Exception as e: logger.error(f"Error during SHAP processing: {e}", exc_info=True)


def plot_roc_auc(y_true, y_pred_proba, ax=None, model_name: str = "", **kwargs):
    if ax is None: fig, ax = plt.subplots(figsize=(8, 6))
    try:
        if len(np.unique(y_true)) > 1:
            RocCurveDisplay.from_predictions(y_true, y_pred_proba, ax=ax, name=model_name, **kwargs)
            auc = roc_auc_score(y_true, y_pred_proba); title = f"ROC Curve (AUC = {auc:.2f})"
        else: auc=np.nan; title="ROC Curve (AUC undefined: 1 class in y_true)"; ax.text(0.5,0.5,"ROC AUC undefined:\n1 class in y_true.",ha='center',va='center',transform=ax.transAxes)
        if model_name: title = f"{model_name} - {title}"; ax.set_title(title)
    except Exception as e: ax.text(0.5,0.5,f"ROC Plot Error: {e}",ha='center',va='center',wrap=True); logger.error(f"Error ROC for {model_name}: {e}")
    return ax

def plot_confusion_matrix(y_true, y_pred, ax=None, class_names=None, model_name: str = "", **kwargs):
    if ax is None: fig, ax = plt.subplots(figsize=(7, 7))
    if class_names is None: class_names = ['No Sig. Move', 'Sig. Move'] 
    try:
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, display_labels=class_names, cmap=plt.cm.Blues, **kwargs)
        title = "Confusion Matrix"; 
        if model_name: title = f"{model_name} - {title}"; ax.set_title(title)
    except Exception as e: ax.text(0.5,0.5,f"Could not plot CM: {e}",ha='center',va='center',wrap=True); logger.error(f"Error CM for {model_name}: {e}")
    return ax

def plot_model_comparison_charts(model_metadata_list: List[Dict[str, Any]], 
                                 metrics_to_plot: Optional[List[str]] = None, 
                                 output_chart_path: str = "model_comparison.png"):
    """
    Generates and saves a bar chart comparing specified metrics across model versions.
    """
    logger.info(f"Generating model comparison chart. Models: {len(model_metadata_list)}, Metrics: {metrics_to_plot or 'default'}, Output: {output_chart_path}")

    if not model_metadata_list:
        logger.warning("No model metadata provided. Skipping chart generation.")
        return

    if metrics_to_plot is None:
        # Default metrics if none are specified
        metrics_to_plot = ["accuracy", "f1_score", "roc_auc", "precision", "recall"] 
        # Try to find at least one common metric present in all models if defaults are all missing
        # For simplicity, we'll proceed with defaults and handle missing ones.
        logger.info(f"No specific metrics provided, using defaults: {metrics_to_plot}")

    chart_data = []
    model_versions_for_chart = []

    for meta in model_metadata_list:
        version = meta.get("model_version", "UnknownVersion")
        model_versions_for_chart.append(version)
        
        metric_values = {"model_version": version}
        model_metrics = meta.get("metrics", {})
        
        for metric_key in metrics_to_plot:
            metric_values[metric_key] = model_metrics.get(metric_key) # Will be None if missing
            if metric_values[metric_key] is None:
                logger.warning(f"Metric '{metric_key}' not found for model version '{version}'. Will plot as 0 or NaN.")
        chart_data.append(metric_values)

    if not chart_data:
        logger.warning("No data extracted for chart generation.")
        return

    df_chart = pd.DataFrame(chart_data)
    df_chart = df_chart.set_index("model_version")
    
    # Filter out metrics that are completely missing from all models to avoid empty plots
    df_chart = df_chart.dropna(axis=1, how='all')
    
    # Re-check metrics_to_plot against available columns in df_chart
    final_metrics_to_plot = [m for m in metrics_to_plot if m in df_chart.columns]
    if not final_metrics_to_plot:
        logger.warning("None of the specified or default metrics are available in the provided metadata. Cannot generate chart.")
        return
    
    df_chart_to_plot = df_chart[final_metrics_to_plot]


    try:
        num_metrics = len(df_chart_to_plot.columns)
        if num_metrics == 0: logger.warning("No valid metrics to plot after filtering."); return

        # Plotting
        # If only one metric, simple bar plot. If multiple, grouped bar plot.
        # pandas plot kind='bar' handles grouped automatically if DataFrame has multiple columns.
        ax = df_chart_to_plot.plot(kind='bar', figsize=(max(10, 2 * len(model_versions_for_chart) * num_metrics), 6), rot=45)
        
        ax.set_title(f"Model Performance Comparison ({', '.join(final_metrics_to_plot)})")
        ax.set_ylabel("Metric Value")
        ax.set_xlabel("Model Version")
        ax.legend(title="Metrics")
        ax.grid(axis='y', linestyle='--')
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        plt.savefig(output_chart_path, bbox_inches='tight')
        plt.clf()
        plt.close()
        logger.info(f"Model comparison chart saved to {output_chart_path}")

    except Exception as e:
        logger.error(f"Error generating model comparison chart: {e}", exc_info=True)


if __name__ == '__main__':
    import os 
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    print("--- Testing Classification Metrics Calculation ---")
    # ... (existing metrics tests) ...
    print("\n--- Classification Metrics Tests Completed ---")

    print("\n--- Testing SHAP Summary Plot Generation (Conceptual) ---")
    # ... (existing SHAP tests) ...
    print("\n--- SHAP Summary Plot Generation Test Completed ---")

    print("\n--- Testing Model Comparison Chart Generation ---")
    dummy_meta_list = [
        {"model_version": "v1.0_abc", "metrics": {"accuracy": 0.85, "f1_score": 0.82, "roc_auc": 0.90, "precision": 0.78, "recall": 0.85}},
        {"model_version": "v1.1_def", "metrics": {"accuracy": 0.87, "f1_score": 0.84, "roc_auc": 0.92, "precision": 0.80, "recall": 0.88}},
        {"model_version": "v1.2_ghi", "metrics": {"accuracy": 0.82, "f1_score": 0.80, "roc_auc": 0.88, "precision": 0.75, "recall": 0.82, "mape": 10.5}},
        {"model_version": "v1.3_jkl", "metrics": {"accuracy": 0.90, "custom_metric": 0.77}} # Missing some common metrics
    ]
    comparison_chart_path = "temp_model_comparison_chart.png"
    
    print("\n1. Plotting default metrics:")
    plot_model_comparison_charts(dummy_meta_list, output_chart_path=comparison_chart_path)
    if os.path.exists(comparison_chart_path): print(f"Chart saved to {comparison_chart_path}"); os.remove(comparison_chart_path)
    else: print(f"Chart {comparison_chart_path} NOT created.")

    print("\n2. Plotting specific metrics (accuracy, roc_auc):")
    plot_model_comparison_charts(dummy_meta_list, metrics_to_plot=["accuracy", "roc_auc"], output_chart_path=comparison_chart_path)
    if os.path.exists(comparison_chart_path): print(f"Chart saved to {comparison_chart_path}"); os.remove(comparison_chart_path)
    else: print(f"Chart {comparison_chart_path} NOT created.")

    print("\n3. Plotting a metric that's partially missing (custom_metric, mape):")
    plot_model_comparison_charts(dummy_meta_list, metrics_to_plot=["accuracy", "custom_metric", "mape"], output_chart_path=comparison_chart_path)
    if os.path.exists(comparison_chart_path): print(f"Chart saved to {comparison_chart_path}"); os.remove(comparison_chart_path)
    else: print(f"Chart {comparison_chart_path} NOT created.")
    
    print("\n4. Plotting with no models:")
    plot_model_comparison_charts([], output_chart_path=comparison_chart_path) # Should log warning and not create file

    print("\n5. Plotting with models but no common metrics for specified list:")
    plot_model_comparison_charts(dummy_meta_list, metrics_to_plot=["non_existent_metric"], output_chart_path=comparison_chart_path)
    if os.path.exists(comparison_chart_path): print(f"Chart saved to {comparison_chart_path} (unexpected)"); os.remove(comparison_chart_path)
    else: print(f"Chart {comparison_chart_path} NOT created (as expected).")


    print("\n--- Model Comparison Chart Tests Completed ---")
    print("\n--- All Evaluation Module Tests Completed ---")
