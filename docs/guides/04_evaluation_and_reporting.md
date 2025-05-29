# Model Evaluation and Reporting

## Overview

This document outlines the tools and reports available within the QLOP project for evaluating model performance and generating insights. Effective evaluation is key to understanding model behavior, comparing different models or versions, and ensuring reliability.

## 1. Calculating Performance Metrics

The core of model performance assessment lies in quantitative metrics.

-   **Function**: `src.evaluation.calculate_classification_metrics(y_true, y_pred, y_pred_proba)`
-   **Calculated Metrics**:
    -   Accuracy
    -   Precision (with `zero_division=0`)
    -   Recall (with `zero_division=0`)
    -   F1-Score (with `zero_division=0`)
    -   ROC-AUC (Area Under the Receiver Operating Characteristic Curve); returns `NaN` if only one class is present in `y_true`.
    -   MAPE (Mean Absolute Percentage Error):
        *Note: Typically for regression. For binary (0/1) classification, it's calculated here by treating labels as continuous, considering only cases where `y_true` is non-zero (i.e., `y_true=1`). Interpretation requires care.*
    -   SMAPE (Symmetric Mean Absolute Percentage Error):
        *Note: Also more common in regression. Calculated here treating labels as continuous. Interpretation requires care.*
-   **Saving Metrics**:
    -   **Primary Method**: The `evaluate` CLI command (see below) is the main way to generate and save these metrics. The metrics dictionary is saved to a JSON file specified by `EvaluateModelConfig.metrics_output_json_path`.
    -   **Model Metadata**: While not fully detailed in this section, these performance metrics are also intended to be (and often are, via placeholder fields) included in the model's `.meta.json` file when a model is registered (see `src/metadata_utils.py`).

## 2. Using the `evaluate` CLI Command

The `evaluate` command is the primary tool for assessing a trained model's performance on a test dataset.

-   **Syntax**:
    ```bash
    python main.py evaluate --config <path_to_eval_config.json>
    ```
-   **Configuration (`EvaluateModelConfig` in `src/config_models.py`)**: The behavior of the command is controlled by a JSON configuration file. Key fields include:
    -   `model_path` (required): Path to the saved trained model file.
    -   `scaler_path` (optional): Path to a saved scaler object (e.g., `MinMaxScaler`) if features need to be scaled before prediction.
    -   `test_data_path` (required): Path to the test dataset (CSV or Parquet) containing features and the target variable.
    -   `target_column` (str, default: "target"): Name of the target variable column in the test data.
    -   `metrics_output_json_path` (required): Path where the calculated metrics (JSON dictionary) will be saved.
    -   `metrics_to_compute` (optional, list of strings): If provided, only these specified metrics will be calculated and saved (e.g., `["accuracy", "roc_auc"]`). If `null` or omitted, all default metrics calculated by `calculate_classification_metrics` are saved.
    -   `model_type` (optional, str): Helps in logging and can be used in the future to select model-specific prediction logic (e.g., for sequence models).
    -   `shap_summary_plot_path` (optional): Path to save the SHAP summary plot image.
    -   `shap_train_data_sample_path` (optional): Path to a sample of training data, required for some SHAP explainers.

## 3. SHAP Value Summary Plots

SHAP (SHapley Additive exPlanations) values help explain the output of machine learning models by quantifying the contribution of each feature to a prediction.

-   **Function**: `src.evaluation.generate_shap_summary_plot(model, X_data, model_type, output_plot_path, X_train_sample)`
-   **Purpose**: Generates a bar plot summarizing the mean absolute SHAP values, indicating overall feature importance.
-   **Triggering via `evaluate` CLI Command**:
    -   If `EvaluateModelConfig.shap_summary_plot_path` is specified in the evaluation configuration, the `evaluate` command will attempt to generate this plot.
    -   `EvaluateModelConfig.shap_train_data_sample_path` may be required for certain SHAP explainers (e.g., `KernelExplainer`, `DeepExplainer`) to provide background data.
    -   `X_test` from the evaluation data is used as the `X_data` for calculating SHAP values.
-   **SHAP Explainers**: The function attempts to select an appropriate SHAP explainer based on the `model_type` string provided:
    -   `TreeExplainer` for "XGBoost", "LGBM", "CatBoost".
    -   `DeepExplainer` for "Keras".
    -   `KernelExplainer` as a default for other types (e.g., "Sklearn-other").
-   **Dependency**: Requires the `shap` library to be installed (`pip install shap`). A warning is logged if it's not found, and plot generation is skipped.

## 4. Comparing Models with Charts

To visually compare the performance of different versions of the same model, a dedicated CLI command is provided.

-   **CLI Command**: `qlop-cli models plot-comparison` (or `python main.py models plot-comparison`)
-   **Syntax**:
    ```bash
    python main.py models plot-comparison <MODEL_NAME> <VERSION1> <VERSION2> [<VERSION3>...] \
        [--metrics <metric1,metric2,...>] \
        --output <output_plot.png>
    ```
-   **Functionality**:
    -   Takes a `MODEL_NAME` (e.g., "XGBoostTest") and at least two `VERSION_STRING`s.
    -   Retrieves metadata for each specified model version from the model registry (`models/model_registry.jsonl`).
    -   Extracts the metrics specified in the `--metrics` option (e.g., "accuracy,f1_score"). If no metrics are specified, it defaults to a predefined list (accuracy, f1_score, roc_auc, precision, recall).
    -   Generates a grouped bar chart comparing these metrics across the selected model versions.
    -   Saves the chart to the path specified by `--output`.
-   **Dependency**: Requires `matplotlib` and `pandas` (which are standard project dependencies).
-   **Function**: The underlying logic is in `src.evaluation.plot_model_comparison_charts()`.

## 5. Other Visualizations

The `src/evaluation.py` module also contains functions for generating other common evaluation plots:

-   **`plot_roc_auc(y_true, y_pred_proba, ax, model_name)`**:
    -   Plots the Receiver Operating Characteristic (ROC) curve.
    -   Calculates and displays the Area Under the Curve (AUC).
    -   Handles cases where AUC is undefined (e.g., only one class in `y_true`).
-   **`plot_confusion_matrix(y_true, y_pred, ax, class_names, model_name)`**:
    -   Plots the confusion matrix.

**Usage**:
These plotting functions are not directly exposed via dedicated CLI commands for generating standalone plot files *yet*. However:
-   They are used internally by the Streamlit GUI (`gui.py`) to display evaluation results.
-   They can be imported and used in Jupyter notebooks or custom Python scripts for more detailed, ad-hoc evaluation and visualization.
-   The `evaluate` CLI command could be extended in the future to optionally save these plots if configured.

This suite of evaluation tools and reports aids in rigorously assessing model performance and making informed decisions.
