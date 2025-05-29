# Feature Analysis: Importance, Drift, and Elimination

## Overview

Analyzing features is a critical step in the machine learning lifecycle. It helps in understanding which features are most influential for model predictions, ensuring model stability by monitoring changes in data distributions (drift), and improving model performance and efficiency by eliminating irrelevant or redundant features. This document outlines the feature analysis capabilities built into the QLOP project.

## 1. Feature Importance Logging

Understanding which features a model considers important is key for model interpretation, debugging, and feature selection.

### Tree-Based Models (XGBoost, LightGBM, CatBoost)

-   **Automatic Calculation**: For tree-based models like XGBoost, LightGBM, and CatBoost, feature importances (e.g., based on gain, split count, or permutation within the model's own calculation) are automatically available after training.
-   **Storage**:
    -   These importances are extracted by the respective `train_<model_type>` functions in `src/modeling.py`.
    -   They are saved as a JSON file named `feature_importances.json` within the model's versioned directory (e.g., `models/<model_name_from_config>/<model_version>/feature_importances.json`).
-   **Format**: The `feature_importances.json` file contains a list of objects, where each object represents a feature and its importance score, sorted by importance in descending order:
    ```json
    [
        {"feature": "feature_name_1", "importance": 0.35},
        {"feature": "feature_name_2", "importance": 0.22},
        ...
    ]
    ```
-   **Metadata Linkage**:
    -   The model's primary metadata file (`model.meta.json`) includes a field `feature_importance_file` (e.g., `"feature_importance_file": "feature_importances.json"`) which points to this artifact within the same directory.
    -   The central model registry (`models/model_registry.jsonl`) also indicates the availability of these importances via a boolean flag `has_feature_importance: true` for the corresponding model version entry.

### Other Model Types (e.g., Keras Deep Learning Models)

-   **Permutation Importance (Planned)**: For models where intrinsic feature importance scores are not readily available (like Keras-based neural networks), the planned approach is to use permutation importance.
-   **Functionality**: The function `src.feature_analysis.calculate_permutation_importance()` is intended for this purpose.
    -   **Current Status**: This function is currently a **STUB**. It logs a warning and returns placeholder dummy importance values.
    -   **Future Implementation**: It will use `sklearn.inspection.permutation_importance` to calculate scores based on how shuffling a feature's values affects model performance on a test set.
-   **Integration Notes**:
    *   `TODO` comments are placed within `src/modeling.py` (specifically in the `_save_keras_model_and_metadata` helper function) to indicate where the call to `calculate_permutation_importance` and the subsequent saving of its results would be integrated.
    *   This would typically occur after model training and require access to a test dataset (`X_test`, `y_test`). The results would also be saved as a JSON artifact (e.g., `permutation_importances.json`) and linked in the model's metadata.

## 2. Feature Drift Tracking

### Concept

Feature drift occurs when the statistical properties of features in the live prediction data change significantly from the data the model was trained on. Monitoring and identifying drift is crucial for maintaining model performance and reliability, as models trained on outdated data distributions may make inaccurate predictions.

### Generating Baseline Statistics

-   **Statistics Calculation**: The function `src.feature_analysis.calculate_feature_statistics()` is responsible for computing a variety of summary statistics for each feature in a given DataFrame.
-   **CLI Integration**:
    -   When you run the `engineer-features` CLI command (`python main.py engineer-features --config <config_file>`), this function is automatically called on the newly engineered features.
    -   The generated statistics are saved to a JSON file. The filename is derived from the `output_features_path` specified in the `FeatureEngineeringConfig`, typically suffixed with `_current_stats.json` (e.g., `data/processed/my_features_current_stats.json`).
    -   This `_current_stats.json` file can then serve as a **baseline** for future drift comparisons.
-   **Statistics Collected** (per feature):
    -   **Common**: Data type, total count, missing count, missing percentage.
    -   **Numerical**: Mean, standard deviation, min, max, median, 25th/75th percentiles.
    -   **Categorical/Object/Boolean**: Unique value count, top N most frequent values (and their counts/percentages), list of unique values (or top N).

### Analyzing Drift

-   **Dedicated CLI Command**: The project provides a command to compare two sets of feature statistics (a "current" set against a "baseline" set):
    ```bash
    python main.py features analyze-drift \
        --current-features-stats <path_to_current_stats.json> \
        --baseline-stats <path_to_baseline_stats.json> \
        --output <drift_report_output.json>
    ```
-   **Inputs**:
    -   `--current-features-stats`: Path to the JSON file containing statistics of the current feature set.
    -   `--baseline-stats`: Path to the JSON file containing statistics of the baseline/reference feature set.
    -   `--output`: Path where the generated drift report (JSON) will be saved.
-   **Drift Report Content**: The output JSON file from `analyze-drift` contains a per-feature comparison, including:
    -   Status if a feature is missing in current or baseline.
    -   Comparison of data types; `dtype_drift_detected: true` if they differ.
    -   For numerical features: Percentage change in mean and standard deviation, comparison of key percentiles. Includes a placeholder for Population Stability Index (PSI) (`"psi": "not_implemented_from_summary_stats"`).
    -   For categorical features: Changes in frequencies for top N categories, Jaccard index for the sets of top N categories. Includes a placeholder for Chi-Squared test p-value (`"chi_squared_p_value": "not_implemented_from_summary_stats"`).
    -   Comparison of missing value percentages.

### Integrated Drift Reporting during Feature Engineering

-   **CLI Option**: The `engineer-features` command has an optional flag:
    ```bash
    python main.py engineer-features --config <config_file> --compare-to-baseline <path_to_baseline_stats.json>
    ```
-   **Functionality**:
    -   If `--compare-to-baseline` is provided, after generating new features and their statistics (`_current_stats.json`), the command will automatically run a drift analysis against the provided baseline statistics file.
    -   The drift report is saved to a file named similarly to `<output_features_base>_drift_report.json`.
    -   This allows for immediate feedback on feature drift when processing new data.
    -   If the option is not provided, only the current feature statistics are saved.

## 3. Auto-Feature Elimination (Framework)

This is a planned capability to automatically identify and select features for removal based on criteria derived from feature importance scores and drift analysis results.

-   **Functionality**: The core logic will reside in `src.feature_analysis.select_features_for_elimination()`.
-   **Current Status**: This function is currently a **STUB**.
    -   It logs its input parameters (paths to importance/drift files, elimination configuration).
    -   It logs a warning that the advanced selection logic is a TODO.
    -   It currently returns an empty list `[]`, meaning no features are selected for elimination by the stub.
-   **Configuration**:
    -   The selection process will be driven by an `elimination_config` dictionary passed to the function.
    -   This configuration would specify thresholds (e.g., minimum importance, maximum drift PSI, maximum change in mean for numerical features) and strategy (e.g., prioritize low importance or high drift).
    -   It is intended that this `elimination_config` will be part of the `FeatureEngineeringConfig` Pydantic model in `src/config_models.py` in the future, allowing it to be specified per feature engineering run.

## 4. Future Extensions

-   **Full Permutation Importance**: Implement the logic in `calculate_permutation_importance()` using `sklearn.inspection.permutation_importance` for all applicable model types, especially Keras models.
-   **PSI and Chi-Squared Tests**: Implement robust calculations for Population Stability Index (PSI) for numerical features and Chi-Squared tests for categorical features in `compare_feature_statistics()`. This may require more detailed distributional information (e.g., histograms) than just summary statistics.
-   **Feature Elimination Logic**: Fully implement `select_features_for_elimination()` with configurable strategies to combine importance and drift metrics for feature removal.
-   **Automated Actions**: Integrate feature elimination suggestions or actions directly into the pipeline (e.g., automatically excluding low-value or highly drifted features from model training based on configuration).
-   **Visualization**: Add capabilities to visualize drift reports and feature importance results.

This structured approach to feature analysis aims to build more robust, interpretable, and maintainable machine learning models.
