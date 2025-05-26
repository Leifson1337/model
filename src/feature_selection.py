# src/feature_selection.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier # Example estimator
import xgboost as xgb # Allow XGBoost as an estimator too
import lightgbm as lgb # Allow LightGBM
import catboost as cb # Allow CatBoost
import shap 
import matplotlib.pyplot as plt # For SHAP plots

def select_features_rfe(
    X: pd.DataFrame, 
    y: pd.Series, 
    estimator, 
    n_features_to_select=None, 
    step: int = 1, 
    verbose: int = 0
) -> list:
    """
    Selects features using Recursive Feature Elimination (RFE).
    (Implementation from previous turn)
    """
    if n_features_to_select is None:
        n_features_to_select = X.shape[1] // 2
        print(f"RFE: n_features_to_select not provided, defaulting to half the features: {n_features_to_select}")
    
    if n_features_to_select <= 0 : # Handle case where 0 or negative features are requested
        print(f"RFE: n_features_to_select ({n_features_to_select}) is invalid. Returning empty list.")
        return []
        
    if n_features_to_select > X.shape[1]:
        print(f"Warning: n_features_to_select ({n_features_to_select}) is greater than total features ({X.shape[1]}). Selecting all features.")
        n_features_to_select = X.shape[1]

    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step, verbose=verbose)
    
    print(f"Fitting RFE with estimator: {type(estimator).__name__} to select {n_features_to_select} features...")
    rfe.fit(X, y)
    
    selected_feature_names = X.columns[rfe.support_].tolist()
    print(f"RFE selected {len(selected_feature_names)} features.")
    
    return selected_feature_names

def select_features_shap(
    X: pd.DataFrame, 
    y: pd.Series, # y is not directly used by SHAP explainer for feature importance but good practice to pass
    model, # A pre-trained model instance
    model_type: str = 'tree', # Helps guide explainer choice, though shap.Explainer often auto-detects
    n_features_to_select = 'auto', 
    threshold: float = 0.01, 
    sample_size: int = None, 
    show_plots: bool = False,
    plot_top_n_dependence: int = 3 # Number of top features for dependence plots
) -> list:
    """
    Selects features using SHAP values from a trained model.

    Args:
        X: Pandas DataFrame of features (ideally the same data used for training the model).
        y: Pandas Series of the target variable (not directly used by explainer but good for context).
        model: A trained model instance (e.g., XGBoost, LightGBM, CatBoost, RandomForest).
        model_type: String hint for explainer type ('tree', 'linear', 'deep', 'kernel').
                    'tree' is generally suitable for tree-based ensemble models.
        n_features_to_select: If 'auto', use the `threshold` method. 
                              If an integer, select the top N features.
        threshold: Mean absolute SHAP value threshold if n_features_to_select is 'auto'.
        sample_size: If provided, use a sample of X for SHAP value calculation.
        show_plots: Boolean, if True, display SHAP summary and dependence plots.
        plot_top_n_dependence: Number of top features for which to generate dependence plots.

    Returns:
        A list of selected feature names.
    """
    print(f"Starting SHAP feature selection with model type: {type(model).__name__}")

    X_shap = X
    if sample_size is not None and sample_size < len(X):
        print(f"Using a sample of {sample_size} from X for SHAP calculation.")
        X_shap = shap.sample(X, sample_size, random_state=42) # Ensure reproducibility of sample

    # Create a SHAP explainer
    # For tree models, shap.TreeExplainer is efficient. shap.Explainer often auto-detects.
    try:
        # explainer = shap.Explainer(model, X_shap) # This can auto-select
        # More specific for tree models to ensure correct handling:
        if model_type == 'tree' and hasattr(shap, 'TreeExplainer') and \
           (isinstance(model, (xgb.XGBModel, lgb.LGBMModel, cb.CatBoost, RandomForestClassifier))):
            explainer = shap.TreeExplainer(model, X_shap)
        else: # Fallback or for other types like 'kernel', 'deep', 'linear'
            print(f"Using generic shap.Explainer. Model type hint was '{model_type}'. This might be slower for non-tree models or require specific explainers.")
            explainer = shap.Explainer(model, X_shap)
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}. Trying KernelExplainer as a fallback.")
        # KernelExplainer is model-agnostic but slower.
        # It requires a summary of the background data (X_shap).
        # For KernelExplainer, X_shap should ideally be a small representative sample.
        # Summarize X_shap if it's large for KernelExplainer background
        if len(X_shap) > 100: # Heuristic for "large"
             X_shap_summary = shap.kmeans(X_shap, 50) # K-means summary with 50 means
        else:
             X_shap_summary = X_shap
        explainer = shap.KernelExplainer(model.predict_proba, X_shap_summary) # predict_proba for classifiers

    # Calculate SHAP values
    print("Calculating SHAP values...")
    try:
        shap_values_obj = explainer(X_shap) # For newer SHAP versions, returns Explanation object
        
        # Determine how to get the raw SHAP values array
        # For binary classification, shap_values_obj.values usually has shape (n_samples, n_features) for positive class,
        # or list of two arrays [(n_samples, n_features), (n_samples, n_features)] for class 0 and 1.
        # Or it might be (n_samples, n_features, n_classes) for some models.
        if isinstance(shap_values_obj.values, list) and len(shap_values_obj.values) == 2: # List for binary case
            shap_values_arr = shap_values_obj.values[1] # Values for the positive class
        elif isinstance(shap_values_obj.values, np.ndarray) and shap_values_obj.values.ndim == 3: # (samples, features, classes)
            shap_values_arr = shap_values_obj.values[:, :, 1] # Values for the positive class
        elif isinstance(shap_values_obj.values, np.ndarray) and shap_values_obj.values.ndim == 2: # (samples, features) - common for tree models
             shap_values_arr = shap_values_obj.values
        else:
            raise ValueError(f"Unexpected SHAP values structure: type {type(shap_values_obj.values)}, shape might be {shap_values_obj.values.shape if hasattr(shap_values_obj.values, 'shape') else 'N/A'}")

    except AttributeError: # Fallback for older SHAP versions where explainer.shap_values() is used
        print("Using explainer.shap_values() (older SHAP syntax).")
        raw_shap_values = explainer.shap_values(X_shap)
        if isinstance(raw_shap_values, list) and len(raw_shap_values) == 2:
            shap_values_arr = raw_shap_values[1] # Positive class for binary classification
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 2:
             shap_values_arr = raw_shap_values
        else:
            raise ValueError(f"Unexpected SHAP values structure from explainer.shap_values(): type {type(raw_shap_values)}")
        # For older versions, need to create an Explanation object manually for summary_plot if it expects one
        # This part is simplified; summary_plot can often take raw shap_values_arr and X_shap.
        # For dependence_plot, shap_values_arr is usually what's needed.

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
    shap_summary = pd.Series(mean_abs_shap, index=X_shap.columns) # Use X_shap columns
    shap_summary_sorted = shap_summary.sort_values(ascending=False)

    print("\nTop 10 features by mean absolute SHAP value:")
    print(shap_summary_sorted.head(10))

    # Feature Selection Logic
    if isinstance(n_features_to_select, int):
        if n_features_to_select <= 0 or n_features_to_select > len(shap_summary_sorted):
             print(f"Warning: Invalid n_features_to_select ({n_features_to_select}). Selecting all features.")
             selected_feature_names = shap_summary_sorted.index.tolist()
        else:
            selected_feature_names = shap_summary_sorted.head(n_features_to_select).index.tolist()
        print(f"\nSelected top {len(selected_feature_names)} features based on SHAP values.")
    elif n_features_to_select == 'auto':
        selected_feature_names = shap_summary_sorted[shap_summary_sorted > threshold].index.tolist()
        print(f"\nSelected {len(selected_feature_names)} features with mean abs SHAP value > {threshold}.")
        if not selected_feature_names: # If no features meet threshold, take top 1 as a fallback
            print(f"Warning: No features met SHAP threshold {threshold}. Selecting the top feature as a fallback.")
            selected_feature_names = [shap_summary_sorted.index[0]] if not shap_summary_sorted.empty else []
    else:
        raise ValueError("n_features_to_select must be an integer or 'auto'.")

    # Visualizations
    if show_plots:
        print("\nGenerating SHAP plots (display depends on environment)...")
        # Ensure using the same X_shap that was used for shap_values calculation
        # For summary_plot, if shap_values_obj exists and is Explanation obj:
        try:
            shap.summary_plot(shap_values_obj, X_shap, plot_type="bar", show=False)
            plt.title("SHAP Summary Plot (Bar)")
            plt.show() # Attempt to show plot

            shap.summary_plot(shap_values_obj, X_shap, show=False) # Dot plot
            plt.title("SHAP Summary Plot (Dot)")
            plt.show()
        except NameError: # If shap_values_obj is not defined (older SHAP path)
             shap.summary_plot(shap_values_arr, X_shap, plot_type="bar", show=False)
             plt.title("SHAP Summary Plot (Bar) - from array")
             plt.show()
             shap.summary_plot(shap_values_arr, X_shap, show=False)
             plt.title("SHAP Summary Plot (Dot) - from array")
             plt.show()


        # Dependence plots for top N selected features
        # Ensure that feature names are strings and exist in X_shap.columns
        valid_top_features_for_dependence = [f for f in selected_feature_names[:plot_top_n_dependence] if f in X_shap.columns]
        
        for feature_name in valid_top_features_for_dependence:
            print(f"Generating SHAP dependence plot for: {feature_name}")
            try:
                # shap.dependence_plot(feature_name, shap_values_arr, X_shap, show=False)
                # Using explainer object and feature name directly can be more robust with new SHAP
                shap.plots.dependence(feature_name, shap_values_obj.values if hasattr(shap_values_obj, 'values') else shap_values_arr, X_shap, show=False)

                plt.title(f"SHAP Dependence Plot: {feature_name}")
                plt.show()
            except Exception as e:
                print(f"Could not generate dependence plot for {feature_name}: {e}")
        
    return selected_feature_names


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=25, random_state=42)
    X_dummy_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(25)])
    y_dummy_series = pd.Series(y_dummy)

    print("--- RFE Example ---")
    rf_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    selected_rfe = select_features_rfe(X_dummy_df, y_dummy_series, rf_estimator, n_features_to_select=5)
    print(f"RFE selected: {selected_rfe}")

    print("\n--- SHAP Example ---")
    # Train a simple model for SHAP example
    model_for_shap = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_dummy_df, y_dummy_series)
    
    print("\nSHAP with n_features_to_select='auto' and threshold:")
    selected_shap_auto = select_features_shap(
        X_dummy_df, y_dummy_series, model_for_shap, 
        model_type='tree', 
        n_features_to_select='auto', threshold=0.05, 
        sample_size=50, # Use a smaller sample for faster example
        show_plots=False # Set to True to see plots if in suitable environment
    )
    print(f"SHAP 'auto' selected: {selected_shap_auto}")

    print("\nSHAP with n_features_to_select=integer:")
    selected_shap_int = select_features_shap(
        X_dummy_df, y_dummy_series, model_for_shap,
        model_type='tree',
        n_features_to_select=3, # Select top 3
        sample_size=50,
        show_plots=False
    )
    print(f"SHAP int selected: {selected_shap_int}")
