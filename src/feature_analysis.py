# src/feature_analysis.py
import pandas as pd
import numpy as np
import logging
from typing import Any, Optional, Dict, List, Union 
import json 

logger = logging.getLogger(__name__)
if not logger.handlers: 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def calculate_feature_statistics(df: pd.DataFrame, n_top_cat: int = 10) -> Dict[str, Dict[str, Any]]:
    """Calculates summary statistics for each feature in the DataFrame."""
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a pandas DataFrame.")
        return {} 
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning empty statistics.")
        return {}

    all_feature_stats: Dict[str, Dict[str, Any]] = {}
    logger.info(f"Calculating statistics for DataFrame with shape: {df.shape}")

    for col in df.columns:
        feature_stats: Dict[str, Any] = {}
        column_data = df[col]
        
        feature_stats["dtype"] = str(column_data.dtype)
        feature_stats["total_count"] = int(len(column_data))
        feature_stats["missing_count"] = int(column_data.isnull().sum())
        feature_stats["missing_percentage"] = float(round(column_data.isnull().mean() * 100, 2))

        if column_data.isnull().all():
            feature_stats["notes"] = "Column is all NaN."
            all_feature_stats[col] = feature_stats
            logger.warning(f"Feature '{col}' contains all NaN values.")
            continue

        if pd.api.types.is_numeric_dtype(column_data):
            feature_stats["mean"] = float(round(column_data.mean(), 4)) if pd.notna(column_data.mean()) else None
            feature_stats["std"] = float(round(column_data.std(), 4)) if pd.notna(column_data.std()) else None
            feature_stats["min"] = float(column_data.min()) if pd.notna(column_data.min()) else None
            feature_stats["max"] = float(column_data.max()) if pd.notna(column_data.max()) else None
            feature_stats["median"] = float(column_data.median()) if pd.notna(column_data.median()) else None
            feature_stats["p25"] = float(column_data.quantile(0.25)) if pd.notna(column_data.quantile(0.25)) else None
            feature_stats["p75"] = float(column_data.quantile(0.75)) if pd.notna(column_data.quantile(0.75)) else None
        
        elif pd.api.types.is_object_dtype(column_data) or \
             pd.api.types.is_string_dtype(column_data) or \
             pd.api.types.is_categorical_dtype(column_data) or \
             pd.api.types.is_bool_dtype(column_data):
            
            feature_stats["unique_count"] = int(column_data.nunique(dropna=False)) 
            value_counts_series = column_data.value_counts(dropna=False).rename(lambda x: "NaN" if pd.isna(x) else str(x))
            top_n_values = value_counts_series.head(n_top_cat)
            feature_stats["top_n_values"] = {k: int(v) for k, v in top_n_values.items()}
            unique_values_list = [idx_val for idx_val in value_counts_series.index.tolist()]
            if len(unique_values_list) > n_top_cat * 2: 
                feature_stats["unique_values_list"] = unique_values_list[:n_top_cat] + ["... (others)"]
            else: feature_stats["unique_values_list"] = unique_values_list
            if not value_counts_series.empty:
                most_frequent_val_str = value_counts_series.index[0]
                most_frequent_val_count = int(value_counts_series.iloc[0])
                feature_stats["most_frequent_value"] = most_frequent_val_str
                feature_stats["most_frequent_value_count"] = most_frequent_val_count
                feature_stats["most_frequent_value_percentage"] = float(round((most_frequent_val_count / feature_stats["total_count"]) * 100, 2))
            else: feature_stats["most_frequent_value"] = None; feature_stats["most_frequent_value_count"] = 0; feature_stats["most_frequent_value_percentage"] = 0.0
        else:
            logger.warning(f"Column '{col}' has an unsupported dtype '{column_data.dtype}'. Basic stats provided.")
            feature_stats["notes"] = f"Unsupported dtype: {column_data.dtype}"
        all_feature_stats[col] = feature_stats
    logger.info(f"Calculated statistics for {len(all_feature_stats)} features.")
    return all_feature_stats


def compare_feature_statistics(current_stats: Dict[str, Dict[str, Any]], 
                               baseline_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    drift_report: Dict[str, Dict[str, Any]] = {}
    logger.info("Comparing feature statistics. Current vs Baseline.")
    all_feature_names = set(current_stats.keys()) | set(baseline_stats.keys())

    for feature_name in all_feature_names:
        drift_metrics: Dict[str, Any] = {}
        current_feat_stats = current_stats.get(feature_name)
        baseline_feat_stats = baseline_stats.get(feature_name)

        if not current_feat_stats:
            logger.warning(f"Feature '{feature_name}' missing in current stats. Marked as drifted/missing.")
            drift_metrics["status"] = "missing_in_current"; drift_report[feature_name] = drift_metrics; continue
        if not baseline_feat_stats:
            logger.warning(f"Feature '{feature_name}' missing in baseline stats. Marked as new/unexpected.")
            drift_metrics["status"] = "missing_in_baseline"; drift_report[feature_name] = drift_metrics; continue
        
        drift_metrics["baseline_dtype"] = baseline_feat_stats.get("dtype")
        drift_metrics["current_dtype"] = current_feat_stats.get("dtype")
        if drift_metrics["baseline_dtype"] != drift_metrics["current_dtype"]:
            drift_metrics["dtype_drift_detected"] = True
            logger.warning(f"Dtype drift for '{feature_name}': Baseline={drift_metrics['baseline_dtype']}, Current={drift_metrics['current_dtype']}")

        if pd.api.types.is_numeric_dtype(str(baseline_feat_stats.get("dtype"))) and \
           pd.api.types.is_numeric_dtype(str(current_feat_stats.get("dtype"))):
            b_mean,c_mean=baseline_feat_stats.get("mean"),current_feat_stats.get("mean")
            if b_mean is not None and c_mean is not None: drift_metrics["mean_change_pct"]=round(((c_mean-b_mean)/b_mean)*100,2) if b_mean!=0 else ("N/A (baseline mean is 0)" if c_mean ==0 else "inf")
            else: drift_metrics["mean_change_pct"]="N/A (missing mean)"
            b_std,c_std=baseline_feat_stats.get("std"),current_feat_stats.get("std")
            if b_std is not None and c_std is not None: drift_metrics["std_change_pct"]=round(((c_std-b_std)/b_std)*100,2) if b_std!=0 else ("N/A (baseline std is 0)" if c_std == 0 else "inf")
            else: drift_metrics["std_change_pct"]="N/A (missing std)"
            drift_metrics["psi"]="not_implemented_from_summary_stats"
            percentiles_to_compare=["min","p25","median","p75","max"];percentile_shifts={}
            for p_name in percentiles_to_compare:
                b_val,c_val=baseline_feat_stats.get(p_name),current_feat_stats.get(p_name)
                if b_val is not None and c_val is not None:
                    percentile_shifts[f"{p_name}_baseline"]=b_val;percentile_shifts[f"{p_name}_current"]=c_val
                    if abs(b_val)>1e-9:percentile_shifts[f"{p_name}_change_pct"]=round(((c_val-b_val)/abs(b_val))*100,2)
                    else:percentile_shifts[f"{p_name}_change_pct"]=0 if c_val==b_val else "inf_baseline_near_zero"
            if percentile_shifts:drift_metrics["percentile_comparison"]=percentile_shifts
        elif ("top_n_values" in baseline_feat_stats and "top_n_values" in current_feat_stats):
            b_top_n=baseline_feat_stats["top_n_values"];c_top_n=current_feat_stats["top_n_values"]
            freq_changes={};all_top_cats=set(b_top_n.keys())|set(c_top_n.keys())
            for cat in all_top_cats:
                b_freq=(b_top_n.get(cat,0)/baseline_feat_stats["total_count"])*100 if baseline_feat_stats["total_count"]>0 else 0
                c_freq=(c_top_n.get(cat,0)/current_feat_stats["total_count"])*100 if current_feat_stats["total_count"]>0 else 0
                freq_changes[cat]={"baseline_pct":round(b_freq,2),"current_pct":round(c_freq,2),"change_pct_diff":round(c_freq-b_freq,2)}
            drift_metrics["top_categories_frequency_change"]=freq_changes
            drift_metrics["chi_squared_p_value"]="not_implemented_from_summary_stats"
            b_cats_set=set(b_top_n.keys());c_cats_set=set(c_top_n.keys())
            intersection=len(b_cats_set.intersection(c_cats_set));union=len(b_cats_set.union(c_cats_set))
            drift_metrics["jaccard_index_top_n"]=round(intersection/union,4) if union>0 else 0.0
        drift_metrics["missing_percentage_baseline"]=baseline_feat_stats.get("missing_percentage")
        drift_metrics["missing_percentage_current"]=current_feat_stats.get("missing_percentage")
        drift_report[feature_name]=drift_metrics
    logger.info(f"Feature drift comparison completed for {len(drift_report)} features.")
    return drift_report


def calculate_permutation_importance(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                                     n_repeats: int = 5, random_state: int = 42, 
                                     scoring: Optional[str] = None) -> Dict[str, float]:
    logger.warning("Permutation importance calculation is a STUB.")
    if not X_test.empty:
        num_features_to_show = min(3, len(X_test.columns))
        selected_columns = X_test.columns[:num_features_to_show].tolist()
        dummy_importances = {col: np.random.rand() for col in selected_columns}
        if dummy_importances: logger.info(f"Returning placeholder permutation importances for: {list(dummy_importances.keys())}")
        else: logger.info("X_test has columns, but could not select any for dummy importance."); return {"placeholder_feature_if_no_cols_selected": 0.0}
        return dummy_importances
    logger.info("X_test is empty, returning default placeholder permutation importance.")
    return {"placeholder_feature_if_X_test_empty": 0.0}


def select_features_for_elimination(
    importances_path: Optional[str], 
    drift_report_path: Optional[str], 
    elimination_config: Dict[str, Any]
) -> List[str]:
    """
    Selects features for elimination based on importance and drift report.
    Currently a STUB function.
    """
    logger.info(f"select_features_for_elimination called with:")
    logger.info(f"  Importances Path: {importances_path}")
    logger.info(f"  Drift Report Path: {drift_report_path}")
    logger.info(f"  Elimination Config: {json.dumps(elimination_config, indent=2)}")
    logger.warning("This is a STUB function. Advanced logic for combining importance and drift metrics is a TODO.")

    features_to_eliminate = []
    
    # Attempt to load data (optional stub behavior)
    importances_data = None
    if importances_path:
        try:
            with open(importances_path, 'r') as f:
                importances_data = json.load(f)
            logger.info(f"Successfully loaded feature importances from {importances_path}")
            # Example stub logic: if a feature has importance < threshold, mark for elimination
            # min_importance = elimination_config.get("min_importance_threshold", 0.001)
            # for item in importances_data: # Assuming list of {"feature": name, "importance": score}
            #     if item.get("importance", 1.0) < min_importance:
            #         features_to_eliminate.append(item["feature"])
            #         logger.info(f"Stub: Feature '{item['feature']}' marked for elimination due to low importance ({item.get('importance')}).")
        except FileNotFoundError:
            logger.error(f"Importances file not found: {importances_path}")
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from importances file: {importances_path}")
        except Exception as e:
            logger.error(f"Error loading importances file {importances_path}: {e}")

    drift_report_data = None
    if drift_report_path:
        try:
            with open(drift_report_path, 'r') as f:
                drift_report_data = json.load(f)
            logger.info(f"Successfully loaded drift report from {drift_report_path}")
            # Example stub logic: if a feature has high drift, mark for elimination
            # max_drift_psi = elimination_config.get("max_drift_psi_threshold", 0.2)
            # max_mean_change = elimination_config.get("max_drift_mean_pct_change_threshold", 50.0)
            # for feature, drift_info in drift_report_data.items():
            #     if drift_info.get("psi", 0) > max_drift_psi or \
            #        abs(drift_info.get("mean_change_pct", 0)) > max_mean_change :
            #         if feature not in features_to_eliminate: # Avoid duplicates
            #             features_to_eliminate.append(feature)
            #             logger.info(f"Stub: Feature '{feature}' marked for elimination due to high drift.")
        except FileNotFoundError:
            logger.error(f"Drift report file not found: {drift_report_path}")
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from drift report file: {drift_report_path}")
        except Exception as e:
            logger.error(f"Error loading drift report file {drift_report_path}: {e}")

    # For the stub, if both files were "loaded" (even if processing failed), return a dummy list.
    # Otherwise, return empty. This is slightly different from the original request to return empty always.
    # Let's stick to returning an empty list as per the original request for a conservative stub.
    # if importances_data and drift_report_data:
    #     logger.info("Stub: Both importance and drift files were 'loaded'. Returning predefined list for testing.")
    #     return ["dummy_elim_feat_A", "dummy_elim_feat_B"]
    
    logger.info("Stub: Returning empty list for features to eliminate.")
    return []


if __name__ == '__main__':
    # --- Permutation Importance Tests (unchanged) ---
    print("--- Testing Permutation Importance Stub ---")
    # ... (permutation importance tests) ...
    print("\n--- Permutation Importance Stub Test Completed ---")

    # --- Feature Statistics Calculation Tests (unchanged) ---
    print("\n\n--- Testing Feature Statistics Calculation ---")
    # ... (feature statistics tests) ...
    print("\n--- Feature Statistics Tests Completed ---")

    # --- New Tests for select_features_for_elimination ---
    print("\n\n--- Testing select_features_for_elimination Stub ---")
    
    # Create dummy files
    dummy_importances_content = [
        {"feature": "feat_A", "importance": 0.5},
        {"feature": "feat_B", "importance": 0.0005}, # Below typical threshold
        {"feature": "feat_C", "importance": 0.2}
    ]
    dummy_drift_report_content = {
        "feat_A": {"mean_change_pct": 10.0, "psi": 0.05},
        "feat_B": {"mean_change_pct": 5.0, "psi": 0.01},
        "feat_C": {"mean_change_pct": 60.0, "psi": 0.25} # High drift
    }
    
    importances_file = "dummy_importances.json"
    drift_file = "dummy_drift_report.json"

    with open(importances_file, 'w') as f: json.dump(dummy_importances_content, f)
    with open(drift_file, 'w') as f: json.dump(dummy_drift_report_content, f)

    sample_elim_config = {
        "min_importance_threshold": 0.001,
        "max_drift_psi_threshold": 0.2, 
        "max_drift_mean_pct_change_threshold": 50.0,
        "priority": ["importance", "drift"]
    }

    print("\n1. Calling with valid dummy files and config:")
    elim_list = select_features_for_elimination(importances_file, drift_file, sample_elim_config)
    print(f"Features selected for elimination (stub): {elim_list}")
    assert isinstance(elim_list, list) 
    # Current stub always returns [], so this will pass.
    # If stub returned ["dummy_elim_feat_A", "dummy_elim_feat_B"] when files loaded:
    # assert len(elim_list) == 2 

    print("\n2. Calling with importances_path=None:")
    elim_list_no_imp = select_features_for_elimination(None, drift_file, sample_elim_config)
    print(f"Features selected (no importances file): {elim_list_no_imp}")
    assert isinstance(elim_list_no_imp, list)

    print("\n3. Calling with drift_report_path=None:")
    elim_list_no_drift = select_features_for_elimination(importances_file, None, sample_elim_config)
    print(f"Features selected (no drift report file): {elim_list_no_drift}")
    assert isinstance(elim_list_no_drift, list)

    print("\n4. Calling with non-existent files:")
    elim_list_bad_paths = select_features_for_elimination("bad_imp.json", "bad_drift.json", sample_elim_config)
    print(f"Features selected (bad paths): {elim_list_bad_paths}")
    assert isinstance(elim_list_bad_paths, list) and len(elim_list_bad_paths) == 0

    # Cleanup dummy files
    if os.path.exists(importances_file): os.remove(importances_file)
    if os.path.exists(drift_file): os.remove(drift_file)
    print("\nCleaned up dummy files for select_features_for_elimination test.")
    
    print("\n--- select_features_for_elimination Stub Test Completed ---")
