# templates/new_feature_pipeline_template.py
import pandas as pd
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

# This is a conceptual base class. In a real implementation, this would be 
# defined in a central location like src/plugins/base.py or similar,
# or feature steps might be functions discovered by naming convention.
class BaseFeatureStepPlugin(ABC):
    """
    Abstract Base Class for individual, reusable feature engineering step plugins.
    
    Each feature step plugin should inherit from this class and implement its methods.
    The plugin is instantiated with its specific parameters, typically derived from
    a list of feature steps in `FeatureEngineeringConfig.feature_pipeline_steps`
    where each step has a 'name' (mapping to the plugin) and 'params'.
    """

    @abstractmethod
    def __init__(self, step_name: str, params: Dict[str, Any]):
        """
        Initialize the feature engineering step plugin.

        Args:
            step_name (str): The name of this feature step.
            params (Dict[str, Any]): A dictionary of parameters specific to this step.
                                     The plugin should validate and use these parameters.
                                     Example: {"window_size": 20, "fill_na_method": "ffill"}
        """
        self.step_name = step_name
        self.params = params
        self._validate_params()

    def _validate_params(self):
        """
        (Optional but Recommended) Validate the params passed during initialization.
        Raise ValueError or a custom exception for invalid parameters.
        """
        # Example:
        # window = self.params.get("window_size")
        # if window is not None and (not isinstance(window, int) or window <= 0):
        #     raise ValueError(f"Parameter 'window_size' must be a positive integer for {self.step_name}")
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame, shared_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply the feature transformation to the input DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame to transform.
            shared_context (Optional[Dict[str, Any]]): A dictionary that can be used to pass
                                                      information between feature steps if needed
                                                      (e.g., names of generated columns from a
                                                      previous step, or global settings like ticker).

        Returns:
            pd.DataFrame: The DataFrame with new/modified features.
                          It's crucial to handle column naming carefully to avoid unintended
                          overwrites or to clearly indicate derived features.
        """
        # Example placeholder logic:
        # print(f"Applying feature step '{self.step_name}' with parameters: {self.params}")
        # print(f"Input data shape: {data.shape}")
        #
        # output_df = data.copy()
        #
        # # Example: Calculate a simple moving average if 'close_column' and 'window_size' are params
        # close_col = self.params.get('close_column', 'Close')
        # window = self.params.get('window_size', 20)
        # if close_col in output_df.columns:
        #     output_df[f'{self.step_name}_SMA_{window}'] = output_df[close_col].rolling(window=window).mean()
        #     if self.params.get('fill_na_method'):
        #         output_df[f'{self.step_name}_SMA_{window}'].fillna(method=self.params['fill_na_method'], inplace=True)
        # else:
        #     print(f"Warning: Column '{close_col}' not found for step '{self.step_name}'. Skipping SMA calculation.")
        #
        # print(f"Output data shape after '{self.step_name}': {output_df.shape}")
        # return output_df
        raise NotImplementedError(f"Transform method not implemented by plugin '{self.step_name}'.")

    def get_output_feature_names(self, input_features: List[str]) -> List[str]:
        """
        (Optional but Recommended) Returns a list of feature names that this step will produce.
        This can help in managing feature namespaces and understanding pipeline outputs.
        
        Args:
            input_features (List[str]): List of column names in the input DataFrame.

        Returns:
            List[str]: List of new or modified feature names this step is expected to generate.
        """
        # Example:
        # if 'close_column' in self.params and 'window_size' in self.params:
        #     return [f"{self.step_name}_SMA_{self.params['window_size']}"]
        return []


# Example of how a concrete feature step plugin might look:
#
# class MovingAverageFeature(BaseFeatureStepPlugin):
#     def __init__(self, step_name: str, params: Dict[str, Any]):
#         super().__init__(step_name, params) # Calls _validate_params
# 
#     def _validate_params(self):
#         super()._validate_params() # Call base validation if any
#         if "window" not in self.params or not isinstance(self.params["window"], int) or self.params["window"] <= 0:
#             raise ValueError("Parameter 'window' (positive integer) is required for MovingAverageFeature.")
#         if "column" not in self.params or not isinstance(self.params["column"], str):
#             raise ValueError("Parameter 'column' (string) is required for MovingAverageFeature.")
# 
#     def transform(self, data: pd.DataFrame, shared_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
#         output_df = data.copy()
#         col_to_avg = self.params["column"]
#         window = self.params["window"]
#         
#         if col_to_avg not in output_df.columns:
#             print(f"Warning for '{self.step_name}': Column '{col_to_avg}' not found in input data. Skipping.")
#             return output_df
#             
#         new_col_name = f"{col_to_avg}_sma_{window}"
#         output_df[new_col_name] = output_df[col_to_avg].rolling(window=window).mean()
#         
#         if self.params.get("fill_na", False): # Example of another param
#             output_df[new_col_name] = output_df[new_col_name].fillna(method=self.params.get("fill_na_method", "ffill"))
#             
#         print(f"Applied '{self.step_name}', new column: {new_col_name}")
#         return output_df
#
#     def get_output_feature_names(self, input_features: List[str]) -> List[str]:
#         col_to_avg = self.params.get("column")
#         window = self.params.get("window")
#         if col_to_avg and window:
#             return [f"{col_to_avg}_sma_{window}"]
#         return []

# How this might be used in a refactored feature_engineering.py:
#
# def process_features_with_pipeline(data: pd.DataFrame, feature_pipeline_config: List[Dict[str,Any]], plugin_manager) -> pd.DataFrame:
#     processed_data = data.copy()
#     shared_context = {} # For inter-step communication if needed
#
#     for step_config in feature_pipeline_config:
#         step_name = step_config.get("name")
#         step_params = step_config.get("params", {})
#         
#         if not step_name:
#             print("Warning: Skipping feature step due to missing 'name'.")
#             continue
#
#         try:
#             # Assume plugin_manager.get_feature_step_plugin(name, params) instantiates the plugin
#             feature_step_plugin = plugin_manager.get_feature_step_plugin(step_name, step_params)
#             processed_data = feature_step_plugin.transform(processed_data, shared_context)
#             # Update shared_context if plugin provides outputs, e.g. names of new columns
#             # shared_context[f"{step_name}_outputs"] = feature_step_plugin.get_output_feature_names(...)
#         except ValueError as e_val: # Parameter validation error from plugin
#             print(f"Configuration error for feature step '{step_name}': {e_val}. Skipping step.")
#         except Exception as e:
#             print(f"Error executing feature step '{step_name}': {e}. Skipping step.")
#
#     return processed_data

```
