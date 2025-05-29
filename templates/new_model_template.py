# templates/new_model_template.py
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# This is a conceptual base class. In a real implementation, this would be 
# defined in a central location like src/plugins/base.py or similar.
class BaseModelPlugin(ABC):
    """
    Abstract Base Class for model plugins.
    
    Each model plugin should inherit from this class and implement its methods.
    The plugin is instantiated with its specific parameters, typically derived from
    the `model_params` section of the `TrainModelConfig`.
    """

    @abstractmethod
    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        """
        Initialize the model plugin.

        Args:
            model_name (str): The name of the model, often matching the key in 
                              `TrainModelConfig.model_params` or `TrainModelConfig.model_type`.
            model_params (Dict[str, Any]): A dictionary of parameters specific to this model.
                                           The plugin should validate and use these parameters.
                                           Example: {"n_estimators": 100, "learning_rate": 0.05}
        """
        self.model_name = model_name
        self.model_params = model_params
        self._validate_params() # It's good practice for plugins to validate their own params
        self.model = None # This will hold the trained model object

    def _validate_params(self):
        """
        (Optional but Recommended) Validate the model_params passed during initialization.
        Raise ValueError or a custom exception for invalid parameters.
        """
        # Example:
        # required_param = "n_estimators"
        # if required_param not in self.model_params:
        #     raise ValueError(f"Missing required parameter '{required_param}' for {self.model_name}")
        # if not isinstance(self.model_params.get(required_param), int):
        #     raise ValueError(f"Parameter '{required_param}' must be an integer for {self.model_name}")
        pass

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Any:
        """
        Train the model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (Optional[pd.DataFrame]): Validation features (optional).
            y_val (Optional[pd.Series]): Validation target variable (optional).

        Returns:
            Any: The trained model artifact (e.g., an XGBoost model object, a Keras model).
                 This artifact will be passed to `save_model` and potentially to `predict`.
        """
        # Example placeholder logic:
        # print(f"Training {self.model_name} with parameters: {self.model_params}")
        # print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
        # if X_val is not None and y_val is not None:
        #     print(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
        #
        # # ... Actual model training logic using self.model_params ...
        # self.model = ... # Store the trained model instance
        # return self.model
        raise NotImplementedError("Train method not implemented by plugin.")

    @abstractmethod
    def predict(self, X: pd.DataFrame, model_artifact: Optional[Any] = None) -> pd.DataFrame: # Or np.ndarray
        """
        Make predictions using the trained model.

        Args:
            X (pd.DataFrame): Data to make predictions on.
            model_artifact (Optional[Any]): The trained model artifact. If None, the plugin
                                           should use its internally stored model (e.g., self.model)
                                           or load it if necessary. This allows flexibility for
                                           batch predictions where the model might be loaded once.

        Returns:
            pd.DataFrame: Predictions. Typically, for classification, this might be probabilities
                          (e.g., a DataFrame with columns for each class probability, or a single
                          column for the positive class probability). For regression, a single
                          column of predicted values.
        """
        # Example placeholder logic:
        # current_model = model_artifact if model_artifact is not None else self.model
        # if current_model is None:
        #     raise RuntimeError("Model has not been trained or loaded.")
        #
        # print(f"Predicting with {self.model_name} on data of shape: {X.shape}")
        # # ... Actual prediction logic ...
        # predictions = ... 
        # return pd.DataFrame(predictions, columns=['prediction']) # Adjust columns as needed
        raise NotImplementedError("Predict method not implemented by plugin.")

    @abstractmethod
    def save_model(self, model_artifact: Any, path: Path):
        """
        Save the trained model artifact to a specified path.

        The path provided will typically be a directory, and the plugin should save
        its model file(s) within this directory. The core system might expect a specific
        filename (e.g., "model.pkl", "model.h5") or the plugin can define its own.
        The `ModelMetadata.model_path` will point to the primary model file.

        Args:
            model_artifact (Any): The trained model artifact (returned by `train`).
            path (Path): Directory path where the model should be saved. The plugin
                         can decide on the filename(s) within this directory.
                         Example: path / "model.joblib"
        """
        # Example placeholder logic:
        # if not path.exists():
        #     path.mkdir(parents=True, exist_ok=True)
        # model_file_path = path / "my_model_filename.ext" # Choose appropriate extension
        # print(f"Saving {self.model_name} model to {model_file_path}")
        # # ... Actual model saving logic (e.g., joblib.dump, model.save()) ...
        raise NotImplementedError("Save_model method not implemented by plugin.")

    @abstractmethod
    def load_model(self, path: Path) -> Any:
        """
        Load a model artifact from a specified path.

        The path provided will typically be the path to the primary model file as
        recorded in `ModelMetadata.model_path`.

        Args:
            path (Path): Path to the model file.

        Returns:
            Any: The loaded model artifact. This artifact will be used for predictions.
        """
        # Example placeholder logic:
        # model_file_path = path # Assuming 'path' is the direct path to the model file
        # print(f"Loading {self.model_name} model from {model_file_path}")
        # # ... Actual model loading logic ...
        # loaded_model = ...
        # self.model = loaded_model # Optionally store in instance
        # return loaded_model
        raise NotImplementedError("Load_model method not implemented by plugin.")


# Example of how a concrete plugin might look (for illustration):
#
# import joblib
# from sklearn.ensemble import RandomForestClassifier
#
# class RandomForestPlugin(BaseModelPlugin):
#     def __init__(self, model_name: str, model_params: Dict[str, Any]):
#         super().__init__(model_name, model_params)
#         # _validate_params will be called by super()
#
#     def _validate_params(self):
#         # Example: RandomForest specific validation
#         allowed_params = {"n_estimators", "max_depth", "random_state"}
#         for p_name in self.model_params:
#             if p_name not in allowed_params:
#                 print(f"Warning: Unknown param '{p_name}' for RandomForestPlugin. It will be ignored.")
#         if "n_estimators" not in self.model_params: self.model_params["n_estimators"] = 100 # Default
#
#     def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Any:
#         self.model = RandomForestClassifier(**self.model_params)
#         self.model.fit(X_train, y_train)
#         print(f"RandomForestPlugin: Trained model with OOB score: {self.model.oob_score_ if hasattr(self.model, 'oob_score_') else 'N/A'}")
#         return self.model
#
#     def predict(self, X: pd.DataFrame, model_artifact: Optional[Any] = None) -> pd.DataFrame:
#         current_model = model_artifact if model_artifact is not None else self.model
#         if current_model is None:
#             raise RuntimeError("RandomForestPlugin: Model not trained or loaded.")
#         # Return probabilities for binary classification (common case)
#         predictions_proba = current_model.predict_proba(X)
#         return pd.DataFrame(predictions_proba, columns=[f"prob_class_{i}" for i in range(predictions_proba.shape[1])], index=X.index)
#
#     def save_model(self, model_artifact: Any, path: Path):
#         # The 'path' is expected to be the directory. Model name is chosen by plugin.
#         model_file = path / "random_forest_model.joblib"
#         joblib.dump(model_artifact, model_file)
#         print(f"RandomForestPlugin: Saved model to {model_file}")
#
#     def load_model(self, path: Path) -> Any:
#         # 'path' is the direct path to the model file here (e.g., from ModelMetadata.model_path)
#         self.model = joblib.load(path)
#         print(f"RandomForestPlugin: Loaded model from {path}")
#         return self.model

```
