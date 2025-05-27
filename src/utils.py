# src/utils.py
import logging
import os
# from ..src.config_models import GlobalAppConfig # Example for type hinting if setup_logging uses it.
# from pydantic import validate_call # For validating inputs
import joblib
import tensorflow as tf # For saving/loading Keras models
from src import config # Assuming config.py is in the same directory or src is in PYTHONPATH

# XGBoost and LightGBM can also be saved using their own methods or joblib
import xgboost as xgb
import lightgbm as lgb

def setup_logging():
    '''Configures logging for the application.'''
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # TODO: These paths and level should ideally come from a validated Pydantic config (e.g., GlobalAppConfig.paths.logs_dir)
    #       rather than directly from src.config module, for consistency and validation.
    #       For now, we use src.config but acknowledge this could be improved.
    logs_dir_from_config = getattr(config, 'LOGS_DIR', 'logs/') # Default if not in src.config
    log_file_from_config = getattr(config, 'LOG_FILE', os.path.join(logs_dir_from_config, 'app.log'))
    log_level_from_config = getattr(config, 'LOG_LEVEL', 'INFO').upper()

    abs_logs_dir = os.path.join(project_root, logs_dir_from_config)
    # Ensure the filename part of LOG_FILE is extracted if log_file_from_config itself is a full path
    log_filename_only = os.path.basename(log_file_from_config) 
    abs_log_file = os.path.join(abs_logs_dir, log_filename_only)

    try:
        os.makedirs(abs_logs_dir, exist_ok=True)
    except OSError as e:
        # This might happen in restricted environments. Fallback or log to stderr.
        print(f"Warning: Could not create logs directory {abs_logs_dir}: {e}. Logging to stdout only.", file=sys.stderr)
        # Configure basic logging to stdout if directory creation fails
        logging.basicConfig(
            level=getattr(logging, log_level_from_config, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler()]
        )
        return

    # Avoid adding handlers multiple times if called repeatedly
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers() or len(root_logger.handlers) == 0 : # Check if handlers are already configured
        # TODO: Consider more advanced logging configurations (e.g., rotating file handlers, structured logging).
        logging.basicConfig(
            level=getattr(logging, log_level_from_config, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(abs_log_file), # TODO: Add error handling for FileHandler instantiation
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging configured. Log file: {abs_log_file}, Level: {log_level_from_config}")
    else:
        logging.info("Logging already configured.")


def save_model(model, model_name: str, models_dir: str = None):
    # TODO: Input validation for model_name (e.g., valid filename characters, expected extension).
    # TODO: `models_dir` should ideally come from a validated config (e.g., GlobalAppConfig.paths.model_registry_path).
    """
    Saves a trained model to the specified directory.
    Handles scikit-learn-like models (joblib), XGBoost, LightGBM, and Keras/TensorFlow models.
    Ensures model versioning and consistent serialization.

    Args:
        model: The trained model object. # TODO: Add type hints for common model types.
        model_name: Filename for the model (e.g., 'xgboost_v1.pkl' or 'lstm_model.h5').
                    # TODO: Consider standardizing model naming convention (e.g., include version, timestamp).
        models_dir: Directory to save the model. Defaults to config.MODELS_DIR.
                    # TODO: This should come from a validated Pydantic config.
    """
    # TODO: Cross-module compatibility: Emphasize consistent serialization (e.g., always use joblib for sklearn-like,
    #       native methods for TF/XGB/LGB if they provide better guarantees or cross-language support).
    #       Document versioning strategy for models (e.g., model_name includes version, or metadata file tracks it).

    if models_dir is None:
        models_dir = getattr(config, 'MODELS_DIR', 'models/') # Default if not in src.config
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    abs_models_dir = os.path.join(project_root, models_dir)
    
    try:
        os.makedirs(abs_models_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create models directory {abs_models_dir}: {e}. Cannot save model.")
        # TODO: Raise a custom exception or handle more gracefully depending on pipeline requirements.
        raise
        
    model_path = os.path.join(abs_models_dir, model_name)
    
    # TODO: Add fault tolerance: try-except for each save operation, log errors.
    # TODO: Save model metadata alongside the model using src.metadata_utils.generate_model_metadata().
    #       This metadata should include model type, parameters, training data reference, performance metrics, etc.
    #       Example:
    #       metadata_path = generate_model_metadata(
    #           model_filepath=model_path,
    #           metrics={"accuracy": 0.85, ...}, # From evaluation step
    #           feature_config={"features_used": ["sma_20", ...], ...}, # From feature engineering config
    #           training_config_obj=model_training_config_object # The Pydantic config obj for training
    #       )
    #       logging.info(f"Metadata for {model_name} saved to {metadata_path}")

    try:
        if isinstance(model, tf.keras.Model): 
            # Keras models: .h5 (legacy), .keras (new preferred), or SavedModel directory format.
            # Using .keras extension for new preferred format.
            # If model_name doesn't specify, choose one. For SavedModel, model_name should be a directory name.
            if not model_name.endswith((".h5", ".keras")):
                 # Assume SavedModel format if no Keras-specific extension; model_path should be a dir.
                 # Or default to .keras: model_path_to_save = model_path + ".keras"
                 pass # Current logic uses model_path as is.
            model.save(model_path) 
            logging.info(f"Keras model '{model_name}' saved to {model_path}")
        elif isinstance(model, xgb.XGBModel): 
            # Use joblib for scikit-learn wrapper of XGBoost for consistency with other sklearn-like models.
            # Native format (.ubj or .json) can be used if cross-language compatibility or specific features are needed.
            if model_name.endswith((".ubj", ".json")):
                model.save_model(model_path) # Native format
                logging.info(f"XGBoost model '{model_name}' saved in native format to {model_path}")
            else: # Default to joblib for .pkl or unspecified extensions for XGB wrapper
                joblib.dump(model, model_path)
                logging.info(f"XGBoost model (sklearn wrapper) '{model_name}' saved using joblib to {model_path}")
        elif isinstance(model, lgb.LGBMModel): 
            if model_name.endswith(".txt"): # Native LightGBM text format
                model.booster_.save_model(model_path) 
                logging.info(f"LightGBM model '{model_name}' saved in native text format to {model_path}")
            else: # Default to joblib for .pkl or unspecified for LGBM wrapper
                joblib.dump(model, model_path)
                logging.info(f"LightGBM model (sklearn wrapper) '{model_name}' saved using joblib to {model_path}")
        else: 
            joblib.dump(model, model_path)
            logging.info(f"Model '{model_name}' saved using joblib to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model '{model_name}' to {model_path}: {e}")
        # TODO: Consider specific error handling for different model types or file system issues.
        raise

def load_model(model_name: str, models_dir: str = None):
    # TODO: Similar input validation and config sourcing for models_dir as in save_model.
    """
    Loads a trained model from the specified directory.
    Handles scikit-learn-like models (joblib), XGBoost, LightGBM, and Keras/TensorFlow models.

    Args:
        model_name: Filename of the model (e.g., 'xgboost_v1.pkl' or 'lstm_model.h5').
        models_dir: Directory where the model is saved. Defaults to config.MODELS_DIR.

    Returns:
        The loaded model object.
    """
    if models_dir is None:
        models_dir = config.MODELS_DIR

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_models_dir = os.path.join(project_root, models_dir)
    model_path = os.path.join(abs_models_dir, model_name)

    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Determine model type by extension or known convention
        if model_name.endswith(".h5") or model_name.endswith(".keras") or isinstance(tf.keras.models.load_model(model_path, compile=False), tf.keras.Model) : # Basic check for Keras
             # The isinstance check is a bit problematic here, as we load first.
             # A better way for Keras is to rely on extension or a try-except block for loading.
            try:
                model = tf.keras.models.load_model(model_path) # `compile=True` by default, can set to False if issues
                logging.info(f"Keras model '{model_name}' loaded from {model_path}")
                return model
            except Exception as e: # Broad exception if it's not a Keras model or other Keras error
                 logging.debug(f"Failed to load {model_name} as Keras model ({e}), trying other methods.")


        # For XGBoost and LightGBM, if they were saved in native format, need specific loading
        # Assuming XGBoost models might be .json, .ubj, or if using joblib, .pkl
        # Assuming LightGBM models might be .txt, or if using joblib, .pkl
        
        # Attempt to load based on typical non-joblib extensions first
        if model_name.endswith((".json", ".ubj")): # XGBoost native
            model = xgb.XGBClassifier() # or Regressor, Booster etc. Need to know type or save type info.
            model.load_model(model_path)
            logging.info(f"XGBoost model '{model_name}' loaded from native format {model_path}")
            return model
        elif model_name.endswith(".txt"): # LightGBM native
            model = lgb.Booster(model_file=model_path) # Loads booster, need to wrap in LGBMClassifier if needed
            # To return an LGBMClassifier, one might need to save/load params or re-instantiate.
            # For now, returning the booster. Or, assume it's a scikit-learn wrapper saved with joblib.
            # A common pattern is to save the scikit-learn wrapper with joblib.
            logging.info(f"LightGBM Booster '{model_name}' loaded from native format {model_path}. Wrap in LGBMClassifier if needed.")
            return model # This is a Booster object, not LGBMClassifier.

        # Default to joblib for .pkl or if other methods failed/not applicable
        model = joblib.load(model_path)
        logging.info(f"Model '{model_name}' loaded using joblib from {model_path}")
        return model
    
    except Exception as e:
        logging.error(f"Error loading model '{model_name}' from {model_path}: {e}")
        raise

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    setup_logging() # Configure logging if running this file directly
    
    # Create dummy models for testing save/load
    # Note: config.MODELS_DIR needs to be valid from project root perspective
    # For this direct test, assume 'models/' exists at project root.
    
    # Scikit-learn like (e.g. LogisticRegression, or even XGB/LGB wrappers)
    from sklearn.linear_model import LogisticRegression
    dummy_sklearn_model = LogisticRegression()
    save_model(dummy_sklearn_model, "dummy_sklearn.pkl")
    loaded_sklearn_model = load_model("dummy_sklearn.pkl")
    if loaded_sklearn_model:
         logging.info(f"Successfully loaded scikit-learn model: {type(loaded_sklearn_model)}")

    # XGBoost (using its scikit-learn wrapper for joblib or native for .json/.ubj)
    # For this test, let's use the wrapper and save as pkl via joblib path in save_model
    # And also save in native format
    dummy_xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    # Example data for fitting, otherwise cannot save
    dummy_xgb_model.fit(np.array([[1],[2]]), np.array([0,1])) 
    
    save_model(dummy_xgb_model, "dummy_xgb.pkl") # Saved via joblib
    loaded_xgb_pkl = load_model("dummy_xgb.pkl")
    if loaded_xgb_pkl:
         logging.info(f"Successfully loaded XGBoost (pkl) model: {type(loaded_xgb_pkl)}")

    save_model(dummy_xgb_model, "dummy_xgb.ubj") # Saved via native save_model
    loaded_xgb_ubj = load_model("dummy_xgb.ubj")
    if loaded_xgb_ubj:
         logging.info(f"Successfully loaded XGBoost (ubj) model: {type(loaded_xgb_ubj)}")


    # LightGBM (using its scikit-learn wrapper for joblib or native for .txt)
    dummy_lgb_model = lgb.LGBMClassifier()
    dummy_lgb_model.fit(np.array([[1],[2]]), np.array([0,1]))

    save_model(dummy_lgb_model, "dummy_lgb.pkl") # Saved via joblib
    loaded_lgb_pkl = load_model("dummy_lgb.pkl")
    if loaded_lgb_pkl:
         logging.info(f"Successfully loaded LightGBM (pkl) model: {type(loaded_lgb_pkl)}")

    # To save LightGBM booster to .txt, save_model expects booster
    # save_model(dummy_lgb_model.booster_, "dummy_lgb_booster.txt") # Save booster
    # loaded_lgb_booster = load_model("dummy_lgb_booster.txt") # Loads booster
    # if loaded_lgb_booster:
    #      logging.info(f"Successfully loaded LightGBM (booster.txt) model: {type(loaded_lgb_booster)}")


    # TensorFlow/Keras
    dummy_keras_model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)])
    dummy_keras_model.compile(optimizer='adam', loss='mse')
    
    save_model(dummy_keras_model, "dummy_keras_model.h5")
    loaded_keras_model = load_model("dummy_keras_model.h5")
    if loaded_keras_model:
        logging.info(f"Successfully loaded Keras model: {type(loaded_keras_model)}")
        loaded_keras_model.summary()

    # Clean up dummy files (optional)
    # for f in ["dummy_sklearn.pkl", "dummy_xgb.pkl", "dummy_xgb.ubj", "dummy_lgb.pkl", "dummy_keras_model.h5"]:
    #     try:
    #         os.remove(os.path.join(config.MODELS_DIR, f))
    #     except OSError:
    #         pass
    # logging.info("Dummy models cleaned up.")
