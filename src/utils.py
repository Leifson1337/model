# src/utils.py
import logging
import os
import joblib
import tensorflow as tf # For saving/loading Keras models
from src import config # Assuming config.py is in the same directory or src is in PYTHONPATH

# XGBoost and LightGBM can also be saved using their own methods or joblib
import xgboost as xgb
import lightgbm as lgb

def setup_logging():
    '''Configures logging for the application.'''
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    abs_logs_dir = os.path.join(project_root, config.LOGS_DIR)
    # Ensure the filename part of LOG_FILE is extracted if LOG_FILE itself is a full path
    log_filename = os.path.basename(config.LOG_FILE) 
    abs_log_file = os.path.join(abs_logs_dir, log_filename)

    os.makedirs(abs_logs_dir, exist_ok=True)
    
    # Avoid adding handlers multiple times if called repeatedly
    logger = logging.getLogger()
    if not logger.handlers: # Check if handlers are already configured
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(abs_log_file),
                logging.StreamHandler()
            ]
        )

def save_model(model, model_name: str, models_dir: str = None):
    """
    Saves a trained model to the specified directory.
    Handles scikit-learn-like models (joblib), XGBoost, LightGBM, and Keras/TensorFlow models.

    Args:
        model: The trained model object.
        model_name: Filename for the model (e.g., 'xgboost_v1.pkl' or 'lstm_model.h5').
        models_dir: Directory to save the model. Defaults to config.MODELS_DIR.
    """
    if models_dir is None:
        models_dir = config.MODELS_DIR
    
    # Ensure models_dir is an absolute path relative to project root for consistency
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src -> project_root
    abs_models_dir = os.path.join(project_root, models_dir)
    
    os.makedirs(abs_models_dir, exist_ok=True)
    model_path = os.path.join(abs_models_dir, model_name)
    
    try:
        if isinstance(model, tf.keras.Model): # TensorFlow/Keras model
            model.save(model_path) # .h5 or SavedModel format based on model_name extension
            logging.info(f"Keras model '{model_name}' saved to {model_path}")
        elif isinstance(model, xgb.XGBModel): # XGBoost model
            # XGBoost recommends its own save_model for full fidelity, or joblib for scikit-learn wrapper
            model.save_model(model_path) # Saves in XGBoost binary format if model_name ends with .json, .ubj, etc.
                                         # For .pkl, it might use joblib. Let's be explicit with joblib for pkl.
            # if model_name.endswith(".pkl"):
            #    joblib.dump(model, model_path)
            # else:
            #    model.save_model(model_path) # Native format
            logging.info(f"XGBoost model '{model_name}' saved to {model_path}")
        elif isinstance(model, lgb.LGBMModel): # LightGBM model
            # LightGBM also has native save_model and can be pickled
            model.booster_.save_model(model_path) # Saves in text format by default, or use .txt extension explicitly
            # if model_name.endswith(".pkl"):
            #    joblib.dump(model, model_path)
            # else:
            #    model.booster_.save_model(model_path) # Native format
            logging.info(f"LightGBM model '{model_name}' saved to {model_path}")
        else: # Default to joblib for other scikit-learn compatible models
            joblib.dump(model, model_path)
            logging.info(f"Model '{model_name}' saved using joblib to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model '{model_name}' to {model_path}: {e}")
        raise

def load_model(model_name: str, models_dir: str = None):
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
