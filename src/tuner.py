# src/tuner.py
import optuna
import numpy as np
import pandas as pd
import os # For path manipulation in run_tuning storage_path

# Model specific imports
import xgboost as xgb
import lightgbm as lgb
import catboost as cb 
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
    Input, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention # Added for Transformer
)
from tensorflow.keras.callbacks import EarlyStopping

# Scikit-learn utilities
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss 

from src import config 
from src import utils 
# Import custom Keras layers from modeling.py
from src.modeling import PositionalEncoding, TransformerEncoderBlock


def objective_xgboost(trial, X: pd.DataFrame, y: pd.Series, cv_splitter: TimeSeriesSplit, eval_metric: str = 'accuracy') -> float:
    """Objective function for XGBoost hyperparameter optimization."""
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params)
    scores = []
    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold, early_stopping_rounds=10, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        if eval_metric == 'accuracy': score = accuracy_score(y_val_fold, model.predict(X_val_fold))
        elif eval_metric == 'logloss': score = log_loss(y_val_fold, model.predict_proba(X_val_fold))
        else: raise ValueError(f"Unsupported eval_metric: {eval_metric}")
        scores.append(score)
    return np.mean(scores)

def objective_lightgbm(trial, X: pd.DataFrame, y: pd.Series, cv_splitter: TimeSeriesSplit, eval_metric: str = 'accuracy') -> float:
    """Objective function for LightGBM hyperparameter optimization."""
    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    scores = []
    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric='logloss', callbacks=[lgb.early_stopping(10, verbose=False)])
        if eval_metric == 'accuracy': score = accuracy_score(y_val_fold, model.predict(X_val_fold))
        elif eval_metric == 'logloss': score = log_loss(y_val_fold, model.predict_proba(X_val_fold))
        else: raise ValueError(f"Unsupported eval_metric: {eval_metric}")
        scores.append(score)
    return np.mean(scores)

def objective_catboost(trial, X: pd.DataFrame, y: pd.Series, cv_splitter: TimeSeriesSplit, eval_metric: str = 'accuracy') -> float:
    """Objective function for CatBoost hyperparameter optimization."""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_state': 42, 'verbose': 0, 'loss_function': 'Logloss',
    }
    model = cb.CatBoostClassifier(**params)
    scores = []
    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=10, verbose=0) 
        if eval_metric == 'accuracy': score = accuracy_score(y_val_fold, model.predict(X_val_fold))
        elif eval_metric == 'logloss': score = log_loss(y_val_fold, model.predict_proba(X_val_fold))
        else: raise ValueError(f"Unsupported eval_metric: {eval_metric}")
        scores.append(score)
    return np.mean(scores)

def objective_lstm(
    trial, X_train_seq_opt: np.ndarray, y_train_seq_opt: np.ndarray, 
    X_val_seq_opt: np.ndarray, y_val_seq_opt: np.ndarray, input_shape: tuple, 
    early_stopping_patience: int = 5, eval_metric: str = 'accuracy'
) -> float:
    """Objective function for LSTM hyperparameter optimization."""
    lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128, step=16)
    dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.5, step=0.1)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model = Sequential()
    model.add(LSTM(units=lstm_units_1, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(rate=dropout_1))
    model.add(Dense(1, activation='sigmoid'))

    if optimizer_name == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else: optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=0)
    model.fit(X_train_seq_opt, y_train_seq_opt, validation_data=(X_val_seq_opt, y_val_seq_opt), 
              epochs=50, batch_size=32, callbacks=[early_stopping_cb], verbose=0)
    
    val_loss, val_accuracy = model.evaluate(X_val_seq_opt, y_val_seq_opt, verbose=0)
    return val_accuracy if eval_metric == 'accuracy' else val_loss

def objective_cnnlstm(
    trial, X_train_seq_opt: np.ndarray, y_train_seq_opt: np.ndarray,
    X_val_seq_opt: np.ndarray, y_val_seq_opt: np.ndarray, input_shape: tuple, 
    early_stopping_patience: int = 5, eval_metric: str = 'accuracy'
) -> float:
    """Objective function for CNN-LSTM hyperparameter optimization."""
    filters = trial.suggest_int('filters', 32, 128, step=32)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    cnn_dropout = trial.suggest_float('cnn_dropout', 0.1, 0.5, step=0.1)
    lstm_units = trial.suggest_int('lstm_units', 32, 128, step=32)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5, step=0.1)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=cnn_dropout))
    model.add(LSTM(units=lstm_units, return_sequences=False))
    model.add(Dropout(rate=lstm_dropout))
    model.add(Dense(1, activation='sigmoid'))

    if optimizer_name == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else: optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=0)
    model.fit(X_train_seq_opt, y_train_seq_opt, validation_data=(X_val_seq_opt, y_val_seq_opt),
              epochs=50, batch_size=32, callbacks=[early_stopping_cb], verbose=0)

    val_loss, val_accuracy = model.evaluate(X_val_seq_opt, y_val_seq_opt, verbose=0)
    return val_accuracy if eval_metric == 'accuracy' else val_loss

def objective_transformer(
    trial, X_train_seq_opt: np.ndarray, y_train_seq_opt: np.ndarray,
    X_val_seq_opt: np.ndarray, y_val_seq_opt: np.ndarray, input_shape: tuple,
    early_stopping_patience: int = 5, eval_metric: str = 'accuracy'
) -> float:
    """Objective function for Transformer hyperparameter optimization."""
    
    # embed_dim must match the number of features if not using a separate projection layer
    # and if PositionalEncoding is used directly on input.
    # For simplicity, assume input_shape[1] (num_features) is the embed_dim.
    embed_dim = input_shape[1] # This is num_features per timestep
    
    # Hyperparameters for Transformer
    # head_size = trial.suggest_int('head_size', 32, 128, step=32) # key_dim for MHA
    # For MultiHeadAttention, num_heads * key_dim (head_size) should ideally be embed_dim if projecting.
    # If key_dim is not embed_dim / num_heads, MHA will project. Let's make key_dim = embed_dim / num_heads
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8]) # Must be a divisor of embed_dim for simple setup
    if embed_dim % num_heads != 0: # Adjust embed_dim or num_heads if not divisible for simplicity
        # This is a simple heuristic. A projection layer would be more robust.
        # For now, just print a warning or adjust. Let's try to adjust num_heads.
        possible_num_heads = [h for h in [1,2,4,8,16] if embed_dim % h == 0]
        if not possible_num_heads: possible_num_heads = [1] # Fallback
        num_heads = trial.suggest_categorical(f'num_heads_adjusted_for_embed_dim_{embed_dim}', possible_num_heads)
        print(f"Adjusted num_heads to {num_heads} for embed_dim {embed_dim}")


    ff_dim = trial.suggest_int('ff_dim', 64, 256, step=64)
    num_transformer_blocks = trial.suggest_int('num_transformer_blocks', 1, 4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1) # For MHA and FFN dropout
    mlp_dropout_rate = trial.suggest_float('mlp_dropout_rate', 0.1, 0.4, step=0.1) # For final MLP head
    mlp_units = trial.suggest_int('mlp_units', 32, 128, step=32)
    
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    inputs = Input(shape=input_shape)
    x = inputs
    
    # Optional: Add a Dense layer here to project num_features to a new embed_dim if they differ
    # For now, embed_dim is taken from input_shape[1] (num_features)
    
    # Positional Encoding
    # sequence_length = input_shape[0]
    # x = PositionalEncoding(position=sequence_length, d_model=embed_dim)(x) # d_model must match x's last dim

    for _ in range(num_transformer_blocks):
        # TransformerEncoderBlock's embed_dim should match the dimension of x
        x = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(mlp_dropout_rate)(x) # Dropout before final MLP layers
    x = Dense(mlp_units, activation="relu")(x)
    x = Dropout(mlp_dropout_rate)(x) # Additional dropout
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    if optimizer_name == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else: optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=0)
    model.fit(X_train_seq_opt, y_train_seq_opt, validation_data=(X_val_seq_opt, y_val_seq_opt),
              epochs=50, batch_size=32, callbacks=[early_stopping_cb], verbose=0)

    val_loss, val_accuracy = model.evaluate(X_val_seq_opt, y_val_seq_opt, verbose=0)
    return val_accuracy if eval_metric == 'accuracy' else val_loss


def run_tuning(
    model_type: str, 
    X: pd.DataFrame = None, y: pd.Series = None, cv_splitter: TimeSeriesSplit = None,
    X_train_seq_opt: np.ndarray = None, y_train_seq_opt: np.ndarray = None, 
    X_val_seq_opt: np.ndarray = None, y_val_seq_opt: np.ndarray = None, 
    input_shape_seq: tuple = None, 
    n_trials: int = 50, 
    study_name_prefix: str = "study", 
    storage_path: str = None, 
    direction: str = "maximize", 
    eval_metric_for_tuning: str = 'accuracy',
    early_stopping_patience_seq: int = 5 
) -> tuple[dict, float]:
    """Runs Optuna hyperparameter tuning for a specified model type."""
    
    objective_map = {
        'xgboost': objective_xgboost,
        'lightgbm': objective_lightgbm,
        'catboost': objective_catboost,
        'lstm': objective_lstm,
        'cnnlstm': objective_cnnlstm,
        'transformer': objective_transformer,
    }

    if model_type not in objective_map:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported types are: {list(objective_map.keys())}")

    objective_func = objective_map[model_type]
    
    if model_type in ['xgboost', 'lightgbm', 'catboost']:
        if X is None or y is None or cv_splitter is None:
            raise ValueError(f"X, y, and cv_splitter must be provided for {model_type} tuning.")
        partial_objective = lambda trial: objective_func(trial, X, y, cv_splitter, eval_metric_for_tuning)
    elif model_type in ['lstm', 'cnnlstm', 'transformer']: 
        if X_train_seq_opt is None or y_train_seq_opt is None or \
           X_val_seq_opt is None or y_val_seq_opt is None or \
           input_shape_seq is None:
            raise ValueError("Sequence data (X_train_seq_opt, etc.) and input_shape_seq must be provided for sequence models.")
        partial_objective = lambda trial: objective_func(
            trial, X_train_seq_opt, y_train_seq_opt, X_val_seq_opt, y_val_seq_opt, 
            input_shape_seq, early_stopping_patience_seq, eval_metric_for_tuning
        )
    else:
        raise NotImplementedError(f"Argument handling for model type {model_type} not implemented in run_tuning.")

    study_name = f"{study_name_prefix}_{model_type}_{eval_metric_for_tuning}_{direction}"
    
    if storage_path and storage_path.startswith("sqlite:///"):
        db_dir = os.path.dirname(storage_path.replace("sqlite:///", ""))
        if db_dir and not os.path.exists(db_dir): os.makedirs(db_dir, exist_ok=True)

    print(f"\n--- Starting Optuna Tuning for {model_type} ---")
    print(f"Study Name: {study_name}, Direction: {direction}, Metric: {eval_metric_for_tuning}, Trials: {n_trials}")

    study = optuna.create_study(study_name=study_name, direction=direction, storage=storage_path, load_if_exists=True)
    study.optimize(partial_objective, n_trials=n_trials, show_progress_bar=True) # show_progress_bar requires tqdm
    
    print(f"\n--- Optuna Tuning Completed for {model_type} ---")
    print(f"Best trial for {study_name}: Value ({eval_metric_for_tuning}): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
        
    return study.best_trial.params, study.best_trial.value

if __name__ == '__main__':
    print("--- Optuna Tuner Module Demonstration ---")
    # ... (Tree model examples from previous turn) ...
    
    # Note: LSTM/CNN-LSTM/Transformer tuning example would require more setup for sequence data here.
    # Example (conceptual - would need to generate X_train_seq_opt etc. first):
    # if False: # Set to True to run a conceptual example
    #     dummy_seq_len = 10
    #     dummy_n_features = 5
    #     dummy_samples_train = 100
    #     dummy_samples_val = 20
    #     dummy_X_train_seq = np.random.rand(dummy_samples_train, dummy_seq_len, dummy_n_features)
    #     dummy_y_train_seq = np.random.randint(0,2,dummy_samples_train)
    #     dummy_X_val_seq = np.random.rand(dummy_samples_val, dummy_seq_len, dummy_n_features)
    #     dummy_y_val_seq = np.random.randint(0,2,dummy_samples_val)
    #     dummy_input_shape = (dummy_seq_len, dummy_n_features)
    #     try:
    #         best_params_tf, best_value_tf = run_tuning(
    #             model_type='transformer', # or 'lstm', 'cnnlstm'
    #             X_train_seq_opt=dummy_X_train_seq, y_train_seq_opt=dummy_y_train_seq,
    #             X_val_seq_opt=dummy_X_val_seq, y_val_seq_opt=dummy_y_val_seq,
    #             input_shape_seq=dummy_input_shape,
    #             n_trials=2, # Very few trials for quick demo
    #             eval_metric_for_tuning='accuracy'
    #         )
    #         print(f"Transformer - Best Accuracy: {best_value_tf:.4f}, Params: {best_params_tf}")
    #     except Exception as e: print(f"Transformer tuning error: {e}")

    print("--- End of Optuna Tuner Module Demonstration ---")
