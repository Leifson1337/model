# src/modeling.py
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # Added Model for Functional API
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input # Added Transformer layers
)
from sklearn.preprocessing import MinMaxScaler

# --- Sequence Creation Helper (used by LSTM, CNN-LSTM, Transformer) ---
def create_sequences(X_data_scaled: np.ndarray, y_data_series: pd.Series, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Creates sequences for time-series models."""
    X_sequences, y_sequences = [], []
    y_data_np = y_data_series.to_numpy() if isinstance(y_data_series, pd.Series) else y_data_series
    for i in range(len(X_data_scaled) - sequence_length):
        X_sequences.append(X_data_scaled[i:i + sequence_length])
        y_sequences.append(y_data_np[i + sequence_length])
    return np.array(X_sequences), np.array(y_sequences)

def create_predict_sequences(X_data_scaled: np.ndarray, sequence_length: int) -> np.ndarray:
    """Creates sequences from input data for prediction (no y values)."""
    X_sequences = []
    for i in range(len(X_data_scaled) - sequence_length + 1):
        X_sequences.append(X_data_scaled[i:i + sequence_length])
    if not X_sequences: return np.array([]) # Handle case where data is too short
    return np.array(X_sequences)

# --- XGBoost ---
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> xgb.XGBClassifier:
    if params is None:
        params = {'objective': 'binary:logistic', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'use_label_encoder': False, 'eval_metric': 'logloss'}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model: xgb.XGBClassifier, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba

# --- LightGBM ---
def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> lgb.LGBMClassifier:
    if params is None:
        params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': -1, 'num_leaves': 31, 'verbose': -1}
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model

def predict_lightgbm(model: lgb.LGBMClassifier, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba
    
# --- LSTM ---
def train_lstm(X_train_df: pd.DataFrame, y_train_series: pd.Series, sequence_length: int, lstm_params: dict = None, fit_params: dict = None) -> tuple[Sequential, MinMaxScaler]:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_series, sequence_length)
    if X_train_seq.shape[0] == 0: raise ValueError("Not enough data for LSTM training sequences.")

    if lstm_params is None: lstm_params = {'lstm_units': 50, 'dropout_rate': 0.2, 'dense_units_factor': 0.5}
    model = Sequential([
        LSTM(lstm_params['lstm_units'], input_shape=(sequence_length, X_train_seq.shape[2]), return_sequences=False),
        Dropout(lstm_params['dropout_rate']),
        Dense(int(lstm_params['lstm_units'] * lstm_params['dense_units_factor']), activation='relu'),
        Dropout(lstm_params['dropout_rate'] / 2),
        Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    if fit_params is None: fit_params = {'epochs': 50, 'batch_size': 32, 'validation_split': 0.1, 'verbose': 1}
    model.fit(X_train_seq, y_train_seq, **fit_params)
    return model, scaler

def predict_lstm(model: Sequential, X_test_df: pd.DataFrame, scaler: MinMaxScaler, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    X_test_scaled = scaler.transform(X_test_df)
    X_test_seq_np = create_predict_sequences(X_test_scaled, sequence_length)
    if X_test_seq_np.shape[0] == 0: return np.array([]), np.array([])
    y_pred_proba = model.predict(X_test_seq_np)
    y_pred = (y_pred_proba > 0.5).astype(int)
    return y_pred.flatten(), y_pred_proba.flatten()

# --- CNN-LSTM ---
def train_cnnlstm(X_train_df: pd.DataFrame, y_train_series: pd.Series, sequence_length: int, cnnlstm_params: dict = None, fit_params: dict = None) -> tuple[Sequential, MinMaxScaler]:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_series, sequence_length)
    if X_train_seq.shape[0] == 0: raise ValueError("Not enough data for CNN-LSTM training sequences.")

    if cnnlstm_params is None:
        cnnlstm_params = {
            'filters': 64, 'kernel_size': 3, 'pool_size': 2,
            'lstm_units': 50, 'dropout_rate': 0.2, 'dense_units_factor': 0.5
        }
    
    model = Sequential([
        Conv1D(filters=cnnlstm_params['filters'], kernel_size=cnnlstm_params['kernel_size'], activation='relu', input_shape=(sequence_length, X_train_seq.shape[2])),
        MaxPooling1D(pool_size=cnnlstm_params['pool_size']),
        LSTM(cnnlstm_params['lstm_units'], return_sequences=False), # Set to True if stacking LSTMs
        Dropout(cnnlstm_params['dropout_rate']),
        Dense(int(cnnlstm_params['lstm_units'] * cnnlstm_params['dense_units_factor']), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    if fit_params is None: fit_params = {'epochs': 50, 'batch_size': 32, 'validation_split': 0.1, 'verbose': 1}
    model.fit(X_train_seq, y_train_seq, **fit_params)
    return model, scaler

def predict_cnnlstm(model: Sequential, X_test_df: pd.DataFrame, scaler: MinMaxScaler, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    X_test_scaled = scaler.transform(X_test_df)
    X_test_seq_np = create_predict_sequences(X_test_scaled, sequence_length)
    if X_test_seq_np.shape[0] == 0: return np.array([]), np.array([])
    y_pred_proba = model.predict(X_test_seq_np)
    y_pred = (y_pred_proba > 0.5).astype(int)
    return y_pred.flatten(), y_pred_proba.flatten()

# --- Transformer Components ---
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(self.position, self.d_model)

    def _get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def _positional_encoding(self, position, d_model):
        angle_rads = self._get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        # Pad if d_model is odd
        if d_model % 2 != 0:
            cosines = tf.pad(cosines, [[0,0],[0,1]])            
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs): # inputs shape: (batch_size, sequence_length, features)
        # Ensure features dimension matches d_model
        # This simple PE adds to the input, assuming input features are already like embeddings
        # Or, a Dense layer projects input features to d_model before PE for more robustness
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def get_config(self): # For model saving/loading
        config = super().get_config()
        config.update({"position": self.position, "d_model": self.d_model})
        return config

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False): # Added training flag for dropout
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self): # For model saving/loading
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim, "num_heads": self.num_heads,
            "ff_dim": self.ff_dim, "rate": self.rate
        })
        return config

# --- Transformer Model ---
def train_transformer(X_train_df: pd.DataFrame, y_train_series: pd.Series, sequence_length: int, transformer_params: dict = None, fit_params: dict = None) -> tuple[Model, MinMaxScaler]:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_series, sequence_length)
    if X_train_seq.shape[0] == 0: raise ValueError("Not enough data for Transformer training sequences.")

    num_features = X_train_seq.shape[2]
    if transformer_params is None:
        transformer_params = {
            'embed_dim': num_features, # Dimension of embeddings (must match num_features if no projection)
            'num_heads': 2, 'ff_dim': 32, 'num_transformer_blocks': 1, 'dropout_rate': 0.1
        }
    # Ensure embed_dim matches input feature dimension if not using a separate projection layer
    if transformer_params['embed_dim'] != num_features:
        print(f"Warning: Transformer embed_dim ({transformer_params['embed_dim']}) "
              f"differs from input features ({num_features}). This might lead to issues "
              "if not handled by a projection layer or if PositionalEncoding assumes match.")
        # For this simplified version, we'll proceed, but typically a Dense layer would project X_train_seq features to embed_dim.

    inputs = Input(shape=(sequence_length, num_features))
    x = inputs
    # Optional: Add a Dense layer here to project num_features to embed_dim if they differ
    # x = Dense(transformer_params['embed_dim'], activation='relu')(x) # Example projection

    # Add Positional Encoding if embed_dim matches feature dim, or after projection
    # Using sequence_length for position arg of PositionalEncoding
    if transformer_params['embed_dim'] == num_features :
         x = PositionalEncoding(position=sequence_length, d_model=transformer_params['embed_dim'])(x)
    
    for _ in range(transformer_params['num_transformer_blocks']):
        x = TransformerEncoderBlock(transformer_params['embed_dim'], transformer_params['num_heads'], transformer_params['ff_dim'], transformer_params['dropout_rate'])(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(transformer_params['dropout_rate'])(x)
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    if fit_params is None: fit_params = {'epochs': 50, 'batch_size': 32, 'validation_split': 0.1, 'verbose': 1}
    model.fit(X_train_seq, y_train_seq, **fit_params)
    return model, scaler

def predict_transformer(model: Model, X_test_df: pd.DataFrame, scaler: MinMaxScaler, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    X_test_scaled = scaler.transform(X_test_df)
    X_test_seq_np = create_predict_sequences(X_test_scaled, sequence_length)
    if X_test_seq_np.shape[0] == 0: return np.array([]), np.array([])
    y_pred_proba = model.predict(X_test_seq_np)
    y_pred = (y_pred_proba > 0.5).astype(int)
    return y_pred.flatten(), y_pred_proba.flatten()

# --- CatBoost ---
def train_catboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None, cat_features: list = None) -> cb.CatBoostClassifier:
    if params is None: params = {'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0}
    if cat_features is None: cat_features = [col for col in X_train.columns if X_train[col].dtype in ['object', 'category']]
    if not cat_features: cat_features = None
    model = cb.CatBoostClassifier(**params)
    # Note: CatBoost early stopping needs an eval_set passed to fit()
    # model.fit(X_train, y_train, cat_features=cat_features, early_stopping_rounds=10 if 'eval_set' in params else None)
    model.fit(X_train, y_train, cat_features=cat_features) # Simplified fit without eval_set for now
    return model

def predict_catboost(model: cb.CatBoostClassifier, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba

# --- Prophet ---
def train_prophet(df_train: pd.DataFrame, params: dict = None) -> Prophet:
    if not all(col in df_train.columns for col in ['ds', 'y']): raise ValueError("Prophet input must have 'ds' and 'y' columns.")
    if params is None: params = {}
    model = Prophet(**params)
    model.fit(df_train)
    return model

def predict_prophet(model: Prophet, periods: int, freq: str = 'D', future_df_custom: pd.DataFrame = None) -> pd.DataFrame:
    if future_df_custom is None: future_df = model.make_future_dataframe(periods=periods, freq=freq)
    else:
        if 'ds' not in future_df_custom.columns: raise ValueError("Custom future_df for Prophet must have 'ds'.")
        future_df = future_df_custom
    forecast = model.predict(future_df)
    return forecast
