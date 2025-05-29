# src/modeling.py
import pandas as pd
import numpy as np
import logging # For logging model saving
import os # For path joining
import json # For saving feature importances

# Pydantic models for type hinting and config access
from src.config_models import TrainModelConfig 
# Utilities for versioning and paths
from src.pipeline_utils import get_model_version_str, get_versioned_model_paths
# Utility for saving models (handles different types)
from src.utils import save_model 
# Utility for generating metadata
from src.metadata_utils import generate_model_metadata
# Utility for model registration
from src.model_registry_utils import register_model 

# Model-specific imports
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from prophet import Prophet 

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input
)
from sklearn.preprocessing import MinMaxScaler

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers: 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Sequence Creation Helper ---
def create_sequences(X_data_scaled: np.ndarray, y_data_series: pd.Series, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    X_sequences, y_sequences = [], []
    y_data_np = y_data_series.to_numpy() if isinstance(y_data_series, pd.Series) else y_data_series
    for i in range(len(X_data_scaled) - sequence_length):
        X_sequences.append(X_data_scaled[i:i + sequence_length])
        y_sequences.append(y_data_np[i + sequence_length])
    return np.array(X_sequences), np.array(y_sequences)

def create_predict_sequences(X_data_scaled: np.ndarray, sequence_length: int) -> np.ndarray:
    X_sequences = []
    for i in range(len(X_data_scaled) - sequence_length + 1):
        X_sequences.append(X_data_scaled[i:i + sequence_length])
    if not X_sequences: return np.array([]) 
    return np.array(X_sequences)

# --- Keras Model Saving Helper ---
def _save_keras_model_and_metadata(
    model: tf.keras.Model, 
    scaler: MinMaxScaler, 
    X_train_df_cols: list, 
    train_config: TrainModelConfig, 
    version_str: str, 
    sequence_length: int 
):
    model_filename = "model.keras" 
    scaler_filename = "scaler.joblib" 
    model_type_or_name_for_path = train_config.model_type 
    
    versioned_model_path, versioned_dir_path = get_versioned_model_paths(
        base_path=train_config.model_output_path_base,
        model_type_or_name=model_type_or_name_for_path,
        version_str=version_str,
        model_filename=model_filename
    )
    versioned_scaler_path = os.path.join(versioned_dir_path, scaler_filename)

    save_model(model, versioned_model_path) 
    save_model(scaler, versioned_scaler_path) 

    placeholder_metrics = {"status": "training_complete", "parameters_from_config": train_config.model_params.model_dump(exclude_none=True)} 
    placeholder_feature_config = {"features_used": X_train_df_cols, "sequence_length": sequence_length}
    placeholder_feature_config["scaler_path_relative"] = scaler_filename 

    meta_json_file_path = generate_model_metadata(
        model_filepath=versioned_model_path,
        metrics=placeholder_metrics,
        feature_config=placeholder_feature_config, # No feature_importance_file here from Keras models directly
        model_version=version_str,
        model_name_from_config=train_config.model_type,
        training_config_obj=train_config,
        feature_importance_artifact=None # Keras models don't produce this directly
    )
    
    if meta_json_file_path:
        registration_success = register_model(meta_json_path=meta_json_file_path)
        if registration_success:
            logger.info(f"Successfully registered Keras model ({train_config.model_type}) version {version_str}.")
        else:
            logger.error(f"Failed to register Keras model ({train_config.model_type}) version {version_str}.")
    else:
        logger.error(f"Skipping registration for Keras model ({train_config.model_type}) version {version_str} due to metadata generation failure.")
        
    logger.info(f"Keras model ({train_config.model_type}) version {version_str} and scaler saved to {versioned_dir_path}")

# --- XGBoost ---
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, train_config: TrainModelConfig, params: dict = None) -> xgb.XGBClassifier:
    if params is None: 
        params = train_config.model_params.model_dump(exclude_none=True) if train_config.model_params else {}
        params.setdefault('objective', 'binary:logistic')
        params.setdefault('use_label_encoder', False) 
        params.setdefault('eval_metric', 'logloss')
        valid_xgb_params = {k: v for k, v in params.items() if k in xgb.XGBClassifier().get_params()}
        params = valid_xgb_params
    
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        version_str = get_model_version_str()
        model_filename = "model.joblib" 
        model_type_or_name_for_path = train_config.model_type
        
        versioned_model_path, versioned_dir_path = get_versioned_model_paths(
            base_path=train_config.model_output_path_base,
            model_type_or_name=model_type_or_name_for_path,
            version_str=version_str,
            model_filename=model_filename
        )
        save_model(model, versioned_model_path)

        feature_importance_artifact_path = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_train.columns.tolist()
            importance_data = sorted(
                [{"feature": name, "importance": float(score)} for name, score in zip(feature_names, importances)],
                key=lambda x: x["importance"], reverse=True )
            importance_file_path_abs = os.path.join(versioned_dir_path, "feature_importances.json")
            try:
                with open(importance_file_path_abs, 'w') as f: json.dump(importance_data, f, indent=4)
                logger.info(f"Feature importances for {train_config.model_type} v{version_str} saved to {importance_file_path_abs}")
                feature_importance_artifact_path = "feature_importances.json" 
            except Exception as e: logger.error(f"Error saving feature importances: {e}")

        placeholder_metrics = {"status": "training_complete", "parameters": params} 
        placeholder_feature_config = {"features_used": X_train.columns.tolist()}
        
        meta_json_file_path = generate_model_metadata(
            model_filepath=versioned_model_path,
            metrics=placeholder_metrics,
            feature_config=placeholder_feature_config, # feature_importance_file removed from here
            model_version=version_str,
            model_name_from_config=train_config.model_type,
            training_config_obj=train_config,
            feature_importance_artifact=feature_importance_artifact_path # Passed as dedicated param
        )
        
        if meta_json_file_path:
            registration_success = register_model(meta_json_path=meta_json_file_path)
            if registration_success: logger.info(f"Successfully registered model {train_config.model_type} version {version_str}.")
            else: logger.error(f"Failed to register model {train_config.model_type} version {version_str}.")
        else: logger.error(f"Skipping registration for model {train_config.model_type} version {version_str} due to metadata generation failure.")

        logger.info(f"XGBoost model version {version_str} training and processing complete. Saved to {versioned_model_path}")
        return model
    except Exception as e: logger.error(f"Error during XGBoost training or saving: {e}"); raise

def predict_xgboost(model: xgb.XGBClassifier, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test); y_pred_proba = model.predict_proba(X_test)[:, 1]; return y_pred, y_pred_proba

# --- LightGBM ---
def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, train_config: TrainModelConfig, params: dict = None) -> lgb.LGBMClassifier:
    if params is None:
        params = train_config.model_params.model_dump(exclude_none=True) if train_config.model_params else {}
        params.setdefault('objective', 'binary'); params.setdefault('metric', 'binary_logloss'); params.setdefault('verbose', -1)
        valid_lgb_params = {k: v for k, v in params.items() if k in lgb.LGBMClassifier().get_params()}; params = valid_lgb_params
    try:
        model = lgb.LGBMClassifier(**params); model.fit(X_train, y_train)
        version_str = get_model_version_str(); model_filename = "model.joblib"; model_type_or_name_for_path = train_config.model_type
        versioned_model_path, versioned_dir_path = get_versioned_model_paths(
            base_path=train_config.model_output_path_base, model_type_or_name=model_type_or_name_for_path,
            version_str=version_str, model_filename=model_filename)
        save_model(model, versioned_model_path)

        feature_importance_artifact_path = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_; feature_names = X_train.columns.tolist()
            importance_data = sorted([{"feature": name, "importance": float(score)} for name, score in zip(feature_names, importances)], key=lambda x: x["importance"], reverse=True)
            importance_file_path_abs = os.path.join(versioned_dir_path, "feature_importances.json")
            try:
                with open(importance_file_path_abs, 'w') as f: json.dump(importance_data, f, indent=4)
                logger.info(f"Feature importances for {train_config.model_type} v{version_str} saved to {importance_file_path_abs}")
                feature_importance_artifact_path = "feature_importances.json"
            except Exception as e: logger.error(f"Error saving feature importances: {e}")

        placeholder_metrics = {"status": "training_complete", "parameters": params}
        placeholder_feature_config = {"features_used": X_train.columns.tolist()}
        meta_json_file_path = generate_model_metadata(
            model_filepath=versioned_model_path, metrics=placeholder_metrics, feature_config=placeholder_feature_config,
            model_version=version_str, model_name_from_config=train_config.model_type,
            training_config_obj=train_config, feature_importance_artifact=feature_importance_artifact_path)
        
        if meta_json_file_path:
            if register_model(meta_json_path=meta_json_file_path): logger.info(f"Registered model {train_config.model_type} v{version_str}.")
            else: logger.error(f"Failed to register model {train_config.model_type} v{version_str}.")
        else: logger.error(f"Skipping registration for {train_config.model_type} v{version_str} (metadata error).")
        logger.info(f"LightGBM model v{version_str} training complete. Saved to {versioned_model_path}"); return model
    except Exception as e: logger.error(f"Error in LightGBM training/saving: {e}"); raise

def predict_lightgbm(model: lgb.LGBMClassifier, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test); y_pred_proba = model.predict_proba(X_test)[:, 1]; return y_pred, y_pred_proba
    
# --- LSTM ---
def train_lstm(X_train_df: pd.DataFrame, y_train_series: pd.Series, train_config: TrainModelConfig, sequence_length: int, lstm_params: dict = None, fit_params: dict = None) -> tuple[Sequential, MinMaxScaler]:
    if lstm_params is None:
        lstm_params = train_config.model_params.model_dump(exclude_none=True) if train_config.model_params else {}
        lstm_params.setdefault('lstm_units', 50); lstm_params.setdefault('dropout_rate', 0.2); lstm_params.setdefault('dense_units_factor', 0.5)
    if fit_params is None: fit_params = {'epochs': 50, 'batch_size': 32, 'validation_split': 0.1, 'verbose': 1}
    scaler = MinMaxScaler(); X_train_scaled = scaler.fit_transform(X_train_df)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_series, sequence_length)
    if X_train_seq.shape[0] == 0: raise ValueError("Not enough data for LSTM training sequences.")
    model = Sequential([
        LSTM(lstm_params['lstm_units'], input_shape=(sequence_length, X_train_seq.shape[2]), return_sequences=False),
        Dropout(lstm_params['dropout_rate']),
        Dense(int(lstm_params['lstm_units'] * lstm_params['dense_units_factor']), activation='relu'),
        Dropout(lstm_params.get('dropout_rate', 0.2) / 2), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    try:
        model.fit(X_train_seq, y_train_seq, **fit_params); version_str = get_model_version_str()
        _save_keras_model_and_metadata(model, scaler, X_train_df.columns.tolist(), train_config, version_str, sequence_length)
        logger.info(f"LSTM model v{version_str} training complete."); return model, scaler
    except Exception as e: logger.error(f"Error in LSTM training/saving: {e}"); raise

def predict_lstm(model: Sequential, X_test_df: pd.DataFrame, scaler: MinMaxScaler, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        X_test_scaled = scaler.transform(X_test_df); X_test_seq_np = create_predict_sequences(X_test_scaled, sequence_length)
        if X_test_seq_np.shape[0] == 0: return np.array([]), np.array([])
        y_pred_proba = model.predict(X_test_seq_np)
    except Exception as e: logger.error(f"Error in LSTM prediction: {e}"); return np.array([]), np.array([])
    y_pred = (y_pred_proba > 0.5).astype(int); return y_pred.flatten(), y_pred_proba.flatten()

# --- CNN-LSTM ---
def train_cnnlstm(X_train_df: pd.DataFrame, y_train_series: pd.Series, train_config: TrainModelConfig, sequence_length: int, cnnlstm_params: dict = None, fit_params: dict = None) -> tuple[Sequential, MinMaxScaler]:
    if cnnlstm_params is None:
        cnnlstm_params = train_config.model_params.model_dump(exclude_none=True) if train_config.model_params else {}
        cnnlstm_params.setdefault('filters', 64); cnnlstm_params.setdefault('kernel_size', 3); cnnlstm_params.setdefault('pool_size', 2)
        cnnlstm_params.setdefault('lstm_units', 50); cnnlstm_params.setdefault('dropout_rate', 0.2); cnnlstm_params.setdefault('dense_units_factor', 0.5)
    if fit_params is None: fit_params = {'epochs': 50, 'batch_size': 32, 'validation_split': 0.1, 'verbose': 1}
    scaler = MinMaxScaler(); X_train_scaled = scaler.fit_transform(X_train_df)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_series, sequence_length)
    if X_train_seq.shape[0] == 0: raise ValueError("Not enough data for CNN-LSTM training sequences.")
    model = Sequential([
        Conv1D(filters=cnnlstm_params['filters'], kernel_size=cnnlstm_params['kernel_size'], activation='relu', input_shape=(sequence_length, X_train_seq.shape[2])),
        MaxPooling1D(pool_size=cnnlstm_params['pool_size']), LSTM(cnnlstm_params['lstm_units'], return_sequences=False), 
        Dropout(cnnlstm_params['dropout_rate']), Dense(int(cnnlstm_params['lstm_units'] * cnnlstm_params['dense_units_factor']), activation='relu'),
        Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    try:
        model.fit(X_train_seq, y_train_seq, **fit_params); version_str = get_model_version_str()
        _save_keras_model_and_metadata(model, scaler, X_train_df.columns.tolist(), train_config, version_str, sequence_length)
        logger.info(f"CNN-LSTM model v{version_str} training complete."); return model, scaler
    except Exception as e: logger.error(f"Error in CNN-LSTM training/saving: {e}"); raise

def predict_cnnlstm(model: Sequential, X_test_df: pd.DataFrame, scaler: MinMaxScaler, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        X_test_scaled = scaler.transform(X_test_df); X_test_seq_np = create_predict_sequences(X_test_scaled, sequence_length)
        if X_test_seq_np.shape[0] == 0: return np.array([]), np.array([])
        y_pred_proba = model.predict(X_test_seq_np)
    except Exception as e: logger.error(f"Error in CNN-LSTM prediction: {e}"); return np.array([]), np.array([])
    y_pred = (y_pred_proba > 0.5).astype(int); return y_pred.flatten(), y_pred_proba.flatten()

# --- Transformer Components ---
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,position,d_model,**kwargs):super(PositionalEncoding,self).__init__(**kwargs);self.position=position;self.d_model=d_model;self.pos_encoding=self._positional_encoding(self.position,self.d_model)
    def _get_angles(self,position,i,d_model):angles=1/tf.pow(10000,(2*(i//2))/tf.cast(d_model,tf.float32));return position*angles
    def _positional_encoding(self,position,d_model):angle_rads=self._get_angles(tf.range(position,dtype=tf.float32)[:,tf.newaxis],tf.range(d_model,dtype=tf.float32)[tf.newaxis,:],d_model);sines=tf.math.sin(angle_rads[:,0::2]);cosines=tf.math.cos(angle_rads[:,1::2]);if d_model%2!=0:cosines=tf.pad(cosines,[[0,0],[0,1]]);pos_encoding=tf.concat([sines,cosines],axis=-1);pos_encoding=pos_encoding[tf.newaxis,...];return tf.cast(pos_encoding,tf.float32)
    def call(self,inputs):return inputs+self.pos_encoding[:,:tf.shape(inputs)[1],:]
    def get_config(self):config=super().get_config();config.update({"position":self.position,"d_model":self.d_model});return config
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1,**kwargs):super(TransformerEncoderBlock,self).__init__(**kwargs);self.embed_dim=embed_dim;self.num_heads=num_heads;self.ff_dim=ff_dim;self.rate=rate;self.att=MultiHeadAttention(num_heads=num_heads,key_dim=embed_dim);self.ffn=tf.keras.Sequential([Dense(ff_dim,activation="relu"),Dense(embed_dim),]);self.layernorm1=LayerNormalization(epsilon=1e-6);self.layernorm2=LayerNormalization(epsilon=1e-6);self.dropout1=Dropout(rate);self.dropout2=Dropout(rate)
    def call(self,inputs,training=False):attn_output=self.att(inputs,inputs);attn_output=self.dropout1(attn_output,training=training);out1=self.layernorm1(inputs+attn_output);ffn_output=self.ffn(out1);ffn_output=self.dropout2(ffn_output,training=training);return self.layernorm2(out1+ffn_output)
    def get_config(self):config=super().get_config();config.update({"embed_dim":self.embed_dim,"num_heads":self.num_heads,"ff_dim":self.ff_dim,"rate":self.rate});return config

# --- Transformer Model ---
def train_transformer(X_train_df: pd.DataFrame, y_train_series: pd.Series, train_config: TrainModelConfig, sequence_length: int, transformer_params: dict = None, fit_params: dict = None) -> tuple[Model, MinMaxScaler]:
    if transformer_params is None:
        transformer_params = train_config.model_params.model_dump(exclude_none=True) if train_config.model_params else {}
        transformer_params.setdefault('num_heads',2);transformer_params.setdefault('ff_dim',32);transformer_params.setdefault('num_transformer_blocks',1);transformer_params.setdefault('dropout_rate',0.1)
    if fit_params is None:fit_params={'epochs':50,'batch_size':32,'validation_split':0.1,'verbose':1}
    scaler=MinMaxScaler();X_train_scaled=scaler.fit_transform(X_train_df);X_train_seq,y_train_seq=create_sequences(X_train_scaled,y_train_series,sequence_length)
    if X_train_seq.shape[0]==0:raise ValueError("Not enough data for Transformer training sequences.")
    num_features=X_train_seq.shape[2];current_embed_dim=transformer_params.get('embed_dim',num_features)
    if 'embed_dim' not in transformer_params:transformer_params['embed_dim']=current_embed_dim
    if current_embed_dim!=num_features:logger.warning(f"Transformer embed_dim ({current_embed_dim}) differs from input features ({num_features}).")
    inputs=Input(shape=(sequence_length,num_features));x=inputs
    if current_embed_dim!=num_features:x=Dense(current_embed_dim,activation='relu',name='feature_projection')(x)
    x=PositionalEncoding(position=sequence_length,d_model=current_embed_dim)(x)
    for i in range(transformer_params.get('num_transformer_blocks',1)):x=TransformerEncoderBlock(embed_dim=current_embed_dim,num_heads=transformer_params.get('num_heads',2),ff_dim=transformer_params.get('ff_dim',32),rate=transformer_params.get('dropout_rate',0.1),name=f'transformer_block_{i}')(x)
    x=GlobalAveragePooling1D()(x);x=Dropout(transformer_params.get('dropout_rate',0.1))(x)
    mlp_units_list=transformer_params.get('mlp_units',[current_embed_dim//2 if current_embed_dim > 1 else 16])
    for units_idx,units in enumerate(mlp_units_list):x=Dense(units,activation="relu",name=f'mlp_dense_{units_idx}')(x);x=Dropout(transformer_params.get('mlp_dropout_rate',transformer_params.get('dropout_rate',0.1)),name=f'mlp_dropout_{units_idx}')(x)
    outputs=Dense(1,activation="sigmoid")(x);model=Model(inputs=inputs,outputs=outputs)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy','AUC'])
    try:
        model.fit(X_train_seq,y_train_seq,**fit_params);version_str=get_model_version_str()
        _save_keras_model_and_metadata(model,scaler,X_train_df.columns.tolist(),train_config,version_str,sequence_length)
        logger.info(f"Transformer model v{version_str} training complete.");return model,scaler
    except Exception as e:logger.error(f"Error in Transformer training/saving: {e}");raise

def predict_transformer(model:Model,X_test_df:pd.DataFrame,scaler:MinMaxScaler,sequence_length:int)->tuple[np.ndarray,np.ndarray]:
    try:
        X_test_scaled=scaler.transform(X_test_df);X_test_seq_np=create_predict_sequences(X_test_scaled,sequence_length)
        if X_test_seq_np.shape[0]==0:return np.array([]),np.array([])
        y_pred_proba=model.predict(X_test_seq_np)
    except Exception as e:logger.error(f"Error in Transformer prediction: {e}");return np.array([]),np.array([])
    y_pred=(y_pred_proba>0.5).astype(int);return y_pred.flatten(),y_pred_proba.flatten()

# --- CatBoost ---
def train_catboost(X_train: pd.DataFrame, y_train: pd.Series, train_config: TrainModelConfig, params: dict = None, cat_features: list = None) -> cb.CatBoostClassifier:
    if params is None:
        params = train_config.model_params.model_dump(exclude_none=True) if train_config.model_params else {}
        params.setdefault('iterations',100);params.setdefault('loss_function','Logloss');params.setdefault('eval_metric','AUC');params.setdefault('verbose',0)
        valid_cb_params = {k:v for k,v in params.items() if k in cb.CatBoostClassifier().get_params()};params=valid_cb_params
    if cat_features is None:cat_features=[col for col in X_train.columns if X_train[col].dtype.name in ['object','category','string']]
    if not cat_features:cat_features=None
    try:
        model=cb.CatBoostClassifier(**params);model.fit(X_train,y_train,cat_features=cat_features)
        version_str=get_model_version_str();model_filename="model.cbm";model_type_or_name_for_path=train_config.model_type
        versioned_model_path,versioned_dir_path=get_versioned_model_paths(base_path=train_config.model_output_path_base,model_type_or_name=model_type_or_name_for_path,version_str=version_str,model_filename=model_filename)
        save_model(model,versioned_model_path)

        feature_importance_artifact_path=None
        if hasattr(model,'feature_importances_'):
            importances=model.feature_importances_;feature_names=X_train.columns.tolist()
            importance_data=sorted([{"feature":name,"importance":float(score)} for name,score in zip(feature_names,importances)],key=lambda x:x["importance"],reverse=True)
            importance_file_path_abs=os.path.join(versioned_dir_path,"feature_importances.json")
            try:
                with open(importance_file_path_abs,'w') as f:json.dump(importance_data,f,indent=4)
                logger.info(f"Feature importances for {train_config.model_type} v{version_str} saved to {importance_file_path_abs}")
                feature_importance_artifact_path="feature_importances.json"
            except Exception as e:logger.error(f"Error saving feature importances: {e}")
        
        placeholder_metrics={"status":"training_complete","parameters":params}
        placeholder_feature_config={"features_used":X_train.columns.tolist()}
        # Note: feature_importance_file is no longer added here directly
        
        meta_json_file_path=generate_model_metadata(
            model_filepath=versioned_model_path,metrics=placeholder_metrics,feature_config=placeholder_feature_config,
            model_version=version_str,model_name_from_config=train_config.model_type,
            training_config_obj=train_config,feature_importance_artifact=feature_importance_artifact_path)
        
        if meta_json_file_path:
            if register_model(meta_json_path=meta_json_file_path):logger.info(f"Registered model {train_config.model_type} v{version_str}.")
            else:logger.error(f"Failed to register model {train_config.model_type} v{version_str}.")
        else:logger.error(f"Skipping registration for {train_config.model_type} v{version_str} (metadata error).")
        logger.info(f"CatBoost model v{version_str} training complete. Saved to {versioned_model_path}");return model
    except Exception as e:logger.error(f"Error in CatBoost training/saving: {e}");raise

def predict_catboost(model:cb.CatBoostClassifier,X_test:pd.DataFrame)->tuple[np.ndarray,np.ndarray]:
    y_pred=model.predict(X_test);y_pred_proba=model.predict_proba(X_test)[:,1];return y_pred,y_pred_proba

# --- Prophet ---
def train_prophet(df_train:pd.DataFrame,params:dict=None)->Prophet:
    if not all(col in df_train.columns for col in ['ds','y']):raise ValueError("Prophet input must have 'ds' and 'y' columns.")
    if params is None:params={}
    model=Prophet(**params);model.fit(df_train);return model
def predict_prophet(model:Prophet,periods:int,freq:str='D',future_df_custom:pd.DataFrame=None)->pd.DataFrame:
    if future_df_custom is None:future_df=model.make_future_dataframe(periods=periods,freq=freq)
    else:
        if 'ds' not in future_df_custom.columns:raise ValueError("Custom future_df for Prophet must have 'ds'.")
        future_df=future_df_custom
    forecast=model.predict(future_df);return forecast

# Main orchestrator function
def train_model_pipeline_step(train_config: TrainModelConfig):
    logger.info(f"Starting model training pipeline step for model type: {train_config.model_type}")
    logger.info(f"Loading features from: {train_config.input_features_path}")
    try:
        features_df=pd.read_csv(train_config.input_features_path)
        if train_config.target_column not in features_df.columns:raise ValueError(f"Target column '{train_config.target_column}' not found.")
        if features_df.empty:raise ValueError("Features data is empty.")
    except Exception as e:logger.error(f"Failed to load or validate input features data: {e}");raise
    if train_config.feature_columns_to_use:feature_cols=[col for col in train_config.feature_columns_to_use if col in features_df.columns and col!=train_config.target_column]
    else:feature_cols=features_df.columns.drop(train_config.target_column,errors='ignore').tolist()
    if not feature_cols:raise ValueError("No feature columns for training.")
    X_train=features_df[feature_cols].fillna(0);y_train=features_df[train_config.target_column].fillna(0)
    logger.info(f"Prepared training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    model_type_dispatch=train_config.model_type.lower().replace('-','')
    train_function_name=f"train_{model_type_dispatch}"
    try:train_function=globals()[train_function_name]
    except KeyError:logger.error(f"Training function '{train_function_name}' not found.");raise NotImplementedError(f"Model type '{train_config.model_type}' training not implemented.")
    logger.info(f"Calling {train_function_name}...")
    trained_model=None
    if model_type_dispatch in ["lstm","cnnlstm","transformer"]:
        seq_len=train_config.model_params.sequence_length
        if not seq_len:raise ValueError("sequence_length must be in model_params for sequence models.")
        trained_model,_=train_function(X_train,y_train,train_config,sequence_length=seq_len)
    else:trained_model=train_function(X_train,y_train,train_config)
    logger.info(f"Successfully executed {train_function_name} for model type {train_config.model_type}.")
    return trained_model

if __name__=='__main__':
    print("--- modeling.py __main__ execution ---")
    if False:
        prophet_data=pd.DataFrame({'ds':pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05']),'y':[10,12,15,13,16]})
        prophet_model=train_prophet(prophet_data);future_prophet=predict_prophet(prophet_model,periods=2)
        print("Prophet Forecast:");print(future_prophet[['ds','yhat','yhat_lower','yhat_upper']].tail())
    else:print("Prophet example in __main__ is currently disabled.")
    print("--- End of modeling.py __main__ ---")
