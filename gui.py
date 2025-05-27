# gui.py
import streamlit as st
from datetime import datetime, date 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from src import config
from src.data_management import download_stock_data 
from src.feature_engineering import add_technical_indicators, add_rolling_lag_features, create_target_variable
from src.sentiment_analysis import get_daily_sentiment_scores 
from src.fundamental_data import add_fundamental_features_to_data
from src import modeling # train_X, predict_X functions are here
from src import utils # save_model, load_model
from src.evaluation import plot_roc_auc, plot_confusion_matrix 

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler # Handled in modeling functions for DL
from sklearn.metrics import accuracy_score, classification_report 

# --- Page Configuration ---
st.set_page_config(page_title="Quantitative Leverage Predictor", layout="wide")
st.title("📈 Quantitative Leverage Opportunity Predictor")
st.markdown("Welcome to the advanced stock analysis and prediction tool.")

# --- Session State Initialization ---
default_ticker = config.DEFAULT_TICKERS[0] if config.DEFAULT_TICKERS else "AAPL"
try: default_start_date = datetime.strptime(config.DEFAULT_START_DATE, "%Y-%m-%d").date()
except: default_start_date = datetime(2020, 1, 1).date()
try: default_end_date = datetime.strptime(config.DEFAULT_END_DATE, "%Y-%m-%d").date()
except: default_end_date = date.today()

session_defaults = {
    'selected_ticker': default_ticker,
    'start_date': default_start_date,
    'end_date': default_end_date,
    'stock_data': None,
    'feature_data': None,
    'load_data_tab1_clicked': False,
    'generate_technical': True,
    'generate_rolling_lag': True,
    'generate_sentiment': False,
    'generate_fundamental': False,
    'generate_target': True,
    'selected_model_type_tab3': "XGBoost", 
    'trained_model_info': {}, 
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Sidebar ---
st.sidebar.header("Global Settings")
st.session_state.selected_ticker = st.sidebar.selectbox("Select Stock Ticker:", options=config.DEFAULT_TICKERS, key='sb_selected_ticker_gui_final_final_final', 
    index=config.DEFAULT_TICKERS.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in config.DEFAULT_TICKERS else 0)
if isinstance(st.session_state.start_date, str): st.session_state.start_date = datetime.strptime(st.session_state.start_date, "%Y-%m-%d").date()
if isinstance(st.session_state.end_date, str): st.session_state.end_date = datetime.strptime(st.session_state.end_date, "%Y-%m-%d").date()
if st.session_state.end_date < st.session_state.start_date: st.session_state.end_date = st.session_state.start_date
st.session_state.start_date = st.sidebar.date_input("Start Date:", value=st.session_state.start_date, min_value=datetime(2010,1,1).date(), max_value=date.today(), key='sb_start_date_gui_final_final_final')
st.session_state.end_date = st.sidebar.date_input("End Date:", value=st.session_state.end_date, min_value=st.session_state.start_date, max_value=date.today(), key='sb_end_date_gui_final_final_final')

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Load & Analyze Data", "🛠️ Feature Engineering", "🧠 Train & Evaluate Model", "📈 Backtest & Visualize"])

with tab1: 
    st.header("Load & Analyze Stock Data")
    st.write(f"**Selected Ticker:** {st.session_state.selected_ticker}, **Date Range:** {st.session_state.start_date.strftime('%Y-%m-%d')} to {st.session_state.end_date.strftime('%Y-%m-%d')}")
    if st.button("Load Data", key="load_data_tab1_button_final_final_final"):
        st.session_state.load_data_tab1_clicked = True
        with st.spinner("Loading data..."):
            try:
                data_df = download_stock_data([st.session_state.selected_ticker], st.session_state.start_date.strftime("%Y-%m-%d"), st.session_state.end_date.strftime("%Y-%m-%d"))
                if data_df is None or data_df.empty: st.warning("No data returned."); st.session_state.stock_data = None
                else:
                    st.session_state.stock_data = data_df.xs(st.session_state.selected_ticker, axis=1, level=0, drop_level=True) if isinstance(data_df.columns, pd.MultiIndex) and st.session_state.selected_ticker in data_df.columns.levels[0] else data_df
                    st.success("Data loaded!")
            except Exception as e: st.error(f"Error: {e}"); st.session_state.stock_data = None
    if st.session_state.stock_data is not None:
        st.subheader("Preview"); st.dataframe(st.session_state.stock_data.head())
        if 'Close' in st.session_state.stock_data: st.subheader("Price Chart"); st.line_chart(st.session_state.stock_data['Close'])
        if 'Volume' in st.session_state.stock_data: st.subheader("Volume Chart"); st.bar_chart(st.session_state.stock_data['Volume'])
        st.subheader("Statistics"); st.dataframe(st.session_state.stock_data.describe())
    elif st.session_state.load_data_tab1_clicked: st.info("Data could not be loaded.")
    else: st.info("Click 'Load Data'.")

with tab2: 
    st.header("Feature Engineering Options")
    if st.session_state.stock_data is None: st.warning("Load data in Tab 1.")
    else:
        st.subheader("Select Features")
        st.session_state.generate_technical = st.checkbox("Technical Indicators", st.session_state.generate_technical, key="cb_tech_gui_final_final_final")
        st.session_state.generate_rolling_lag = st.checkbox("Rolling & Lag", st.session_state.generate_rolling_lag, key="cb_roll_gui_final_final_final")
        st.session_state.generate_sentiment = st.checkbox("Sentiment (NewsAPI Key needed)", st.session_state.generate_sentiment, key="cb_sent_gui_final_final_final")
        st.session_state.generate_fundamental = st.checkbox("Fundamental Data", st.session_state.generate_fundamental, key="cb_fund_gui_final_final_final")
        st.session_state.generate_target = st.checkbox("Target Variable", st.session_state.generate_target, key="cb_target_gui_final_final_final")
        if st.button("Generate Features", key="gen_features_tab2_button_final_final_final"):
            with st.spinner("Generating..."):
                try:
                    df = st.session_state.stock_data.copy()
                    if st.session_state.generate_technical: df = add_technical_indicators(df.copy(), fillna=False)
                    if st.session_state.generate_rolling_lag: df = add_rolling_lag_features(df.copy())
                    if st.session_state.generate_sentiment:
                        if config.NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE" or not config.NEWS_API_KEY: st.warning("NewsAPI key not set.")
                        else:
                            s_start = st.session_state.end_date - pd.Timedelta(days=60); s_start = max(s_start, st.session_state.start_date)
                            sent_df = get_daily_sentiment_scores(st.session_state.selected_ticker, s_start.strftime("%Y-%m-%d"), st.session_state.end_date.strftime("%Y-%m-%d"))
                            if not sent_df.empty: df = df.merge(sent_df, left_index=True, right_index=True, how='left'); df[sent_df.columns] = df[sent_df.columns].ffill().fillna(0)
                    if st.session_state.generate_fundamental: df = add_fundamental_features_to_data(df.copy(), st.session_state.selected_ticker)
                    if st.session_state.generate_target: df = create_target_variable(df.copy(), 5, 0.03) 
                    st.session_state.feature_data = df; st.success("Features generated!")
                except Exception as e: st.error(f"Error: {e}"); st.session_state.feature_data = None
        if st.session_state.feature_data is not None:
            st.subheader("Preview with Features"); st.dataframe(st.session_state.feature_data.head())
            st.write(f"Shape: {st.session_state.feature_data.shape}, Nulls: {st.session_state.feature_data.isnull().sum().sum()}")

with tab3:
    st.header("Model Training and Evaluation")
    if 'feature_data' not in st.session_state or st.session_state.feature_data is None:
        st.warning("Please generate features in the 'Feature Engineering' tab first.")
    else:
        st.subheader("1. Select Model")
        available_models = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "CNN-LSTM", "Transformer"] 
        st.session_state.selected_model_type_tab3 = st.selectbox("Choose a Model:", options=available_models,
            index=available_models.index(st.session_state.selected_model_type_tab3), key="model_choice_tab3_final_final")
        
        if 'target' not in st.session_state.feature_data.columns:
            st.error("Target variable 'target' not found. Please generate it in Tab 2.")
        else:
            df_model_input = st.session_state.feature_data.dropna(subset=['target'])
            potential_feature_cols = [col for col in df_model_input.columns if col not in ['target','Open','High','Low','Close','Volume','Adj Close'] and df_model_input[col].dtype in [np.int64,np.float64,np.int32,np.float32,int,float]]
            X_all_features = df_model_input[potential_feature_cols].copy()
            X_all_features.fillna(method='ffill', inplace=True); X_all_features.fillna(method='bfill', inplace=True)
            X_all_features.dropna(axis=1, how='all', inplace=True); X_all_features.dropna(axis=0, how='any', inplace=True)
            y_all_features = df_model_input['target'].loc[X_all_features.index]

            if X_all_features.empty or y_all_features.empty or len(X_all_features) < 30:
                st.error(f"Not enough data after preprocessing ({len(X_all_features)} samples).")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_all_features, y_all_features, test_size=0.2, shuffle=False)
                st.write(f"Overall training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

                model_key_selected = st.session_state.selected_model_type_tab3
                
                # --- Tree Models (XGBoost, LightGBM, CatBoost) ---
                if model_key_selected in ["XGBoost", "LightGBM", "CatBoost"]:
                    st.subheader(f"{model_key_selected} Model")
                    if st.button(f"Train {model_key_selected} Model", key=f"train_{model_key_selected}_final_final_btn"):
                        st.session_state.trained_model_info[model_key_selected] = {} 
                        with st.spinner(f"Training {model_key_selected}..."):
                            try:
                                train_func = getattr(modeling, f"train_{model_key_selected.lower()}", None)
                                model = train_func(X_train, y_train) 
                                st.session_state.trained_model_info[model_key_selected].update({'model': model, 'feature_columns': list(X_train.columns), 'X_test': X_test, 'y_test': y_test})
                                st.success(f"{model_key_selected} trained!")
                            except Exception as e: st.error(f"Error: {e}"); st.session_state.trained_model_info.pop(model_key_selected, None)
                    
                    if st.session_state.get('trained_model_info', {}).get(model_key_selected, {}).get('model'):
                        st.markdown(f"#### Evaluation: {model_key_selected}")
                        info = st.session_state.trained_model_info[model_key_selected]
                        model_eval, X_test_eval, y_test_eval = info['model'], info['X_test'][info['feature_columns']], info['y_test']
                        if 'y_pred_class' not in info or 'y_pred_proba' not in info:
                             with st.spinner("Generating predictions..."):
                                predict_func = getattr(modeling, f"predict_{model_key_selected.lower()}", None)
                                info['y_pred_class'], info['y_pred_proba'] = predict_func(model_eval, X_test_eval)
                        st.text(f"Accuracy: {accuracy_score(y_test_eval, info['y_pred_class']):.4f}")
                        st.text_area("Report:", classification_report(y_test_eval, info['y_pred_class'], zero_division=0), height=200, key=f"{model_key_selected}_report_final_final")
                        col1, col2 = st.columns(2)
                        with col1: fig, ax = plt.subplots(); plot_roc_auc(y_test_eval, info['y_pred_proba'], ax=ax, model_name=model_key_selected); st.pyplot(fig); plt.close(fig)
                        with col2: fig, ax = plt.subplots(); plot_confusion_matrix(y_test_eval, info['y_pred_class'], ax=ax, model_name=model_key_selected); st.pyplot(fig); plt.close(fig)
                        if hasattr(model_eval, 'feature_importances_') or (model_key_selected == "CatBoost" and hasattr(model_eval, 'get_feature_importance')):
                            st.subheader("Feature Importances"); 
                            f_imp_vals = model_eval.feature_importances_ if hasattr(model_eval, 'feature_importances_') else model_eval.get_feature_importance()
                            f_imp = pd.Series(f_imp_vals, index=info['feature_columns']).sort_values(ascending=False).head(15)
                            fig_fi, ax_fi = plt.subplots(figsize=(10,6)); ax_fi.barh(f_imp.index, f_imp.values); ax_fi.set_title(f"Top 15 ({model_key_selected})"); ax_fi.invert_yaxis(); plt.tight_layout(); st.pyplot(fig_fi); plt.close(fig_fi)
                        fn_ext_map = {"XGBoost": ".json", "LightGBM": ".txt", "CatBoost": ".cbm"}
                        if st.button(f"Save {model_key_selected} Model", key=f"save_{model_key_selected}_final_final_btn"): 
                            utils.save_model(model_eval, f"{st.session_state.selected_ticker}_{model_key_selected.lower()}_model{fn_ext_map.get(model_key_selected)}", config.MODELS_DIR); st.success("Model saved.")
                
                # --- Sequence Models (LSTM, CNN-LSTM, Transformer) ---
                elif model_key_selected in ["LSTM", "CNN-LSTM", "Transformer"]:
                    st.subheader(f"{model_key_selected} Model")
                    fit_params_dl = {'epochs': 10, 'batch_size': 32, 'validation_split': 0.1, 'verbose': 0} # Reduced for GUI
                    sequence_length_dl = 20 

                    if st.button(f"Train {model_key_selected} Model", key=f"train_{model_key_selected}_final_final_btn"):
                        st.session_state.trained_model_info[model_key_selected] = {}
                        with st.spinner(f"Training {model_key_selected}... (This can take a while)"):
                            try:
                                train_func = getattr(modeling, f"train_{model_key_selected.lower().replace('-', '')}", None)
                                # Default model_params from modeling.py are used by passing None or {}
                                model, scaler = train_func(X_train.copy(), y_train.copy(), sequence_length=sequence_length_dl, fit_params=fit_params_dl) 
                                st.session_state.trained_model_info[model_key_selected].update({'model': model, 'scaler': scaler, 'sequence_length': sequence_length_dl, 'feature_columns': list(X_train.columns), 'X_test': X_test, 'y_test': y_test})
                                st.success(f"{model_key_selected} trained!")
                            except Exception as e: st.error(f"Error: {e}"); st.session_state.trained_model_info.pop(model_key_selected, None)
                    
                    if st.session_state.get('trained_model_info', {}).get(model_key_selected, {}).get('model'):
                        st.markdown(f"#### Evaluation: {model_key_selected}")
                        info = st.session_state.trained_model_info[model_key_selected]
                        model_eval, scaler_eval, seq_len_eval = info['model'], info['scaler'], info['sequence_length']
                        X_test_eval, y_test_eval = info['X_test'][info['feature_columns']], info['y_test'] # Ensure X_test_eval uses correct columns
                        
                        if 'y_pred_class' not in info or 'y_pred_proba' not in info:
                            with st.spinner("Generating predictions..."):
                                predict_func = getattr(modeling, f"predict_{model_key_selected.lower().replace('-', '')}", None)
                                info['y_pred_class'], info['y_pred_proba'] = predict_func(model_eval, X_test_eval.copy(), scaler_eval, seq_len_eval)
                        
                        y_pred_class_dl = info['y_pred_class']
                        y_pred_proba_dl = info['y_pred_proba']
                        
                        if len(y_pred_class_dl) > 0:
                            y_test_dl_aligned = y_test_eval.iloc[len(y_test_eval) - len(y_pred_class_dl):]
                            st.text(f"Accuracy: {accuracy_score(y_test_dl_aligned, y_pred_class_dl):.4f}")
                            st.text_area("Report:", classification_report(y_test_dl_aligned, y_pred_class_dl, zero_division=0), height=200, key=f"{model_key_selected}_report_final_final")
                            col1, col2 = st.columns(2)
                            with col1: fig, ax = plt.subplots(); plot_roc_auc(y_test_dl_aligned, y_pred_proba_dl, ax=ax, model_name=model_key_selected); st.pyplot(fig); plt.close(fig)
                            with col2: fig, ax = plt.subplots(); plot_confusion_matrix(y_test_dl_aligned, y_pred_class_dl, ax=ax, model_name=model_key_selected); st.pyplot(fig); plt.close(fig)
                        else: st.warning(f"No {model_key_selected} predictions to evaluate (test set too short for sequence?).")
                        
                        if st.button(f"Save {model_key_selected} Model & Scaler", key=f"save_{model_key_selected}_final_final_btn"):
                            model_fn = f"{st.session_state.selected_ticker}_{model_key_selected.lower().replace('-','_')}_model.h5"
                            scaler_fn = f"{st.session_state.selected_ticker}_{model_key_selected.lower().replace('-','_')}_scaler.pkl"
                            utils.save_model(model_eval, model_fn, config.MODELS_DIR)
                            utils.save_model(scaler_eval, scaler_fn, config.MODELS_DIR)
                            st.success(f"{model_key_selected} Model and Scaler saved.")
                else:
                     st.info(f"All model types in this tab have been implemented.")

with tab4: 
    st.header("Backtesting and Visualization")
    st.info("Backtesting controls and results will be implemented here.")

st.sidebar.markdown("---")
st.sidebar.info("App for stock analysis and prediction.")
