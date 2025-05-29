```mermaid
graph TD
    subgraph "User Interfaces"
        CLI["CLI (main.py)"]
        API_Server["API (api/controllers.py, api/models.py)"]
        GUI["Streamlit GUI (gui.py)"]
    end

    subgraph "Core Logic (src/)"
        DM["Data Management (data_management.py)"]
        FE["Feature Engineering (feature_engineering.py)"]
        SA["Sentiment Analysis (sentiment_analysis.py)"]
        FD["Fundamental Data (fundamental_data.py)"]
        MOD["Modeling (modeling.py)"]
        EVAL["Evaluation (evaluation.py)"]
        BT["Backtesting (backtesting.py)"]
        REG["Model Registry (model_registry_utils.py)"]
        FA["Feature Analysis (feature_analysis.py)"]
        UTILS["Cross-cutting (utils.py, exceptions.py)"]
        CONFIG_MODELS["Configuration Models (config_models.py, config.py)"]
    end

    subgraph "Data & Artifacts Storage"
        Config["Configuration Files (config/*.json)"]
        RawData["Raw Data (data/raw/*.parquet)"]
        ProcessedData["Processed Features (data/processed/*.parquet)"]
        StatsFiles["Feature Stats (baseline_stats.json, current_stats.json)"]
        ModelsDir["Trained Models & Scalers (models/MODEL_NAME/VERSION/)"]
        ModelRegistryFile["Model Registry Log (models/model_registry.jsonl)"]
        EvalReports["Evaluation Reports (data/reports/*.json, *.png)"]
        BacktestResults["Backtest Results (data/reports/backtests/*.json)"]
        DriftReports["Drift Reports (data/reports/feature_drift/*.json)"]
        Logs["Application Logs (logs/app.log)"]
    end

    subgraph "External Data Sources"
        NewsAPIService["News API Service"]
        AlphaVantageService["AlphaVantage Service"]
        YFinance["Yahoo Finance (yfinance lib)"]
    end

    %% CLI Interactions
    CLI -->|uses| CONFIG_MODELS
    CLI -->|invokes| DM
    CLI -->|invokes| FE
    CLI -->|invokes| MOD
    CLI -->|invokes| EVAL
    CLI -->|invokes| BT
    CLI -->|invokes| REG
    CLI -->|invokes| FA
    CLI -->|uses| UTILS

    %% API Interactions
    API_Server -->|uses| CONFIG_MODELS
    API_Server -->|delegates to| DM
    API_Server -->|delegates to| FE
    API_Server -->|delegates to| MOD
    API_Server -->|delegates to| EVAL
    API_Server -->|delegates to| BT
    API_Server -->|delegates to| REG
    API_Server -->|delegates to| FA
    API_Server -->|uses| UTILS

    %% GUI Interactions
    GUI -->|constructs & runs CLI commands via| CLI
    %% Potentially, GUI could directly use API if API is running
    %% GUI -->|makes requests to| API_Server

    %% Core Logic Interactions with Data & Artifacts
    CONFIG_MODELS -->|reads| Config
    
    DM -->|uses| YFinance
    DM -->|writes to| RawData
    DM -->|uses| UTILS
    
    FE -->|reads from| RawData
    FE -->|uses| SA
    FE -->|uses| FD
    FE -->|writes to| ProcessedData
    FE -->|writes to| StatsFiles
    FE -->|uses| UTILS

    SA -->|uses| NewsAPIService
    FD -->|uses| AlphaVantageService

    MOD -->|reads from| ProcessedData
    MOD -->|writes to| ModelsDir
    MOD -->|updates| ModelRegistryFile
    MOD -->|uses| REG
    MOD -->|uses| UTILS
    
    EVAL -->|reads from| ProcessedData
    EVAL -->|reads from| ModelsDir
    EVAL -->|writes to| EvalReports
    EVAL -->|updates| ModelRegistryFile
    EVAL -->|uses| REG
    EVAL -->|uses| UTILS

    BT -->|reads from| RawData %% or ProcessedData if signals are feature-based
    BT -->|reads from| ModelsDir %% for predictions or model application
    BT -->|writes to| BacktestResults
    BT -->|uses| UTILS

    REG -->|reads/writes| ModelRegistryFile
    REG -->|reads from| ModelsDir %% to check metadata files

    FA -->|reads from| StatsFiles
    FA -->|writes to| DriftReports
    
    %% Logging - most components write to logs
    CLI -->|writes to| Logs
    API_Server -->|writes to| Logs
    DM -->|writes to| Logs
    FE -->|writes to| Logs
    MOD -->|writes to| Logs
    EVAL -->|writes to| Logs
    BT -->|writes to| Logs
    UTILS -->|provides logging setup for| Logs
```
