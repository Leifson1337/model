# src/config.py
# API Keys (placeholders - actual keys should not be stored here directly in a real app)
NEWS_API_KEY = "YOUR_NEWS_API_KEY_HERE" 
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_KEY_HERE"

# Data paths
DATA_DIR = "data/"
MODELS_DIR = "models/"
LOGS_DIR = "logs/"

# Default stock tickers for analysis
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA"]

# Default date range
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2023-12-31"

# Logging configuration
LOG_FILE = LOGS_DIR + "app.log" # Corrected to use LOGS_DIR
LOG_LEVEL = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
