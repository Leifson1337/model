# src/config.py

# This file is intended for simple, global constants or default values
# that are NOT expected to change between environments (dev, test, prod).
# For environment-specific configurations or complex structures,
# prefer using JSON/YAML files parsed by Pydantic models in src/config_models.py.

# Example: Default Tickers if no config is provided by user
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA"]

# Example: Default Date Range if not specified in a config
DEFAULT_START_DATE = "2020-01-01" # YYYY-MM-DD format
DEFAULT_END_DATE = "2023-12-31"   # YYYY-MM-DD format

# --- DEPRECATION NOTICE for some variables ---
# The following variables might be better managed by the Pydantic-based configuration system
# (e.g., loaded from dev.json, test.json, prod.json via GlobalAppConfig in src/config_models.py).
# Keeping them here might be for legacy reasons or for truly global, non-environment-specific defaults.
# Consider migrating them to your JSON configs and accessing via a loaded GlobalAppConfig instance.

# API Keys (placeholders - actual keys should NOT be stored here directly in a real app)
# These should ideally be loaded from environment variables or a secure vault,
# and then populated into a Pydantic config model instance.
NEWS_API_KEY = "YOUR_NEWS_API_KEY_HERE" # TODO: Load from env var (e.g., os.getenv('NEWS_API_KEY'))
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_KEY_HERE" # TODO: Load from env var

# Data paths - These are often environment-dependent and better suited for JSON configs + DataPathsConfig Pydantic model.
DATA_DIR = "data/"          # Base data directory
MODELS_DIR = "models/"      # Base models directory
LOGS_DIR = "logs/"          # Base logs directory

# Logging configuration - Also often environment-dependent.
# These settings are used by src/utils.py:setup_logging()
LOG_FILENAME = "app.log"    # Log filename
LOG_LEVEL = "INFO"          # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_MAX_BYTES = 10 * 1024 * 1024  # Max log file size (10MB)
LOG_BACKUP_COUNT = 5        # Number of backup log files to keep

# --- End DEPRECATION NOTICE ---

# It's recommended to minimize the use of this global config.py for complex settings.
# Instead, encourage use of:
# 1. Environment variables for secrets (API keys, DB URLs).
# 2. JSON/YAML config files for parameters, paths, and pipeline settings, validated by Pydantic models.
#    These can be loaded into a single GlobalAppConfig object at application startup (e.g., in main.py or app initialization).

# Example of how this file might be simplified in the future:
#
# import os
#
# # Secrets loaded from environment
# NEWS_API_KEY = os.getenv("NEWS_API_KEY", "default_if_not_set_and_dev_mode")
#
# # Global constants (if any truly global and non-configurable ones exist)
# SOME_PROJECT_CONSTANT = "value"
#
# # Default values that might be overridden by specific Pydantic configs
# DEFAULT_MODEL_TYPE = "XGBoost"

# Ensure .gitkeep files are present in directories if they are meant to be tracked by Git when empty.
# For example, if LOGS_DIR is "logs/", make sure "logs/.gitkeep" exists if you want the "logs" directory in Git.
