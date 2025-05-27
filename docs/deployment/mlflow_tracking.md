# MLflow Tracking Setup

This document will outline how MLflow is integrated into the project for experiment tracking and model management.

## Setup

-   Installing MLflow (`pip install mlflow`).
-   Setting up a local MLflow tracking server or connecting to a remote one.
-   Environment variables for MLflow (e.g., `MLFLOW_TRACKING_URI`).

## Usage

-   Logging parameters, metrics, and artifacts during model training (`src/modeling.py`, `src/tuner.py`).
-   Registering models in the MLflow Model Registry.
-   Versioning experiments and models.

(Details on specific `mlflow.log_param`, `mlflow.log_metric`, `mlflow.log_artifact`, `mlflow.sklearn.log_model`, etc. calls will be added.)
