# Pydantic models for API requests/responses will be defined here
# This layer will provide the data validation and serialization/deserialization
# for the API, ensuring that data exchanged with clients (e.g., a web UI or
# other services) is well-formed and adheres to defined schemas.

# Example:
# from pydantic import BaseModel

# class DataPath(BaseModel):
#     path: str

# class TrainRequest(BaseModel):
#     data_config: DataPath
#     model_params: dict
#     feature_config: dict

# class TrainResponse(BaseModel):
#     model_id: str
#     status: str
#     metrics: dict

# class EvaluateRequest(BaseModel):
#     model_id: str
#     data_config: DataPath
#     evaluation_metrics: list[str]

# class EvaluateResponse(BaseModel):
#     model_id: str
#     metrics: dict

# class BacktestRequest(BaseModel):
#     strategy_config: dict
#     data_config: DataPath
#     model_id: str # Optional, if using a trained model

# class BacktestResponse(BaseModel):
#     backtest_id: str
#     results: dict # e.g., PnL, Sharpe ratio, etc.

# class ExportRequest(BaseModel):
#     model_id: str
#     export_format: str # e.g., 'onnx', 'pickle'
#     destination_path: str

# class ExportResponse(BaseModel):
#     export_path: str
#     status: str
