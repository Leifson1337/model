# src/exceptions.py

class AppBaseException(Exception):
    """Base class for custom exceptions in this application."""
    def __init__(self, message="An application error occurred.", details=None):
        super().__init__(message)
        self.message = message
        self.details = details if details is not None else {}

    def __str__(self):
        return f"{self.message} (Details: {self.details})"

class ConfigError(AppBaseException):
    """Exception raised for errors in application configuration."""
    def __init__(self, message="Configuration error.", missing_key=None, invalid_value=None, details=None):
        super().__init__(message, details)
        if missing_key:
            self.message = f"Configuration error: Missing key '{missing_key}'."
        elif invalid_value:
            self.message = f"Configuration error: Invalid value for '{invalid_value['key']}' - {invalid_value['value']} (Reason: {invalid_value.get('reason', 'N/A')})."

class DataValidationError(AppBaseException):
    """Exception raised for errors in data validation."""
    def __init__(self, message="Data validation failed.", validation_errors=None, details=None):
        super().__init__(message, details)
        self.validation_errors = validation_errors if validation_errors is not None else {}
        if self.validation_errors:
            self.message = f"Data validation failed. Errors: {self.validation_errors}"

class ModelTrainingError(AppBaseException):
    """Exception raised for errors during model training."""
    def __init__(self, message="Model training failed.", model_name=None, stage=None, details=None):
        super().__init__(message, details)
        self.model_name = model_name
        self.stage = stage
        if model_name and stage:
            self.message = f"Model training failed for '{model_name}' during stage '{stage}'."
        elif model_name:
            self.message = f"Model training failed for '{model_name}'."


class PipelineError(AppBaseException):
    """Exception raised for errors within a data or model pipeline stage."""
    def __init__(self, message="Pipeline error occurred.", stage_name=None, details=None):
        super().__init__(message, details)
        self.stage_name = stage_name
        if stage_name:
            self.message = f"Pipeline error during stage '{stage_name}'."

class FileOperationError(AppBaseException):
    """Exception raised for errors during file operations (read, write, delete)."""
    def __init__(self, message="File operation failed.", filepath=None, operation=None, details=None):
        super().__init__(message, details)
        self.filepath = filepath
        self.operation = operation
        if filepath and operation:
            self.message = f"File operation '{operation}' failed for path '{filepath}'."
        elif filepath:
            self.message = f"File operation failed for path '{filepath}'."

class APIError(AppBaseException):
    """Base exception for API related errors."""
    def __init__(self, message="API error occurred.", status_code=500, details=None):
        super().__init__(message, details)
        self.status_code = status_code

class ExternalServiceError(APIError):
    """Exception for errors when interacting with external services or APIs."""
    def __init__(self, message="External service communication error.", service_name=None, status_code=503, details=None):
        super().__init__(message, status_code, details)
        self.service_name = service_name
        if service_name:
            self.message = f"Error communicating with external service '{service_name}'."

# Example of how to use these exceptions:
#
# def load_config(path):
#     if not os.path.exists(path):
#         raise ConfigError(missing_key=path)
#     # ... further processing ...
#
# def process_data(data):
#     if not data.get("required_field"):
#         raise DataValidationError(validation_errors={"required_field": "is missing"})
#     # ... further processing ...

if __name__ == '__main__':
    # Example usage demonstration
    try:
        raise ConfigError(missing_key="DATABASE_URL")
    except ConfigError as e:
        print(e)

    try:
        raise DataValidationError(validation_errors={"age": "must be a positive integer"}, details={"problem_field": "age"})
    except DataValidationError as e:
        print(e)
        print(e.validation_errors)
        print(e.details)

    try:
        raise ModelTrainingError(model_name="ResNet50", stage="data preprocessing", details={"error_code": 123})
    except ModelTrainingError as e:
        print(e)

    try:
        raise FileOperationError(filepath="/tmp/data.csv", operation="read", details={"os_error": "Permission denied"})
    except FileOperationError as e:
        print(e)

    try:
        raise ExternalServiceError(service_name="AlphaVantage", details={"request_url": "http://example.com"})
    except ExternalServiceError as e:
        print(e)
        print(f"Status Code: {e.status_code}")
