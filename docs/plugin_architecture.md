# Plugin Architecture for QLOP

## 1. Introduction and Goals

This document outlines a conceptual design for a plugin architecture for the Quantitative Leverage Opportunity Predictor (QLOP) project. The primary goals of this architecture are:

-   **Extensibility**: Allow new components (model types, data sources, feature engineering steps, backtesting strategies, export formats) to be added with minimal modification to the core codebase.
-   **Modularity**: Decouple specific implementations of components from the core system, making the system easier to maintain and understand.
-   **Contribution**: Facilitate contributions from different developers by providing clear interfaces and integration points for new components.
-   **Customization**: Enable users to easily select and configure different components based on their needs.

## 2. Core Concepts

The plugin system will revolve around clearly defined interfaces for different types of extensible components, a discovery mechanism, and a registration system.

### 2.1. Plugin Types

The system will initially support the following plugin types:

1.  **Model Plugins**: For implementing different types of predictive models (e.g., XGBoost, LSTM, custom neural networks).
2.  **Data Source Plugins**: For fetching data from various sources (e.g., yfinance, AlphaVantage, custom database, local files with specific formats).
3.  **Feature Engineering Step Plugins**: For adding reusable, complex feature transformation steps that can be part of a larger feature engineering pipeline.
4.  **Backtesting Strategy Plugins**: For defining different trading strategies and their logic for backtesting.
5.  **Export Format Plugins**: For exporting trained models or predictions into different formats (e.g., ONNX, PMML, custom binary).

### 2.2. Plugin Interfaces (Base Classes)

Each plugin type will have a corresponding abstract base class (ABC) or a well-defined protocol that plugins must implement. These interfaces will ensure that the core system can interact with any plugin of a given type in a consistent manner.

**Example Base Classes (Conceptual):**

```python
# In a future src/plugins/base.py or similar

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

# Assuming relevant Pydantic config models are defined elsewhere
# from src.config_models import (
#     LoadDataConfig, TrainModelConfig, ExportConfig, 
#     FeatureEngineeringStepConfig, BacktestStrategyConfig
# )

class BaseModelPlugin(ABC):
    """Interface for model plugins."""
    
    @abstractmethod
    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        self.model_name = model_name
        self.model_params = model_params
        self.model = None # To store the trained model instance

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Any:
        """Trains the model. Returns the trained model artifact."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, model_artifact: Optional[Any] = None) -> pd.DataFrame: # Or np.ndarray
        """Makes predictions using the trained model."""
        # model_artifact can be self.model or loaded if not already in instance
        pass

    @abstractmethod
    def save_model(self, model_artifact: Any, path: Path):
        """Saves the trained model artifact."""
        pass

    @abstractmethod
    def load_model(self, path: Path) -> Any:
        """Loads a model artifact from a path."""
        pass

class BaseDataSourcePlugin(ABC):
    """Interface for data source plugins."""

    @abstractmethod
    def __init__(self, source_name: str, config_params: Dict[str, Any]):
        self.source_name = source_name
        self.config_params = config_params # e.g., API keys, paths

    @abstractmethod
    def load_data(self, tickers: List[str], start_date: str, end_date: str, interval: str, **kwargs) -> pd.DataFrame:
        """Loads data from the source. Configuration details passed via __init__ or kwargs."""
        pass

class BaseFeatureStepPlugin(ABC):
    """Interface for individual feature engineering step plugins."""

    @abstractmethod
    def __init__(self, step_name: str, params: Dict[str, Any]):
        self.step_name = step_name
        self.params = params

    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Applies the feature transformation to the data."""
        pass

class BaseStrategyPlugin(ABC):
    """Interface for backtesting strategy plugins."""

    @abstractmethod
    def __init__(self, strategy_name: str, params: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, model_predictions: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """Generates trading signals based on data and/or model predictions."""
        pass
    
    # run_backtest might be part of the core backtesting engine,
    # with the strategy plugin mainly providing signals or specific execution logic.

class BaseExportFormatPlugin(ABC):
    """Interface for model/data export format plugins."""

    @abstractmethod
    def __init__(self, format_name: str, params: Dict[str, Any]):
        self.format_name = format_name
        self.params = params # e.g., compression level, metadata options

    @abstractmethod
    def export(self, artifact_to_export: Any, output_path: Path, **kwargs):
        """Exports the given artifact to the specified path in the plugin's format."""
        pass

```

### 2.3. Discovery Mechanism

The system needs a way to find available plugins. Two common approaches are:

1.  **Entry Points (Recommended for distributable plugins)**:
    -   Python packages can declare "entry points" in their `pyproject.toml` (or `setup.py`/`setup.cfg`).
    -   The main application can then query these entry points for registered plugins under specific group names (e.g., `qlorp.models`, `qlorp.datasources`).
    -   Example `pyproject.toml` for a plugin package:
        ```toml
        [project.entry-points."qlorp.models"]
        my_custom_model = "my_plugin_package.models:MyCustomModelPlugin"
        ```
    -   This method is robust and standard for Python package extensibility.

2.  **Directory Scanning (Simpler for internal plugins)**:
    -   The system scans predefined directories (e.g., `src/plugins/models/`, `src/plugins/datasources/`) for Python modules.
    -   Plugins within these modules could be identified by:
        -   Adhering to a naming convention (e.g., class name ends with `Plugin`).
        -   Implementing the required base class (checked via `issubclass()`).
        -   Having a specific registration decorator.
    -   This is simpler for plugins developed directly within the main project structure.

A hybrid approach can also be used, supporting both entry points and a local plugins directory.

### 2.4. Registration

A central `PluginManager` class will be responsible for discovering, loading, and providing access to plugins.

-   **Loading**: The `PluginManager` will use the chosen discovery mechanism (e.g., `importlib.metadata.entry_points` for entry points, or `os.listdir` and `importlib.import_module` for directory scanning).
-   **Storage**: It will maintain dictionaries mapping plugin names (e.g., "XGBoost", "MyCustomModel") to the loaded plugin classes or instances.
-   **Access**: Core components will request a plugin by its name from the `PluginManager`.

```python
# Conceptual PluginManager
class PluginManager:
    def __init__(self):
        self.model_plugins = {}
        self.datasource_plugins = {}
        # ... other plugin type registries

    def discover_plugins(self):
        # Logic for entry points discovery
        # Logic for directory scanning
        pass

    def get_model_plugin(self, name: str, params: Dict[str, Any]) -> BaseModelPlugin:
        plugin_class = self.model_plugins.get(name)
        if not plugin_class:
            raise ValueError(f"Model plugin '{name}' not found.")
        return plugin_class(model_name=name, model_params=params) # Instantiate with specific params

    # Similar get_xxx_plugin methods for other types
```

### 2.5. Configuration

Plugin-specific configurations will be handled as follows:

-   **Main Configuration**: Core Pydantic configuration models (e.g., `TrainModelConfig`, `LoadDataConfig`) will have a dedicated field for plugin-specific parameters.
    -   For `TrainModelConfig`: `model_type: str` (plugin name) and `model_params: Dict[str, Any]` (parameters for that specific model plugin).
    -   For `LoadDataConfig`: `data_source_type: str` (plugin name) and `data_source_params: Dict[str, Any]`.
-   **Plugin Responsibility**: Each plugin's `__init__` method will receive its specific parameters (e.g., the content of `model_params`). The plugin itself can use Pydantic to validate these parameters if needed.
-   **Core System Role**: The core system reads the main configuration, identifies the chosen plugin by its type/name, and passes the relevant sub-dictionary of parameters to the plugin instance.

### 2.6. Integration with Core System

Core modules like `src/modeling.py`, `src/data_management.py`, etc., will be refactored to use the `PluginManager`.

-   **Example (`src/modeling.py`)**:
    ```python
    # plugin_manager = PluginManager() # Instantiated globally or passed around
    # plugin_manager.discover_plugins()

    def train_model_with_plugin(config: TrainModelConfig, X_train, y_train, ...):
        model_plugin = plugin_manager.get_model_plugin(config.model_type, config.model_params)
        trained_artifact = model_plugin.train(X_train, y_train, ...)
        
        # Core logic might still handle saving via a generic mechanism or delegate to plugin
        # If plugin handles saving:
        # model_plugin.save_model(trained_artifact, Path(config.model_output_path_base) / "model.ext") 
        # Or the core system uses src.utils.save_model which might internally understand the artifact type.
        
        # Metadata generation would still happen in core, potentially enriched by plugin-specific info.
        # ...
    ```

## 3. Workflow Example (Adding a New Model Plugin)

1.  **Implement Interface**: A developer creates `my_new_model_plugin.py` with a class `MyNewModelPlugin(BaseModelPlugin)`.
2.  **Implement Methods**: `__init__`, `train`, `predict`, `save_model`, `load_model` are implemented.
3.  **Register Plugin**:
    -   **Entry Point**: Add to `pyproject.toml` of their plugin package.
    -   **Directory**: Place the file in `src/plugins/models/`.
4.  **Configuration**: User updates their training configuration JSON to specify `model_type: "MyNewModel"` and provides relevant parameters in `model_params`.
5.  **Execution**:
    -   `PluginManager` discovers `MyNewModelPlugin`.
    -   The training script requests "MyNewModel" from the manager.
    -   The manager instantiates `MyNewModelPlugin` with its specific `model_params`.
    -   The training script calls `plugin_instance.train(...)`.

## 4. Benefits

-   **Clear Separation**: Core logic is separated from specific implementations.
-   **Reduced Core Changes**: Adding new components doesn't require modifying core files like `modeling.py` or `data_management.py` extensively, only the plugin registration/discovery and potentially UI elements to select them.
-   **Independent Development**: Plugins can be developed and tested independently.
-   **Community Contributions**: Easier for others to extend the system.

## 5. Future Considerations

-   **Plugin Versioning**: How to handle versions of plugins and their compatibility with the core system.
-   **Dependency Management**: Plugins might have their own dependencies. Using entry points handles this well as plugins can be separate packages.
-   **Security**: If loading third-party plugins, security implications need to be considered.
-   **User Interface**: The GUI and CLI would need to be updated to dynamically list and allow selection of available plugins.

This conceptual design provides a flexible foundation for making QLOP more modular and extensible.

---

## 6. Live GUI Updates with WebSockets (Conceptual)

### 6.1. Problem Statement

Currently, when the Streamlit GUI triggers long-running CLI tasks (e.g., model training, extensive backtests, large data downloads), the GUI might appear unresponsive or lack real-time progress indication. The user only sees the final output or errors after the subprocess completes. This can lead to a poor user experience for time-consuming operations.

### 6.2. Proposed Solution

To address this, a WebSocket-based communication channel can be established to stream logs, progress updates, and status messages from the backend CLI processes directly to the Streamlit GUI in real-time.

### 6.3. Components

1.  **WebSocket Server**:
    *   A lightweight server process, potentially built using Python libraries like `websockets` or integrated within a broader API framework if one is established (e.g., FastAPI WebSockets).
    *   It would manage connections from CLI processes (acting as publishers) and GUI sessions (acting as subscribers).
    *   It could use topics/channels based on a unique `task_id` or `session_id` to route messages correctly from a specific CLI task to the corresponding GUI user.

2.  **CLI Process (WebSocket Client/Publisher)**:
    *   When a CLI command is initiated from the GUI with a request for live updates (e.g., via a specific flag like `--stream-updates <task_id>`), it would establish a WebSocket connection to the server.
    *   Throughout its execution, the CLI command would send messages to the WebSocket server. These messages could include:
        *   Log lines (e.g., from the `logging` module).
        *   Progress updates (e.g., percentage completion, current epoch).
        *   Status changes (e.g., "data download complete", "starting feature engineering").
        *   Paths to intermediate or final artifacts as they are created.

3.  **Streamlit GUI (WebSocket Client/Subscriber)**:
    *   When the GUI launches a CLI task for which live updates are desired, it would also initiate a WebSocket connection to the server, subscribing to the relevant `task_id`.
    *   A dedicated area in the GUI (e.g., an expandable section with `st.text_area` for logs, `st.progress` for progress bars) would be updated dynamically as messages are received over the WebSocket.
    *   JavaScript might be needed within Streamlit (`st.components.v1.html` or custom components) for robust WebSocket client handling and DOM manipulation if Streamlit's native capabilities for this are limited. However, libraries like `streamlit-ws-client` or simple polling mechanisms with `st.experimental_rerun` could be explored for Python-only solutions.

### 6.4. Workflow Example

1.  **Task Initiation**: User clicks "Train Model" in the Streamlit GUI.
2.  **Task ID Generation**: The GUI generates a unique `task_id` (e.g., `uuid.uuid4()`).
3.  **CLI Invocation**: The GUI launches the CLI command: `python main.py train-model --config <path_to_config> --stream-to-ws <task_id>`.
4.  **WebSocket Connections**:
    *   The `train-model` CLI process starts, connects to the WebSocket server, and registers itself with `<task_id>`.
    *   The Streamlit GUI frontend also connects to the WebSocket server and subscribes to messages for `<task_id>`.
5.  **Message Streaming**:
    *   As `train-model` executes, it uses a helper function (potentially integrated with the `logging` system or called explicitly) to send messages:
        ```json
        // Example log message
        {"task_id": "<task_id>", "type": "log", "level": "INFO", "message": "Epoch 1/10 completed, loss: 0.567"}
        // Example progress message
        {"task_id": "<task_id>", "type": "progress", "value": 10, "total": 100, "step": "Epoch"}
        ```
6.  **Message Relaying**: The WebSocket server receives these messages and forwards them to all clients subscribed to `<task_id>` (i.e., the user's GUI session).
7.  **GUI Update**: The Streamlit GUI's WebSocket client receives the messages.
    *   Log messages are appended to a text area.
    *   Progress messages update a progress bar.
    *   Status messages update a status indicator.
8.  **Task Completion**: The CLI command sends a final "completed" or "failed" message and disconnects. The GUI updates accordingly and closes its WebSocket connection for that task.

### 6.5. Message Format (Example)

A simple JSON-based message format could be used:

```json
{
  "task_id": "string (UUID)",
  "timestamp_utc": "string (ISO 8601)",
  "type": "string (e.g., 'log', 'progress', 'status', 'artifact_path')",
  "payload": {
    // Content depends on 'type'
    // For 'log': {"level": "INFO", "message": "Log message here"}
    // For 'progress': {"value": 25, "total": 100, "unit": "epochs", "step_name": "Training"}
    // For 'status': {"current_status": "Feature Engineering complete", "next_step": "Model Training"}
    // For 'artifact_path': {"artifact_name": "trained_model", "path": "/path/to/model.pkl"}
  }
}
```

### 6.6. Challenges

-   **WebSocket Server Management**: Deciding whether to run it as a standalone process, or integrate it (e.g., if the main app becomes a FastAPI server that also serves Streamlit).
-   **State Management**: Handling disconnections and reconnections for both CLI and GUI clients.
-   **Scalability**: For many concurrent users/tasks, the WebSocket server needs to be robust.
-   **Security**: If the WebSocket server is exposed externally, authentication and authorization for publishing and subscribing to task updates would be crucial.
-   **Streamlit Integration**: Implementing a seamless WebSocket client within Streamlit that updates components dynamically might require careful handling of its execution model (e.g., using `st.session_state`, custom components, or newer Streamlit features for background tasks/callbacks).

### 6.7. Benefits

-   **Improved User Experience**: Provides real-time feedback for long-running tasks, making the GUI feel more responsive and informative.
-   **Better Monitoring**: Allows users to see logs and progress directly in the GUI without needing to check console outputs or log files manually during execution.
-   **Foundation for Interactive Features**: Could be extended for more interactive elements if needed in the future.
```
