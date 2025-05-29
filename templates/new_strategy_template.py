# templates/new_strategy_template.py
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod

# This is a conceptual base class. In a real implementation, this would be 
# defined in a central location like src/plugins/base.py or similar.
class BaseStrategyPlugin(ABC):
    """
    Abstract Base Class for backtesting strategy plugins.
    
    Each strategy plugin should inherit from this class and implement its methods.
    The plugin is instantiated with its specific parameters, typically derived from
    the `strategy_config` section of the `BacktestConfig`.
    """

    @abstractmethod
    def __init__(self, strategy_name: str, params: Dict[str, Any]):
        """
        Initialize the strategy plugin.

        Args:
            strategy_name (str): The name of this strategy.
            params (Dict[str, Any]): A dictionary of parameters specific to this strategy.
                                     The plugin should validate and use these parameters.
                                     Example: {"long_threshold": 0.7, "short_threshold": 0.3, "stop_loss_pct": 0.05}
        """
        self.strategy_name = strategy_name
        self.params = params
        self._validate_params()

    def _validate_params(self):
        """
        (Optional but Recommended) Validate the params passed during initialization.
        Raise ValueError or a custom exception for invalid parameters.
        """
        # Example:
        # if "long_threshold" not in self.params or not (0 < self.params["long_threshold"] < 1):
        #     raise ValueError(f"Parameter 'long_threshold' must be between 0 and 1 for {self.strategy_name}")
        # if "short_threshold" not in self.params or not (0 < self.params["short_threshold"] < 1): # Assuming short_threshold is positive if it's a probability
        #     raise ValueError(f"Parameter 'short_threshold' must be between 0 and 1 for {self.strategy_name}")
        # if self.params.get("long_threshold", 1) <= self.params.get("short_threshold", 0): # Example cross-param validation
        #      raise ValueError("'long_threshold' must be greater than 'short_threshold'")
        pass

    @abstractmethod
    def generate_signals(self, ohlcv_data: pd.DataFrame, model_predictions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generates trading signals based on OHLCV data and/or model predictions.

        Args:
            ohlcv_data (pd.DataFrame): DataFrame with at least 'Open', 'High', 'Low', 'Close', 'Volume' columns.
                                       The index should be a DatetimeIndex.
            model_predictions (Optional[pd.DataFrame]): DataFrame containing model predictions, indexed
                                                       identically to ohlcv_data. This might include
                                                       probabilities or predicted price movements.

        Returns:
            pd.DataFrame: A DataFrame with the same index as `ohlcv_data`, containing at least a 'signal' column.
                          Signal values could be:
                          -  1: Long entry/hold long
                          - -1: Short entry/hold short
                          -  0: Neutral/exit position
                          Additional columns like 'confidence', 'stop_loss_price', 'take_profit_price'
                          can also be included if the strategy generates them.
                          Example output column: 'signal'
        """
        # Example placeholder logic:
        # print(f"Generating signals for strategy '{self.strategy_name}' with params: {self.params}")
        # print(f"OHLCV data shape: {ohlcv_data.shape}")
        # if model_predictions is not None:
        #     print(f"Model predictions shape: {model_predictions.shape}")
        #
        # signals_df = pd.DataFrame(index=ohlcv_data.index)
        # signals_df['signal'] = 0 # Default to neutral
        #
        # # Example: Simple threshold-based strategy using model predictions
        # # Assume model_predictions has a 'probability_long' column
        # if model_predictions is not None and 'probability_long' in model_predictions.columns:
        #     long_thresh = self.params.get('long_threshold', 0.7)
        #     short_thresh = self.params.get('short_threshold', 0.3) # Assuming this means prob_long < short_thresh
        #
        #     signals_df.loc[model_predictions['probability_long'] > long_thresh, 'signal'] = 1
        #     signals_df.loc[model_predictions['probability_long'] < short_thresh, 'signal'] = -1 # If strategy involves shorting
        # else:
        #     # Example: Simple moving average crossover if no predictions
        #     if 'Close' in ohlcv_data.columns:
        #         short_window = self.params.get('short_ma_window', 20)
        #         long_window = self.params.get('long_ma_window', 50)
        #         if short_window < long_window:
        #             sma_short = ohlcv_data['Close'].rolling(window=short_window).mean()
        #             sma_long = ohlcv_data['Close'].rolling(window=long_window).mean()
        #             signals_df.loc[sma_short > sma_long, 'signal'] = 1
        #             signals_df.loc[sma_short < sma_long, 'signal'] = -1
        #
        # print(f"Generated signals: {signals_df['signal'].value_counts().to_dict()}")
        # return signals_df
        raise NotImplementedError(f"Generate_signals method not implemented by plugin '{self.strategy_name}'.")

    # The actual backtesting execution (calculating P&L, Sharpe, etc.) might be handled
    # by a core backtesting engine in `src/backtesting.py`. This engine would take the
    # OHLCV data and the signals DataFrame generated by this plugin.
    #
    # However, a strategy plugin *could* also implement its own P&L calculation logic
    # if it needs very specific handling not covered by a generic engine.
    # For now, we assume the main role is signal generation.

# Example of how a concrete strategy plugin might look:
#
# class MovingAverageCrossoverStrategy(BaseStrategyPlugin):
#     def __init__(self, strategy_name: str, params: Dict[str, Any]):
#         super().__init__(strategy_name, params)
#
#     def _validate_params(self):
#         super()._validate_params()
#         if not all(k in self.params for k in ["short_window", "long_window", "ohlc_column_to_use"]):
#             raise ValueError("Parameters 'short_window', 'long_window', and 'ohlc_column_to_use' are required.")
#         if self.params["short_window"] >= self.params["long_window"]:
#             raise ValueError("'short_window' must be less than 'long_window'.")
#
#     def generate_signals(self, ohlcv_data: pd.DataFrame, model_predictions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
#         signals_df = pd.DataFrame(index=ohlcv_data.index)
#         signals_df['signal'] = 0
#
#         col = self.params["ohlc_column_to_use"] # e.g., "Close"
#         short_w = self.params["short_window"]
#         long_w = self.params["long_window"]
#
#         if col not in ohlcv_data.columns:
#             print(f"Warning for '{self.strategy_name}': Column '{col}' not found. Cannot generate signals.")
#             return signals_df
#
#         sma_short = ohlcv_data[col].rolling(window=short_w, min_periods=short_w).mean()
#         sma_long = ohlcv_data[col].rolling(window=long_w, min_periods=long_w).mean()
#
#         # Buy signal: short MA crosses above long MA
#         signals_df.loc[(sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1)), 'signal'] = 1
#         # Sell signal: short MA crosses below long MA
#         signals_df.loc[(sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1)), 'signal'] = -1
#
#         # Hold signal (maintain position) - can be more complex, e.g. fill forward
#         # signals_df['signal'] = signals_df['signal'].ffill().fillna(0) # Example to hold signals
#
#         print(f"Strategy '{self.strategy_name}' generated signals: {signals_df['signal'].value_counts().to_dict()}")
#         return signals_df

# How this might be used in src/backtesting.py:
#
# def run_backtest_with_plugin(
#     ohlcv_data: pd.DataFrame, 
#     strategy_plugin_name: str, 
#     strategy_params: Dict[str, Any], 
#     plugin_manager,
#     model_predictions: Optional[pd.DataFrame] = None # If strategy uses predictions
# ) -> Dict[str, Any]:
#
#     strategy_plugin = plugin_manager.get_strategy_plugin(strategy_plugin_name, strategy_params)
#     signals_df = strategy_plugin.generate_signals(ohlcv_data, model_predictions)
#
#     # Core backtesting engine uses signals_df and ohlcv_data to calculate P&L, metrics, etc.
#     # results = core_backtest_engine(ohlcv_data, signals_df, transaction_cost_pct=0.001, ...)
#     # return results
#     return {"status": "conceptual backtest completed", "strategy_used": strategy_plugin_name}
```
