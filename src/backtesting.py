import pandas as pd
import numpy as np
import backtrader as bt
# from ..src.config_models import BacktestConfig # Example for type hinting
# from pydantic import validate_call # For validating inputs

# TODO: Define expected input/output schemas for data_df and predictions_series.
#       Ensure consistency with how predictions are formatted by the modeling module.
# TODO: Log key library versions (backtrader, pandas) for reproducibility.

# --- Existing simple_backtest function (commented out as per instructions) ---
# def simple_backtest(...):
#     ... (entire function content from previous version) ...
# (Assuming the full simple_backtest function was here and is now commented out or removed)


class SignalStrategy(bt.Strategy):
    """
    A Backtrader strategy that converts raw predictions/scores into trading signals
    based on configurable thresholds.
    - Prediction > long_threshold: Go Long
    - Prediction < short_threshold: Go Short
    - Otherwise: Close position / Stay Neutral
    """
    params = (
        ('predictions', None), # Pass the raw prediction Series as a parameter
        ('strategy_config', { # Default strategy configuration
            'long_threshold': 0.5,
            'short_threshold': -0.5,
            'target_percent': 0.95, # Target percentage of portfolio for trades
        }),
    )

    def __init__(self):
        if self.params.predictions is None:
            raise ValueError("Predictions parameter must be provided to SignalStrategy.")
        
        self.raw_predictions = self.params.predictions
        if not isinstance(self.raw_predictions.index, pd.DatetimeIndex):
            try:
                self.raw_predictions.index = pd.to_datetime(self.raw_predictions.index)
            except Exception as e:
                raise ValueError(f"Could not convert predictions index to DatetimeIndex: {e}")

        # Extract parameters from strategy_config
        config = self.params.strategy_config
        self.long_threshold = config.get('long_threshold', 0.5)
        self.short_threshold = config.get('short_threshold', -0.5)
        self.target_percent = config.get('target_percent', 0.95)
        
        if self.long_threshold <= self.short_threshold:
            raise ValueError("long_threshold must be greater than short_threshold.")

        self.order = None # To keep track of pending orders

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()} {txt}') # Optional: print logs

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        current_prediction = self.raw_predictions.get(current_date, np.nan) # Default to NaN if missing

        current_signal = 0 # Default to neutral
        if not np.isnan(current_prediction):
            if current_prediction > self.long_threshold:
                current_signal = 1
            elif current_prediction < self.short_threshold:
                current_signal = -1
        
        current_position_size = self.getposition(self.datas[0]).size

        if self.order: # Check if an order is pending
            return

        if current_signal == 1: # Long signal
            if current_position_size == 0:
                self.log(f'BUY CREATE, Pred: {current_prediction:.2f}, Close: {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=self.target_percent)
            elif current_position_size < 0: # Currently short, close short and go long
                self.log(f'CLOSE SHORT & BUY CREATE, Pred: {current_prediction:.2f}, Close: {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=self.target_percent)
        
        elif current_signal == -1: # Short signal
            if current_position_size == 0:
                self.log(f'SELL CREATE (SHORT), Pred: {current_prediction:.2f}, Close: {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=-self.target_percent)
            elif current_position_size > 0: # Currently long, close long and go short
                self.log(f'CLOSE LONG & SELL CREATE (SHORT), Pred: {current_prediction:.2f}, Close: {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=-self.target_percent)

        elif current_signal == 0: # Neutral signal (or prediction is NaN)
            if current_position_size != 0:
                self.log(f'CLOSE POSITION, Pred: {current_prediction:.2f}, Close: {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=0.0) # Close position

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.Status[order.status]}')

        self.order = None # Reset order status


def run_backtrader_backtest(
    data_df: pd.DataFrame,
    predictions_series: pd.Series, # Expects a Series with datetime index and raw prediction scores
    strategy_config: dict, # Configuration for the strategy (thresholds, target_percent)
    initial_capital: float = 100000.0,
    leverage: float = 1.0, # Note: leverage is now primarily handled by strategy_config's target_percent if sizer is PercentSizer
    commission_bps: float = 2.0, # e.g. 2 bps = 0.02%
    slippage_bps: float = 1.0,   # e.g. 1 bps = 0.01%
    knock_out_long_bps: float = None, # Placeholder, not implemented in this version
    knock_out_short_bps: float = None # Placeholder, not implemented in this version
    ):
    """
    Runs a backtest using Backtrader with the provided data, raw predictions, and strategy configuration.

    Args:
        data_df: Pandas DataFrame with Datetime index and OHLCV columns.
                 # TODO: Validate schema (required columns: Open, High, Low, Close, Volume).
                 'OpenInterest' can be zero if not available.
        predictions_series: Pandas Series with Datetime index and raw prediction scores/values.
                 # TODO: Validate that index aligns with data_df and values are numeric.
        strategy_config: Dictionary containing strategy parameters like 'long_threshold',
                         'short_threshold', 'target_percent'.
                         # TODO: Validate this dict, ideally using a Pydantic model from config_models.
        initial_capital: Starting capital.
        leverage: Leverage factor. Note: The sizer's `percents` parameter is derived from
                  `strategy_config['target_percent']` and this leverage factor.
        commission_bps: Commission in basis points.
        slippage_bps: Slippage in basis points.
        knock_out_long_bps: Placeholder for knock-out level for long positions (not implemented).
        knock_out_short_bps: Placeholder for knock-out level for short positions (not implemented).

    Returns:
        A dictionary of performance metrics.
        Optionally, could also return the cerebro object for plotting if needed.
        # TODO: Formalize the structure of the returned metrics dictionary.
    """
    # TODO: Add try-except block for robust error handling during backtest execution.
    # TODO: Consider caching backtest results if inputs (data, predictions, config) are unchanged.

    if knock_out_long_bps is not None or knock_out_short_bps is not None:
        print("Warning: Knock-out parameters are provided but not implemented in this Backtrader version.")

    # TODO: Validate input DataFrames and Series (e.g., non-empty, correct dtypes, matching indices for predictions and data).
    if data_df.empty:
        print("Error: Input data_df for backtest is empty.")
        return {}, None # Return empty metrics and no cerebro object
    if predictions_series.empty:
        print("Error: Input predictions_series for backtest is empty.")
        return {}, None
    # Further checks: index alignment, OHLCV column presence, numeric predictions.

    try:
        cerebro = bt.Cerebro()
    except Exception as e:
        print(f"Error initializing Backtrader Cerebro: {e}")
        return {}, None

    # Add data feed

    # Add data feed
    if 'OpenInterest' not in data_df.columns:
        data_df['OpenInterest'] = 0
    
    data_df_bt = data_df.copy()
    data_df_bt.columns = [col.lower() for col in data_df_bt.columns]
    
    data_feed = bt.feeds.PandasData(dataname=data_df_bt)
    cerebro.adddata(data_feed)

    # Add strategy, passing predictions and the strategy_config dictionary
    cerebro.addstrategy(SignalStrategy, predictions=predictions_series, strategy_config=strategy_config)

    # Set broker parameters
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission_bps / 10000.0) # Convert bps to decimal
    
    # Slippage
    cerebro.broker.set_slippage_perc(
        perc=slippage_bps / 10000.0, # Convert bps to decimal percentage
        slip_open=True,    # Slippage on market orders at open
        slip_limit=True,   # Slippage on limit orders
        slip_match=True,   # Slippage on matching limit orders at open
        slip_out=True      # Slippage on stop/market orders for exiting
    )

    # Sizer for leverage
    # PercentSizer applies to the total portfolio value.
    # `percents` dictates what percentage of portfolio to allocate to a new position.
    # Leverage here means we are willing to use more than 100% of current cash for a position,
    # up to `leverage * 100%` of portfolio value (e.g., 200% for 2x leverage).
    # However, PercentSizer's `percents` is typically < 100.
    # A common way to use leverage with PercentSizer is to size as if for a larger portfolio.
    # Or, more directly, Backtrader has setmargin() for futures-like leverage.
    # For this simplified version, we'll use PercentSizer with a target percentage
    # that's amplified by leverage. Max `percents` is 100.
    # So, if target_percent_trade is 95, and leverage is 2, it means we want to use 190% of portfolio.
    # This is tricky with PercentSizer alone. A value like 95 means 95% of current portfolio equity.
    # True leverage usually involves borrowing.
    # For now, we'll use a simpler interpretation: the 'percents' param of sizer is scaled by leverage.
    # This means the sizer will attempt to allocate `target_trade_percent * leverage` of portfolio.
    # This might not be standard if `target_trade_percent * leverage > 100`.
    # A more standard PercentSizer use would be `percents=95` (trade with 95% of equity), and leverage
    # would be managed by `setmargin` or by a custom sizer.
    # For this task, we'll use the target_percent from strategy_config, potentially amplified by leverage.
    # The SignalStrategy itself uses strategy_config['target_percent'] for order_target_percent.
    # The sizer here ensures the strategy doesn't try to allocate more than possible.
    # A common approach is to let the strategy define its desired target size (e.g. via order_target_percent)
    # and the sizer manages the actual allocation based on cash/portfolio value.
    # If strategy_config['target_percent'] is, say, 0.95 (95%), and leverage is 1,
    # then up to 95% of equity is used. If leverage is 2, it could mean 190% if margin is available.
    # Backtrader's PercentSizer applies to current portfolio equity.
    # We can make `effective_sizer_percent` dependent on `strategy_config['target_percent']` and `leverage`.
    # This makes `leverage` a multiplier on the strategy's defined `target_percent`.
    
    # The strategy's internal self.target_percent IS the percentage of portfolio to use for a trade.
    # The sizer's `percents` parameter is what PercentSizer uses.
    # If the strategy issues `order_target_percent(target=self.target_percent)`,
    # this target is relative to current portfolio value.
    # The leverage parameter in run_backtrader_backtest can inform the sizer or broker margin.
    # For simplicity with PercentSizer:
    # Let strategy's target_percent be the primary guide. Leverage can scale this if desired.
    # The strategy's `target_percent` is already in `strategy_config`.
    # `effective_sizer_percent` could be `strategy_config.get('target_percent', 0.95) * leverage`
    # capped at 99 for safety with PercentSizer.
    
    # The strategy itself uses `self.order_target_percent(target=self.target_percent)`
    # or `target=-self.target_percent`. This function in Backtrader directly calculates
    # the number of shares/contracts to reach that percentage of the current portfolio value.
    # So, the `leverage` parameter in `run_backtrader_backtest` is somewhat redundant if
    # `strategy_config['target_percent']` is already considered leveraged (e.g. > 1.0 for futures).
    # However, if `target_percent` is for a single asset in a multi-asset portfolio, or if it's
    # meant to be a fraction of allocated capital which is itself leveraged, it gets complex.

    # Sticking to a simple interpretation for now:
    # The strategy defines its trade size with `strategy_config['target_percent']`.
    # `bt.sizers.PercentSizer` with `percents=strategy_config['target_percent']*100` would be redundant
    # because `order_target_percent` already does this.
    # If we want to apply a global leverage that might allow `target_percent` to effectively exceed 100%
    # of initial capital (e.g. through margin), that's usually set on the broker.
    # `cerebro.broker.setmargin(margin)` for futures.
    # For now, the sizer will just use the strategy's defined target_percent directly.
    # The `leverage` parameter to `run_backtrader_backtest` will be used to scale the
    # `percents` for the sizer, to ensure it aligns with the strategy's intent if `target_percent`
    # is meant to be a fraction of *leveraged* capital.
    
    # Let's assume strategy_config['target_percent'] is the desired portfolio fraction (e.g., 0.95 for 95%).
    # The `leverage` variable scales this.
    sizer_target_percent_effective = strategy_config.get('target_percent', 0.95) * leverage
    # Cap at 0.99 because 1.0 (100%) can sometimes lead to issues if commissions/slippage are involved.
    # Or, if leverage implies >100% of portfolio, this sizer is not the way to achieve that directly.
    # True margin trading would be `cerebro.broker.setmargin()`.
    # For now, we'll assume PercentSizer is used and `target_percent` in strategy means % of current equity.
    # The strategy uses `order_target_percent` which is usually sufficient.
    # Adding a sizer like PercentSizer ensures that if the strategy *didn't* use `order_target_percent`
    # but used `order_target_value` or `buy()`, the sizer would enforce the percentage.
    # Since SignalStrategy *does* use `order_target_percent`, an explicit sizer here
    # with `percents` derived from the strategy's own `target_percent` is mostly for ensuring consistency
    # or if we wanted to globally cap allocation.
    
    # Let's use the `strategy_config['target_percent']` directly for the sizer,
    # assuming it's a value like 95 for 95%.
    # The `leverage` parameter is more of a global setting; its interaction with
    # `PercentSizer` can be tricky. If `leverage > 1`, it implies we might want to
    # trade more than our cash. This usually means setting margin requirements.
    # For this iteration, we'll use `strategy_config['target_percent']` in the sizer,
    # and acknowledge `leverage` is not fully integrated with margin here.
    effective_sizer_percents = strategy_config.get('target_percent', 0.95) * 100 # Convert to percentage for sizer
    effective_sizer_percents = min(effective_sizer_percents * leverage, 99.0) # Apply leverage and cap

    cerebro.addsizer(bt.sizers.PercentSizer, percents=effective_sizer_percents)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn') # System Quality Number
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)


    # Run the backtest
    results = cerebro.run()
    strat_results = results[0] # Get results from the first (and only) strategy

    # Extract metrics
    metrics = {}
    metrics['Initial Capital'] = initial_capital
    metrics['Final Portfolio Value'] = cerebro.broker.getvalue()
    metrics['Total Return (%)'] = (metrics['Final Portfolio Value'] / initial_capital - 1) * 100
    
    sharpe_analysis = strat_results.analyzers.sharpe.get_analysis()
    metrics['Sharpe Ratio'] = sharpe_analysis.get('sharperatio', 0.0) if sharpe_analysis else 0.0

    annual_return_analysis = strat_results.analyzers.annual_return.get_analysis()
    # AnnualReturn gives a dict {year: return}, so we might want an average or specific year
    # For simplicity, let's try to get an overall annualized figure if possible, or average.
    # Backtrader's AnnualReturn analyzer returns a dictionary of returns for each year.
    # We'll calculate a compounded annual growth rate (CAGR) like for Calmar.
    num_years = (data_df.index[-1] - data_df.index[0]).days / 365.25
    if num_years > 0 and metrics['Final Portfolio Value'] > 0:
         cagr = ((metrics['Final Portfolio Value'] / initial_capital)**(1/num_years) - 1) * 100
    elif metrics['Final Portfolio Value'] <=0:
        cagr = -100.0
    else:
        cagr = 0.0
    metrics['Annualized Return (%)'] = cagr

    drawdown_analysis = strat_results.analyzers.drawdown.get_analysis()
    metrics['Max Drawdown (%)'] = drawdown_analysis.get('max', {}).get('drawdown', 0.0)

    trade_analysis = strat_results.analyzers.trade_analyzer.get_analysis()
    metrics['Number of Trades'] = trade_analysis.get('total', {}).get('total', 0)
    metrics['Number of Winning Trades'] = trade_analysis.get('won', {}).get('total', 0)
    metrics['Number of Losing Trades'] = trade_analysis.get('lost', {}).get('total', 0)
    metrics['Win Rate (%)'] = (metrics['Number of Winning Trades'] / metrics['Number of Trades'] * 100) if metrics['Number of Trades'] > 0 else 0.0
    metrics['Gross Profit'] = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0.0)
    metrics['Gross Loss'] = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0.0)) # abs for gross loss
    if metrics['Gross Loss'] > 0:
        metrics['Profit Factor'] = metrics['Gross Profit'] / metrics['Gross Loss']
    else:
        metrics['Profit Factor'] = np.inf if metrics['Gross Profit'] > 0 else 0.0
    
    metrics['Average Trade P&L'] = trade_analysis.get('pnlnet', {}).get('average', 0.0) # Net P&L per trade

    # Calmar Ratio
    if metrics['Max Drawdown (%)'] != 0:
        metrics['Calmar Ratio'] = metrics['Annualized Return (%)'] / metrics['Max Drawdown (%)']
    else:
        metrics['Calmar Ratio'] = np.nan if metrics['Annualized Return (%)'] !=0 else 0.0

    # Sortino Ratio (Simplified - using portfolio daily returns)
    # To do this properly, we need daily portfolio values from the backtest.
    # `Returns` analyzer gives daily returns of the portfolio.
    returns_analysis = strat_results.analyzers.returns.get_analysis()
    daily_returns_series = pd.Series(returns_analysis.values()) # dict_values to series
    
    if not daily_returns_series.empty:
        downside_returns = daily_returns_series[daily_returns_series < 0]
        expected_return = daily_returns_series.mean()
        downside_std = downside_returns.std()
        if downside_std != 0 and not np.isnan(downside_std):
            # Assuming risk_free_rate = 0 for Sortino for simplicity
            metrics['Sortino Ratio (Portfolio Daily)'] = (expected_return / downside_std) * np.sqrt(252) 
        else:
            metrics['Sortino Ratio (Portfolio Daily)'] = 0.0 if expected_return == 0 else np.nan 
            # If expected_return is non-zero but downside_std is zero (no losses), sortino is inf. Capping at a large number or np.nan.
            if expected_return > 0 and downside_std == 0: metrics['Sortino Ratio (Portfolio Daily)'] = np.inf

    else:
        metrics['Sortino Ratio (Portfolio Daily)'] = 0.0

    metrics['SQN'] = strat_results.analyzers.sqn.get_analysis().get('sqn', 0.0)
    
    # Note: Knock-outs are not implemented in this Backtrader version.
    metrics['Number of Knock-Outs'] = "N/A (Not implemented in Backtrader version)"

    return metrics, cerebro # Return cerebro for optional plotting

# Placeholder for walk_forward_test function stub
def walk_forward_test(data_df: pd.DataFrame, model: any, walk_forward_config: dict, strategy_config: dict) -> dict:
    """
    Stub for performing walk-forward testing of a trading strategy.

    In walk-forward testing, the model is periodically retrained on a sliding or expanding window
    of past data and then tested on the subsequent out-of-sample period. This process is repeated
    over the entire dataset.

    Args:
        data_df: Pandas DataFrame with historical market data (OHLCV).
        model: The predictive model to be used. This could be a trained model object,
               a function to train a model, or a path to a model.
        walk_forward_config: Dictionary defining parameters for the walk-forward process, e.g.:
            - 'train_period_size': Length of the training window.
            - 'test_period_size': Length of the out-of-sample testing window.
            - 'step_size': How much the window slides forward for the next iteration.
            - 'retrain_every_test_period': Boolean, if true, retrain before each test period.
        strategy_config: Dictionary defining the trading strategy parameters (e.g., thresholds)
                         to be used with the model's predictions.

    Returns:
        A dictionary containing aggregated performance metrics from all walk-forward periods
        and potentially individual period results. For this stub, returns placeholder results.
    """
    print(f"\n[STUB] walk_forward_test called.")
    print(f"  Data shape: {data_df.shape}")
    print(f"  Model: {model}")
    print(f"  Walk-forward config: {walk_forward_config}")
    print(f"  Strategy config: {strategy_config}")
    
    # Placeholder logic:
    # 1. Loop through data based on train/test splits defined by walk_forward_config.
    # 2. In each loop:
    #    a. Train/retrain `model` on the current training slice of `data_df`.
    #    b. Generate predictions using the trained `model` on the test slice.
    #    c. Run a backtest (e.g., using `run_backtrader_backtest`) on the test slice
    #       with these predictions and the `strategy_config`.
    #    d. Collect performance metrics for this period.
    # 3. Aggregate metrics.

    return {
        "status": "walk_forward_test stub executed",
        "total_return_percent": 10.0, # Placeholder
        "sharpe_ratio": 0.5,          # Placeholder
        "num_periods": (len(data_df) - walk_forward_config.get('train_period_size',0)) // walk_forward_config.get('test_period_size',1) if walk_forward_config.get('test_period_size',1) > 0 else 0
    }

# Placeholder for dynamic_rolling_window_evaluation function stub
def dynamic_rolling_window_evaluation(data_df: pd.DataFrame, model: any, window_params: dict, strategy_config: dict) -> dict:
    """
    Stub for performing dynamic rolling-window evaluation of a trading strategy.

    This involves evaluating the strategy's performance over a series of rolling windows,
    which can help assess its stability and adaptability over time. The model might be
    retrained at each window or use a pre-trained model.

    Args:
        data_df: Pandas DataFrame with historical market data (OHLCV).
        model: The predictive model or a function to obtain predictions.
        window_params: Dictionary defining parameters for the rolling window, e.g.:
            - 'window_size': The length of each evaluation window.
            - 'step_size': How much the window slides forward for the next evaluation.
            - 'retrain_model_per_window': Boolean, whether to retrain the model for each window.
        strategy_config: Dictionary defining the trading strategy parameters.

    Returns:
        A dictionary containing aggregated or time-series performance metrics
        from the rolling window evaluations. For this stub, returns placeholder results.
    """
    print(f"\n[STUB] dynamic_rolling_window_evaluation called.")
    print(f"  Data shape: {data_df.shape}")
    print(f"  Model: {model}")
    print(f"  Window parameters: {window_params}")
    print(f"  Strategy config: {strategy_config}")

    # Placeholder logic:
    # 1. Iterate through `data_df` using a rolling window defined by `window_params`.
    # 2. For each window:
    #    a. Optionally retrain `model` or use a pre-trained one.
    #    b. Generate predictions for the current window.
    #    c. Run a backtest on this window's data with the predictions and `strategy_config`.
    #    d. Record performance metrics for this window.
    # 3. Aggregate or present metrics as a time series.

    return {
        "status": "dynamic_rolling_window_evaluation stub executed",
        "average_sharpe_over_windows": 0.6, # Placeholder
        "performance_stability_metric": 0.8 # Placeholder
    }

if __name__ == '__main__':
    # Generate dummy OHLCV data
    np.random.seed(42)
    num_periods = 252 * 2 # 2 years of daily data
    dates = pd.date_range(start='2022-01-01', periods=num_periods, freq='B')
    
    data = pd.DataFrame(index=dates)
    data['Open'] = 100 + np.random.randn(num_periods).cumsum() * 0.5
    data['High'] = data['Open'] + np.random.rand(num_periods) * 2
    data['Low'] = data['Open'] - np.random.rand(num_periods) * 2
    data['Close'] = (data['Open'] + data['High'] + data['Low']) / 3 # Simplistic close
    data['Volume'] = np.random.randint(100000, 500000, size=num_periods)
    
    # Ensure High is max and Low is min
    data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
    data['OpenInterest'] = 0 # Dummy OpenInterest

    # Generate dummy predictions (raw scores, e.g., from a model)
    # These are not 1, 0, -1 signals yet. They are scores to be thresholded by the strategy.
    # Example: scores between -1.5 and 1.5
    prediction_values = np.random.randn(num_periods) * 0.8 
    predictions_series = pd.Series(prediction_values, index=dates, name="prediction_score")
    
    # Ensure first prediction is not neutral to initiate a trade for testing if desired
    if len(predictions_series) > 1:
        if predictions_series.iloc[0] > -0.1 and predictions_series.iloc[0] < 0.1: # if close to zero
             predictions_series.iloc[0] = 0.6 # Make it a buy signal based on default thresholds

    print("--- Example Backtrader Backtest with Refactored Strategy ---")
    initial_cap = 100000.0

    # Define strategy configurations
    strategy_config_1 = {
        'long_threshold': 0.5,
        'short_threshold': -0.5,
        'target_percent': 0.95, # Use 95% of portfolio for trades
    }
    
    strategy_config_2 = {
        'long_threshold': 0.2,
        'short_threshold': -0.2,
        'target_percent': 0.50, # Use 50% of portfolio for trades
    }

    print("\nTest 1: Strategy Config 1 (Leverage 1x, default costs)")
    metrics1, cerebro1 = run_backtrader_backtest(
        data_df=data.copy(), 
        predictions_series=predictions_series.copy(),
        strategy_config=strategy_config_1,
        initial_capital=initial_cap,
        leverage=1.0 # Leverage applied to sizer's percent
    )
    print("Performance (Test 1 - Strategy Config 1):")
    for metric, value in metrics1.items(): print(f"  {metric}: {value}")

    print("\nTest 2: Strategy Config 2 (Leverage 1.5x, Higher Commission 5bps)")
    metrics2, cerebro2 = run_backtrader_backtest(
        data_df=data.copy(), 
        predictions_series=predictions_series.copy(),
        strategy_config=strategy_config_2,
        initial_capital=initial_cap,
        leverage=1.5, # Leverage applied to sizer's percent
        commission_bps=5.0,
        slippage_bps=2.0
    )
    print("Performance (Test 2 - Strategy Config 2):")
    for metric, value in metrics2.items(): print(f"  {metric}: {value}")
    
    # Optional: Plotting (might not work in all environments directly from script)
    # try:
    #     print("\nPlotting for Test 1 (Strategy Config 1)...")
    #     cerebro1.plot(style='candlestick', barup='green', bardown='red')
    # except Exception as e:
    #     print(f"Could not plot: {e}")

    print("\n--- Testing New Stub Functions ---")
    
    # Dummy data for stub testing
    dummy_model = "dummy_model_object_or_path"
    # Corrected keys to match stub's expectations
    dummy_walk_forward_config = {
        "train_period_size": 100, 
        "test_period_size": 30, 
        "step_size": 30 # step_size is mentioned in docstring, good to have
    }
    dummy_rolling_eval_config = {
        "window_size": 60, 
        "step_size": 30, # step_size is mentioned in docstring
        "evaluation_metric": "sharpe"
    }

    print("\nTesting walk_forward_test stub:")
    wf_results = walk_forward_test(
        data_df=data.copy(), 
        model=dummy_model, 
        walk_forward_config=dummy_walk_forward_config,
        strategy_config=strategy_config_1 # Reuse strategy config for consistency
    )
    print(f"Walk-forward test results: {wf_results}")

    print("\nTesting dynamic_rolling_window_evaluation stub:")
    rolling_results = dynamic_rolling_window_evaluation(
        data_df=data.copy(), 
        model=dummy_model, 
        window_params=dummy_rolling_eval_config,
        strategy_config=strategy_config_1 # Reuse strategy config
    )
    print(f"Dynamic rolling window evaluation results: {rolling_results}")
