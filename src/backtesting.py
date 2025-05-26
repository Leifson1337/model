import pandas as pd
import numpy as np

def simple_backtest(price_data: pd.Series, 
                    signals: pd.Series, 
                    initial_capital: float, 
                    leverage: float = 1.0) -> tuple[pd.Series, dict]:
    """
    Simulates a trading strategy and calculates performance.

    Args:
        price_data: Pandas Series of historical prices (e.g., 'Close').
        signals: Pandas Series of trading signals (1: Long, -1: Short, 0: Neutral).
                 Index must align with price_data.
        initial_capital: Starting capital for the backtest.
        leverage: Leverage to apply (default 1.0).

    Returns:
        A tuple containing:
            - equity_curve: Pandas Series representing portfolio value over time.
            - performance_metrics: Dictionary of performance metrics.
    """
    if not isinstance(price_data, pd.Series):
        raise ValueError("price_data must be a pandas Series.")
    if not isinstance(signals, pd.Series):
        raise ValueError("signals must be a pandas Series.")
    if not price_data.index.equals(signals.index):
        raise ValueError("price_data and signals must have the same index.")

    capital = initial_capital
    position_size = 0  # Number of shares/units
    entry_price = 0
    current_position_type = 0  # 0: None, 1: Long, -1: Short

    equity_curve = pd.Series(index=price_data.index, dtype=float)
    equity_curve.iloc[0] = initial_capital
    
    trades = [] # To store individual trade outcomes (profit/loss)

    for i in range(len(price_data)):
        current_price = price_data.iloc[i]
        signal = signals.iloc[i]
        
        # Update equity based on current holdings before processing new signal
        if current_position_type == 1: # Long position
            unrealized_pnl = (current_price - entry_price) * position_size
            capital_at_current_price = (entry_price * position_size) + unrealized_pnl # More direct way: current_price * position_size
        elif current_position_type == -1: # Short position
            unrealized_pnl = (entry_price - current_price) * position_size
            capital_at_current_price = (entry_price * position_size) + unrealized_pnl # More direct way: (2 * entry_price - current_price) * position_size
        else: # No position
            unrealized_pnl = 0
            capital_at_current_price = capital
        
        equity_curve.iloc[i] = capital_at_current_price # This is before any trade action for the current day

        # Process Signal
        # 1. If Neutral signal or changing position, close current position first
        if (signal == 0 or (signal == 1 and current_position_type == -1) or \
           (signal == -1 and current_position_type == 1)) and current_position_type != 0:
            
            if current_position_type == 1: # Closing Long
                trade_pnl = (current_price - entry_price) * position_size
                capital += trade_pnl
                trades.append(trade_pnl)
            elif current_position_type == -1: # Closing Short
                trade_pnl = (entry_price - current_price) * position_size
                capital += trade_pnl
                trades.append(trade_pnl)
            
            position_size = 0
            entry_price = 0
            current_position_type = 0
            equity_curve.iloc[i] = capital # Update equity after closing position

        # 2. Open new position if signal is Long or Short and no conflicting position
        if signal == 1 and current_position_type == 0: # Open Long
            position_size = (capital * leverage) / current_price
            entry_price = current_price
            current_position_type = 1
        elif signal == -1 and current_position_type == 0: # Open Short
            position_size = (capital * leverage) / current_price
            entry_price = current_price
            current_position_type = -1
        
        # If signal is 0 and no position, or signal matches current position type, capital remains as is for the end of this period
        # (equity_curve was already updated based on holding or just capital)
        if current_position_type == 0 :
             equity_curve.iloc[i] = capital


    # Ensure last equity value is up-to-date if a position is held till the end
    if current_position_type == 1:
        equity_curve.iloc[-1] = (price_data.iloc[-1] * position_size) 
    elif current_position_type == -1:
         equity_curve.iloc[-1] = (entry_price * position_size) + ((entry_price - price_data.iloc[-1]) * position_size)


    # --- Performance Metrics ---
    total_return_pct = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    
    # Annualized Return (assuming daily data, 252 trading days a year)
    num_days = len(price_data)
    num_years = num_days / 252.0
    annualized_return_pct = ((equity_curve.iloc[-1] / initial_capital)**(1/num_years) - 1) * 100 if num_years > 0 else 0
    
    # Sharpe Ratio (Risk-Free Rate = 0)
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown_pct = drawdown.min() * 100
    
    # Win Rate & Profit Factor
    profitable_trades = sum(1 for t in trades if t > 0)
    losing_trades = sum(1 for t in trades if t < 0)
    win_rate_pct = (profitable_trades / len(trades)) * 100 if len(trades) > 0 else 0
    
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0)) # abs because gross loss is a positive number
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    performance_metrics = {
        "Total Return (%)": total_return_pct,
        "Annualized Return (%)": annualized_return_pct,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown_pct,
        "Number of Trades": len(trades),
        "Win Rate (%)": win_rate_pct,
        "Profit Factor": profit_factor,
        "Gross Profit": gross_profit,
        "Gross Loss": gross_loss
    }
    
    return equity_curve, performance_metrics

if __name__ == '__main__':
    # Example Usage
    # Generate dummy price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B') # Business days
    price_data_example = pd.Series(100 + np.random.randn(100).cumsum(), index=dates)

    # Generate dummy signals (randomly 1, -1, or 0)
    # More realistic: 0s should be more frequent
    signal_values = np.random.choice([-1, 0, 1], size=100, p=[0.2, 0.6, 0.2])
    signals_example = pd.Series(signal_values, index=dates)
    
    # Ensure first signal allows for an entry if non-zero
    if signals_example.iloc[0] == 0 and len(signals_example) > 1:
        non_zero_signals = signals_example[signals_example != 0]
        if not non_zero_signals.empty:
            signals_example.iloc[0] = non_zero_signals.iloc[0] 
        else: # all signals are 0, make first one 1 for test
            signals_example.iloc[0] = 1


    print("--- Example Backtest ---")
    print("Price Data Head:")
    print(price_data_example.head())
    print("\nSignals Head:")
    print(signals_example.head())

    initial_cap = 10000.0
    equity_curve_result, metrics_result = simple_backtest(price_data_example, signals_example, initial_cap)

    print("\nEquity Curve Head:")
    print(equity_curve_result.head())
    print("\nEquity Curve Tail:")
    print(equity_curve_result.tail())
    print("\nPerformance Metrics:")
    for metric, value in metrics_result.items():
        print(f"{metric}: {value}")

    # Optional: Plotting for quick verification if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        equity_curve_result.plot(title='Example Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.show()
    except ImportError:
        print("\nMatplotlib not found, skipping plot in example.")
    
    print("\n--- Test Case: Always Long ---")
    signals_always_long = pd.Series(1, index=dates)
    equity_long, metrics_long = simple_backtest(price_data_example, signals_always_long, initial_cap)
    print("Performance (Always Long):")
    for metric, value in metrics_long.items():
        print(f"{metric}: {value}")
    # Expected: Total return should match price_data.iloc[-1]/price_data.iloc[0] - 1

    print("\n--- Test Case: Always Short (Hypothetical - not realistic for stocks) ---")
    signals_always_short = pd.Series(-1, index=dates)
    equity_short, metrics_short = simple_backtest(price_data_example, signals_always_short, initial_cap)
    print("Performance (Always Short):")
    for metric, value in metrics_short.items():
        print(f"{metric}: {value}")

    print("\n--- Test Case: No Trades ---")
    signals_no_trades = pd.Series(0, index=dates)
    equity_no_trades, metrics_no_trades = simple_backtest(price_data_example, signals_no_trades, initial_cap)
    print("Performance (No Trades):") # Should be 0 for most metrics, capital unchanged
    for metric, value in metrics_no_trades.items():
        print(f"{metric}: {value}")
        
    print("\n--- Test Case: Single Trade ---")
    single_trade_signals = pd.Series(0, index=dates)
    single_trade_signals.iloc[1] = 1 # Buy on day 1
    single_trade_signals.iloc[5] = 0 # Close on day 5
    equity_single, metrics_single = simple_backtest(price_data_example, single_trade_signals, initial_cap)
    print("Performance (Single Trade):")
    for metric, value in metrics_single.items():
        print(f"{metric}: {value}")
    print(f"Entry price: {price_data_example.iloc[1]}, Exit price: {price_data_example.iloc[5]}")
    expected_pnl_single_trade = (price_data_example.iloc[5] - price_data_example.iloc[1]) * (initial_cap / price_data_example.iloc[1])
    print(f"Expected PnL: {expected_pnl_single_trade}, Actual Gross Profit: {metrics_single['Gross Profit']}")
    
    print("\n--- Test Case: Short Trade ---")
    short_trade_signals = pd.Series(0, index=dates)
    short_trade_signals.iloc[1] = -1 # Short on day 1
    short_trade_signals.iloc[5] = 0  # Close on day 5
    equity_short_trade, metrics_short_trade = simple_backtest(price_data_example, short_trade_signals, initial_cap)
    print("Performance (Short Trade):")
    for metric, value in metrics_short_trade.items():
        print(f"{metric}: {value}")
    print(f"Entry price (short): {price_data_example.iloc[1]}, Exit price: {price_data_example.iloc[5]}")
    expected_pnl_short_trade = (price_data_example.iloc[1] - price_data_example.iloc[5]) * (initial_cap / price_data_example.iloc[1])
    print(f"Expected PnL: {expected_pnl_short_trade}, Actual Gross Profit/Loss: {metrics_short_trade['Gross Profit'] if expected_pnl_short_trade > 0 else metrics_short_trade['Gross Loss']}")
