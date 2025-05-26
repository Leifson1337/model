import pandas as pd
import numpy as np
import backtrader as bt

# --- Existing simple_backtest function (commented out as per instructions) ---
# def simple_backtest(...):
#     ... (entire function content from previous version) ...
# (Assuming the full simple_backtest function was here and is now commented out or removed)


class SignalStrategy(bt.Strategy):
    """
    A simple Backtrader strategy based on an external signal series.
    - Signal = 1: Go Long
    - Signal = -1: Go Short
    - Signal = 0: Close position / Stay Neutral
    """
    params = (
        ('signals', None), # Pass the signals DataFrame/Series as a parameter
        ('target_percent', 0.95), # Target percentage of portfolio for trades
    )

    def __init__(self):
        if self.params.signals is None:
            raise ValueError("Signals parameter must be provided to SignalStrategy.")
        
        # Store signals, ensuring the index is datetime for lookup
        self.signals_series = self.params.signals
        if not isinstance(self.signals_series.index, pd.DatetimeIndex):
            try:
                self.signals_series.index = pd.to_datetime(self.signals_series.index)
            except Exception as e:
                raise ValueError(f"Could not convert signals index to DatetimeIndex: {e}")

        self.order = None # To keep track of pending orders

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()} {txt}') # Optional: print logs

    def next(self):
        # Get current date from data feed
        current_date = self.datas[0].datetime.date(0)
        
        # Get signal for the current date
        # Using .get(current_date, 0) to default to 0 (neutral) if date is missing in signals
        current_signal = self.signals_series.get(current_date, 0) 

        current_position_size = self.getposition(self.datas[0]).size

        if self.order: # Check if an order is pending
            return

        if current_signal == 1: # Long signal
            if current_position_size == 0: # No position
                self.log(f'BUY CREATE, {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=self.params.target_percent)
            elif current_position_size < 0: # Currently short, close short and go long
                self.log(f'CLOSE SHORT & BUY CREATE, {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=self.params.target_percent) # This will close short and open long
        
        elif current_signal == -1: # Short signal
            if current_position_size == 0: # No position
                self.log(f'SELL CREATE (SHORT), {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=-self.params.target_percent)
            elif current_position_size > 0: # Currently long, close long and go short
                self.log(f'CLOSE LONG & SELL CREATE (SHORT), {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
                self.order = self.order_target_percent(target=-self.params.target_percent) # This will close long and open short

        elif current_signal == 0: # Neutral signal
            if current_position_size != 0: # If a position exists
                self.log(f'CLOSE POSITION, {self.datas[0].close[0]:.2f}, Signal: {current_signal}')
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
    signals_df: pd.Series, # Expects a Series with datetime index and signal column
    initial_capital: float = 100000.0, 
    leverage: float = 1.0, 
    commission_bps: float = 2.0, # e.g. 2 bps = 0.02%
    slippage_bps: float = 1.0,   # e.g. 1 bps = 0.01%
    knock_out_long_bps: float = None, # Placeholder, not implemented in this version
    knock_out_short_bps: float = None # Placeholder, not implemented in this version
    ):
    """
    Runs a backtest using Backtrader with the provided data and signals.

    Args:
        data_df: Pandas DataFrame with Datetime index and OHLCV columns.
                 Column names must be standard (Open, High, Low, Close, Volume, OpenInterest).
                 'OpenInterest' can be zero if not available.
        signals_df: Pandas Series with Datetime index and signals (1, -1, 0).
        initial_capital: Starting capital.
        leverage: Leverage factor for position sizing.
        commission_bps: Commission in basis points.
        slippage_bps: Slippage in basis points.
        knock_out_long_bps: Placeholder for knock-out level for long positions (not implemented).
        knock_out_short_bps: Placeholder for knock-out level for short positions (not implemented).

    Returns:
        A dictionary of performance metrics.
        Optionally, could also return the cerebro object for plotting if needed.
    """
    if knock_out_long_bps is not None or knock_out_short_bps is not None:
        print("Warning: Knock-out parameters are provided but not implemented in this Backtrader version.")

    cerebro = bt.Cerebro()

    # Add data feed
    # Ensure columns are named as Backtrader expects: open, high, low, close, volume, openinterest
    # If 'Adj Close' is present and preferred, rename it to 'close'. For now, assumes 'Close' is the primary.
    # If 'OpenInterest' is not present, create a dummy one.
    if 'OpenInterest' not in data_df.columns:
        data_df['OpenInterest'] = 0 
    
    # Ensure column names are lowercase as Backtrader might prefer
    data_df_bt = data_df.copy()
    data_df_bt.columns = [col.lower() for col in data_df_bt.columns] # open, high, low, close, volume, openinterest
    
    data_feed = bt.feeds.PandasData(dataname=data_df_bt)
    cerebro.adddata(data_feed)

    # Add strategy
    cerebro.addstrategy(SignalStrategy, signals=signals_df) # Pass signals Series

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
    # For this task, we'll stick to the prompt's suggestion:
    target_trade_percent = 95 # Strategy aims to use 95% for a trade
    effective_sizer_percent = min(target_trade_percent * leverage, 99) # Cap at 99% to avoid issues with 100%
    cerebro.addsizer(bt.sizers.PercentSizer, percents=effective_sizer_percent)


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

    # Generate dummy signals (ensure index is DatetimeIndex)
    signal_values = np.random.choice([-1, 0, 1], size=num_periods, p=[0.1, 0.8, 0.1]) # Fewer trades
    signals_series = pd.Series(signal_values, index=dates, name="signal")
    if signals_series.iloc[0] == 0 and len(signals_series) > 1:
        signals_series.iloc[0] = 1 

    print("--- Example Backtrader Backtest ---")
    initial_cap = 100000.0

    print("\nTest 1: Basic (Leverage 1x, default costs)")
    metrics1, cerebro1 = run_backtrader_backtest(
        data_df=data.copy(), 
        signals_df=signals_series.copy(), 
        initial_capital=initial_cap,
        leverage=1.0
    )
    print("Performance (Test 1 - Basic):")
    for metric, value in metrics1.items(): print(f"  {metric}: {value}")

    print("\nTest 2: Leverage 2x, Higher Commission (5bps)")
    metrics2, cerebro2 = run_backtrader_backtest(
        data_df=data.copy(), 
        signals_df=signals_series.copy(), 
        initial_capital=initial_cap,
        leverage=2.0,
        commission_bps=5.0,
        slippage_bps=2.0
    )
    print("Performance (Test 2 - Leverage & Costs):")
    for metric, value in metrics2.items(): print(f"  {metric}: {value}")
    
    # Optional: Plotting (might not work in all environments directly from script)
    # try:
    #     print("\nPlotting for Test 1 (Basic)...")
    #     cerebro1.plot(style='candlestick', barup='green', bardown='red')
    # except Exception as e:
    #     print(f"Could not plot: {e}")

    # print("\n--- Old Simple Backtester (for comparison, if it were still here and compatible) ---")
    # This part is just a note, as the old backtester is removed/commented.
    # To compare, you'd need to adapt the old one for OHLCV input or this one for just Close price input.
    # For now, we are just testing the new backtrader implementation.
