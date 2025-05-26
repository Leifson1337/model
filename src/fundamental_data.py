# src/fundamental_data.py
import yfinance as yf
import pandas as pd

def get_fundamental_data(ticker_symbol: str, data_frequency: str = 'quarterly') -> pd.DataFrame:
    """
    Fetches and processes fundamental data for a given ticker symbol.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., "AAPL").
        data_frequency: 'quarterly' or 'annual'. Determines the frequency of
                        financial statements fetched.

    Returns:
        A Pandas DataFrame with fundamental data indexed by date.
        Features will be forward-filled to cover daily frequency.
    """
    print(f"Fetching fundamental data for {ticker_symbol} ({data_frequency})...")
    ticker = yf.Ticker(ticker_symbol)
    
    all_fund_data = []

    # 1. Financials (Income Statement)
    try:
        if data_frequency == 'quarterly':
            financials = ticker.quarterly_financials
        else:
            financials = ticker.financials
        
        if not financials.empty:
            financials_proc = financials.T # Transpose to have dates as rows
            selected_financials = {}
            if 'Total Revenue' in financials_proc.columns:
                selected_financials['fund_TotalRevenue'] = financials_proc['Total Revenue']
            if 'Net Income' in financials_proc.columns:
                selected_financials['fund_NetIncome'] = financials_proc['Net Income']
            if 'Gross Profit' in financials_proc.columns:
                selected_financials['fund_GrossProfit'] = financials_proc['Gross Profit']
            if 'Operating Income' in financials_proc.columns: # Often called EBIT
                selected_financials['fund_OperatingIncome'] = financials_proc['Operating Income']
            if 'EBITDA' in financials_proc.columns: # Check if EBITDA is directly available
                 selected_financials['fund_EBITDA'] = financials_proc['EBITDA']

            if selected_financials:
                all_fund_data.append(pd.DataFrame(selected_financials))
        else:
            print(f"No {data_frequency} financials data found for {ticker_symbol}.")
    except Exception as e:
        print(f"Error fetching/processing financials for {ticker_symbol}: {e}")

    # 2. Balance Sheet
    try:
        if data_frequency == 'quarterly':
            balance_sheet = ticker.quarterly_balance_sheet
        else:
            balance_sheet = ticker.balance_sheet
            
        if not balance_sheet.empty:
            balance_sheet_proc = balance_sheet.T
            selected_bs = {}
            if 'Total Assets' in balance_sheet_proc.columns:
                selected_bs['fund_TotalAssets'] = balance_sheet_proc['Total Assets']
            if 'Total Liab' in balance_sheet_proc.columns: # Liabilities
                selected_bs['fund_TotalLiabilities'] = balance_sheet_proc['Total Liab']
            if 'Total Stockholder Equity' in balance_sheet_proc.columns:
                selected_bs['fund_TotalEquity'] = balance_sheet_proc['Total Stockholder Equity']
            if 'Total Current Assets' in balance_sheet_proc.columns:
                selected_bs['fund_TotalCurrentAssets'] = balance_sheet_proc['Total Current Assets']
            if 'Total Current Liabilities' in balance_sheet_proc.columns:
                selected_bs['fund_TotalCurrentLiabilities'] = balance_sheet_proc['Total Current Liabilities']
            
            if selected_bs:
                all_fund_data.append(pd.DataFrame(selected_bs))
        else:
            print(f"No {data_frequency} balance sheet data found for {ticker_symbol}.")
    except Exception as e:
        print(f"Error fetching/processing balance sheet for {ticker_symbol}: {e}")

    # 3. Cash Flow
    try:
        if data_frequency == 'quarterly':
            cashflow = ticker.quarterly_cashflow
        else:
            cashflow = ticker.cashflow
            
        if not cashflow.empty:
            cashflow_proc = cashflow.T
            selected_cf = {}
            if 'Total Cash From Operating Activities' in cashflow_proc.columns: # Often named this or 'Operating Cash Flow'
                selected_cf['fund_OperatingCashFlow'] = cashflow_proc['Total Cash From Operating Activities']
            if 'Total Cash From Investing Activities' in cashflow_proc.columns:
                selected_cf['fund_InvestingCashFlow'] = cashflow_proc['Total Cash From Investing Activities']
            if 'Total Cash From Financing Activities' in cashflow_proc.columns:
                selected_cf['fund_FinancingCashFlow'] = cashflow_proc['Total Cash From Financing Activities']
            if 'Capital Expenditures' in cashflow_proc.columns:
                 selected_cf['fund_CapitalExpenditures'] = cashflow_proc['Capital Expenditures'] # Usually negative
            if 'Free Cash Flow' in cashflow_proc.columns: # yfinance sometimes provides this directly
                 selected_cf['fund_FreeCashFlow'] = cashflow_proc['Free Cash Flow']


            if selected_cf:
                all_fund_data.append(pd.DataFrame(selected_cf))
        else:
            print(f"No {data_frequency} cash flow data found for {ticker_symbol}.")
    except Exception as e:
        print(f"Error fetching/processing cash flow for {ticker_symbol}: {e}")

    if not all_fund_data:
        print(f"No fundamental statement data could be processed for {ticker_symbol}.")
        return pd.DataFrame()

    # Combine all fundamental statement data
    combined_statements_df = pd.concat(all_fund_data, axis=1)
    
    # Ensure index is DatetimeIndex
    combined_statements_df.index = pd.to_datetime(combined_statements_df.index)
    
    # Sort by date to ensure forward fill works correctly
    combined_statements_df.sort_index(inplace=True)
    
    # Note on ticker.info:
    # `ticker.info` provides current snapshot data. For historical daily ratios from this source,
    # one would need to fetch and store this daily, which yfinance doesn't do retroactively.
    # For this function, we focus on quarterly/annual statements.
    # If specific current ratios from ticker.info are desired as constant features,
    # they should be fetched separately and merged carefully, or a dedicated historical source used.
    # Example: info = ticker.info; pe_ratio = info.get('trailingPE') # This is current P/E
    
    print(f"Successfully processed fundamental statements for {ticker_symbol}. Shape: {combined_statements_df.shape}")
    return combined_statements_df


def add_fundamental_features_to_data(
    data_df: pd.DataFrame, 
    ticker_symbol: str, 
    data_frequency: str = 'quarterly'
) -> pd.DataFrame:
    """
    Fetches fundamental data and merges it with the main data_df.
    Also creates lag and delta features from the fundamental metrics.

    Args:
        data_df: The main DataFrame (indexed by Date, typically daily prices/TIs).
        ticker_symbol: Stock ticker symbol.
        data_frequency: 'quarterly' or 'annual' for fundamental data.

    Returns:
        DataFrame with fundamental features added.
    """
    fund_data_orig_freq = get_fundamental_data(ticker_symbol, data_frequency)

    if fund_data_orig_freq.empty:
        print(f"No fundamental data to add for {ticker_symbol}.")
        return data_df

    # 1. Resample fundamental data to daily and merge (forward-fill)
    # Ensure data_df index is DatetimeIndex
    if not isinstance(data_df.index, pd.DatetimeIndex):
        data_df.index = pd.to_datetime(data_df.index)
        
    # Reindex fundamental data to match the main df's index, then forward fill
    # This aligns quarterly/annual data to daily trading days
    fund_data_daily = fund_data_orig_freq.reindex(data_df.index, method='ffill')
    
    # Merge with main data_df
    # Suffix '_fund' to avoid collisions if data_df already has columns with same name
    data_with_fund = data_df.merge(fund_data_daily, left_index=True, right_index=True, how='left', suffixes=('', '_fund_dup'))
    
    # 2. Create Lag and Delta Features from the *original frequency* data
    # These represent previous period's value and change from previous period.
    
    # Determine number of periods for lag based on frequency (approx)
    # For 'quarterly', 1 period lag means previous quarter.
    # For 'annual', 1 period lag means previous year.
    # We'll use a simple shift on the original frequency data.
    
    lag_periods = 1 # Lag by one period (1 quarter or 1 year)
    
    for col in fund_data_orig_freq.columns:
        # Lag feature (value from previous period)
        lagged_col_name = f"{col}_lag{lag_periods}"
        data_with_fund[lagged_col_name] = fund_data_orig_freq[col].shift(lag_periods)
        
        # Delta feature (change from previous period)
        delta_col_name = f"{col}_delta{lag_periods}"
        data_with_fund[delta_col_name] = fund_data_orig_freq[col].diff(lag_periods)

        # Percentage change feature
        pct_change_col_name = f"{col}_pct_change{lag_periods}"
        # Avoid division by zero or by NaN; result in NaN if prev value is 0/NaN
        prev_val = fund_data_orig_freq[col].shift(lag_periods)
        data_with_fund[pct_change_col_name] = (fund_data_orig_freq[col] - prev_val) / prev_val.abs()
        data_with_fund[pct_change_col_name].replace([np.inf, -np.inf], 0, inplace=True) # Handle inf


    # The lag/delta features are at original frequency. Forward fill them to daily.
    lag_delta_cols = [col for col in data_with_fund.columns if '_lag' in col or '_delta' in col or '_pct_change' in col]
    for col in lag_delta_cols:
        if col in data_with_fund.columns: # Check if column was actually created (e.g. if fund_data_orig_freq had enough rows for shift)
             # Reindex this specific series to data_df.index and ffill
             temp_series = data_with_fund[col].dropna() # Get the series at original frequency points
             if not temp_series.empty:
                data_with_fund[col] = temp_series.reindex(data_df.index, method='ffill')


    # Final forward fill for any remaining NaNs after merge (e.g. at the very beginning)
    # And then backfill for any still at the start.
    data_with_fund.ffill(inplace=True)
    data_with_fund.bfill(inplace=True) # Fill any remaining NaNs at the beginning
    
    print(f"Added fundamental features, lags, and deltas for {ticker_symbol}. New shape: {data_with_fund.shape}")
    return data_with_fund


if __name__ == '__main__':
    sample_ticker = "MSFT" # Example
    print(f"--- Fundamental Data Module Demonstration for {sample_ticker} ---")

    # 1. Get fundamental data (quarterly, as DataFrame with original frequency)
    fund_data_q = get_fundamental_data(sample_ticker, data_frequency='quarterly')
    if not fund_data_q.empty:
        print("\nQuarterly Fundamental Data (Head):")
        print(fund_data_q.head())
        print("\nQuarterly Fundamental Data (Tail):")
        print(fund_data_q.tail())
        print(f"Columns: {fund_data_q.columns.tolist()}")

    # 2. Create a dummy daily price DataFrame to merge with
    # Use dates that would typically overlap with quarterly data
    if not fund_data_q.empty:
        # Create daily dates from before first fund date to after last fund date
        if fund_data_q.index.min() is not pd.NaT and fund_data_q.index.max() is not pd.NaT :
            start_date_price = fund_data_q.index.min() - pd.Timedelta(days=30)
            end_date_price = fund_data_q.index.max() + pd.Timedelta(days=30)
            daily_dates = pd.date_range(start=start_date_price, end=end_date_price, freq='B')
            dummy_price_df = pd.DataFrame(index=daily_dates)
            dummy_price_df['Close'] = np.random.rand(len(daily_dates)) * 100 + 100 # Dummy prices
            dummy_price_df['Volume'] = np.random.randint(100000, 1000000, size=len(daily_dates))
            
            print(f"\nDummy Daily Price Data created, shape: {dummy_price_df.shape}")

            # 3. Add fundamental features to the daily data
            data_with_all_features = add_fundamental_features_to_data(
                dummy_price_df.copy(), 
                sample_ticker,
                data_frequency='quarterly'
            )
            print("\nData with Fundamental Features Added (Tail):")
            # Print columns that show raw fund data, ffilled, and lag/delta
            cols_to_show = ['Close']
            if 'fund_TotalRevenue' in data_with_all_features.columns: cols_to_show.append('fund_TotalRevenue')
            if 'fund_TotalRevenue_lag1' in data_with_all_features.columns: cols_to_show.append('fund_TotalRevenue_lag1')
            if 'fund_TotalRevenue_delta1' in data_with_all_features.columns: cols_to_show.append('fund_TotalRevenue_delta1')
            if 'fund_TotalRevenue_pct_change1' in data_with_all_features.columns: cols_to_show.append('fund_TotalRevenue_pct_change1')
            
            # Filter for columns that actually exist to prevent KeyError
            cols_to_show = [col for col in cols_to_show if col in data_with_all_features.columns]

            if cols_to_show:
                 print(data_with_all_features[cols_to_show].tail(10))
            else:
                print("No fundamental columns were added or could be displayed.")
            print(f"Final combined data shape: {data_with_all_features.shape}")
            # Check for NaNs after all processing
            print(f"NaNs remaining in combined data: {data_with_all_features.isnull().sum().sum()}")
        else:
            print("Could not determine date range for dummy price data from fund_data_q.")
    else:
        print(f"Skipping merge example as no fundamental data was fetched for {sample_ticker}.")
        
    print("\n--- End of Fundamental Data Module Demonstration ---")
