# src/fundamental_data.py
import yfinance as yf
import pandas as pd
# from ..src.config_models import FundamentalDataConfig # Example for type hinting
# from pydantic import validate_call # For validating inputs

# TODO: Define expected input/output schemas for DataFrames.
#       Input: Price data (for merging). Output: DataFrame with fundamental features.
# TODO: Log key library versions (yfinance, pandas) for reproducibility.

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
        # TODO: Formalize output schema.
    """
    # TODO: Input validation for ticker_symbol and data_frequency.
    # TODO: Implement retry logic for yf.Ticker API calls.
    # TODO: Consider caching yfinance results to avoid repeated API calls.
    
    print(f"Fetching fundamental data for {ticker_symbol} ({data_frequency})...")
    try:
        ticker = yf.Ticker(ticker_symbol)
    except Exception as e:
        # TODO: Log this error.
        print(f"Error creating yf.Ticker object for {ticker_symbol}: {e}")
        return pd.DataFrame() # Return empty DataFrame on critical error
    
    all_fund_data = []
    statement_types = ['financials', 'balance_sheet', 'cashflow']
    
    for stmt_type in statement_types:
        try:
            if data_frequency == 'quarterly':
                data_stmt = getattr(ticker, f"quarterly_{stmt_type}")
            else: # Default to annual
                data_stmt = getattr(ticker, stmt_type)
            
            if not data_stmt.empty:
                data_stmt_proc = data_stmt.T # Transpose to have dates as rows
                
                # TODO: Standardize selected features, possibly driven by config (e.g., from FeatureEngineeringConfig).
                #       The current selection is hardcoded.
                selected_cols = {}
                if stmt_type == 'financials':
                    if 'Total Revenue' in data_stmt_proc.columns: selected_cols['fund_TotalRevenue'] = data_stmt_proc['Total Revenue']
                    if 'Net Income' in data_stmt_proc.columns: selected_cols['fund_NetIncome'] = data_stmt_proc['Net Income']
                    if 'Gross Profit' in data_stmt_proc.columns: selected_cols['fund_GrossProfit'] = data_stmt_proc['Gross Profit']
                    if 'Operating Income' in data_stmt_proc.columns: selected_cols['fund_OperatingIncome'] = data_stmt_proc['Operating Income']
                    if 'EBITDA' in data_stmt_proc.columns: selected_cols['fund_EBITDA'] = data_stmt_proc['EBITDA']
                elif stmt_type == 'balance_sheet':
                    if 'Total Assets' in data_stmt_proc.columns: selected_cols['fund_TotalAssets'] = data_stmt_proc['Total Assets']
                    if 'Total Liab' in data_stmt_proc.columns: selected_cols['fund_TotalLiabilities'] = data_stmt_proc['Total Liab']
                    if 'Total Stockholder Equity' in data_stmt_proc.columns: selected_cols['fund_TotalEquity'] = data_stmt_proc['Total Stockholder Equity']
                    if 'Total Current Assets' in data_stmt_proc.columns: selected_cols['fund_TotalCurrentAssets'] = data_stmt_proc['Total Current Assets']
                    if 'Total Current Liabilities' in data_stmt_proc.columns: selected_cols['fund_TotalCurrentLiabilities'] = data_stmt_proc['Total Current Liabilities']
                elif stmt_type == 'cashflow':
                    if 'Total Cash From Operating Activities' in data_stmt_proc.columns: selected_cols['fund_OperatingCashFlow'] = data_stmt_proc['Total Cash From Operating Activities']
                    if 'Total Cash From Investing Activities' in data_stmt_proc.columns: selected_cols['fund_InvestingCashFlow'] = data_stmt_proc['Total Cash From Investing Activities']
                    if 'Total Cash From Financing Activities' in data_stmt_proc.columns: selected_cols['fund_FinancingCashFlow'] = data_stmt_proc['Total Cash From Financing Activities']
                    if 'Capital Expenditures' in data_stmt_proc.columns: selected_cols['fund_CapitalExpenditures'] = data_stmt_proc['Capital Expenditures']
                    if 'Free Cash Flow' in data_stmt_proc.columns: selected_cols['fund_FreeCashFlow'] = data_stmt_proc['Free Cash Flow']

                if selected_cols:
                    all_fund_data.append(pd.DataFrame(selected_cols))
                else:
                    print(f"No relevant columns found in {data_frequency} {stmt_type} for {ticker_symbol}.")
            else:
                print(f"No {data_frequency} {stmt_type} data found for {ticker_symbol}.")
        except Exception as e:
            # TODO: Log this error.
            print(f"Error fetching/processing {stmt_type} for {ticker_symbol}: {e}")

    if not all_fund_data:
        print(f"No fundamental statement data could be processed for {ticker_symbol}.")
        return pd.DataFrame()

    # Combine all fundamental statement data
    # Ensure robust concat for cases where some statements might be empty or have different date ranges.
    # Using outer join and then handling NaNs might be safer if date indices differ significantly.
    try:
        combined_statements_df = pd.concat(all_fund_data, axis=1) # axis=1 joins on index (dates)
    except Exception as e:
        print(f"Error combining fundamental data for {ticker_symbol}: {e}")
        return pd.DataFrame()
        
    if combined_statements_df.empty:
        print(f"Combined fundamental data is empty for {ticker_symbol}.")
        return pd.DataFrame()
        
    combined_statements_df.index = pd.to_datetime(combined_statements_df.index)
    combined_statements_df.sort_index(inplace=True)
    
    # Note on ticker.info: (comment retained)
    # ...
    
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
