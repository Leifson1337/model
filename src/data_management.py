import yfinance as yf
import pandas as pd
# from ..src.config_models import LoadDataConfig # Example for type hinting if passing full config
# from pydantic import validate_call # For validating inputs to functions if not using full config obj

# TODO: Define expected schema for the output DataFrame (e.g., using pandera or a Pydantic model for rows)

def download_stock_data(
    tickers: list[str], 
    start_date: str, 
    end_date: str, 
    interval: str = '1d'
    # config: Optional[LoadDataConfig] = None # Alternative: pass the validated config object
) -> pd.DataFrame | None:
    """
    Downloads historical stock data using the yfinance library.

    Args:
        tickers: A list of stock tickers (e.g., ['AAPL', 'MSFT']).
        start_date: A string representing the start date (e.g., '2020-01-01').
        end_date: A string representing the end date (e.g., '2023-12-31').
        interval: A string for data resolution (e.g., '1d' for daily, '1h' for hourly).
        # config: Optional LoadDataConfig object if using direct config passing.

    Returns:
        A pandas DataFrame containing the downloaded data, with a MultiIndex for tickers
        if multiple tickers are provided. Returns None if an error occurs.
    """
    # TODO: Input validation for tickers, dates, interval using Pydantic's validate_call or similar if not using full config object.
    # Example:
    # @validate_call
    # def download_stock_data_validated(tickers: List[str], start_date: str, ...): ...
    
    # TODO: Implement retry logic for API calls (yf.download can fail due to network issues)
    # TODO: Consider caching results to avoid re-downloading frequently for same parameters.
    #       E.g., using a simple file-based cache or a library like `diskcache`.

    try:
        print(f"Attempting to download data for tickers: {tickers} from {start_date} to {end_date} with interval {interval}.")
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False) # progress=False for cleaner logs in non-interactive mode
        
        if data.empty:
            print("No data downloaded. Check tickers, date range, and internet connectivity.")
            # Potentially raise a custom exception or return an error object for better upstream handling.
            return None
            
        # TODO: Validate the structure and content of the downloaded 'data' DataFrame.
        #       - Check for expected columns (Open, High, Low, Close, Volume, Adj Close).
        #       - Check for data types.
        #       - Check for excessive NaNs or unexpected values.
        #       Example using comments:
        #       # validate_dataframe_schema(data, expected_ohlcv_schema)

        if len(tickers) == 1 and not isinstance(data.columns, pd.MultiIndex):
             # For a single ticker, yfinance might not return a MultiIndex if group_by='column' is not used.
             # Ensure consistent structure if downstream code expects it.
             # If direct use, this is fine. If stacking, it might need adjustment.
             pass # Default yfinance structure for single ticker is usually fine.
        elif len(tickers) > 1 : # yfinance default for multiple tickers often has MultiIndex columns
            # Standardize to a common format if needed, e.g. all dataframes having 'Ticker' column or part of index.
            # The current stacking logic is one way to standardize.
            try:
                # yfinance returns a DataFrame with a MultiIndex columns when multiple tickers are requested.
                # We want the tickers to be the first level of a row MultiIndex.
                data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reorder_levels(['Ticker', 'Date'])
            except Exception as e:
                print(f"Could not stack data for multiple tickers, possibly due to unexpected format: {e}")
                # This might happen if only one ticker was valid and yf returned a single-ticker format.
                # Or if all tickers were invalid, data would be empty (handled above).
                # If one valid ticker, data might be a simple DataFrame.
                if len(data.columns) > 0 and not data.empty: # Check if it's a non-empty single ticker df
                    # Add ticker column manually if possible (assuming the valid ticker is known, which is tricky here)
                    # For simplicity, if stacking fails, return data as is with a warning or handle error.
                    print("Warning: Stacking failed. Returning data in its original yfinance format.")


        # TODO: Save schema/metadata alongside the data.
        #       - e.g., data.to_parquet(config.output_path)
        #       - with open(config.output_path + ".schema.json", "w") as f: f.write(data_schema.to_json())
        print(f"Data downloaded successfully. Shape: {data.shape}")
        return data
        
    except Exception as e:
        # TODO: More specific error handling (network errors, API errors, data processing errors)
        print(f"Error downloading stock_data for {tickers}: {e}")
        # Consider logging the error to a file or monitoring system.
        return None

# TODO: Add a function to load data from a file, validating against a schema.
# def load_processed_data(file_path: str, expected_schema: Any) -> pd.DataFrame | None:
#     try:
#         df = pd.read_parquet(file_path) # Or other format
#         # validate_dataframe_schema(df, expected_schema)
#         return df
#     except FileNotFoundError:
#         print(f"Error: Data file not found at {file_path}")
#     except Exception as e: # Schema validation error, etc.
#         print(f"Error loading or validating data from {file_path}: {e}")
#     return None


if __name__ == '__main__':
    # Example usage:
    tickers = ['AAPL', 'GOOGL']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # Download daily data
    daily_data = download_stock_data(tickers, start_date, end_date)
    if daily_data is not None:
        print("\nDaily Data:")
        print(daily_data.head())

    # Example with a single ticker
    single_ticker_data = download_stock_data(['NVDA'], start_date, end_date)
    if single_ticker_data is not None:
        print("\nSingle Ticker (NVDA) Daily Data:")
        print(single_ticker_data.head())

    # Example with an invalid ticker (should print an error and None)
    invalid_ticker_data = download_stock_data(['INVALIDTICKER'], start_date, end_date)
    if invalid_ticker_data is None:
        print("\nCorrectly handled invalid ticker.")

    # Example of hourly data for a shorter period
    hourly_tickers = ['MSFT']
    hourly_start_date = '2023-12-01'
    hourly_end_date = '2023-12-05'
    hourly_data = download_stock_data(hourly_tickers, hourly_start_date, hourly_end_date, interval='1h')
    if hourly_data is not None:
        print("\nHourly Data (MSFT):")
        print(hourly_data.head())
