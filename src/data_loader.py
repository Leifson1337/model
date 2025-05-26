import yfinance as yf
import pandas as pd

def download_stock_data(tickers: list[str], start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame | None:
    """
    Downloads historical stock data using the yfinance library.

    Args:
        tickers: A list of stock tickers (e.g., ['AAPL', 'MSFT']).
        start_date: A string representing the start date (e.g., '2020-01-01').
        end_date: A string representing the end date (e.g., '2023-12-31').
        interval: A string for data resolution (e.g., '1d' for daily, '1h' for hourly).

    Returns:
        A pandas DataFrame containing the downloaded data, with a MultiIndex for tickers
        if multiple tickers are provided. Returns None if an error occurs.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        if data.empty:
            print("No data downloaded. Check tickers and date range.")
            return None
        if len(tickers) > 1:
            # yfinance returns a DataFrame with a MultiIndex columns when multiple tickers are requested.
            # We want the tickers to be the first level of a row MultiIndex.
            data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reorder_levels(['Ticker', 'Date'])
        return data
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        return None

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
