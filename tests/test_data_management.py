import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from click.testing import CliRunner
import json
import sys

# Add project root to sys.path to allow importing src modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_management import download_stock_data
from src.config_models import LoadDataConfig
from main import cli as main_cli # For testing CLI commands

# Sample DataFrame to be returned by mocked yf.download
SAMPLE_TICKER = "TESTICKER"
SAMPLE_DATA = pd.DataFrame({
    'Open': [100, 101, 102],
    'High': [105, 106, 107],
    'Low': [99, 100, 101],
    'Close': [102, 103, 104],
    'Volume': [1000, 1100, 1200]
}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
SAMPLE_DATA.index.name = 'Date'


@pytest.fixture
def mock_yfinance_download():
    with patch('yfinance.download') as mock_download:
        yield mock_download

@pytest.fixture
def sample_load_data_config(tmp_path: Path) -> LoadDataConfig:
    output_path = tmp_path / "raw_data" / f"{SAMPLE_TICKER.lower()}.parquet"
    return LoadDataConfig(
        ticker=SAMPLE_TICKER,
        start_date="2023-01-01",
        end_date="2023-01-03",
        output_parquet_path=str(output_path) # Pydantic model expects str for FilePath/DirectoryPath
    )

# --- Tests for download_stock_data function ---

def test_download_stock_data_success(mock_yfinance_download):
    mock_yfinance_download.return_value = SAMPLE_DATA
    
    data = download_stock_data(
        tickers=[SAMPLE_TICKER],
        start_date="2023-01-01",
        end_date="2023-01-03"
    )
    
    mock_yfinance_download.assert_called_once_with(
        [SAMPLE_TICKER], 
        start="2023-01-01", 
        end="2023-01-03", 
        interval='1d', 
        progress=False
    )
    assert data is not None
    pd.testing.assert_frame_equal(data, SAMPLE_DATA)

def test_download_stock_data_empty_return(mock_yfinance_download):
    mock_yfinance_download.return_value = pd.DataFrame() # Empty DataFrame
    
    data = download_stock_data(
        tickers=["EMPTYTICKER"],
        start_date="2023-01-01",
        end_date="2023-01-03"
    )
    
    assert data is None

def test_download_stock_data_yf_exception(mock_yfinance_download):
    mock_yfinance_download.side_effect = Exception("Yahoo Finance API error")
    
    data = download_stock_data(
        tickers=["ERRORTICKER"],
        start_date="2023-01-01",
        end_date="2023-01-03"
    )
    
    assert data is None

def test_download_multiple_tickers_stacked(mock_yfinance_download):
    # Create a sample multi-ticker DataFrame as yfinance might return it
    multi_ticker_df = pd.concat([SAMPLE_DATA.copy(), SAMPLE_DATA.copy()], keys=['TICK1', 'TICK2'], axis=1)
    mock_yfinance_download.return_value = multi_ticker_df

    data = download_stock_data(
        tickers=['TICK1', 'TICK2'],
        start_date="2023-01-01",
        end_date="2023-01-03"
    )
    assert data is not None
    assert isinstance(data.index, pd.MultiIndex)
    assert 'TICK1' in data.index.get_level_values('Ticker')
    assert 'TICK2' in data.index.get_level_values('Ticker')
    # Further checks on structure can be added if needed


# --- Tests for CLI 'load-data' command ---

@pytest.fixture
def cli_runner():
    return CliRunner()

@patch('src.data_management.download_stock_data') # Patch the function called by the CLI
def test_cli_load_data_success(mock_download_stock_data_func, cli_runner: CliRunner, sample_load_data_config: LoadDataConfig, tmp_path: Path):
    mock_download_stock_data_func.return_value = SAMPLE_DATA.copy() # Simulate successful download
    
    config_file = tmp_path / "test_load_config.json"
    # The sample_load_data_config already has output_parquet_path set within tmp_path
    # We need to ensure the directory for this output_parquet_path exists for the CLI command
    output_dir = Path(sample_load_data_config.output_parquet_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        # Use .model_dump_json() for Pydantic v2, or .json() for v1
        f.write(sample_load_data_config.model_dump_json() if hasattr(sample_load_data_config, 'model_dump_json') else sample_load_data_config.json())

    # The CLI command 'load-data' in main.py currently has a stub for actual saving.
    # It calls load_and_validate_config, then prints "load-data logic (stub) completed."
    # To test saving, the CLI handler would need to be updated to actually save the DataFrame.
    # For now, we test that the CLI command runs, calls the (mocked) download function,
    # and that the Parquet file (which the CLI stub *should* save) is created.
    # We'll add a patch for pd.DataFrame.to_parquet to check this.
    
    with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
        result = cli_runner.invoke(main_cli, ['load-data', '--config', str(config_file)])

    assert result.exit_code == 0, f"CLI Error: {result.output}"
    # Check if download_stock_data was called (it is, inside the stub which we are not fully mocking here)
    # The current CLI stub in main.py for load-data is very basic.
    # It logs, loads config, then has a stub for data loading using the config
    # and then saves a dummy file.
    # Let's check if our mocked download_stock_data (if it were called by a real implementation) would be called.
    # The current `load_data` CLI command in `main.py` does not directly call `src.data_management.download_stock_data`.
    # It has a placeholder: pd.DataFrame({'dummy_data': [1,2,3]}).to_csv(loaded_config.output_path, index=False)
    # So, we check if `to_csv` was called on a DataFrame.
    
    # The mock_download_stock_data_func won't be called by the current stub.
    # Instead, the stub creates a dummy df and saves it.
    # Let's verify the dummy file creation:
    expected_output_path = Path(sample_load_data_config.output_parquet_path) # Path from config
    # The current stub saves as CSV, not Parquet as per config. This is a discrepancy.
    # For the test to pass with current stub, we check for CSV.
    # This highlights that the CLI stub needs to honor the LoadDataConfig.output_parquet_path.
    
    # Assuming the CLI stub is updated to save to sample_load_data_config.output_parquet_path
    # and uses the dummy DataFrame it creates:
    # For now, the stub saves a CSV like this:
    # pd.DataFrame({'dummy_data': [1,2,3]}).to_csv(loaded_config.output_path, index=False)
    # The output_path in LoadDataConfig is output_parquet_path.
    # If the stub is fixed to use output_parquet_path from config:
    assert expected_output_path.exists(), f"Output file {expected_output_path} was not created by CLI. CLI output: {result.output}"
    # And we can check if mock_to_parquet was called if the stub used it.
    # For the current stub saving CSV:
    # mock_to_csv.assert_called_once_with(Path(sample_load_data_config.output_path), index=False)
    # Since the stub does its own DF creation and saving, we cannot easily check if SAMPLE_DATA was written
    # without further patching the DataFrame constructor or read_csv within the CLI command.

def test_cli_load_data_invalid_config_path(cli_runner: CliRunner):
    result = cli_runner.invoke(main_cli, ['load-data', '--config', 'non_existent_config.json'])
    assert result.exit_code != 0
    assert "Error" in result.output # Check for some error indication
    assert "Configuration file not found" in result.output or "Path 'non_existent_config.json' does not exist" in result.output # Click might say path doesn't exist if type=click.Path(exists=True) is used

def test_cli_load_data_invalid_config_content(cli_runner: CliRunner, tmp_path: Path):
    config_file = tmp_path / "invalid_config.json"
    with open(config_file, 'w') as f:
        f.write("{'ticker': 'AAPL', 'start_date': '2023-01-01', ") # Invalid JSON

    result = cli_runner.invoke(main_cli, ['load-data', '--config', str(config_file)])
    assert result.exit_code != 0
    assert "Error" in result.output
    assert "Invalid JSON" in result.output or "Could not decode JSON" in result.output # Depending on where error is caught

def test_cli_load_data_missing_required_field_in_config(cli_runner: CliRunner, tmp_path: Path):
    config_file = tmp_path / "missing_field_config.json"
    # Missing 'ticker' which is required by LoadDataConfig
    invalid_config_data = {"start_date": "2023-01-01", "end_date": "2023-01-03", "output_parquet_path": str(tmp_path / "data.parquet")}
    with open(config_file, 'w') as f:
        json.dump(invalid_config_data, f)

    result = cli_runner.invoke(main_cli, ['load-data', '--config', str(config_file)])
    assert result.exit_code != 0
    assert "Error" in result.output
    # Pydantic v2 error message for missing field: "Field required"
    # Pydantic v1 error message for missing field: "field required"
    assert "Field required" in result.output or "field required" in result.output or "validation failed" in result.output.lower()

# To run these tests:
# Ensure yfinance is mocked or tests are run with network access disabled for yfinance parts if not mocked.
# Ensure main.py and other src imports are resolvable (e.g. by running pytest from project root).
# The CLI tests depend on the error handling and Pydantic validation in main.py's load_and_validate_config.
```
