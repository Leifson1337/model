import click
import json
from pathlib import Path
from pydantic import ValidationError # Import ValidationError
from src.config_models import (
    GlobalAppConfig,
    LoadDataConfig,
    FeatureEngineeringConfig,
    TrainModelConfig,
    EvaluateModelConfig,
    BacktestConfig,
    ExportConfig,
    PipelineStepConfig
)

# Helper function to load and validate a step-specific config
def load_and_validate_config(config_path: str, model_class: type):
    if not config_path:
        click.echo("Error: No config file provided.", err=True)
        # Try to load a default if one exists for the command, e.g. from a global config
        # For now, just exit if specific config is expected and not given.
        # Alternatively, allow creating a default model_class instance if appropriate.
        # Example: return model_class() if defaults are sufficient
        return None 

    try:
        config_json = json.loads(Path(config_path).read_text())
        # If the config_json is for a PipelineStepConfig which wraps the actual step's config
        if 'config_inline' in config_json and isinstance(config_json.get('config_inline'), dict):
            # If using PipelineStepConfig to wrap, extract the inline part for the specific model_class
            parsed_config = model_class(**config_json.get('config_inline'))
        elif 'config_path' in config_json and isinstance(config_json.get('config_path'), str):
             # If PipelineStepConfig points to another file, this needs to be resolved here
             # For simplicity, assume direct config or inline for now when passing to CLI commands.
             # This part means the CLI is receiving a config that itself points to another config.
             # The calling context (e.g. GUI) should resolve this to a direct config for the step.
             click.echo(f"Warning: Config file '{config_path}' points to another config path '{config_json.get('config_path')}'." 
                        " This should ideally be resolved before calling the CLI step. Attempting to parse directly.", err=True)
             # Fallback: try to parse as if it's the direct config for model_class
             parsed_config = model_class(**config_json)
        else:
            # Assume the file directly contains the config for model_class
            parsed_config = model_class(**config_json)
        
        click.echo(f"Successfully loaded and validated config from '{config_path}' for {model_class.__name__}.")
        # click.echo(f"Config content: {parsed_config.model_dump_json(indent=2)}") # Optional: print full config
        return parsed_config
    except FileNotFoundError:
        click.echo(f"Error: Config file '{config_path}' not found.", err=True)
    except json.JSONDecodeError:
        click.echo(f"Error: Could not decode JSON from '{config_path}'.", err=True)
    except ValidationError as e:
        click.echo(f"Error: Config validation failed for '{config_path}':\n{e}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred while loading config '{config_path}': {e}", err=True)
    return None

@click.group()
@click.option('--app-config', default=None, help='Path to the global application configuration file (e.g., dev.json).')
@click.pass_context
def cli(ctx, app_config):
    """A CLI tool for the ML pipeline."""
    ctx.obj = {}
    if app_config:
        try:
            global_cfg_json = json.loads(Path(app_config).read_text())
            ctx.obj['GLOBAL_CONFIG'] = GlobalAppConfig(**global_cfg_json)
            click.echo(f"Global application config loaded from: {app_config}")
        except Exception as e:
            click.echo(f"Error loading global app config '{app_config}': {e}", err=True)
            ctx.obj['GLOBAL_CONFIG'] = GlobalAppConfig() # Use defaults
    else:
        ctx.obj['GLOBAL_CONFIG'] = GlobalAppConfig() # Use defaults if no global config is provided


@cli.command('load-data')
@click.option('--config', default=None, help='Path to the LoadDataConfig JSON file.')
@click.pass_context
def load_data(ctx, config):
    """Loads data for the pipeline based on LoadDataConfig."""
    click.echo("--- load-data command ---")
    loaded_config = load_and_validate_config(config, LoadDataConfig)
    if loaded_config:
        click.echo(f"Operation: Load data for ticker {loaded_config.ticker} from {loaded_config.start_date} to {loaded_config.end_date}.")
        # TODO: Call actual data loading logic from src.data_management
        # e.g., src.data_management.download_stock_data_from_config(loaded_config)
        # For now, just print config.
        click.echo(f"Output would be saved to: {loaded_config.output_parquet_path}")
        click.echo("load-data logic (stub) completed.")
    else:
        click.echo("load-data command failed due to config issues.", err=True)

@cli.command('engineer-features')
@click.option('--config', default=None, help='Path to the FeatureEngineeringConfig JSON file.')
@click.pass_context
def engineer_features(ctx, config):
    """Engineers features for the model based on FeatureEngineeringConfig."""
    click.echo("--- engineer-features command ---")
    loaded_config = load_and_validate_config(config, FeatureEngineeringConfig)
    if loaded_config:
        click.echo(f"Operation: Engineer features from {loaded_config.input_data_path} to {loaded_config.output_features_path}.")
        # TODO: Call actual feature engineering logic from src.feature_engineering
        click.echo("engineer-features logic (stub) completed.")
    else:
        click.echo("engineer-features command failed due to config issues.", err=True)

@cli.command('train-model')
@click.option('--config', default=None, help='Path to the TrainModelConfig JSON file.')
@click.pass_context
def train_model(ctx, config):
    """Trains the model based on TrainModelConfig."""
    click.echo("--- train-model command ---")
    loaded_config = load_and_validate_config(config, TrainModelConfig)
    if loaded_config:
        click.echo(f"Operation: Train a {loaded_config.model_type} model.")
        # TODO: Call actual model training logic from src.modeling
        click.echo(f"Model would be saved to base path: {loaded_config.model_output_path_base}")
        click.echo("train-model logic (stub) completed.")
    else:
        click.echo("train-model command failed due to config issues.", err=True)

@cli.command('evaluate')
@click.option('--config', default=None, help='Path to the EvaluateModelConfig JSON file.')
@click.pass_context
def evaluate(ctx, config):
    """Evaluates the model based on EvaluateModelConfig."""
    click.echo("--- evaluate command ---")
    loaded_config = load_and_validate_config(config, EvaluateModelConfig)
    if loaded_config:
        click.echo(f"Operation: Evaluate model {loaded_config.model_path} using test data {loaded_config.test_data_path}.")
        # TODO: Call actual model evaluation logic from src.evaluation
        click.echo(f"Metrics would be saved to: {loaded_config.metrics_output_json_path}")
        click.echo("evaluate logic (stub) completed.")
    else:
        click.echo("evaluate command failed due to config issues.", err=True)

@cli.command('backtest')
@click.option('--config', default=None, help='Path to the BacktestConfig JSON file.')
@click.pass_context
def backtest(ctx, config):
    """Backtests the model based on BacktestConfig."""
    click.echo("--- backtest command ---")
    loaded_config = load_and_validate_config(config, BacktestConfig)
    if loaded_config:
        click.echo(f"Operation: Backtest using predictions from {loaded_config.predictions_path} and OHLCV data from {loaded_config.ohlcv_data_path}.")
        # TODO: Call actual backtesting logic from src.backtesting
        click.echo(f"Backtest results would be saved to: {loaded_config.results_output_path}")
        click.echo("backtest logic (stub) completed.")
    else:
        click.echo("backtest command failed due to config issues.", err=True)

@cli.command('export')
@click.option('--config', default=None, help='Path to the ExportConfig JSON file.')
@click.pass_context
def export(ctx, config):
    """Exports the model or predictions based on ExportConfig."""
    click.echo("--- export command ---")
    loaded_config = load_and_validate_config(config, ExportConfig)
    if loaded_config:
        click.echo(f"Operation: Export model {loaded_config.trained_model_path} as {loaded_config.export_type} to {loaded_config.export_output_path}.")
        # TODO: Call actual export logic
        click.echo("export logic (stub) completed.")
    else:
        click.echo("export command failed due to config issues.", err=True)

if __name__ == '__main__':
    # Example of running CLI with a global app config (e.g. dev.json)
    # This demonstrates loading and validation of configs.
    # Create dummy valid and invalid config files for testing
    
    Path("config").mkdir(exist_ok=True) # Ensure config dir exists

    # Valid LoadDataConfig (as used by GUI)
    valid_load_data_content = {
        "ticker": "TESTAAPL",
        "start_date": "2022-01-01",
        "end_date": "2022-02-01",
        "output_parquet_path": "data/raw/testaapl_data.parquet"
    }
    with open("config/valid_load_data_test.json", "w") as f:
        json.dump(valid_load_data_content, f, indent=4)

    # Invalid LoadDataConfig (bad date format)
    invalid_load_data_content = {
        "ticker": "TESTBAD",
        "start_date": "01/01/2022", # Invalid format
        "end_date": "2022-02-01"
    }
    with open("config/invalid_load_data_test.json", "w") as f:
        json.dump(invalid_load_data_content, f, indent=4)

    print("\n--- Testing CLI with Valid LoadDataConfig ---")
    cli_args_valid = ['load-data', '--config', 'config/valid_load_data_test.json']
    try:
        cli(cli_args_valid, standalone_mode=False)
    except SystemExit: # Catch click's sys.exit on error
        pass


    print("\n--- Testing CLI with Invalid LoadDataConfig ---")
    cli_args_invalid = ['load-data', '--config', 'config/invalid_load_data_test.json']
    try:
        cli(cli_args_invalid, standalone_mode=False)
    except SystemExit:
        pass
        
    print("\n--- Testing CLI with Non-existent Config ---")
    cli_args_non_existent = ['load-data', '--config', 'config/non_existent.json']
    try:
        cli(cli_args_non_existent, standalone_mode=False)
    except SystemExit:
        pass

    # Clean up dummy files
    Path("config/valid_load_data_test.json").unlink(missing_ok=True)
    Path("config/invalid_load_data_test.json").unlink(missing_ok=True)
    
    # Example of using a global config (e.g. dev.json)
    # Assume dev.json has a structure parseable by GlobalAppConfig
    # And that it might contain a load_data sub-configuration.
    # For this test, we'll assume dev.json is structured for GlobalAppConfig.
    # The commands themselves still expect their specific config files.
    # The global config is more for setting context (paths, environment).
    print("\n--- Testing CLI with Global App Config (e.g., dev.json from project) ---")
    # This will try to load config/dev.json if it exists and is valid GlobalAppConfig
    # It won't run a command unless specified.
    # To run a command with global config: python main.py --app-config config/dev.json load-data --config <specific_load_data_config.json>
    
    # If config/dev.json exists and is a GlobalAppConfig, it will be loaded.
    # We can then test a command that might implicitly use parts of it if its own --config is not given (not yet implemented).
    # For now, the global config is loaded, but commands still require their own --config.
    if Path("config/dev.json").exists():
        cli_args_global = ['--app-config', 'config/dev.json', 'load-data', '--config', 'config/valid_load_data_test.json']
        # Recreate valid_load_data_test.json for this test
        with open("config/valid_load_data_test.json", "w") as f: json.dump(valid_load_data_content, f, indent=4)
        try:
            print(f"Executing: python main.py {' '.join(cli_args_global)}")
            cli(cli_args_global, standalone_mode=False)
        except SystemExit:
            pass
        Path("config/valid_load_data_test.json").unlink(missing_ok=True)
    else:
        print("Skipping global config test as config/dev.json does not exist.")
    
    print("\n--- CLI Test Examples Completed ---")
    # To run normally: cli()
    # For these tests, standalone_mode=False is used to integrate with direct calls.
    # If running the script directly, cli() at the end would parse sys.argv.
    # The if __name__ == '__main__': block here is for demonstration.
    # For actual CLI use: python main.py load-data --config path/to/your/load_config.json
    
    # Example of how the GUI would use this:
    # It would generate a JSON file matching, e.g. LoadDataConfig,
    # then call `python main.py load-data --config /tmp/generated_config.json`
    
    # Final call to cli() to make script runnable as `python main.py --help` etc.
    # For testing, this is commented out to avoid conflict with programmatic calls above.
    # cli() 
