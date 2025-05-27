## CLI Usage

The project includes a Command Line Interface (CLI) built with `click` for interacting with the pipeline.

To see available commands:
```bash
python main.py --help
```

**Key Commands:**

-   `python main.py load-data --config <path_to_load_config.json>`
-   `python main.py engineer-features --config <path_to_feature_config.json>`
-   `python main.py train-model --config <path_to_train_config.json>`
-   `python main.py evaluate --config <path_to_eval_config.json>`
-   `python main.py backtest --config <path_to_backtest_config.json>`
-   `python main.py export --config <path_to_export_config.json>`

<!-- AUTOGEN:CLI_COMMANDS_LIST -->
(CLI command list will be auto-generated here in the future)
<!-- END_AUTOGEN:CLI_COMMANDS_LIST -->

For detailed CLI workflows and configuration options, see [docs/workflows/](./docs/workflows/). The configuration for each command is defined using Pydantic models in `src/config_models.py`.
