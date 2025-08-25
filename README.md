# Options Trader Template

This repository provides a minimal starting point for building a machine
learning model that suggests future options trades based on historical data.

## Getting Started

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ensure the source directory is on your Python path and run one of the
   example training scripts:

   ```bash
   export PYTHONPATH=src
   # Option price example
   python -m options_trader.train

   # Tesla stock price example via QuantConnect
   export QC_USER_ID=<your-user-id>
   export QC_API_TOKEN=<your-api-token>
   python -m options_trader.tsla_example
   ```

`options_trader.train` downloads a single option chain using `yfinance`,
prepares a simple feature matrix and fits a `RandomForestRegressor` to predict
option prices.  `options_trader.tsla_example` uses QuantConnect's Data API to
download daily Tesla prices for 2025, trains a model to predict the next day's
close and reports the RMSE. These pipelines are intentionally basic and serve as
foundations for more advanced research into profitable strategies. QuantConnect
credentials are required for the stock example and may incur data costs; see the
QuantConnect terms of use for details.

## Project Structure

- `src/options_trader/` – core package containing data utilities, model
  definitions and training script.
- `tests/` – unit tests for the package.
- `requirements.txt` – Python dependencies.

Feel free to expand on this template by adding data storage, more sophisticated
models and evaluation metrics.
