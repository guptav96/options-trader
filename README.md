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

2. Ensure the source directory is on your Python path and run the example
   training script:

   ```bash
   export PYTHONPATH=src
   python -m options_trader.train
   ```

The script downloads a single option chain using `yfinance`, prepares a simple
feature matrix and fits a `RandomForestRegressor` to predict option prices.  The
pipeline is intentionally basic and serves as a foundation for more advanced
research into profitable options strategies.

## Project Structure

- `src/options_trader/` – core package containing data utilities, model
  definitions and training script.
- `tests/` – unit tests for the package.
- `requirements.txt` – Python dependencies.

Feel free to expand on this template by adding data storage, more sophisticated
models and evaluation metrics.
