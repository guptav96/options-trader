"""Example training script.

This module demonstrates how the pieces of the template fit together.  It
fetches an option chain using :mod:`yfinance`, prepares a feature matrix and
trains a simple model.
"""

from __future__ import annotations

import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .data_loader import fetch_option_chain, prepare_features
from .model import build_model


def main(ticker: str = "AAPL") -> None:
    """Run a minimal training experiment for ``ticker``.

    The function fetches the nearest expiration option chain for ``ticker`` and
    trains a :class:`~sklearn.ensemble.RandomForestRegressor` to predict the
    ``lastPrice`` of options contracts.  The reported metric is the root mean
    squared error on a hold-out test set.
    """
    stock = yf.Ticker(ticker)
    expiration = stock.options[0]
    chain = fetch_option_chain(ticker, expiration)
    features = prepare_features(chain)
    target = chain["lastPrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Trained on {ticker} options expiring {expiration}. RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
