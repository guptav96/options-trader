"""Train a simple model on Tesla stock prices from QuantConnect."""

from __future__ import annotations

import io
import os
from typing import Literal

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

API_URL = "https://www.quantconnect.com/api/v2/data/read"


def load_data(
    ticker: str = "TSLA",
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    *,
    user_id: str | None = None,
    api_token: str | None = None,
    market: Literal["USA"] = "USA",
    resolution: Literal["Daily"] = "Daily",
) -> pd.DataFrame:
    """Download daily prices from QuantConnect and add next-day close.

    Parameters are passed directly to QuantConnect's Data API. `user_id` and
    `api_token` default to the ``QC_USER_ID`` and ``QC_API_TOKEN`` environment
    variables. Visit https://www.quantconnect.com/docs/v2/our-platform/lean-cli/api
    to obtain your credentials.
    """

    user_id = user_id or os.getenv("QC_USER_ID")
    api_token = api_token or os.getenv("QC_API_TOKEN")
    if not user_id or not api_token:
        raise RuntimeError("Set QC_USER_ID and QC_API_TOKEN environment variables")

    params = {
        "type": "Equity",
        "ticker": ticker,
        "market": market,
        "resolution": resolution,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "dataNormalizationMode": "Adjusted",
    }
    resp = requests.get(API_URL, params=params, auth=(user_id, api_token), timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.rename(
        columns={
            "time": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    df["close_next"] = df["close"].shift(-1)
    return df.dropna()


def train_model(data: pd.DataFrame) -> float:
    """Train RandomForestRegressor to predict next-day close."""
    features = ["open", "high", "low", "close", "volume"]
    X = data[features]
    y = data["close_next"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    print(f"RMSE: {rmse:.2f}")
    return rmse


def main() -> None:
    """Execute a training run and report RMSE."""
    data = load_data()
    train_model(data)


if __name__ == "__main__":
    main()
