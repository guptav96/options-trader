"""Utilities for downloading and preparing options data."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_option_chain(ticker: str, expiration: str) -> pd.DataFrame:
    """Fetch the option chain for ``ticker`` and ``expiration``.

    Parameters
    ----------
    ticker:
        Ticker symbol such as ``"AAPL"``.
    expiration:
        Expiration date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pandas.DataFrame
        Combined calls and puts for the requested expiration.
    """
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiration)
    calls = chain.calls.assign(option_type="call")
    puts = chain.puts.assign(option_type="put")
    data = pd.concat([calls, puts], ignore_index=True)
    return data


def prepare_features(option_chain: pd.DataFrame) -> pd.DataFrame:
    """Generate a simple feature matrix from an option chain.

    The function keeps a subset of numeric columns and adds a binary
    ``is_call`` flag which can be used as a target variable or feature.
    Missing values are filled with zeros.
    """
    features = option_chain[
        [
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "change",
            "volume",
            "openInterest",
            "impliedVolatility",
            "option_type",
        ]
    ].copy()
    features["is_call"] = (features.pop("option_type") == "call").astype(int)
    return features.fillna(0)
