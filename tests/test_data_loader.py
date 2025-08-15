import pandas as pd

from options_trader.data_loader import prepare_features


def test_prepare_features_generates_is_call_flag():
    sample = pd.DataFrame(
        {
            "strike": [100],
            "lastPrice": [1.2],
            "bid": [1.1],
            "ask": [1.3],
            "change": [0.1],
            "volume": [100],
            "openInterest": [200],
            "impliedVolatility": [0.2],
            "option_type": ["call"],
        }
    )
    features = prepare_features(sample)
    assert "is_call" in features.columns
    assert features["is_call"].iloc[0] == 1
