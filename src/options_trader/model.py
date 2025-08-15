"""Model definitions used by the options trader template."""

from sklearn.ensemble import RandomForestRegressor


def build_model() -> RandomForestRegressor:
    """Return a simple random forest regressor with sensible defaults."""
    return RandomForestRegressor(n_estimators=100, random_state=42)
