from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trading.data import normalize_ohlcv


@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=260, tz="UTC")
    base = 100 + np.linspace(0, 25, len(dates)) + 3 * np.sin(np.linspace(0, 10 * np.pi, len(dates)))
    close = base + rng.normal(0, 0.8, len(dates))
    open_ = close + rng.normal(0, 0.5, len(dates))
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.2, len(dates))
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.2, len(dates))
    volume = rng.integers(50_000, 120_000, len(dates))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume,
        }
    )


@pytest.fixture()
def market_data(synthetic_frame):
    return normalize_ohlcv(synthetic_frame, symbol="TEST", source="fixture")
