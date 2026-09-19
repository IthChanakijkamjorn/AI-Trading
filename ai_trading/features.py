from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "volatility_5",
    "volatility_14",
    "rsi_14",
    "sma_10_distance",
    "sma_20_distance",
    "atr_14",
    "atr_pct_14",
    "volume_change_1",
    "volume_zscore_20",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.rolling(window=window, min_periods=window).mean()
    average_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    close = features["close"]
    returns = close.pct_change()

    features["return_1"] = returns
    features["return_5"] = close.pct_change(5)
    features["volatility_5"] = returns.rolling(window=5, min_periods=5).std()
    features["volatility_14"] = returns.rolling(window=14, min_periods=14).std()
    features["rsi_14"] = _rsi(close, 14)

    sma_10 = close.rolling(window=10, min_periods=10).mean()
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    features["sma_10_distance"] = (close / sma_10) - 1
    features["sma_20_distance"] = (close / sma_20) - 1

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (features["high"] - features["low"]),
            (features["high"] - previous_close).abs(),
            (features["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()
    features["atr_pct_14"] = features["atr_14"] / close

    features["volume_change_1"] = features["volume"].pct_change()
    volume_mean = features["volume"].rolling(window=20, min_periods=20).mean()
    volume_std = features["volume"].rolling(window=20, min_periods=20).std()
    features["volume_zscore_20"] = (features["volume"] - volume_mean) / volume_std.replace(0, np.nan)

    return features
