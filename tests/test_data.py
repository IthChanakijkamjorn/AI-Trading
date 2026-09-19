from __future__ import annotations

import pandas as pd
import pytest

from ai_trading.data import DataValidationError, load_csv_bytes, normalize_ohlcv


def test_validate_prices_and_high_low_consistency(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.copy()
    broken.loc[0, "high"] = broken.loc[0, "low"] - 1
    with pytest.raises(DataValidationError):
        normalize_ohlcv(broken, symbol="BAD", source="test")


def test_csv_loader_rejects_oversized_payload(synthetic_frame: pd.DataFrame) -> None:
    payload = synthetic_frame.to_csv(index=False).encode("utf-8")
    with pytest.raises(DataValidationError):
        load_csv_bytes(payload, filename="oversized.csv", symbol="TEST", max_bytes=10)


def test_duplicate_timestamps_fail_validation(synthetic_frame: pd.DataFrame) -> None:
    broken = pd.concat([synthetic_frame.iloc[[0]], synthetic_frame.iloc[[0]], synthetic_frame.iloc[1:]], ignore_index=True)
    with pytest.raises(DataValidationError):
        normalize_ohlcv(broken, symbol="DUP", source="test")
