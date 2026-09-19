from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from ai_trading.constants import MAX_UPLOAD_BYTES, SAMPLE_DATA_PATH


class DataValidationError(ValueError):
    """Raised when market data fails validation."""


class DataSourceError(RuntimeError):
    """Raised when market data cannot be loaded from a source."""


@dataclass(slots=True)
class MarketData:
    frame: pd.DataFrame
    symbol: str
    source: str
    interval: str
    timezone: str
    last_completed_candle: pd.Timestamp
    freshness: str
    provenance_notes: tuple[str, ...] = field(default_factory=tuple)
    is_synthetic: bool = False


_COLUMN_ALIASES = {
    "timestamp": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "volume": "volume",
}


_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _standardize_columns(columns: Iterable[str]) -> list[str]:
    standardized = []
    for column in columns:
        key = str(column).strip().lower()
        standardized.append(_COLUMN_ALIASES.get(key, key))
    return standardized


def _gap_notes(timestamps: pd.Series) -> tuple[str, ...]:
    if len(timestamps) < 3:
        return tuple()
    deltas = timestamps.sort_values().diff().dropna()
    if deltas.empty:
        return tuple()
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return tuple()
    gaps = deltas[deltas > max(median_delta * 3, pd.Timedelta(days=4))]
    if gaps.empty:
        return tuple()
    return (f"Detected {len(gaps)} gaps larger than the median interval of {median_delta}.",)


def normalize_ohlcv(
    raw: pd.DataFrame,
    *,
    symbol: str,
    source: str,
    interval: str = "1d",
    timezone: str = "UTC",
    is_synthetic: bool = False,
) -> MarketData:
    if raw.empty:
        raise DataValidationError("No rows were provided.")

    frame = raw.copy()
    frame.columns = _standardize_columns(frame.columns)
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")

    frame = frame.loc[:, list(_REQUIRED_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise DataValidationError("One or more timestamps could not be parsed.")
    if not (frame["timestamp"].diff().dropna() > pd.Timedelta(0)).all():
        raise DataValidationError("Timestamps must be strictly increasing and unique.")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not pd.Series(frame[numeric_columns].to_numpy().ravel()).map(pd.notna).all():
        raise DataValidationError("OHLCV values must be numeric.")

    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("Open, high, low, and close must be positive.")
    if (frame["volume"] < 0).any():
        raise DataValidationError("Volume must be non-negative.")

    if not ((frame["high"] >= frame[["open", "close", "low"]].max(axis=1)).all()):
        raise DataValidationError("High must be greater than or equal to open, low, and close.")
    if not ((frame["low"] <= frame[["open", "close", "high"]].min(axis=1)).all()):
        raise DataValidationError("Low must be less than or equal to open, high, and close.")

    frame = frame.reset_index(drop=True)

    notes = list(_gap_notes(frame["timestamp"]))
    last_completed_candle = frame["timestamp"].iloc[-1]
    freshness = "Synthetic demo data" if is_synthetic else "Historical/delayed data"

    return MarketData(
        frame=frame,
        symbol=symbol.upper().strip() or "UNKNOWN",
        source=source,
        interval=interval,
        timezone=timezone,
        last_completed_candle=last_completed_candle,
        freshness=freshness,
        provenance_notes=tuple(notes),
        is_synthetic=is_synthetic,
    )


def load_sample_data(symbol: str = "DEMO") -> MarketData:
    frame = pd.read_csv(SAMPLE_DATA_PATH)
    return normalize_ohlcv(
        frame,
        symbol=symbol,
        source="bundled-demo-csv",
        interval="1d",
        timezone="UTC",
        is_synthetic=True,
    )


def load_csv_bytes(
    payload: bytes,
    *,
    filename: str,
    symbol: str,
    interval: str = "1d",
    timezone: str = "UTC",
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> MarketData:
    if len(payload) > max_bytes:
        raise DataValidationError(f"CSV file is too large; maximum size is {max_bytes} bytes.")
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # pragma: no cover - exercised by tests via raised message
        raise DataValidationError(f"Could not parse CSV: {exc}") from exc
    return normalize_ohlcv(
        frame,
        symbol=symbol,
        source=f"csv:{filename}",
        interval=interval,
        timezone=timezone,
    )


def download_yfinance_data(symbol: str, start: str, end: str, interval: str = "1d") -> MarketData:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency is part of requirements
        raise DataSourceError("yfinance is not installed.") from exc

    try:
        frame = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    except Exception as exc:
        raise DataSourceError(f"Failed to download data: {exc}") from exc

    if frame.empty:
        raise DataSourceError("No rows were returned for that ticker/date range.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] if isinstance(column, tuple) else column for column in frame.columns]
    frame = frame.reset_index()
    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "timestamp"})
    elif "Datetime" in frame.columns:
        frame = frame.rename(columns={"Datetime": "timestamp"})

    return normalize_ohlcv(
        frame,
        symbol=symbol,
        source="yfinance",
        interval=interval,
        timezone="UTC",
        is_synthetic=False,
    )
