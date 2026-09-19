from __future__ import annotations

import pandas as pd
import pytest

from ai_trading.features import FEATURE_COLUMNS
from ai_trading.labels import LABEL_HOLD
from ai_trading.model import InsufficientClassCoverage, ResearchConfig, run_research_pipeline


def test_pipeline_generates_held_out_predictions(market_data) -> None:
    artifacts = run_research_pipeline(market_data, ResearchConfig(min_history=120, horizon=5, neutral_threshold=0.01))
    first_test_ts = pd.Timestamp(artifacts.test_predictions["timestamp"].min())
    train_end_ts = pd.Timestamp(artifacts.labeled_frame.iloc[artifacts.split.train_idx[-1]]["timestamp"])
    assert first_test_ts > train_end_ts
    assert set(["prob_down", "prob_neutral", "prob_up", "prediction", "forecast"]).issubset(artifacts.test_predictions.columns)
    assert artifacts.latest_signal.timestamp >= artifacts.latest_signal.training_cutoff


def test_pipeline_rejects_insufficient_class_coverage(synthetic_frame) -> None:
    flat = synthetic_frame.copy()
    flat["close"] = 100.0
    flat["open"] = 100.0
    flat["high"] = 100.0
    flat["low"] = 100.0
    market_data = market_data = __import__("ai_trading.data", fromlist=["normalize_ohlcv"]).normalize_ohlcv(flat, symbol="FLAT", source="fixture")
    with pytest.raises((InsufficientClassCoverage, ValueError)):
        run_research_pipeline(market_data, ResearchConfig(min_history=80, horizon=5, neutral_threshold=0.01))


def test_latest_signal_has_complete_feature_row(market_data) -> None:
    artifacts = run_research_pipeline(market_data, ResearchConfig(min_history=120, horizon=5))
    latest_row = artifacts.featured_frame.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    assert latest_row[FEATURE_COLUMNS].notna().all()
    assert artifacts.latest_signal.forecast in {"DOWN", "NEUTRAL", "UP"}
