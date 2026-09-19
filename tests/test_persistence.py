from __future__ import annotations

import pandas as pd

from ai_trading.persistence import JournalStore, safe_csv_bytes


def _payload(timestamp: str) -> dict[str, object]:
    return {
        "record_type": "recommendation",
        "symbol": "TEST",
        "source": "fixture",
        "interval": "1d",
        "as_of_timestamp": timestamp,
        "created_timestamp": timestamp,
        "action": "BUY",
        "forecast": "UP",
        "reference_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,
        "horizon": 2,
        "feature_version": "f1",
        "model_version": "m1",
        "training_cutoff": timestamp,
        "configuration": {"horizon": 2},
        "probabilities": {"UP": 0.6, "DOWN": 0.2, "NEUTRAL": 0.2},
        "data_fingerprint": "abc123",
        "is_synthetic": False,
    }


def test_recommendation_save_is_idempotent(tmp_path) -> None:
    store = JournalStore(tmp_path / "journal.db")
    first_id = store.save_recommendation(_payload("2024-01-01T00:00:00+00:00"))
    second_id = store.save_recommendation(_payload("2024-01-01T00:00:00+00:00"))
    assert first_id == second_id
    recommendations = store.list_recommendations()
    assert len(recommendations) == 1


def test_pending_then_resolved_outcome_refresh(tmp_path) -> None:
    store = JournalStore(tmp_path / "journal.db")
    store.save_recommendation(_payload("2024-01-01T00:00:00+00:00"))
    partial = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100, 101],
            "volume": [1, 1],
        }
    )
    assert store.evaluate_matured_recommendations(partial, neutral_threshold=0.01, evaluated_timestamp=pd.Timestamp("2024-01-03", tz="UTC")) == 0
    full = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
            "open": [100, 101, 103, 104],
            "high": [101, 102, 104, 105],
            "low": [99, 100, 102, 103],
            "close": [100, 101, 104, 105],
            "volume": [1, 1, 1, 1],
        }
    )
    assert store.evaluate_matured_recommendations(full, neutral_threshold=0.01, evaluated_timestamp=pd.Timestamp("2024-01-04", tz="UTC")) == 1
    resolved = store.list_recommendations().iloc[0]
    assert resolved["status"] == "resolved"
    assert resolved["observed_label"] == "UP"


def test_safe_csv_export_escapes_formula_cells() -> None:
    payload = pd.DataFrame({"value": ["=1+1", "plain"]})
    exported = safe_csv_bytes(payload).decode("utf-8")
    assert "'=1+1" in exported
