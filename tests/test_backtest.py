from __future__ import annotations

import pandas as pd

from ai_trading.backtest import BacktestConfig, run_backtest
from ai_trading.labels import LABEL_DOWN, LABEL_HOLD, LABEL_UP


def _prediction_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 102.0, 104.0, 103.0, 102.0],
            "high": [101.0, 103.0, 104.5, 104.8, 104.0, 103.0],
            "low": [99.0, 100.0, 101.6, 102.5, 101.0, 100.0],
            "close": [100.5, 102.5, 104.0, 103.0, 102.0, 101.0],
            "atr_14": [1.0] * 6,
            "label": [LABEL_DOWN, LABEL_UP, LABEL_HOLD, LABEL_DOWN, LABEL_HOLD, LABEL_HOLD],
            "prob_down": [0.8, 0.1, 0.1, 0.8, 0.3, 0.3],
            "prob_neutral": [0.1, 0.2, 0.8, 0.1, 0.4, 0.4],
            "prob_up": [0.1, 0.7, 0.1, 0.1, 0.3, 0.3],
            "prediction": [LABEL_DOWN, LABEL_UP, LABEL_HOLD, LABEL_DOWN, LABEL_HOLD, LABEL_HOLD],
        }
    )


def test_backtest_uses_next_bar_entry_and_long_only_sell() -> None:
    result = run_backtest(_prediction_frame(), BacktestConfig(entry_threshold=0.45, max_holding_bars=5, allocation=0.5, commission_bps=0, slippage_bps=0))
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert str(trade["entry_time"]) == "2024-01-03 00:00:00+00:00"
    assert str(trade["exit_time"]) == "2024-01-05 00:00:00+00:00"
    assert trade["exit_reason"] == "signal"


def test_backtest_prefers_stop_when_stop_and_target_touch_same_bar() -> None:
    frame = _prediction_frame()
    frame.loc[2, ["open", "high", "low", "close"]] = [101.5, 105.5, 99.5, 102.0]
    result = run_backtest(frame, BacktestConfig(entry_threshold=0.45, max_holding_bars=5, allocation=0.5, commission_bps=0, slippage_bps=0, stop_atr=1.0, target_atr=1.0))
    assert result.trades.iloc[0]["exit_reason"] == "stop_first_same_bar"


def test_backtest_gap_stop_and_zero_trade_case() -> None:
    frame = _prediction_frame()
    frame.loc[3, ["open", "high", "low", "close"]] = [103.0, 104.0, 102.2, 103.4]
    frame.loc[4, ["open", "high", "low", "close"]] = [98.0, 99.0, 97.5, 98.5]
    frame.loc[3, ["prob_down", "prob_neutral", "prob_up", "prediction"]] = [0.2, 0.7, 0.1, LABEL_HOLD]
    result = run_backtest(frame, BacktestConfig(entry_threshold=0.45, max_holding_bars=5, allocation=0.5, commission_bps=0, slippage_bps=0, stop_atr=1.0, target_atr=3.0))
    assert result.trades.iloc[0]["exit_reason"] == "stop_gap"

    hold_frame = frame.copy()
    hold_frame[["prob_up", "prob_down"]] = 0.1
    hold_frame["prob_neutral"] = 0.8
    hold_frame["prediction"] = LABEL_HOLD
    zero_result = run_backtest(hold_frame, BacktestConfig(entry_threshold=0.45, allocation=0.5, commission_bps=0, slippage_bps=0))
    assert zero_result.summary["closed_trades"] == 0
    assert zero_result.summary["win_rate"] is None
