from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ai_trading.labels import LABEL_DOWN, LABEL_NAMES, LABEL_UP


@dataclass(slots=True)
class BacktestConfig:
    starting_capital: float = 10_000.0
    allocation: float = 0.95
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    stop_atr: float = 1.5
    target_atr: float = 3.0
    max_holding_bars: int = 5
    entry_threshold: float = 0.45


@dataclass(slots=True)
class BacktestResult:
    summary: dict[str, object]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


def _commission_rate(config: BacktestConfig) -> float:
    return config.commission_bps / 10_000


def _slippage_rate(config: BacktestConfig) -> float:
    return config.slippage_bps / 10_000


def _action_from_probabilities(prob_up: float, prob_down: float, threshold: float, has_position: bool) -> str:
    if has_position:
        return "SELL" if prob_down >= threshold and prob_down > prob_up else "HOLD"
    return "BUY" if prob_up >= threshold and prob_up > prob_down else "HOLD"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def run_backtest(test_predictions: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    if test_predictions.empty:
        raise ValueError("Backtest requires held-out predictions.")
    if config.starting_capital <= 0:
        raise ValueError("Starting capital must be positive.")
    if not (0 < config.allocation <= 1):
        raise ValueError("Allocation must be between 0 and 1.")

    frame = test_predictions.sort_values("timestamp").reset_index(drop=True).copy()
    cash = config.starting_capital
    quantity = 0.0
    entry_price = 0.0
    entry_time = None
    signal_time = None
    stop_loss = None
    take_profit = None
    holding_bars = 0
    costs_paid = 0.0
    pending_entry = None
    pending_exit = None
    trade_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = []
    bars_with_position = 0

    def close_position(price: float, timestamp: pd.Timestamp, reason: str) -> None:
        nonlocal cash, quantity, entry_price, entry_time, signal_time, stop_loss, take_profit, holding_bars, costs_paid, pending_exit
        if quantity <= 0:
            return
        commission = quantity * price * _commission_rate(config)
        proceeds = (quantity * price) - commission
        pnl = proceeds - (quantity * entry_price)
        cash += proceeds
        costs_paid += commission
        trade_records.append(
            {
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": timestamp,
                "entry_price": entry_price,
                "exit_price": price,
                "quantity": quantity,
                "gross_entry_value": quantity * entry_price,
                "gross_exit_value": quantity * price,
                "net_pnl": pnl,
                "return_pct": (price / entry_price) - 1,
                "exit_reason": reason,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "total_costs": commission,
            }
        )
        quantity = 0.0
        entry_price = 0.0
        entry_time = None
        signal_time = None
        stop_loss = None
        take_profit = None
        holding_bars = 0
        pending_exit = None

    for idx, row in frame.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])

        if pending_exit and pending_exit["execute_at"] == timestamp and quantity > 0:
            exit_fill = open_price * (1 - _slippage_rate(config))
            close_position(exit_fill, timestamp, pending_exit["reason"])

        if pending_entry and pending_entry["execute_at"] == timestamp and quantity == 0:
            entry_fill = open_price * (1 + _slippage_rate(config))
            budget = cash * config.allocation
            quantity = budget / (entry_fill * (1 + _commission_rate(config)))
            commission = quantity * entry_fill * _commission_rate(config)
            total_cost = (quantity * entry_fill) + commission
            if total_cost > cash:
                quantity = cash / (entry_fill * (1 + _commission_rate(config)))
                commission = quantity * entry_fill * _commission_rate(config)
                total_cost = (quantity * entry_fill) + commission
            cash -= total_cost
            costs_paid += commission
            entry_price = entry_fill
            entry_time = timestamp
            signal_time = pending_entry["signal_time"]
            stop_loss = entry_fill - (pending_entry["atr"] * config.stop_atr)
            take_profit = entry_fill + (pending_entry["atr"] * config.target_atr)
            holding_bars = 0
            pending_entry = None

        if quantity > 0:
            bars_with_position += 1
            if open_price <= stop_loss:
                exit_fill = open_price * (1 - _slippage_rate(config))
                close_position(exit_fill, timestamp, "stop_gap")
            elif open_price >= take_profit:
                exit_fill = open_price * (1 - _slippage_rate(config))
                close_position(exit_fill, timestamp, "target_gap")
            elif low_price <= stop_loss and high_price >= take_profit:
                exit_fill = stop_loss * (1 - _slippage_rate(config))
                close_position(exit_fill, timestamp, "stop_first_same_bar")
            elif low_price <= stop_loss:
                exit_fill = stop_loss * (1 - _slippage_rate(config))
                close_position(exit_fill, timestamp, "stop")
            elif high_price >= take_profit:
                exit_fill = take_profit * (1 - _slippage_rate(config))
                close_position(exit_fill, timestamp, "target")
            else:
                holding_bars += 1
                if holding_bars >= config.max_holding_bars:
                    if idx < len(frame) - 1:
                        pending_exit = {"execute_at": pd.Timestamp(frame.iloc[idx + 1]["timestamp"]), "reason": "time_limit"}
                    else:
                        close_position(close_price * (1 - _slippage_rate(config)), timestamp, "end_of_test")

        if idx < len(frame) - 1:
            action = _action_from_probabilities(
                prob_up=float(row["prob_up"]),
                prob_down=float(row["prob_down"]),
                threshold=config.entry_threshold,
                has_position=quantity > 0,
            )
            next_timestamp = pd.Timestamp(frame.iloc[idx + 1]["timestamp"])
            if action == "BUY" and quantity == 0:
                pending_entry = {
                    "execute_at": next_timestamp,
                    "signal_time": timestamp,
                    "atr": float(row["atr_14"]),
                }
            elif action == "SELL" and quantity > 0:
                pending_exit = {"execute_at": next_timestamp, "reason": "signal"}

        if idx == len(frame) - 1 and quantity > 0:
            close_position(close_price * (1 - _slippage_rate(config)), timestamp, "end_of_test")

        market_value = quantity * close_price
        equity_records.append({"timestamp": timestamp, "cash": cash, "market_value": market_value, "equity": cash + market_value})

    trades = pd.DataFrame.from_records(trade_records)
    equity_curve = pd.DataFrame.from_records(equity_records)
    final_equity = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else config.starting_capital
    total_return = (final_equity / config.starting_capital) - 1
    running_max = equity_curve["equity"].cummax() if not equity_curve.empty else pd.Series(dtype=float)
    drawdown = (equity_curve["equity"] / running_max) - 1 if not equity_curve.empty else pd.Series(dtype=float)
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    losses = trades.loc[trades["net_pnl"] <= 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.abs().sum()) if not losses.empty else 0.0
    benchmark = (frame["close"].iloc[-1] / frame["open"].iloc[0]) - 1
    benchmark -= (2 * _commission_rate(config)) + (2 * _slippage_rate(config))

    summary = {
        "starting_capital": config.starting_capital,
        "final_equity": final_equity,
        "net_total_return": total_return,
        "buy_and_hold_return": float(benchmark),
        "max_drawdown": max_drawdown,
        "closed_trades": int(len(trades)),
        "win_rate": None if trades.empty else float((trades["net_pnl"] > 0).mean()),
        "average_win": None if wins.empty else float(wins.mean()),
        "average_loss": None if losses.empty else float(losses.mean()),
        "expectancy": None if trades.empty else float(trades["net_pnl"].mean()),
        "profit_factor": None if trades.empty else (float("inf") if gross_loss == 0 and gross_profit > 0 else _safe_ratio(gross_profit, gross_loss)),
        "exposure": float(bars_with_position / len(frame)),
        "directional_accuracy": float((frame["prediction"].astype(int) == frame["label"].astype(int)).mean()),
        "signal_coverage": float((frame["prediction"].astype(int) != 0).mean()),
        "settings": asdict(config),
    }
    return BacktestResult(summary=summary, trades=trades, equity_curve=equity_curve)
