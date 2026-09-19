from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def price_figure(frame: pd.DataFrame, title: str, signal_frame: pd.DataFrame | None = None) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=frame["timestamp"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Price",
        )
    )
    if "sma_10_distance" in frame.columns:
        sma_10 = frame["close"].rolling(10, min_periods=10).mean()
        sma_20 = frame["close"].rolling(20, min_periods=20).mean()
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=sma_10, mode="lines", name="SMA 10"))
        figure.add_trace(go.Scatter(x=frame["timestamp"], y=sma_20, mode="lines", name="SMA 20"))
    if signal_frame is not None and not signal_frame.empty:
        buys = signal_frame.loc[signal_frame["prediction"] == 1]
        sells = signal_frame.loc[signal_frame["prediction"] == -1]
        if not buys.empty:
            figure.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["close"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10),
                    name="BUY signal",
                )
            )
        if not sells.empty:
            figure.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["close"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10),
                    name="SELL signal",
                )
            )
    figure.update_layout(title=title, xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False, height=520)
    return figure


def equity_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame["equity"], mode="lines", name="Equity"))
    figure.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Equity", height=360)
    return figure
