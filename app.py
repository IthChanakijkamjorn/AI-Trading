from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from ai_trading import __version__
from ai_trading.backtest import BacktestConfig, run_backtest
from ai_trading.data import DataSourceError, DataValidationError, download_yfinance_data, load_csv_bytes, load_sample_data
from ai_trading.model import ResearchConfig, ResearchArtifacts, artifacts_to_jsonable, run_research_pipeline
from ai_trading.persistence import JournalStore, data_fingerprint, safe_csv_bytes
from ai_trading.ui import equity_figure, price_figure

st.set_page_config(page_title="AI Trading Research", layout="wide")


@st.cache_resource
def get_store() -> JournalStore:
    return JournalStore()


def _load_market_data(source_name: str, symbol: str, start: str, end: str, upload) -> object:
    if source_name == "Bundled synthetic demo CSV":
        return load_sample_data(symbol=symbol or "DEMO")
    if source_name == "Upload CSV":
        if upload is None:
            raise DataValidationError("Upload a CSV file first.")
        return load_csv_bytes(upload.getvalue(), filename=upload.name, symbol=symbol or "CSV")
    return download_yfinance_data(symbol=symbol, start=start, end=end)


def _show_metrics(metrics: dict[str, object], label: str) -> None:
    st.subheader(label)
    first, second, third = st.columns(3)
    first.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    second.metric("DOWN precision", f"{metrics['precision']['DOWN']:.2%}")
    third.metric("UP recall", f"{metrics['recall']['UP']:.2%}")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["DOWN", "NEUTRAL", "UP"], columns=["DOWN", "NEUTRAL", "UP"]))


def _signal_recommendation_record(artifacts: ResearchArtifacts, position_context: str) -> dict[str, object]:
    signal = artifacts.latest_signal
    action = signal.action_if_flat if position_context == "Flat" else signal.action_if_long
    return {
        "record_type": "demo_recommendation" if artifacts.market_data.is_synthetic else "recommendation",
        "symbol": artifacts.market_data.symbol,
        "source": artifacts.market_data.source,
        "interval": artifacts.market_data.interval,
        "as_of_timestamp": signal.timestamp,
        "created_timestamp": pd.Timestamp.utcnow(),
        "action": action,
        "forecast": signal.forecast,
        "reference_price": signal.reference_price,
        "stop_loss": signal.stop_loss if action == "BUY" else None,
        "take_profit": signal.take_profit if action == "BUY" else None,
        "horizon": signal.horizon,
        "feature_version": artifacts.feature_version,
        "model_version": artifacts.model_version,
        "training_cutoff": signal.training_cutoff,
        "configuration": artifacts_to_jsonable(artifacts),
        "probabilities": signal.probabilities,
        "data_fingerprint": data_fingerprint(artifacts.market_data.frame),
        "is_synthetic": artifacts.market_data.is_synthetic,
    }


def main() -> None:
    st.title("AI Trading Research")
    st.caption("Educational research app only. No broker connectivity, no order execution, and no guarantee of accuracy or profitability.")

    store = get_store()
    page = st.sidebar.radio("Navigation", ["Data & Training", "Latest Signal", "Backtest", "Journal"])

    with st.sidebar:
        st.markdown(f"**Version:** {__version__}")
        position_context = st.selectbox("Position context", ["Flat", "Long"], help="Used only when mapping the directional forecast into a position-aware BUY/SELL/HOLD recommendation.")

    with st.expander("Model and execution settings", expanded=(page == "Data & Training")):
        col1, col2, col3 = st.columns(3)
        symbol = col1.text_input("Ticker / symbol", value=st.session_state.get("symbol", "DEMO"))
        start = col2.date_input("Start date", value=pd.Timestamp("2022-01-01")).strftime("%Y-%m-%d")
        end = col3.date_input("End date", value=pd.Timestamp("2024-12-31")).strftime("%Y-%m-%d")
        source_name = st.selectbox("Data source", ["Bundled synthetic demo CSV", "Upload CSV", "Yahoo Finance (daily historical)"])
        upload = st.file_uploader("CSV file", type=["csv"], disabled=source_name != "Upload CSV")

        c1, c2, c3, c4 = st.columns(4)
        horizon = c1.number_input("Forecast horizon (bars)", min_value=1, max_value=30, value=5)
        neutral_threshold = c2.number_input("Neutral threshold", min_value=0.0, max_value=0.2, value=0.01, step=0.001, format="%.3f")
        entry_threshold = c3.number_input("Entry probability threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05, format="%.2f")
        min_history = c4.number_input("Minimum labeled history", min_value=40, max_value=500, value=120, step=10)

        c5, c6, c7 = st.columns(3)
        stop_atr = c5.number_input("Stop ATR multiple", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        target_atr = c6.number_input("Target ATR multiple", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
        allocation = c7.number_input("Backtest allocation", min_value=0.1, max_value=1.0, value=0.95, step=0.05)

        if st.button("Load data", type="primary"):
            try:
                market_data = _load_market_data(source_name, symbol, start, end, upload)
                st.session_state["market_data"] = market_data
                st.session_state["symbol"] = symbol
                st.success(f"Loaded {len(market_data.frame)} rows from {market_data.source}.")
            except (DataValidationError, DataSourceError, ValueError) as exc:
                st.error(str(exc))

        if st.button("Train / evaluate model"):
            market_data = st.session_state.get("market_data")
            if market_data is None:
                st.error("Load market data first.")
            else:
                try:
                    artifacts = run_research_pipeline(
                        market_data,
                        ResearchConfig(
                            horizon=int(horizon),
                            neutral_threshold=float(neutral_threshold),
                            min_history=int(min_history),
                            entry_threshold=float(entry_threshold),
                            stop_atr=float(stop_atr),
                            target_atr=float(target_atr),
                        ),
                    )
                    st.session_state["artifacts"] = artifacts
                    st.success("Model trained and evaluated on chronological held-out data.")
                except Exception as exc:
                    st.error(str(exc))

    market_data = st.session_state.get("market_data")
    artifacts: ResearchArtifacts | None = st.session_state.get("artifacts")

    if page == "Data & Training":
        if market_data is None:
            st.info("Load the bundled synthetic demo data, your own completed-bar CSV, or supported Yahoo Finance history to begin.")
            return
        meta1, meta2, meta3, meta4 = st.columns(4)
        meta1.metric("Source", market_data.source)
        meta2.metric("Interval", market_data.interval)
        meta3.metric("Last completed candle", str(market_data.last_completed_candle))
        meta4.metric("Freshness", market_data.freshness)
        if market_data.provenance_notes:
            for note in market_data.provenance_notes:
                st.warning(note)
        if market_data.is_synthetic:
            st.warning("This bundled dataset is synthetic demo data only and is not evidence of real market predictive success.")
        st.plotly_chart(price_figure(market_data.frame, f"{market_data.symbol} price history"), use_container_width=True)
        if artifacts is not None:
            split_summary = artifacts.split.summary(artifacts.labeled_frame["timestamp"])
            st.json({"split_summary": {key: str(value) for key, value in split_summary.items()}, "selected_entry_threshold": artifacts.selected_entry_threshold})
            _show_metrics(artifacts.validation_metrics, "Validation metrics")
            _show_metrics(artifacts.test_metrics, "Held-out test metrics")
            st.caption(f"Baseline test accuracy (causal prior-return heuristic): {artifacts.baseline_metrics['accuracy']:.2%}")

    elif page == "Latest Signal":
        if artifacts is None:
            st.info("Train the model first to generate a latest-signal recommendation.")
            return
        signal = artifacts.latest_signal
        action = signal.action_if_flat if position_context == "Flat" else signal.action_if_long
        col1, col2, col3 = st.columns(3)
        col1.metric("Directional forecast", signal.forecast)
        col2.metric("Position-aware action", action)
        col3.metric("Reference price", f"{signal.reference_price:.2f}")
        st.caption(
            f"As of {signal.timestamp}, trained on history through {signal.training_cutoff}. Probabilities are model scores from a held-out-tuned classifier and are not calibrated guarantees of profit."
        )
        prob_df = pd.DataFrame([signal.probabilities])
        st.bar_chart(prob_df.T)
        if action == "BUY":
            st.info(
                f"Estimated long-only levels from the latest completed bar: stop {signal.stop_loss:.2f} < reference {signal.reference_price:.2f} < target {signal.take_profit:.2f}. Actual simulated fills use the next bar open, not this reference price."
            )
        else:
            st.info("No new long stop-loss / take-profit is suggested for HOLD or SELL-to-close actions.")
        st.write("Feature contribution summary (descriptive only; not causal proof):")
        st.dataframe(pd.DataFrame(signal.explanation))
        if st.button("Save recommendation to journal"):
            recommendation_id = store.save_recommendation(_signal_recommendation_record(artifacts, position_context))
            st.success(f"Recommendation stored with id {recommendation_id}. Saving is idempotent for the same signal/configuration.")

    elif page == "Backtest":
        if artifacts is None:
            st.info("Train the model first to run the honest held-out backtest.")
            return
        backtest = run_backtest(
            artifacts.test_predictions,
            BacktestConfig(
                allocation=float(allocation),
                commission_bps=10.0,
                slippage_bps=5.0,
                stop_atr=artifacts.config.stop_atr,
                target_atr=artifacts.config.target_atr,
                max_holding_bars=artifacts.config.horizon,
                entry_threshold=artifacts.selected_entry_threshold,
            ),
        )
        st.plotly_chart(price_figure(artifacts.test_predictions, "Held-out test window with signals", artifacts.test_predictions), use_container_width=True)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Net return", f"{backtest.summary['net_total_return']:.2%}")
        metric_cols[1].metric("Max drawdown", f"{backtest.summary['max_drawdown']:.2%}")
        metric_cols[2].metric("Win rate", "N/A" if backtest.summary["win_rate"] is None else f"{backtest.summary['win_rate']:.2%}")
        metric_cols[3].metric("Buy & hold", f"{backtest.summary['buy_and_hold_return']:.2%}")
        st.caption("Signals come from chronologically held-out predictions. BUY opens a simulated long at the next bar open. SELL only closes an existing simulated long. Gap stops are handled conservatively, and stop is assumed first when stop and target both touch in the same bar.")
        st.plotly_chart(equity_figure(backtest.equity_curve), use_container_width=True)
        st.dataframe(backtest.trades)
        if not backtest.trades.empty:
            st.download_button("Download trades CSV", data=safe_csv_bytes(backtest.trades), file_name="backtest_trades.csv", mime="text/csv")
        if st.button("Save backtest run"):
            run_id = store.save_backtest_run(
                symbol=artifacts.market_data.symbol,
                summary=backtest.summary,
                configuration={**asdict(artifacts.config), "entry_threshold": artifacts.selected_entry_threshold},
                trades=backtest.trades,
            )
            st.success(f"Saved backtest run {run_id}.")

    else:
        recommendations = store.list_recommendations()
        if st.button("Refresh matured outcomes using current loaded data"):
            if market_data is None:
                st.error("Load matching historical data first.")
            else:
                count = store.evaluate_matured_recommendations(
                    market_data.frame,
                    neutral_threshold=artifacts.config.neutral_threshold if artifacts else 0.01,
                    evaluated_timestamp=pd.Timestamp.utcnow(),
                )
                st.success(f"Resolved {count} matured recommendations.")
                recommendations = store.list_recommendations()
        if recommendations.empty:
            st.info("No saved recommendations or backtests yet.")
        else:
            st.dataframe(recommendations)
            st.download_button("Download journal CSV", data=safe_csv_bytes(recommendations), file_name="recommendation_journal.csv", mime="text/csv")
        runs = store.list_backtest_runs()
        if not runs.empty:
            st.subheader("Saved backtest runs")
            st.dataframe(runs)


if __name__ == "__main__":
    main()
