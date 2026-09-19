# AI Trading Research

Local-first Streamlit app for **educational trading research only**. It trains a simple historical classifier on daily OHLCV data, shows a latest BUY / SELL / HOLD recommendation, runs an honest held-out long-only backtest, and stores recommendation/backtest logs in local SQLite.

## Important disclaimer

- This project does **not** place trades, connect to brokers, or store trading credentials.
- It is **not** financial advice.
- Historical out-of-sample results do **not** guarantee future performance.
- The bundled demo dataset is **synthetic** and is only for offline walkthroughs/testing.
- The model outputs are classifier scores/probabilities, **not guaranteed probabilities of profit**.

## Supported Python

- Python 3.12

## Stack and chosen scope

- Streamlit UI
- pandas / numpy for data work
- scikit-learn logistic regression with training-only preprocessing
- Plotly charts
- SQLite journal stored outside the repo by default at `~/.ai_trading/journal.db`
- Daily OHLCV workflow with:
  - bundled synthetic demo CSV (offline)
  - user CSV upload
  - optional Yahoo Finance historical download

This MVP intentionally avoids broker integration, order execution, short selling, deep learning, and multi-user deployment concerns.

## Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Launch

```bash
streamlit run app.py
```

## Run tests

```bash
pytest -q
```

## Offline demo walkthrough

1. Install dependencies.
2. Run `streamlit run app.py`.
3. Keep **Bundled synthetic demo CSV** selected.
4. Click **Load data**.
5. Click **Train / evaluate model**.
6. Review:
   - chronological split boundaries and held-out metrics
   - latest directional forecast and position-aware BUY/SELL/HOLD action
   - held-out long-only backtest and trade log
7. Save a recommendation or backtest run to the local journal.
8. Restart the app and open **Journal** to confirm persistence.

## CSV format

CSV uploads must contain completed bars with these columns (case-insensitive aliases accepted):

- `timestamp` (or `date` / `datetime`)
- `open`
- `high`
- `low`
- `close`
- `volume`

Rules:
- one row per completed bar
- strictly increasing unique timestamps
- positive open/high/low/close
- non-negative volume
- `high >= open, low, close`
- `low <= open, high, close`

## Signal semantics

- The classifier predicts **UP / NEUTRAL / DOWN** over a configurable future horizon.
- Position-aware actions are long-only:
  - **BUY**: open a simulated long if currently flat
  - **SELL**: close an existing simulated long
  - **HOLD**: do nothing
- A bearish forecast while flat does **not** open a short.
- For the latest signal page:
  - BUY shows **estimated** ATR-based stop loss / take profit around the latest completed-bar reference price.
  - SELL and HOLD show no new stop/target because no new long is opened.
- In backtests, actual trade levels are based on the **next bar open fill** and ATR known when the signal was made.

## Backtest assumptions

- Uses only chronologically held-out predictions.
- Signal from completed bar `t` executes at the **next available bar open**.
- One long position at a time; no leverage and no negative cash.
- Commission and slippage are applied on both sides.
- Stop/target execution uses only future observable OHLC bars.
- Gap moves through the stop/target execute conservatively at the next open.
- If both stop and target are touched in the same OHLC bar and intrabar ordering is unknown, the app assumes **stop first**.
- Final-bar signals are not executed if no next bar exists.
- Reported metrics separate directional accuracy, signal coverage, and trade win rate.

## Modeling notes

- Features are causal/rolling only: returns, volatility, RSI, moving-average distance, ATR, and volume changes.
- Labels use future-return thresholds for **UP / NEUTRAL / DOWN** and leave unavailable future labels blank.
- Preprocessing (imputation/scaling) is fitted on training data only.
- Validation and test evaluation are chronological with a purge gap equal to the forecast horizon.
- The latest-signal model is retrained on available labeled history for inference, but that model is not reused to fabricate out-of-sample backtests.

## Architecture

- `app.py` — Streamlit UI and workflows
- `ai_trading/data.py` — input loading, normalization, validation, provenance
- `ai_trading/features.py` — causal feature engineering
- `ai_trading/labels.py` — horizon-based labeling
- `ai_trading/splits.py` — chronological splits and purge logic
- `ai_trading/model.py` — training, held-out prediction, latest signal generation
- `ai_trading/backtest.py` — long-only event-loop simulator
- `ai_trading/persistence.py` — SQLite journal/backtest persistence and safe CSV export
- `ai_trading/ui.py` — charts
- `data/demo_daily.csv` — synthetic offline dataset
- `tests/` — offline pytest coverage

## Persistence and journal behavior

- Journal data is stored locally in SQLite outside the tracked source tree by default.
- Recommendation saves are idempotent for the same signal/configuration payload.
- Outcome refresh resolves only recommendations with enough later historical bars.
- Pending recommendations are left unresolved rather than marked wrong.
- CSV exports are sanitized to reduce spreadsheet formula injection risk.

## Security and operational notes

- No pickle/joblib uploads or deserialization are supported.
- SQL writes use parameterized statements.
- Uploaded CSV size is bounded in code.
- The app is local-first; a public multi-user deployment would need authentication, access controls, and isolated storage.

## Limitations

- Daily data only in this MVP.
- Yahoo Finance can fail or return no data depending on ticker/range/network.
- Model probabilities are not calibrated.
- The app focuses on a coherent, tested research workflow rather than maximizing win rate.
