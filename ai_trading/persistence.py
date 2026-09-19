from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from ai_trading.constants import CSV_EXPORT_PREFIXES, DEFAULT_DB_PATH
from ai_trading.labels import LABEL_DOWN, LABEL_HOLD, LABEL_UP


class JournalStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_key TEXT NOT NULL UNIQUE,
                    record_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    as_of_timestamp TEXT NOT NULL,
                    created_timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    forecast TEXT NOT NULL,
                    reference_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    horizon INTEGER NOT NULL,
                    feature_version TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    training_cutoff TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    data_fingerprint TEXT NOT NULL,
                    is_synthetic INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS recommendation_outcomes (
                    recommendation_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    observed_timestamp TEXT,
                    observed_close REAL,
                    realized_return REAL,
                    observed_label TEXT,
                    is_correct INTEGER,
                    evaluated_timestamp TEXT NOT NULL,
                    FOREIGN KEY(recommendation_id) REFERENCES recommendations(id)
                );
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    created_timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    configuration_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    signal_time TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    gross_entry_value REAL,
                    gross_exit_value REAL,
                    net_pnl REAL,
                    return_pct REAL,
                    exit_reason TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    total_costs REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id)
                );
                """
            )

    def save_recommendation(self, payload: dict[str, object]) -> int:
        serializable = json.dumps(payload, sort_keys=True, default=str)
        recommendation_key = hashlib.sha256(serializable.encode("utf-8")).hexdigest()
        values = {
            "recommendation_key": recommendation_key,
            "record_type": payload["record_type"],
            "symbol": payload["symbol"],
            "source": payload["source"],
            "interval": payload["interval"],
            "as_of_timestamp": str(payload["as_of_timestamp"]),
            "created_timestamp": str(payload["created_timestamp"]),
            "action": payload["action"],
            "forecast": payload["forecast"],
            "reference_price": payload["reference_price"],
            "stop_loss": payload.get("stop_loss"),
            "take_profit": payload.get("take_profit"),
            "horizon": payload["horizon"],
            "feature_version": payload["feature_version"],
            "model_version": payload["model_version"],
            "training_cutoff": str(payload["training_cutoff"]),
            "configuration_json": json.dumps(payload["configuration"], sort_keys=True, default=str),
            "probabilities_json": json.dumps(payload["probabilities"], sort_keys=True, default=str),
            "data_fingerprint": payload["data_fingerprint"],
            "is_synthetic": int(bool(payload.get("is_synthetic", False))),
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join(f":{key}" for key in values)
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR IGNORE INTO recommendations ({columns}) VALUES ({placeholders})",
                values,
            )
            row = connection.execute(
                "SELECT id FROM recommendations WHERE recommendation_key = ?",
                (recommendation_key,),
            ).fetchone()
        return int(row["id"])

    def list_recommendations(self, record_type: str | None = None) -> pd.DataFrame:
        query = """
            SELECT r.*, o.status, o.observed_timestamp, o.observed_close, o.realized_return, o.observed_label, o.is_correct
            FROM recommendations r
            LEFT JOIN recommendation_outcomes o ON o.recommendation_id = r.id
        """
        params: tuple[object, ...] = tuple()
        if record_type:
            query += " WHERE r.record_type = ?"
            params = (record_type,)
        query += " ORDER BY r.as_of_timestamp DESC"
        with self._connect() as connection:
            return pd.read_sql_query(query, connection, params=params)

    def evaluate_matured_recommendations(
        self,
        market_frame: pd.DataFrame,
        *,
        neutral_threshold: float,
        evaluated_timestamp: pd.Timestamp,
    ) -> int:
        market = market_frame.copy().reset_index(drop=True)
        market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
        timestamp_to_index = {timestamp.isoformat(): idx for idx, timestamp in enumerate(market["timestamp"])}
        inserted = 0
        with self._connect() as connection:
            recommendations = connection.execute(
                """
                SELECT r.* FROM recommendations r
                LEFT JOIN recommendation_outcomes o ON o.recommendation_id = r.id
                WHERE o.recommendation_id IS NULL
                ORDER BY r.as_of_timestamp ASC
                """
            ).fetchall()
            for recommendation in recommendations:
                as_of = pd.Timestamp(recommendation["as_of_timestamp"])
                idx = timestamp_to_index.get(as_of.isoformat())
                if idx is None:
                    continue
                target_idx = idx + int(recommendation["horizon"])
                if target_idx >= len(market):
                    continue
                observed_close = float(market.iloc[target_idx]["close"])
                reference_price = float(recommendation["reference_price"])
                realized_return = (observed_close / reference_price) - 1
                if realized_return > neutral_threshold:
                    observed_label = "UP"
                elif realized_return < -neutral_threshold:
                    observed_label = "DOWN"
                else:
                    observed_label = "NEUTRAL"
                is_correct = int(observed_label == recommendation["forecast"])
                connection.execute(
                    """
                    INSERT OR REPLACE INTO recommendation_outcomes (
                        recommendation_id, status, observed_timestamp, observed_close, realized_return,
                        observed_label, is_correct, evaluated_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recommendation["id"],
                        "resolved",
                        str(market.iloc[target_idx]["timestamp"]),
                        observed_close,
                        realized_return,
                        observed_label,
                        is_correct,
                        str(evaluated_timestamp),
                    ),
                )
                inserted += 1
        return inserted

    def save_backtest_run(
        self,
        *,
        symbol: str,
        summary: dict[str, object],
        configuration: dict[str, object],
        trades: pd.DataFrame,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO backtest_runs (run_id, created_timestamp, symbol, summary_json, configuration_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, str(pd.Timestamp.utcnow()), symbol, json.dumps(summary, default=str), json.dumps(configuration, default=str)),
            )
            for row in trades.to_dict(orient="records"):
                connection.execute(
                    """
                    INSERT INTO backtest_trades (
                        run_id, signal_time, entry_time, exit_time, entry_price, exit_price, quantity,
                        gross_entry_value, gross_exit_value, net_pnl, return_pct, exit_reason,
                        stop_loss, take_profit, total_costs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        str(row.get("signal_time")),
                        str(row.get("entry_time")),
                        str(row.get("exit_time")),
                        row.get("entry_price"),
                        row.get("exit_price"),
                        row.get("quantity"),
                        row.get("gross_entry_value"),
                        row.get("gross_exit_value"),
                        row.get("net_pnl"),
                        row.get("return_pct"),
                        row.get("exit_reason"),
                        row.get("stop_loss"),
                        row.get("take_profit"),
                        row.get("total_costs"),
                    ),
                )
        return run_id

    def list_backtest_runs(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query("SELECT * FROM backtest_runs ORDER BY created_timestamp DESC", connection)

    def list_backtest_trades(self, run_id: str) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY entry_time ASC",
                connection,
                params=(run_id,),
            )


def data_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_csv_bytes(frame: pd.DataFrame) -> bytes:
    sanitized = frame.copy()
    for column in sanitized.columns:
        sanitized[column] = sanitized[column].map(
            lambda value: f"'{value}" if isinstance(value, str) and value.startswith(CSV_EXPORT_PREFIXES) else value
        )
    return sanitized.to_csv(index=False).encode("utf-8")
