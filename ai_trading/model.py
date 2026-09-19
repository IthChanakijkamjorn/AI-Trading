from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_trading.constants import FEATURE_VERSION, MODEL_VERSION
from ai_trading.data import MarketData
from ai_trading.features import FEATURE_COLUMNS, build_features
from ai_trading.labels import LABEL_DOWN, LABEL_HOLD, LABEL_NAMES, LABEL_UP, labeled_subset, make_direction_labels
from ai_trading.splits import SplitDefinition, chronological_split, purged_history_indices


class InsufficientClassCoverage(ValueError):
    """Raised when a split does not contain enough class coverage for training/evaluation."""


@dataclass(slots=True)
class ResearchConfig:
    horizon: int = 5
    neutral_threshold: float = 0.01
    min_history: int = 120
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    entry_threshold: float = 0.45
    stop_atr: float = 1.5
    target_atr: float = 3.0
    random_state: int = 42


@dataclass(slots=True)
class LatestSignal:
    timestamp: pd.Timestamp
    forecast: str
    action_if_flat: str
    action_if_long: str
    reference_price: float
    stop_loss: float | None
    take_profit: float | None
    horizon: int
    training_cutoff: pd.Timestamp
    probabilities: dict[str, float]
    explanation: list[dict[str, float | str]]


@dataclass(slots=True)
class ResearchArtifacts:
    config: ResearchConfig
    market_data: MarketData
    featured_frame: pd.DataFrame
    labeled_frame: pd.DataFrame
    split: SplitDefinition
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    validation_metrics: dict[str, object]
    test_metrics: dict[str, object]
    baseline_metrics: dict[str, object]
    latest_signal: LatestSignal
    selected_entry_threshold: float
    model_version: str = MODEL_VERSION
    feature_version: str = FEATURE_VERSION


def _build_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="newton-cg",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _probability_frame(probabilities: np.ndarray, classes: Iterable[int]) -> pd.DataFrame:
    frame = pd.DataFrame(probabilities, columns=list(classes))
    for label in (LABEL_DOWN, LABEL_HOLD, LABEL_UP):
        if label not in frame.columns:
            frame[label] = 0.0
    return frame[[LABEL_DOWN, LABEL_HOLD, LABEL_UP]].rename(
        columns={LABEL_DOWN: "prob_down", LABEL_HOLD: "prob_neutral", LABEL_UP: "prob_up"}
    )


def _apply_threshold(prob_frame: pd.DataFrame, threshold: float) -> pd.Series:
    predictions = pd.Series(LABEL_HOLD, index=prob_frame.index, dtype="int64")
    buy_mask = (prob_frame["prob_up"] >= threshold) & (prob_frame["prob_up"] > prob_frame["prob_down"])
    sell_mask = (prob_frame["prob_down"] >= threshold) & (prob_frame["prob_down"] > prob_frame["prob_up"])
    predictions.loc[buy_mask] = LABEL_UP
    predictions.loc[sell_mask] = LABEL_DOWN
    return predictions


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, object]:
    labels = [LABEL_DOWN, LABEL_HOLD, LABEL_UP]
    precision, recall, _, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": {LABEL_NAMES[label]: float(value) for label, value in zip(labels, precision)},
        "recall": {LABEL_NAMES[label]: float(value) for label, value in zip(labels, recall)},
        "support": {LABEL_NAMES[label]: int(value) for label, value in zip(labels, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _baseline_predictions(frame: pd.DataFrame, neutral_threshold: float) -> pd.Series:
    signal = pd.Series(LABEL_HOLD, index=frame.index, dtype="int64")
    signal.loc[frame["return_1"] > neutral_threshold] = LABEL_UP
    signal.loc[frame["return_1"] < -neutral_threshold] = LABEL_DOWN
    return signal


def _ensure_class_coverage(values: pd.Series, *, minimum_classes: int = 2) -> None:
    observed = values.dropna().astype(int).nunique()
    if observed < minimum_classes:
        raise InsufficientClassCoverage("Insufficient class coverage for training/evaluation.")


def _walk_forward_probabilities(
    frame: pd.DataFrame,
    evaluation_idx: np.ndarray,
    *,
    purge_gap: int,
    random_state: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    X = frame[FEATURE_COLUMNS]
    y = frame["label"].astype(int)

    for target_idx in evaluation_idx:
        history_idx = purged_history_indices(int(target_idx), purge_gap)
        if len(history_idx) == 0:
            continue
        y_train = y.iloc[history_idx]
        _ensure_class_coverage(y_train)
        pipeline = _build_pipeline(random_state=random_state)
        pipeline.fit(X.iloc[history_idx], y_train)
        probabilities = pipeline.predict_proba(X.iloc[[target_idx]])
        prob_frame = _probability_frame(probabilities, pipeline.named_steps["model"].classes_)
        records.append(
            {
                "row_index": int(target_idx),
                "timestamp": frame.iloc[target_idx]["timestamp"],
                "label": int(y.iloc[target_idx]),
                "open": float(frame.iloc[target_idx]["open"]),
                "high": float(frame.iloc[target_idx]["high"]),
                "low": float(frame.iloc[target_idx]["low"]),
                "close": float(frame.iloc[target_idx]["close"]),
                "atr_14": float(frame.iloc[target_idx]["atr_14"]),
                "return_1": float(frame.iloc[target_idx]["return_1"]),
                **prob_frame.iloc[0].to_dict(),
            }
        )
    return pd.DataFrame.from_records(records)


def _choose_entry_threshold(probabilities: pd.DataFrame, default_threshold: float) -> float:
    grid = sorted({default_threshold, 0.40, 0.45, 0.50, 0.55, 0.60})
    best_threshold = default_threshold
    best_score = -np.inf
    for threshold in grid:
        predicted = _apply_threshold(probabilities, threshold)
        precision, recall, _, _ = precision_recall_fscore_support(
            probabilities["label"],
            predicted,
            labels=[LABEL_DOWN, LABEL_HOLD, LABEL_UP],
            average="macro",
            zero_division=0,
        )
        score = float(precision + recall)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _final_model_explanation(pipeline: Pipeline, row: pd.DataFrame, predicted_label: int) -> list[dict[str, float | str]]:
    transformed = pipeline.named_steps["scaler"].transform(pipeline.named_steps["imputer"].transform(row[FEATURE_COLUMNS]))
    model = pipeline.named_steps["model"]
    classes = list(model.classes_)
    if predicted_label not in classes:
        return []
    class_index = classes.index(predicted_label)
    coefficients = model.coef_[class_index]
    contributions = coefficients * transformed[0]
    ranked = sorted(zip(FEATURE_COLUMNS, contributions), key=lambda item: abs(item[1]), reverse=True)[:3]
    explanation = []
    for feature, contribution in ranked:
        explanation.append({"feature": feature, "contribution": float(contribution), "value": float(row.iloc[0][feature])})
    return explanation


def _probabilities_to_dict(prob_row: pd.Series) -> dict[str, float]:
    return {
        "DOWN": float(prob_row["prob_down"]),
        "NEUTRAL": float(prob_row["prob_neutral"]),
        "UP": float(prob_row["prob_up"]),
    }


def _latest_signal(pipeline: Pipeline, featured_frame: pd.DataFrame, config: ResearchConfig, threshold: float, training_cutoff: pd.Timestamp) -> LatestSignal:
    latest_row = featured_frame.iloc[[-1]].copy()
    if latest_row[FEATURE_COLUMNS].isna().any(axis=None):
        raise ValueError("The latest completed bar does not yet have enough history for inference.")
    probability_frame = _probability_frame(pipeline.predict_proba(latest_row[FEATURE_COLUMNS]), pipeline.named_steps["model"].classes_)
    predicted_label = int(_apply_threshold(probability_frame, threshold).iloc[0])
    probs = _probabilities_to_dict(probability_frame.iloc[0])
    forecast = LABEL_NAMES[predicted_label]
    action_if_flat = "BUY" if predicted_label == LABEL_UP else "HOLD"
    action_if_long = "SELL" if predicted_label == LABEL_DOWN else "HOLD"
    reference_price = float(latest_row.iloc[0]["close"])
    atr_value = float(latest_row.iloc[0]["atr_14"])
    stop_loss = None
    take_profit = None
    if action_if_flat == "BUY" and np.isfinite(atr_value):
        stop_loss = reference_price - (atr_value * config.stop_atr)
        take_profit = reference_price + (atr_value * config.target_atr)
    return LatestSignal(
        timestamp=pd.Timestamp(latest_row.iloc[0]["timestamp"]),
        forecast=forecast,
        action_if_flat=action_if_flat,
        action_if_long=action_if_long,
        reference_price=reference_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        horizon=config.horizon,
        training_cutoff=training_cutoff,
        probabilities=probs,
        explanation=_final_model_explanation(pipeline, latest_row, predicted_label),
    )


def run_research_pipeline(market_data: MarketData, config: ResearchConfig) -> ResearchArtifacts:
    if config.horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")
    if config.min_history < 40:
        raise ValueError("Minimum history must be at least 40 rows.")
    if not (0 < config.entry_threshold < 1):
        raise ValueError("Entry threshold must be between 0 and 1.")
    if config.stop_atr <= 0 or config.target_atr <= 0:
        raise ValueError("ATR stop/target multipliers must be positive.")

    featured_frame = build_features(market_data.frame)
    labeled_frame = make_direction_labels(featured_frame, horizon=config.horizon, neutral_threshold=config.neutral_threshold)
    complete_rows = labeled_frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    labeled_complete = labeled_subset(complete_rows).reset_index(drop=True)

    if len(labeled_complete) < config.min_history:
        raise ValueError("Not enough labeled history after feature creation for training and evaluation.")
    _ensure_class_coverage(labeled_complete["label"])

    split = chronological_split(
        len(labeled_complete),
        val_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        purge_gap=config.horizon,
        min_train_rows=config.min_history,
        min_segment_rows=max(10, config.horizon * 2),
    )

    _ensure_class_coverage(labeled_complete.iloc[split.val_idx]["label"])
    _ensure_class_coverage(labeled_complete.iloc[split.test_idx]["label"])

    validation_predictions = _walk_forward_probabilities(
        labeled_complete,
        split.val_idx,
        purge_gap=config.horizon,
        random_state=config.random_state,
    )
    if validation_predictions.empty:
        raise ValueError("Validation predictions could not be generated.")
    selected_entry_threshold = _choose_entry_threshold(validation_predictions, config.entry_threshold)
    validation_predictions["prediction"] = _apply_threshold(validation_predictions, selected_entry_threshold)
    validation_predictions["forecast"] = validation_predictions["prediction"].map(LABEL_NAMES)

    test_predictions = _walk_forward_probabilities(
        labeled_complete,
        split.test_idx,
        purge_gap=config.horizon,
        random_state=config.random_state,
    )
    if test_predictions.empty:
        raise ValueError("Test predictions could not be generated.")
    test_predictions["prediction"] = _apply_threshold(test_predictions, selected_entry_threshold)
    test_predictions["forecast"] = test_predictions["prediction"].map(LABEL_NAMES)

    validation_metrics = _metrics(validation_predictions["label"].astype(int), validation_predictions["prediction"].astype(int))
    test_metrics = _metrics(test_predictions["label"].astype(int), test_predictions["prediction"].astype(int))

    baseline_pred = _baseline_predictions(test_predictions, config.neutral_threshold)
    baseline_metrics = _metrics(test_predictions["label"].astype(int), baseline_pred.astype(int))

    final_train = labeled_complete.copy()
    final_pipeline = _build_pipeline(config.random_state)
    final_pipeline.fit(final_train[FEATURE_COLUMNS], final_train["label"].astype(int))
    latest_featured_frame = complete_rows.copy()
    latest_signal = _latest_signal(
        final_pipeline,
        latest_featured_frame,
        config,
        selected_entry_threshold,
        training_cutoff=pd.Timestamp(final_train["timestamp"].iloc[-1]),
    )

    return ResearchArtifacts(
        config=config,
        market_data=market_data,
        featured_frame=featured_frame,
        labeled_frame=labeled_complete,
        split=split,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        latest_signal=latest_signal,
        selected_entry_threshold=selected_entry_threshold,
    )


def artifacts_to_jsonable(artifacts: ResearchArtifacts) -> dict[str, object]:
    payload = asdict(artifacts.config)
    payload.update(
        {
            "model_version": artifacts.model_version,
            "feature_version": artifacts.feature_version,
            "selected_entry_threshold": artifacts.selected_entry_threshold,
            "split_summary": artifacts.split.summary(artifacts.labeled_frame["timestamp"]),
            "validation_metrics": artifacts.validation_metrics,
            "test_metrics": artifacts.test_metrics,
            "baseline_metrics": artifacts.baseline_metrics,
            "latest_signal": {
                **asdict(artifacts.latest_signal),
                "timestamp": str(artifacts.latest_signal.timestamp),
                "training_cutoff": str(artifacts.latest_signal.training_cutoff),
            },
        }
    )
    return json.loads(json.dumps(payload, default=str))
