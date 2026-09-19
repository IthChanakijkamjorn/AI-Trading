from __future__ import annotations

import pandas as pd

LABEL_DOWN = -1
LABEL_HOLD = 0
LABEL_UP = 1
LABEL_NAMES = {LABEL_DOWN: "DOWN", LABEL_HOLD: "NEUTRAL", LABEL_UP: "UP"}


def make_direction_labels(frame: pd.DataFrame, *, horizon: int, neutral_threshold: float) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1 bar.")
    if neutral_threshold < 0:
        raise ValueError("Neutral threshold must be non-negative.")

    labeled = frame.copy()
    labeled["future_close"] = labeled["close"].shift(-horizon)
    labeled["future_return"] = (labeled["future_close"] / labeled["close"]) - 1
    labeled["label"] = pd.NA
    labeled.loc[labeled["future_return"] > neutral_threshold, "label"] = LABEL_UP
    labeled.loc[labeled["future_return"] < -neutral_threshold, "label"] = LABEL_DOWN
    labeled.loc[
        labeled["future_return"].between(-neutral_threshold, neutral_threshold, inclusive="both"),
        "label",
    ] = LABEL_HOLD
    return labeled


def labeled_subset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["label"].notna()].copy()
