from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SplitDefinition:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    purge_gap: int

    def summary(self, timestamps: pd.Series) -> dict[str, object]:
        return {
            "train_rows": int(len(self.train_idx)),
            "val_rows": int(len(self.val_idx)),
            "test_rows": int(len(self.test_idx)),
            "purge_gap": int(self.purge_gap),
            "train_start": timestamps.iloc[self.train_idx[0]] if len(self.train_idx) else None,
            "train_end": timestamps.iloc[self.train_idx[-1]] if len(self.train_idx) else None,
            "val_start": timestamps.iloc[self.val_idx[0]] if len(self.val_idx) else None,
            "val_end": timestamps.iloc[self.val_idx[-1]] if len(self.val_idx) else None,
            "test_start": timestamps.iloc[self.test_idx[0]] if len(self.test_idx) else None,
            "test_end": timestamps.iloc[self.test_idx[-1]] if len(self.test_idx) else None,
        }


class SplitError(ValueError):
    """Raised when chronological split constraints cannot be met."""


def chronological_split(
    n_rows: int,
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    purge_gap: int = 1,
    min_train_rows: int = 60,
    min_segment_rows: int = 20,
) -> SplitDefinition:
    if n_rows <= 0:
        raise SplitError("At least one labeled row is required.")
    if not (0 < val_fraction < 0.4 and 0 < test_fraction < 0.4 and val_fraction + test_fraction < 0.8):
        raise SplitError("Validation and test fractions must both be between 0 and 0.4.")

    val_count = max(min_segment_rows, int(n_rows * val_fraction))
    test_count = max(min_segment_rows, int(n_rows * test_fraction))
    train_count = n_rows - val_count - test_count - purge_gap
    if train_count < min_train_rows:
        raise SplitError("Not enough rows for the requested chronological split.")

    train_end = train_count
    val_start = train_end + purge_gap
    test_start = val_start + val_count
    val_end = test_start
    test_end = test_start + test_count

    if test_end > n_rows:
        overflow = test_end - n_rows
        train_end -= overflow
        val_start = train_end + purge_gap
        test_start = val_start + val_count
        val_end = test_start
        test_end = n_rows

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(val_start, val_end)
    test_idx = np.arange(test_start, test_end)
    if len(train_idx) < min_train_rows or len(val_idx) < min_segment_rows or len(test_idx) < min_segment_rows:
        raise SplitError("Insufficient data after applying the purge gap.")
    return SplitDefinition(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, purge_gap=purge_gap)


def purged_history_indices(target_index: int, purge_gap: int) -> np.ndarray:
    cutoff = max(target_index - purge_gap, 0)
    return np.arange(0, cutoff)
