from __future__ import annotations

import pandas as pd

from ai_trading.features import FEATURE_COLUMNS, build_features
from ai_trading.labels import labeled_subset, make_direction_labels
from ai_trading.splits import chronological_split, purged_history_indices


def test_future_changes_do_not_modify_prior_features(synthetic_frame: pd.DataFrame) -> None:
    original = build_features(synthetic_frame)
    modified_input = synthetic_frame.copy()
    modified_input.loc[modified_input.index[-1], "close"] *= 5
    modified = build_features(modified_input)
    pd.testing.assert_frame_equal(original.iloc[:-1][FEATURE_COLUMNS], modified.iloc[:-1][FEATURE_COLUMNS])


def test_labels_leave_unavailable_future_rows_unlabeled(synthetic_frame: pd.DataFrame) -> None:
    labeled = make_direction_labels(build_features(synthetic_frame), horizon=5, neutral_threshold=0.01)
    assert labeled["label"].tail(5).isna().all()
    subset = labeled_subset(labeled)
    assert len(subset) == len(labeled) - 5


def test_purged_split_excludes_horizon_overlap() -> None:
    split = chronological_split(200, purge_gap=5, min_train_rows=60, min_segment_rows=20)
    assert split.val_idx[0] - split.train_idx[-1] > 1
    assert purged_history_indices(split.test_idx[0], 5)[-1] == split.test_idx[0] - 6
