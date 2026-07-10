from __future__ import annotations

import pandas as pd

from jp_signal.walkforward import (
    make_walk_forward_folds,
    purge_overlapping_labels,
)


def test_walk_forward_has_embargo_gap():
    dates = pd.date_range(
        "2024-01-01",
        periods=20,
        freq="B",
    )

    folds = make_walk_forward_folds(
        dates,
        train_size=10,
        test_size=3,
        embargo_size=2,
    )

    assert folds

    first = folds[0]

    assert len(first.train_dates) == 10
    assert len(first.test_dates) == 3

    train_end_index = dates.get_loc(first.train_end)
    test_start_index = dates.get_loc(first.test_start)

    assert test_start_index - train_end_index - 1 == 2


def test_rolling_train_window_is_fixed():
    dates = pd.date_range(
        "2024-01-01",
        periods=30,
        freq="B",
    )

    folds = make_walk_forward_folds(
        dates,
        train_size=10,
        test_size=5,
        embargo_size=1,
        anchored=False,
    )

    assert all(len(fold.train_dates) == 10 for fold in folds)


def test_purge_removes_overlapping_labels():
    samples = pd.DataFrame(
        [
            {
                "signal_date": "2024-01-01",
                "label_end": "2024-01-04",
            },
            {
                "signal_date": "2024-01-02",
                "label_end": "2024-01-06",
            },
        ]
    )

    purged = purge_overlapping_labels(
        samples,
        test_start="2024-01-05",
    )

    assert len(purged) == 1
    assert purged.iloc[0] == pd.Timestamp("2024-01-01")
