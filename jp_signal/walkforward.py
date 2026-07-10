"""Walk-forward and purged time-series split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]


def make_walk_forward_folds(
    dates: list[str] | pd.Series | pd.Index,
    *,
    train_size: int,
    test_size: int,
    embargo_size: int = 0,
    step_size: int | None = None,
    anchored: bool = False,
) -> list[WalkForwardFold]:
    """営業日配列からwalk-forward foldを生成する。

    train_size/test_size/embargo_sizeは観測営業日数単位。
    """
    if train_size < 1:
        raise ValueError(f"train_size must be >= 1: {train_size}")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1: {test_size}")
    if embargo_size < 0:
        raise ValueError(f"embargo_size must be >= 0: {embargo_size}")

    step = test_size if step_size is None else step_size

    if step < 1:
        raise ValueError(f"step_size must be >= 1: {step}")

    unique_dates = sorted(pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique())

    folds: list[WalkForwardFold] = []
    test_start_index = train_size + embargo_size
    fold_number = 0

    while test_start_index + test_size <= len(unique_dates):
        train_end_index = test_start_index - embargo_size - 1
        train_start_index = 0 if anchored else train_end_index - train_size + 1

        if train_start_index < 0:
            train_start_index = 0

        if train_end_index < train_start_index:
            break

        train_dates_tuple = tuple(unique_dates[train_start_index : train_end_index + 1])
        test_dates_tuple = tuple(unique_dates[test_start_index : test_start_index + test_size])

        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=unique_dates[train_start_index],
                train_end=unique_dates[train_end_index],
                test_start=unique_dates[test_start_index],
                test_end=unique_dates[test_start_index + test_size - 1],
                train_dates=train_dates_tuple,
                test_dates=test_dates_tuple,
            )
        )

        fold_number += 1
        test_start_index += step

        if not anchored:
            train_start_index = train_end_index - train_size + 1

    return folds


def purge_overlapping_labels(
    samples: pd.DataFrame,
    test_start: str | pd.Timestamp,
    *,
    label_end_col: str = "label_end",
    signal_date_col: str = "signal_date",
) -> pd.Series:
    """test開始以後までlabel期間が重なるtrain sampleを除外する。

    train label_end >= test_start となる行を除外し、
    残った行のsignal_dateを返す。
    """
    dates = samples.copy()

    if signal_date_col not in dates.columns:
        raise ValueError(f"samples missing column: {signal_date_col}")
    if label_end_col not in dates.columns:
        raise ValueError(f"samples missing column: {label_end_col}")

    dates[signal_date_col] = pd.to_datetime(
        dates[signal_date_col],
        errors="raise",
    )
    dates[label_end_col] = pd.to_datetime(
        dates[label_end_col],
        errors="raise",
    )

    test_start_ts = pd.Timestamp(test_start)

    purged = dates.loc[
        dates[label_end_col] < test_start_ts,
        signal_date_col,
    ]

    return purged
