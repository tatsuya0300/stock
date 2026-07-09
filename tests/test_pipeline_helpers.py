"""pipeline helper tests."""

from datetime import date

import pandas as pd

from jp_signal.pipeline import _latest_shortable


def test_latest_shortable_true():
    df = pd.DataFrame(
        [
            ("7203", "2024-01-04", 1, 0),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    assert _latest_shortable(df, "7203", date(2024, 1, 5)) is True


def test_latest_shortable_missing_is_false():
    df = pd.DataFrame(
        [
            ("7203", "2024-01-04", 1, 0),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    assert _latest_shortable(df, "6758", date(2024, 1, 5)) is False


def test_latest_shortable_restricted_is_false():
    df = pd.DataFrame(
        [
            ("7203", "2024-01-04", 1, 1),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    assert _latest_shortable(df, "7203", date(2024, 1, 5)) is False


def test_latest_shortable_future_snapshot_ignored():
    df = pd.DataFrame(
        [
            ("7203", "2024-01-06", 1, 0),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    assert _latest_shortable(df, "7203", date(2024, 1, 5)) is False
