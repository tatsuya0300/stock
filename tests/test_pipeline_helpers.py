"""pipeline helper tests (shortability → order_builder に移譲済み)."""

from datetime import date

import pandas as pd

from jp_signal.order_builder import is_shortable_asof


def test_latest_shortable_empty_df_is_false():
    assert is_shortable_asof(pd.DataFrame(), "7203", date(2024, 1, 5)) is False


def test_latest_shortable_unknown_code_is_false():
    df = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "is_margin_lendable": 1,
                "short_restricted": 0,
            }
        ]
    )
    assert is_shortable_asof(df, "9999", date(2024, 1, 5)) is False


def test_latest_shortable_confirmed_true():
    df = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "is_margin_lendable": 1,
                "short_restricted": 0,
            }
        ]
    )
    assert is_shortable_asof(df, "7203", date(2024, 1, 5)) is True


def test_latest_shortable_true():
    df = pd.DataFrame(
        [
            ("7203", "2024-01-04", 1, 0),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    assert is_shortable_asof(df, "7203", date(2024, 1, 5)) is True


def test_latest_shortable_missing_is_false():
    df = pd.DataFrame(
        [
            ("7203", "2024-01-04", 1, 0),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    assert is_shortable_asof(df, "6758", date(2024, 1, 5)) is False


def test_latest_shortable_restricted_is_false():
    df = pd.DataFrame(
        [
            ("7203", "2024-01-04", 1, 1),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    assert is_shortable_asof(df, "7203", date(2024, 1, 5)) is False


def test_latest_shortable_future_snapshot_ignored():
    df = pd.DataFrame(
        [
            ("7203", "2024-01-06", 1, 0),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    assert is_shortable_asof(df, "7203", date(2024, 1, 5)) is False
