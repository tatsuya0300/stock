"""price data quality tests."""

import pandas as pd
import pytest

from jp_signal.data_quality import REQUIRED_PRICE_COLS, validate_prices


def _valid_prices():
    return pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "adj_open": 100,
                "adj_high": 110,
                "adj_low": 90,
                "adj_close": 105,
                "volume": 1000,
                "turnover": 105000,
            }
        ]
    )


def test_validate_prices_accepts_valid_rows():
    df = validate_prices(_valid_prices(), strict=True)
    assert list(df.columns) == REQUIRED_PRICE_COLS
    assert len(df) == 1


def test_validate_prices_drops_invalid_rows_when_not_strict():
    df = _valid_prices()
    df.loc[0, "high"] = 80

    out = validate_prices(df, strict=False)

    assert out.empty


def test_validate_prices_raises_when_strict():
    df = _valid_prices()
    df.loc[0, "high"] = 80

    with pytest.raises(ValueError):
        validate_prices(df, strict=True)


def test_missing_required_columns_raises():
    df = _valid_prices().drop(columns=["adj_close"])

    with pytest.raises(ValueError):
        validate_prices(df)
