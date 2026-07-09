"""J-Quants V2 datasource tests."""

import pandas as pd

from jp_signal.datasource import _from_jquants_code, _standardize_jquants_v2_frame


def test_from_jquants_code():
    assert _from_jquants_code("86970") == "8697"
    assert _from_jquants_code("7203") == "7203"


def test_standardize_jquants_v2_frame():
    raw = pd.DataFrame(
        [
            {
                "Date": "2023-03-24",
                "Code": "86970",
                "O": 2047.0,
                "H": 2069.0,
                "L": 2035.0,
                "C": 2045.0,
                "Vo": 2202500.0,
                "Va": 4507051850.0,
                "AdjO": 2047.0,
                "AdjH": 2069.0,
                "AdjL": 2035.0,
                "AdjC": 2045.0,
            }
        ]
    )
    out = _standardize_jquants_v2_frame(raw)
    assert list(out.columns) == [
        "code", "date",
        "open", "high", "low", "close",
        "adj_open", "adj_high", "adj_low", "adj_close",
        "volume", "turnover",
    ]
    assert out.iloc[0]["code"] == "8697"
    assert out.iloc[0]["close"] == 2045.0
    assert out.iloc[0]["turnover"] == 4507051850.0
