"""datasource tests."""

from datetime import date

import pandas as pd
import pytest

from jp_signal.datasource import (
    JQuantsSource,
    YFinanceSource,
    _standardize_jquants_v2_frame,
    _standardize_yfinance_frame,
)


def test_standardize_yfinance_frame():
    idx = pd.to_datetime(["2024-01-04", "2024-01-05"])
    raw = pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [105.0, 115.0],
            "Low": [95.0, 108.0],
            "Close": [104.0, 112.0],
            "Adj Close": [52.0, 56.0],
            "Volume": [1000, 2000],
        },
        index=idx,
    )
    raw.index.name = "Date"

    out = _standardize_yfinance_frame(raw, "7203")

    assert list(out.columns) == [
        "code",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
        "turnover",
    ]
    assert out.iloc[0]["code"] == "7203"
    assert out.iloc[0]["date"] == "2024-01-04"
    assert out.iloc[0]["open"] == 100.0
    assert out.iloc[0]["close"] == 104.0
    assert out.iloc[0]["adj_close"] == 52.0
    assert out.iloc[0]["adj_open"] == 50.0
    assert out.iloc[0]["turnover"] == 104000.0


def test_standardize_yfinance_frame_without_adj_close():
    idx = pd.to_datetime(["2024-01-04"])
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [104.0],
            "Volume": [1000],
        },
        index=idx,
    )
    raw.index.name = "Date"

    out = _standardize_yfinance_frame(raw, "7203")

    assert out.iloc[0]["adj_close"] == 104.0
    assert out.iloc[0]["adj_open"] == 100.0


def test_standardize_yfinance_missing_raw_column_raises():
    idx = pd.to_datetime(["2024-01-04"])
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            # Close missing
            "Volume": [1000],
        },
        index=idx,
    )
    raw.index.name = "Date"

    with pytest.raises(KeyError):
        _standardize_yfinance_frame(raw, "7203")


def test_standardize_jquants_v2_frame():
    raw = pd.DataFrame(
        [
            {
                "Date": "2024-01-04",
                "Code": "7203",
                "Open": 100.0,
                "High": 105.0,
                "Low": 95.0,
                "Close": 104.0,
                "AdjustmentOpen": 50.0,
                "AdjustmentHigh": 52.5,
                "AdjustmentLow": 47.5,
                "AdjustmentClose": 52.0,
                "Volume": 1000,
                "TurnoverValue": 104000.0,
            }
        ]
    )

    out = _standardize_jquants_v2_frame(raw)

    assert out.iloc[0]["code"] == "7203"
    assert out.iloc[0]["open"] == 100.0
    assert out.iloc[0]["adj_open"] == 50.0
    assert out.iloc[0]["turnover"] == 104000.0


def test_standardize_jquants_v2_missing_column_raises():
    raw = pd.DataFrame(
        [
            {
                "Date": "2024-01-04",
                "Code": "7203",
                "Open": 100.0,
                "High": 105.0,
                "Low": 95.0,
                "Close": 104.0,
                # AdjustmentOpen missing
                "AdjustmentHigh": 52.5,
                "AdjustmentLow": 47.5,
                "AdjustmentClose": 52.0,
                "Volume": 1000,
                "TurnoverValue": 104000.0,
            }
        ]
    )

    with pytest.raises(KeyError):
        _standardize_jquants_v2_frame(raw)


def test_yfinance_source_fetch_daily(monkeypatch):
    import yfinance as yf

    idx = pd.to_datetime(["2024-01-04"])
    fake = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [104.0],
            "Adj Close": [52.0],
            "Volume": [1000],
        },
        index=idx,
    )
    fake.index.name = "Date"

    def fake_download(*args, **kwargs):
        assert kwargs.get("auto_adjust") is False
        assert kwargs.get("repair") is True
        return fake

    monkeypatch.setattr(yf, "download", fake_download)

    ds = YFinanceSource()
    out = ds.fetch_daily(["7203"], date(2024, 1, 1), date(2024, 1, 6))

    assert len(out) == 1
    assert out.iloc[0]["code"] == "7203"
    assert out.iloc[0]["date"] == "2024-01-04"


def test_jquants_source_requires_api_key():
    with pytest.raises(ValueError):
        JQuantsSource("")
