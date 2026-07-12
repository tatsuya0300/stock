"""pipeline.make_datasource() tests."""

import pytest

from jp_signal.datasource import JQuantsSource, YFinanceSource
from jp_signal.pipeline import make_datasource


def test_jquants_source_defaults():
    """make_datasource が JQuantsSource を正しいデフォルトで生成する。"""
    cfg = {
        "data": {
            "source": "jquants",
            "jquants_api_key": "test-key",
        },
    }
    ds = make_datasource(cfg)
    assert isinstance(ds, JQuantsSource)
    assert ds.plan == "free"
    assert ds.sleep_sec == 12.0


def test_jquants_source_with_plan():
    """プラン指定が反映される。"""
    cfg = {
        "data": {
            "source": "jquants",
            "jquants_api_key": "test-key",
            "jquants_plan": "light",
        },
    }
    ds = make_datasource(cfg)
    assert isinstance(ds, JQuantsSource)
    assert ds.plan == "light"
    assert ds.sleep_sec == 1.0


def test_jquants_source_explicit_sleep():
    """明示的な sleep_sec が反映される。"""
    cfg = {
        "data": {
            "source": "jquants",
            "jquants_api_key": "test-key",
            "jquants_plan": "free",
            "jquants_sleep_sec": 30.0,
        },
    }
    ds = make_datasource(cfg)
    assert isinstance(ds, JQuantsSource)
    assert ds.sleep_sec == 30.0


def test_yfinance_source_defaults():
    """make_datasource が YFinanceSource を正しいデフォルトで生成する。"""
    cfg = {
        "data": {
            "source": "yfinance",
        },
    }
    ds = make_datasource(cfg)
    assert isinstance(ds, YFinanceSource)
    assert ds.chunk_size == 50


def test_unsupported_source_raises():
    """未対応の source は ValueError。"""
    cfg = {
        "data": {
            "source": "bloomberg",
        },
    }
    with pytest.raises(ValueError, match="bloomberg"):
        make_datasource(cfg)


def test_missing_data_section_uses_defaults():
    """data セクションが空でもエラーにならない。"""
    cfg = {"data": {}}
    with pytest.raises(ValueError, match="unsupported"):
        make_datasource(cfg)
