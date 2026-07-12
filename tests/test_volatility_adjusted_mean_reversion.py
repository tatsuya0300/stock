"""Tests for VolatilityAdjustedMeanReversion model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jp_signal.model import (
    MeanReversionConfig,
    MeanReversionRule,
    SignalModel,
    VolatilityAdjustedMeanReversion,
)


def test_signal_model_is_abstract():
    with pytest.raises(TypeError):
        SignalModel()  # type: ignore[abstract]


def test_mean_reversion_config_defaults():
    cfg = MeanReversionConfig()
    assert cfg.lookback == 20
    assert cfg.z_entry == 0.5
    assert cfg.top_n == 10


def test_volatility_adjusted_mean_reversion_empty_prices():
    model = VolatilityAdjustedMeanReversion()
    result = model.generate(pd.DataFrame(), "2026-07-10")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_volatility_adjusted_mean_reversion_missing_columns():
    model = VolatilityAdjustedMeanReversion()
    prices = pd.DataFrame({"code": ["1001"], "date": ["2026-07-09"]})
    with pytest.raises(ValueError, match="missing columns"):
        model.generate(prices, "2026-07-10")


def test_volatility_adjusted_mean_reversion_from_dict():
    model = VolatilityAdjustedMeanReversion({"lookback": 10, "z_entry": 0.3})
    assert model.config.lookback == 10
    assert model.config.z_entry == 0.3


def _make_price_data(codes: list[str], days: int) -> pd.DataFrame:
    rows = []
    for i, code in enumerate(codes):
        base = 1000 + i * 100
        for d in range(days):
            rows.append({
                "code": code,
                "date": pd.Timestamp(f"2026-06-{10 + d:02d}"),
                "close": base + np.random.randn() * 5,
                "adj_close": base + np.random.randn() * 5,
                "turnover": 1_000_000 + np.random.randn() * 100_000,
            })
    return pd.DataFrame(rows)


def test_volatility_adjusted_mean_reversion_generates_signals():
    codes = [f"{1000 + i}" for i in range(20)]
    prices = _make_price_data(codes, 60)
    model = VolatilityAdjustedMeanReversion(MeanReversionConfig(lookback=10, top_n=5))
    result = model.generate(prices, "2026-07-10")
    if not result.empty:
        assert "code" in result.columns
        assert "side" in result.columns
        assert "score" in result.columns


def test_volatility_adjusted_mean_reversion_config():
    config = MeanReversionConfig(lookback=15, z_entry=0.8, vol_lookback=40)
    model = VolatilityAdjustedMeanReversion(config)
    assert model.config.lookback == 15
    assert model.config.z_entry == 0.8
    assert model.config.vol_lookback == 40


def test_mean_reversion_rule_still_works():
    prices = _make_price_data([f"{1000 + i}" for i in range(20)], 20)
    model = MeanReversionRule(lookback=5, top_n=3)
    result = model.generate(prices, "2026-07-10")
    assert isinstance(result, pd.DataFrame)
