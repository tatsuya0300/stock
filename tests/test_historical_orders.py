"""Tests for historical_orders module."""

from __future__ import annotations

from datetime import date

import pandas as pd

from jp_signal.historical_orders import HistoricalOrderResult, generate_historical_orders


def test_historical_order_result_empty():
    result = HistoricalOrderResult(
        orders=pd.DataFrame(),
        rejections=pd.DataFrame(),
    )
    assert result.orders.empty
    assert result.rejections.empty
    assert result.diagnostics == {}


def test_generate_historical_orders_empty_dates():
    def dummy_model_factory(cfg):
        return None

    def dummy_price_loader(codes, start, end):
        return pd.DataFrame()

    def dummy_universe_loader(as_of):
        return pd.DataFrame({"code": []})

    result = generate_historical_orders(
        trading_dates=[],
        model_factory=dummy_model_factory,
        price_loader=dummy_price_loader,
        universe_loader=dummy_universe_loader,
        model_cfg={},
        sizing_cfg={},
        risk_cfg={},
    )
    assert result.orders.empty
    assert result.diagnostics["dates_processed"] == 0


def test_generate_historical_orders_with_one_date():
    """minimal smoke test: our implementation actually calls the model factory
    and produces both diagnostics and (possibly empty/failed) orders."""
    calls = []

    class DummyModel:
        def generate(self, prices, as_of):
            return pd.DataFrame(
                {"code": ["1001"], "side": ["BUY"], "score": [1.0]}
            )

    def dummy_model_factory(cfg):
        calls.append(cfg)
        return DummyModel()

    def dummy_price_loader(codes, start, end):
        rows = []
        for code in codes:
            for d in pd.bdate_range(
                end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=30
            ):
                rows.append(
                    {
                        "code": code,
                        "date": d.strftime("%Y-%m-%d"),
                        "open": 100.0,
                        "high": 110.0,
                        "low": 90.0,
                        "close": 105.0,
                        "adj_open": 100.0,
                        "adj_high": 110.0,
                        "adj_low": 90.0,
                        "adj_close": 105.0,
                        "volume": 1_000_000,
                        "turnover": 105_000_000,
                    }
                )
        return pd.DataFrame(rows)

    def dummy_universe_loader(as_of):
        return pd.DataFrame({"code": ["1001"], "name": ["Test"]})

    result = generate_historical_orders(
        trading_dates=[date(2026, 7, 10)],
        model_factory=dummy_model_factory,
        price_loader=dummy_price_loader,
        universe_loader=dummy_universe_loader,
        model_cfg={"lookback": 5},
        sizing_cfg={
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "adv_window": 20,
            "min_adv_periods": 20,
            "require_full_adv_history": True,
        },
        risk_cfg={
            "require_both_sides": False,
            "max_orders_per_day": 10,
            "max_gross_exposure_yen": 100_000_000,
        },
    )
    assert result.diagnostics["dates_processed"] == 1
    assert len(calls) == 1


def test_historical_order_result_with_diagnostics():
    result = HistoricalOrderResult(
        orders=pd.DataFrame(),
        rejections=pd.DataFrame(),
        diagnostics={"dates_processed": 5, "errors": 0},
    )
    assert result.diagnostics["dates_processed"] == 5
