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
    calls = []

    def dummy_model_factory(cfg):
        calls.append(cfg)

        class DummyModel:
            def generate(self, prices, as_of):
                return pd.DataFrame({"code": ["1001"], "side": ["BUY"], "score": [1.0]})

        return DummyModel()

    def dummy_price_loader(codes, start, end):
        return pd.DataFrame({"code": ["1001"], "date": [str(start)], "close": [100.0]})

    def dummy_universe_loader(as_of):
        return pd.DataFrame({"code": ["1001"]})

    result = generate_historical_orders(
        trading_dates=[date(2026, 7, 10)],
        model_factory=dummy_model_factory,
        price_loader=dummy_price_loader,
        universe_loader=dummy_universe_loader,
        model_cfg={"lookback": 20, "top_n": 5},
        sizing_cfg={},
        risk_cfg={},
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
