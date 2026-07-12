"""Tests for risk_optimizer module."""

from __future__ import annotations

import pytest

from jp_signal.risk import RiskConfig
from jp_signal.risk_optimizer import OptimizationDiagnostics, optimize_orders


def _make_risk_config(**kwargs) -> RiskConfig:
    defaults = dict(
        max_orders_per_day=10,
        max_gross_exposure_yen=100_000_000.0,
        max_single_name_exposure_yen=20_000_000.0,
        max_long_exposure_yen=100_000_000.0,
        max_short_exposure_yen=100_000_000.0,
        max_net_exposure_yen=100_000_000.0,
        allow_short_without_confirmed_shortability=False,
        require_both_sides=False,
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


def test_optimize_orders_no_candidates():
    risk = _make_risk_config()
    selected, diag = optimize_orders([], risk)
    assert selected == []
    assert diag.n_candidates == 0


def test_optimize_orders_simple():
    candidates = [
        {"code": "1001", "side": "BUY", "score": 1.0, "entry_value": 1_000_000},
        {"code": "1002", "side": "SELL", "score": 0.8, "entry_value": 2_000_000},
        {"code": "1003", "side": "BUY", "score": 0.5, "entry_value": 500_000},
    ]
    risk = _make_risk_config()
    selected, diag = optimize_orders(candidates, risk)
    assert len(selected) > 0
    assert isinstance(diag, OptimizationDiagnostics)


def test_optimize_orders_order_limit():
    candidates = [
        {"code": f"{i:04d}", "side": "BUY", "score": float(i), "entry_value": 1_000_000}
        for i in range(20)
    ]
    risk = _make_risk_config(max_orders_per_day=3)
    selected, diag = optimize_orders(candidates, risk)
    assert len(selected) <= 3


def test_optimize_orders_gross_limit():
    candidates = [
        {"code": "1001", "side": "BUY", "score": 1.0, "entry_value": 80_000_000},
        {"code": "1002", "side": "BUY", "score": 0.9, "entry_value": 80_000_000},
    ]
    risk = _make_risk_config(max_gross_exposure_yen=100_000_000)
    selected, diag = optimize_orders(candidates, risk)
    assert len(selected) <= 1


def test_optimize_orders_with_existing():
    candidates = [
        {"code": "1001", "side": "BUY", "score": 1.0, "entry_value": 60_000_000},
    ]
    risk = _make_risk_config(max_long_exposure_yen=100_000_000, max_gross_exposure_yen=100_000_000)
    selected, diag = optimize_orders(candidates, risk, existing_long=50_000_000)
    assert len(selected) == 1


def test_optimize_orders_existing_blocks():
    candidates = [
        {"code": "1001", "side": "BUY", "score": 1.0, "entry_value": 60_000_000},
    ]
    risk = _make_risk_config(max_long_exposure_yen=100_000_000)
    selected, diag = optimize_orders(candidates, risk, existing_long=60_000_000)
    assert len(selected) == 1


def test_optimization_diagnostics_defaults():
    diag = OptimizationDiagnostics(status=0, fun=0.0, message="test")
    assert diag.n_candidates == 0
    assert diag.n_selected == 0
    assert diag.details == {}


def test_optimization_diagnostics_full():
    diag = OptimizationDiagnostics(
        status=1,
        fun=-5.0,
        message="optimization succeeded",
        n_candidates=10,
        n_selected=3,
        details={"solver": "scipy.milp"},
    )
    assert diag.status == 1
    assert diag.n_selected == 3
