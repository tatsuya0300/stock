"""order_builder監査（reject理由保存）の回帰テスト。"""

from __future__ import annotations

import pandas as pd

from jp_signal.order_builder import build_orders_with_audit
from jp_signal.risk import RiskConfig


def make_risk() -> RiskConfig:
    return RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=100_000_000,
        max_single_name_exposure_yen=20_000_000,
        max_long_exposure_yen=50_000_000,
        max_short_exposure_yen=50_000_000,
        max_net_exposure_yen=50_000_000,
        require_both_sides=False,
        allow_short_without_confirmed_shortability=False,
    )


def test_missing_reference_price_is_audited():
    signals = pd.DataFrame(
        [
            {
                "code": "9999",
                "side": "BUY",
                "score": 1.0,
            }
        ]
    )

    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "close": 100.0,
                "turnover": 100_000_000.0,
            }
        ]
    )

    result = build_orders_with_audit(
        signals,
        prices,
        as_of="2024-01-05",
        sizing_cfg={
            "adv_window": 1,
            "min_adv_periods": 1,
            "require_full_adv_history": True,
            "allow_single_day_turnover_fallback": False,
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "market_open_unit_cap": 50,
        },
        risk_cfg=make_risk(),
    )

    assert result.selected.empty
    assert not result.rejected.empty
    assert result.rejected.iloc[0]["reason"] == "NO_REFERENCE_PRICE"
    assert result.rejected.iloc[0]["stage"] == "REFERENCE_DATA"


def test_unconfirmed_short_is_audited():
    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-03",
                "close": 100.0,
                "turnover": 100_000_000.0,
            },
            {
                "code": "7203",
                "date": "2024-01-04",
                "close": 100.0,
                "turnover": 100_000_000.0,
            },
        ]
    )

    signals = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "SELL",
                "score": 1.0,
            }
        ]
    )

    result = build_orders_with_audit(
        signals,
        prices,
        as_of="2024-01-05",
        sizing_cfg={
            "adv_window": 1,
            "min_adv_periods": 1,
            "require_full_adv_history": True,
            "allow_single_day_turnover_fallback": False,
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "market_open_unit_cap": 50,
        },
        risk_cfg=make_risk(),
    )

    assert result.selected.empty
    assert not result.rejected.empty
    assert "NOT_SHORTABLE" in set(result.rejected["reason"])


def test_empty_prices_returns_all_rejected():
    signals = pd.DataFrame(
        [
            {"code": "7203", "side": "BUY", "score": 1.0},
            {"code": "6758", "side": "SELL", "score": 2.0},
        ]
    )

    result = build_orders_with_audit(
        signals,
        prices=pd.DataFrame(),
        as_of="2024-01-05",
        sizing_cfg={
            "adv_window": 1,
            "min_adv_periods": 1,
            "require_full_adv_history": True,
            "allow_single_day_turnover_fallback": False,
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "market_open_unit_cap": 50,
        },
        risk_cfg=make_risk(),
    )

    assert result.selected.empty
    assert len(result.rejected) == 2
    assert set(result.rejected["reason"]) == {"EMPTY_PRICES"}
