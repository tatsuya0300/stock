"""ロングショート制約のテスト。"""

import pandas as pd

from jp_signal.risk import RiskConfig, apply_order_risk_limits


def test_long_only_orders_are_rejected_when_both_sides_required() -> None:
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": False,
            },
            {
                "code": "6758",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 0.8,
                "shortable": False,
            },
        ]
    )

    risk = RiskConfig(
        require_both_sides=True,
        max_net_exposure_yen=500_000,
    )

    result = apply_order_risk_limits(orders, risk)

    assert result.empty


def test_balanced_long_short_orders_are_accepted() -> None:
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": False,
            },
            {
                "code": "6758",
                "side": "SELL",
                "value_yen": 1_000_000,
                "score": 0.9,
                "shortable": True,
            },
        ]
    )

    risk = RiskConfig(
        require_both_sides=True,
        max_net_exposure_yen=100_000,
    )

    result = apply_order_risk_limits(orders, risk)

    assert len(result) == 2
    assert set(result["side"]) == {"BUY", "SELL"}


def test_invalid_side_is_rejected() -> None:
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "HOLD",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": True,
            }
        ]
    )

    risk = RiskConfig(require_both_sides=False)

    result = apply_order_risk_limits(orders, risk)

    assert result.empty


def test_unconfirmed_short_is_rejected() -> None:
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": False,
            },
            {
                "code": "6758",
                "side": "SELL",
                "value_yen": 1_000_000,
                "score": 0.9,
                "shortable": False,
            },
        ]
    )

    risk = RiskConfig(
        require_both_sides=True,
        allow_short_without_confirmed_shortability=False,
    )

    result = apply_order_risk_limits(orders, risk)

    assert result.empty
