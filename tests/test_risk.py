"""risk limit tests."""

import pandas as pd

from jp_signal.risk import RiskConfig, apply_order_risk_limits


def _orders():
    return pd.DataFrame(
        [
            {
                "code": "A",
                "side": "BUY",
                "value_yen": 100_000,
                "shortable": True,
                "score": 0.9,
            },
            {
                "code": "B",
                "side": "SELL",
                "value_yen": 100_000,
                "shortable": False,
                "score": 0.8,
            },
            {
                "code": "C",
                "side": "BUY",
                "value_yen": 1_000_000,
                "shortable": True,
                "score": 0.7,
            },
        ]
    )


def test_drops_unconfirmed_short():
    risk = RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=10_000_000,
        max_single_name_exposure_yen=10_000_000,
        max_long_exposure_yen=10_000_000,
        max_short_exposure_yen=10_000_000,
        require_both_sides=False,
        allow_short_without_confirmed_shortability=False,
    )

    out = apply_order_risk_limits(_orders(), risk)

    assert "B" not in out["code"].tolist()


def test_single_name_limit():
    risk = RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=10_000_000,
        max_single_name_exposure_yen=500_000,
        max_long_exposure_yen=10_000_000,
        max_short_exposure_yen=10_000_000,
        require_both_sides=False,
    )

    out = apply_order_risk_limits(_orders(), risk)

    assert "C" not in out["code"].tolist()


def test_max_orders_per_day():
    risk = RiskConfig(
        max_orders_per_day=1,
        max_gross_exposure_yen=10_000_000,
        max_single_name_exposure_yen=10_000_000,
        max_long_exposure_yen=10_000_000,
        max_short_exposure_yen=10_000_000,
        require_both_sides=False,
    )

    out = apply_order_risk_limits(_orders(), risk)

    assert len(out) == 1


def test_risk_value_yen_used_for_exposure_check():
    """risk_value_yen が value_yen より大きい場合、exposure制限に使われること。"""
    orders = pd.DataFrame(
        [
            {
                "code": "A",
                "side": "BUY",
                "value_yen": 50_000_000,
                "risk_value_yen": 75_000_000,
                "shortable": True,
                "score": 0.9,
            },
            {
                "code": "B",
                "side": "SELL",
                "value_yen": 20_000_000,
                "risk_value_yen": 30_000_000,
                "shortable": True,
                "score": 0.8,
            },
        ]
    )
    risk = RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=100_000_000,
        max_single_name_exposure_yen=60_000_000,
        max_long_exposure_yen=100_000_000,
        max_short_exposure_yen=100_000_000,
        max_net_exposure_yen=100_000_000,
        require_both_sides=False,
        allow_short_without_confirmed_shortability=True,
    )
    out = apply_order_risk_limits(orders, risk)
    # A は risk_value_yen=75M > 60M single name limit により reject
    assert "A" not in out["code"].tolist()
    assert "B" in out["code"].tolist()
