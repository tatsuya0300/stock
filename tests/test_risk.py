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
