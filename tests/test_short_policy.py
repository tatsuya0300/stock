"""shortability 運用ルール: 未確認売りを出さない。"""

import pandas as pd

from jp_signal.order_builder import signals_to_orders
from jp_signal.risk import RiskConfig, apply_order_risk_limits


def _prices():
    return pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-09",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 100,
                "adj_close": 100,
                "volume": 1e6,
                "turnover": 1e10,
            },
            {
                "code": "7203",
                "date": "2024-01-10",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "adj_close": 105,
                "volume": 1e6,
                "turnover": 1e10,
            },
        ]
    )


def test_unconfirmed_short_dropped_by_default():
    sig = pd.DataFrame(
        [{"code": "7203", "side": "SELL", "score": 1.0, "limit_price": None}]
    )
    risk = RiskConfig(allow_short_without_confirmed_shortability=False)
    orders = signals_to_orders(
        sig,
        _prices(),
        as_of="2024-01-11",
        sizing_cfg={
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "market_open_unit_cap": 50,
        },
        risk_cfg=risk,
        shortability=None,
    )
    assert orders.empty


def test_confirmed_short_allowed():
    sig = pd.DataFrame(
        [{"code": "7203", "side": "SELL", "score": 1.0, "limit_price": None}]
    )
    sh = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-10",
                "is_margin_lendable": 1,
                "short_restricted": 0,
            }
        ]
    )
    risk = RiskConfig(allow_short_without_confirmed_shortability=False)
    orders = signals_to_orders(
        sig,
        _prices(),
        as_of="2024-01-11",
        sizing_cfg={
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "market_open_unit_cap": 50,
        },
        risk_cfg=risk,
        shortability=sh,
    )
    assert not orders.empty
    assert (orders["side"] == "SELL").all()
    assert bool(orders.iloc[0]["shortable"]) is True


def test_risk_filter_drops_shortable_false():
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "SELL",
                "value_yen": 100_000,
                "shortable": False,
                "score": 1.0,
            }
        ]
    )
    risk = RiskConfig(allow_short_without_confirmed_shortability=False)
    out = apply_order_risk_limits(orders, risk)
    assert out.empty
