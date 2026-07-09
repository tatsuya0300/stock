"""order_builder tests."""


import pandas as pd

from jp_signal.order_builder import is_shortable_asof, signals_to_orders
from jp_signal.risk import RiskConfig


def test_is_shortable_missing_is_false():
    assert is_shortable_asof(None, "7203", "2024-01-10") is False
    assert is_shortable_asof(pd.DataFrame(), "7203", "2024-01-10") is False


def test_is_shortable_no_lookahead():
    sh = pd.DataFrame(
        [
            {"code": "7203", "date": "2024-01-10", "is_margin_lendable": 1, "short_restricted": 0},
            {"code": "7203", "date": "2024-01-12", "is_margin_lendable": 1, "short_restricted": 0},
        ]
    )
    # 1/11 時点では 1/10 のみ見える
    assert is_shortable_asof(sh, "7203", "2024-01-11") is True
    # 制限あり
    sh2 = pd.DataFrame(
        [{"code": "7203", "date": "2024-01-10", "is_margin_lendable": 1, "short_restricted": 1}]
    )
    assert is_shortable_asof(sh2, "7203", "2024-01-11") is False


def test_signals_to_orders_drops_unshortable_sell():
    prices = pd.DataFrame(
        [
            {
                "code": "7203", "date": "2024-01-09",
                "open": 100, "high": 110, "low": 90,
                "close": 100, "adj_close": 100, "volume": 1e6, "turnover": 1e10,
            },
            {
                "code": "7203", "date": "2024-01-10",
                "open": 100, "high": 110, "low": 90,
                "close": 105, "adj_close": 105, "volume": 1e6, "turnover": 1e10,
            },
        ]
    )
    # 営業日カレンダーに依存するため、as_of を平日に
    sig = pd.DataFrame([{"code": "7203", "side": "SELL", "score": 1.0, "limit_price": None}])
    risk = RiskConfig(allow_short_without_confirmed_shortability=False)
    orders = signals_to_orders(
        sig,
        prices,
        as_of="2024-01-11",
        sizing_cfg={"adv_ratio": 0.001, "adv_ratio_cap": 0.002, "market_open_unit_cap": 50},
        risk_cfg=risk,
        shortability=None,
    )
    assert orders.empty or (orders["side"] != "SELL").all()
