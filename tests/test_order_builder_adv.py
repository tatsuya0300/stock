"""Order builder rolling ADV tests."""

from __future__ import annotations

import pandas as pd

from jp_signal.order_builder import signals_to_orders
from jp_signal.risk import RiskConfig


def test_sizing_uses_rolling_adv_not_last_turnover() -> None:
    dates = pd.bdate_range(end="2026-01-19", periods=20)

    turnovers = [1_000_000.0] * 19 + [100_000_000.0]

    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": d.strftime("%Y-%m-%d"),
                "open": 1000.0,
                "high": 1010.0,
                "low": 990.0,
                "close": 1000.0,
                "adj_open": 1000.0,
                "adj_high": 1010.0,
                "adj_low": 990.0,
                "adj_close": 1000.0,
                "volume": 100000,
                "turnover": t,
            }
            for d, t in zip(dates, turnovers, strict=True)
        ]
    )

    signals = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "score": 1.0,
                "limit_price": None,
            }
        ]
    )

    sizing_cfg = {
        "adv_ratio": 0.001,
        "adv_ratio_cap": 0.01,
        "adv_window": 20,
        "min_adv_periods": 20,
        "market_open_unit_cap": 1000,
    }

    risk_cfg = RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=1_000_000_000,
        max_single_name_exposure_yen=1_000_000_000,
        max_long_exposure_yen=1_000_000_000,
        max_short_exposure_yen=1_000_000_000,
        max_net_exposure_yen=1_000_000_000,
        require_both_sides=False,
        allow_short_without_confirmed_shortability=False,
    )

    orders = signals_to_orders(
        signals,
        prices,
        as_of="2026-01-20",
        sizing_cfg=sizing_cfg,
        risk_cfg=risk_cfg,
        shortability=None,
        universe=None,
        unit=1,
        order_type="MKT_OPEN",
        for_backtest=False,
    )

    assert len(orders) == 1

    # rolling ADV = (19 * 1,000,000 + 100,000,000) / 20 = 5,950,000
    # target = 5,950,000 * 0.001 = 5,950 yen
    # price = 1,000 yen, unit=1 なので qty=5
    assert int(orders.iloc[0]["qty"]) == 5
