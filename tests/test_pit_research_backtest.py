import pandas as pd

from jp_signal.portfolio import (
    PortfolioBacktester,
)
from jp_signal.risk import RiskConfig


def test_positions_are_carried_to_next_day():
    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2026-07-09",
                "open": 1000.0,
                "close": 1010.0,
                "turnover": 1_000_000_000.0,
            },
            {
                "code": "7203",
                "date": "2026-07-10",
                "open": 1010.0,
                "close": 1020.0,
                "turnover": 1_000_000_000.0,
            },
            {
                "code": "7203",
                "date": "2026-07-13",
                "open": 1020.0,
                "close": 1030.0,
                "turnover": 1_000_000_000.0,
            },
        ]
    )

    orders = pd.DataFrame(
        [
            {
                "date": "2026-07-10",
                "code": "7203",
                "name": "Toyota",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    risk = RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=100_000_000,
        max_single_name_exposure_yen=20_000_000,
        max_long_exposure_yen=100_000_000,
        max_short_exposure_yen=100_000_000,
        max_net_exposure_yen=100_000_000,
        require_both_sides=False,
    )

    bt = PortfolioBacktester(
        initial_capital=10_000_000,
        risk=risk,
        impact_k_bp=0,
        commission_bp=0,
        half_spread_bp=0,
        adv_window=1,
        min_adv_periods=1,
        require_liquidity_data=True,
    )

    result = bt.run(
        orders,
        prices,
        start_date="2026-07-09",
        end_date="2026-07-13",
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    assert trade["entry_date"] == "2026-07-10"
    assert trade["exit_date"] == "2026-07-13"
    assert trade["entry"] == 1010.0
    assert trade["exit"] == 1030.0

    ledger = result.daily_ledger.set_index(
        "date"
    )

    assert (
        ledger.loc[
            "2026-07-10",
            "open_position_count",
        ]
        == 1
    )
    assert (
        ledger.loc[
            "2026-07-13",
            "open_position_count",
        ]
        == 0
    )
