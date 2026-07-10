"""Portfolio-level backtester tests."""

from __future__ import annotations

import pandas as pd

from jp_signal.portfolio import PortfolioBacktester
from jp_signal.risk import RiskConfig


def make_risk(max_net: float = 10_000_000) -> RiskConfig:
    return RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=20_000_000,
        max_single_name_exposure_yen=10_000_000,
        max_long_exposure_yen=10_000_000,
        max_short_exposure_yen=10_000_000,
        max_net_exposure_yen=max_net,
        require_both_sides=False,
        allow_short_without_confirmed_shortability=True,
    )


def make_prices() -> pd.DataFrame:
    values = {
        "7203": [
            ("2024-01-01", 100.0, 100.0),
            ("2024-01-02", 100.0, 101.0),
            ("2024-01-03", 101.0, 103.0),
            ("2024-01-04", 103.0, 102.0),
            ("2024-01-05", 102.0, 104.0),
        ],
        "6758": [
            ("2024-01-01", 200.0, 200.0),
            ("2024-01-02", 200.0, 199.0),
            ("2024-01-03", 199.0, 197.0),
            ("2024-01-04", 197.0, 198.0),
            ("2024-01-05", 198.0, 196.0),
        ],
    }

    rows = []
    for code, code_rows in values.items():
        for dt, open_price, close_price in code_rows:
            rows.append(
                {
                    "code": code,
                    "date": dt,
                    "open": open_price,
                    "high": max(open_price, close_price),
                    "low": min(open_price, close_price),
                    "close": close_price,
                    "adj_open": open_price,
                    "adj_high": max(open_price, close_price),
                    "adj_low": min(open_price, close_price),
                    "adj_close": close_price,
                    "volume": 1_000_000,
                    "turnover": 100_000_000,
                }
            )

    return pd.DataFrame(rows)


def make_backtester(max_net: float = 10_000_000) -> PortfolioBacktester:
    return PortfolioBacktester(
        initial_capital=1_000_000,
        risk=make_risk(max_net),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        annual_interest_rate=0.0,
        annual_lending_rate=0.0,
        adv_window=2,
        min_adv_periods=2,
    )


def test_long_trade_updates_nav():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
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

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    assert trade["entry"] == 101.0
    assert trade["exit"] == 102.0
    assert trade["pnl"] == 100.0

    assert result.daily_ledger.iloc[-1]["nav"] == 1_000_100.0


def test_existing_position_blocks_duplicate_code():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 2,
                "score": 1.0,
                "shortable": False,
            },
            {
                "date": "2024-01-04",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert "EXISTING_POSITION" in set(result.rejected_orders["reason"])


def test_net_limit_rejects_unbalanced_order():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester(max_net=5_000).run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert result.trades.empty
    assert "NET_LIMIT" in set(result.rejected_orders["reason"])


def test_balanced_long_short_passes_net_limit():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
            {
                "date": "2024-01-03",
                "code": "6758",
                "side": "SELL",
                "qty": 50,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": True,
            },
        ]
    )

    result = make_backtester(max_net=1_000).run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 2
    assert result.rejected_orders.empty


def test_nav_accounting_identity():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 2,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    ledger = result.daily_ledger

    expected_nav = ledger["cash"] + ledger["long_exposure"] - ledger["short_exposure"]
    error = (ledger["nav"] - expected_nav).abs()

    assert float(error.max()) < 0.01
