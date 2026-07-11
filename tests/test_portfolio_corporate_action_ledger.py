"""Portfolio-level corporate action ledger separation tests."""

from __future__ import annotations

import pandas as pd

from jp_signal.corporate_actions import CorporateAction
from jp_signal.portfolio import PortfolioBacktester
from jp_signal.risk import RiskConfig


def _risk() -> RiskConfig:
    return RiskConfig(
        require_both_sides=False,
        max_net_exposure_yen=100_000_000,
        allow_short_without_confirmed_shortability=True,
    )


def test_dividend_event_does_not_create_extra_daily_ledger_row():
    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100,
                "close": 100,
                "turnover": 1_000_000_000,
            },
            {
                "code": "7203",
                "date": "2024-01-05",
                "open": 100,
                "close": 100,
                "turnover": 1_000_000_000,
            },
            {
                "code": "7203",
                "date": "2024-01-08",
                "open": 100,
                "close": 100,
                "turnover": 1_000_000_000,
            },
        ]
    )

    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "shortable": True,
            }
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=_risk(),
        impact_k_bp=0,
        commission_bp=0,
        half_spread_bp=0,
        adv_window=1,
        min_adv_periods=1,
    )

    result = backtester.run(
        orders,
        prices,
        corporate_actions=[
            CorporateAction(
                code="7203",
                ex_date="2024-01-08",
                action_type="CASH_DIVIDEND",
                amount=10,
            )
        ],
    )

    assert result.daily_ledger["date"].is_unique
    assert "nav" in result.daily_ledger.columns
    assert len(result.corporate_action_events) == 1
    assert result.corporate_action_events.iloc[0]["action_type"] == "CASH_DIVIDEND"


def test_empty_orders_produce_flat_nav():
    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100,
                "close": 100,
                "turnover": 1_000_000_000,
            },
            {
                "code": "7203",
                "date": "2024-01-05",
                "open": 100,
                "close": 100,
                "turnover": 1_000_000_000,
            },
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=_risk(),
        adv_window=1,
        min_adv_periods=1,
    )

    result = backtester.run(
        pd.DataFrame(),
        prices,
    )

    assert len(result.daily_ledger) == 2
    assert (result.daily_ledger["nav"] == 1_000_000).all()
    assert result.trades.empty
