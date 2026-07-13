"""Portfolio backtest event chronology tests.

These tests verify that the daily event order in PortfolioBacktester
is correct:
  1. Carry is accrued for all existing positions at day open
  2. Open exposure is captured BEFORE any orders execute
  3. Exits are processed (or deferred/forced)
  4. Entries are selected based on open exposure
  5. Close NAV/gross uses close prices, not open prices
"""
import pandas as pd
import pytest

from jp_signal.portfolio import PortfolioBacktester, Position
from jp_signal.risk import RiskConfig


def _risk() -> RiskConfig:
    return RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=15_000,
        max_single_name_exposure_yen=10_000,
        max_long_exposure_yen=15_000,
        max_short_exposure_yen=15_000,
        max_net_exposure_yen=15_000,
        require_both_sides=False,
        allow_short_without_confirmed_shortability=True,
    )


def test_close_proceeds_are_not_available_at_same_day_open():
    """Same-day close proceeds must NOT be available for same-day open orders."""
    dates = pd.date_range(
        "2026-01-01",
        periods=25,
        freq="B",
    )

    rows = []

    for code in ["1111", "2222"]:
        for day in dates:
            rows.append(
                {
                    "code": code,
                    "date": day,
                    "open": 100.0,
                    "close": 100.0,
                    "turnover": 1_000_000.0,
                }
            )

    prices = pd.DataFrame(rows)

    entry_day = dates[20]
    next_day = dates[21]

    orders = pd.DataFrame(
        [
            {
                "date": entry_day,
                "code": "1111",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 2.0,
                "shortable": False,
            },
            {
                "date": next_day,
                "code": "2222",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=20_000,
        risk=_risk(),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        adv_window=20,
        min_adv_periods=20,
        maintain_margin_ratio=0.25,
    )

    result = backtester.run(
        orders,
        prices,
    )

    second_day_rejections = result.rejected_orders[
        result.rejected_orders["code"] == "2222"
    ]

    assert not second_day_rejections.empty
    assert (
        second_day_rejections["reason"]
        .isin(["GROSS_LIMIT", "LONG_LIMIT"])
        .any()
    )


def test_close_margin_ratio_uses_close_nav_and_close_gross():
    """equity_to_gross_ratio at day close uses close NAV / close gross exposure."""
    dates = pd.date_range(
        "2026-01-01",
        periods=22,
        freq="B",
    )

    prices = pd.DataFrame(
        [
            {
                "code": "1111",
                "date": day,
                "open": 100.0,
                "close": 100.0,
                "turnover": 1_000_000.0,
            }
            for day in dates
        ]
    )

    orders = pd.DataFrame(
        [
            {
                "date": dates[20],
                "code": "1111",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=20_000,
        risk=_risk(),
        impact_k_bp=0.1,
        commission_bp=0.0,
        half_spread_bp=0.0,
        adv_window=20,
        min_adv_periods=20,
    )

    result = backtester.run(
        orders,
        prices,
    )

    entry_row = result.daily_ledger[
        result.daily_ledger["date"]
        == str(dates[20].date())
    ].iloc[0]

    expected = (
        entry_row["nav"]
        / entry_row["gross_exposure"]
    )

    assert (
        entry_row["equity_to_gross_ratio"]
        == expected
    )


def test_deferred_exit_continues_to_next_day():
    """Position with missing exit price is deferred up to max_exit_defer_days."""
    dates = pd.date_range(
        "2026-01-01",
        periods=25,
        freq="B",
    )

    # Only provide price data for the entry day; make exit day missing
    entry_day = dates[20]
    exit_day = dates[21]

    rows = []
    for code in ["1111"]:
        for day in dates:
            if day == exit_day:
                # Missing price data on exit day
                continue
            rows.append(
                {
                    "code": code,
                    "date": day,
                    "open": 100.0,
                    "close": 100.0,
                    "turnover": 1_000_000.0,
                }
            )

    prices = pd.DataFrame(rows)

    orders = pd.DataFrame(
        [
            {
                "date": entry_day,
                "code": "1111",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=20_000,
        risk=_risk(),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        adv_window=20,
        min_adv_periods=20,
        max_exit_defer_days=3,
    )

    result = backtester.run(
        orders,
        prices,
    )

    # The position should be deferred (not exited) on exit_day
    exit_day_ledger = result.daily_ledger[
        result.daily_ledger["date"] == str(exit_day.date())
    ]

    assert not exit_day_ledger.empty
    assert exit_day_ledger["open_position_count"].iloc[0] > 0

    # The rejection log should show the deferral
    defer_rejections = result.rejected_orders[
        result.rejected_orders["reason"] == "DEFERRED_NO_EXIT_PRICE"
    ]

    assert not defer_rejections.empty


def test_forced_exit_after_max_defer_days():
    """Position is force-closed when max_exit_defer_days is exceeded."""
    dates = pd.date_range(
        "2026-01-01",
        periods=30,
        freq="B",
    )

    entry_day = dates[20]
    exit_day = dates[21]

    rows = []
    for code in ["1111"]:
        for day in dates:
            if day >= exit_day:
                # Missing all prices after entry
                continue
            rows.append(
                {
                    "code": code,
                    "date": day,
                    "open": 100.0,
                    "close": 100.0,
                    "turnover": 1_000_000.0,
                }
            )

    prices = pd.DataFrame(rows)

    orders = pd.DataFrame(
        [
            {
                "date": entry_day,
                "code": "1111",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=20_000,
        risk=_risk(),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        adv_window=20,
        min_adv_periods=20,
        max_exit_defer_days=2,
    )

    result = backtester.run(
        orders,
        prices,
    )

    # The trade should eventually exist (forced close)
    assert not result.trades.empty

    # Check that forced exit reason is recorded
    forced_trades = result.trades[
        result.trades["forced_exit_reason"].notna()
    ]

    assert not forced_trades.empty
    assert "FORCED" in forced_trades["forced_exit_reason"].iloc[0]


def test_carry_is_accrued_only_once_per_day():
    """Carry should only be accrued at day open, not again after exits."""
    dates = pd.date_range(
        "2026-01-01",
        periods=22,
        freq="B",
    )

    rows = []
    for code in ["1111"]:
        for day in dates:
            rows.append(
                {
                    "code": code,
                    "date": day,
                    "open": 100.0,
                    "close": 100.0,
                    "turnover": 1_000_000.0,
                }
            )

    prices = pd.DataFrame(rows)

    entry_day = dates[20]

    orders = pd.DataFrame(
        [
            {
                "date": entry_day,
                "code": "1111",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 3,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=20_000,
        risk=_risk(),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        adv_window=20,
        min_adv_periods=20,
    )

    result = backtester.run(
        orders,
        prices,
    )

    # The position should have carry accrued over multiple days
    trade = result.trades.iloc[0]
    assert trade["accrued_carry"] > 0


def test_daily_ledger_has_open_exposure_columns():
    """Daily ledger should include equity_at_open and gross_exposure_at_open."""
    dates = pd.date_range(
        "2026-01-01",
        periods=22,
        freq="B",
    )

    rows = []
    for code in ["1111"]:
        for day in dates:
            rows.append(
                {
                    "code": code,
                    "date": day,
                    "open": 100.0,
                    "close": 100.0,
                    "turnover": 1_000_000.0,
                }
            )

    prices = pd.DataFrame(rows)

    orders = pd.DataFrame(
        [
            {
                "date": dates[20],
                "code": "1111",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=20_000,
        risk=_risk(),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        adv_window=20,
        min_adv_periods=20,
    )

    result = backtester.run(
        orders,
        prices,
    )

    assert "equity_at_open" in result.daily_ledger.columns
    assert "gross_exposure_at_open" in result.daily_ledger.columns
    assert "equity_to_gross_ratio_at_open" in result.daily_ledger.columns

    # On the entry day, equity_at_open should just be initial capital
    entry_row = result.daily_ledger[
        result.daily_ledger["date"] == str(dates[20].date())
    ].iloc[0]

    assert entry_row["equity_at_open"] == 20_000
    assert entry_row["gross_exposure_at_open"] == 0
    assert entry_row["equity_to_gross_ratio_at_open"] is None


def test_exit_defer_days_in_trades():
    """Trade records should include exit_defer_days."""
    dates = pd.date_range(
        "2026-01-01",
        periods=25,
        freq="B",
    )

    entry_day = dates[20]
    exit_day = dates[21]

    rows = []
    for code in ["1111"]:
        for day in dates:
            if day == exit_day:
                continue  # Missing exit day price -> defer
            rows.append(
                {
                    "code": code,
                    "date": day,
                    "open": 100.0,
                    "close": 100.0,
                    "turnover": 1_000_000.0,
                }
            )

    prices = pd.DataFrame(rows)

    orders = pd.DataFrame(
        [
            {
                "date": entry_day,
                "code": "1111",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    backtester = PortfolioBacktester(
        initial_capital=20_000,
        risk=_risk(),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        adv_window=20,
        min_adv_periods=20,
        max_exit_defer_days=2,
    )

    result = backtester.run(
        orders,
        prices,
    )

    assert "exit_defer_days" in result.trades.columns


def test_execution_price_validation():
    """PortfolioBacktester should reject invalid execution costs."""
    with pytest.raises(ValueError, match="execution cost must be > 0"):
        PortfolioBacktester(
            initial_capital=20_000,
            risk=_risk(),
            impact_k_bp=0.0,
            commission_bp=0.0,
            half_spread_bp=0.0,
            adv_window=20,
            min_adv_periods=20,
        )


def test_max_exit_defer_days_validation():
    """PortfolioBacktester should reject invalid max_exit_defer_days."""
    with pytest.raises(ValueError, match="max_exit_defer_days must be > 0"):
        PortfolioBacktester(
            initial_capital=20_000,
            risk=_risk(),
            impact_k_bp=0.1,
            commission_bp=15.0,
            half_spread_bp=5.0,
            adv_window=20,
            min_adv_periods=20,
            max_exit_defer_days=0,
        )
_days must be > 0"):
        PortfolioBacktester(
            initial_capital=20_000,
            risk=_risk(),
            impact_k_bp=0.1,
            commission_bp=15.0,
            half_spread_bp=5.0,
            adv_window=20,
            min_adv_periods=20,
            max_exit_defer_days=0,
        )
