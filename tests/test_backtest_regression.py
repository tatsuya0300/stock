"""バックテストの回帰テスト。"""

import pandas as pd
import pytest

from jp_signal.backtest import Backtester


def make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", "2026-07-03", 100, 101, 99, 100, 1_000, 100_000),
            ("A", "2026-07-06", 100, 102, 99, 101, 1_000, 101_000),
            ("A", "2026-07-07", 101, 103, 100, 102, 1_000, 102_000),
        ],
        columns=[
            "code",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ],
    )


def make_buy_signal(
    *,
    order_type: str = "MKT_OPEN",
    limit_price: float | None = None,
    holding_days: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "A",
                "date": "2026-07-06",
                "side": "BUY",
                "qty": 100,
                "order_type": order_type,
                "limit_price": limit_price,
                "holding_days": holding_days,
            }
        ]
    )


def test_overnight_position_charges_carry() -> None:
    bt = Backtester(
        impact_k_bp=0,
        annual_interest_rate=0.0365,
        require_liquidity_data=True,
        zero_carry_for_intraday=True,
    )

    result = bt.run(
        make_buy_signal(),
        make_prices(),
    )
    row = result.iloc[0]

    assert row["status"] == "FILLED"
    assert row["entry_date"] == "2026-07-06"
    assert row["exit_date"] == "2026-07-07"
    assert row["carry_days"] == 1
    assert row["carry_cost"] == pytest.approx(1.0)
    # PnL = (102 - 100) * 100 - 1.0 = 199.0
    assert row["pnl"] == pytest.approx(199.0)


def test_holding_days_less_than_one_is_rejected() -> None:
    bt = Backtester(
        impact_k_bp=0,
        require_liquidity_data=True,
    )

    result = bt.run(
        make_buy_signal(holding_days=0),
        make_prices(),
    )

    assert result.iloc[0]["status"] == "INVALID_HOLDING_DAYS"


def test_limit_buy_does_not_execute_above_limit() -> None:
    bt = Backtester(
        impact_k_bp=1_000,
        require_liquidity_data=True,
    )

    result = bt.run(
        make_buy_signal(
            order_type="LIMIT",
            limit_price=100.0,
        ),
        make_prices(),
    )

    assert result.iloc[0]["status"] == "NO_FILL"


def test_invalid_exit_liquidity_is_rejected() -> None:
    prices = pd.DataFrame(
        [
            ("A", "2026-07-03", 100, 101, 99, 100, 1_000, 100_000),
            ("A", "2026-07-06", 100, 102, 99, 101, 1_000, float("nan")),
            ("A", "2026-07-07", 101, 103, 100, 102, 1_000, 102_000),
        ],
        columns=[
            "code",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ],
    )

    bt = Backtester(
        impact_k_bp=0,
        require_liquidity_data=True,
        adv_window=1,
    )

    result = bt.run(
        make_buy_signal(),
        prices,
    )

    assert result.iloc[0]["status"] == "NO_EXIT_LIQUIDITY_DATA"


def test_entry_and_exit_date_saved() -> None:
    bt = Backtester(
        impact_k_bp=0,
        require_liquidity_data=True,
    )

    result = bt.run(
        make_buy_signal(),
        make_prices(),
    )
    row = result.iloc[0]

    assert row["entry_date"] == "2026-07-06"
    assert row["exit_date"] == "2026-07-07"


def test_market_impact_bp_raises_on_invalid_adv() -> None:
    bt = Backtester()

    with pytest.raises(ValueError, match="invalid ADV"):
        bt.market_impact_bp(1_000_000, 0.0)

    with pytest.raises(ValueError, match="invalid ADV"):
        bt.market_impact_bp(1_000_000, float("nan"))

    with pytest.raises(ValueError, match="invalid order_value"):
        bt.market_impact_bp(0, 1_000_000)

    with pytest.raises(ValueError, match="invalid order_value"):
        bt.market_impact_bp(float("nan"), 1_000_000)


def test_short_side_carry_uses_lending_rate() -> None:
    prices = make_prices()

    short = pd.DataFrame(
        [
            ("A", "2026-07-03", 1, 0),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    signals = pd.DataFrame(
        [
            {
                "code": "A",
                "date": "2026-07-06",
                "side": "SELL",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "limit_price": None,
                "holding_days": 1,
            }
        ]
    )

    bt = Backtester(
        impact_k_bp=0,
        annual_interest_rate=0.02,
        annual_lending_rate=0.05,
        require_liquidity_data=True,
    )

    result = bt.run(signals, prices, shortability=short)
    row = result.iloc[0]

    assert row["status"] == "FILLED"
    # 貸株レート 5% / 365 * 1日
    expected_carry = 100.0 * (0.05 / 365.0) * 1 * 100
    assert abs(row["carry_cost"] - expected_carry) < 0.01
